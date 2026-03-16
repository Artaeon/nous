package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/assistant"
	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/cognitive"
)

// Server exposes Nous as an HTTP API for remote access.
// This enables deployment on a server where clients connect via HTTP.
type Server struct {
	board     *blackboard.Blackboard
	perceiver *cognitive.Perceiver
	assistant *assistant.Store
	fastPath  *cognitive.FastPathResponder
	conv      *cognitive.Conversation
	classifier *cognitive.FastPathClassifier
	jobs      *JobManager
	addr      string
	apiKey    string
	server    *http.Server
}

// ChatRequest is the JSON body for POST /api/chat.
type ChatRequest struct {
	Message string `json:"message"`
}

// ChatResponse is the JSON response from POST /api/chat.
type ChatResponse struct {
	Answer   string `json:"answer"`
	Duration int64  `json:"duration_ms"`
}

// StatusResponse is the JSON response from GET /api/status.
type StatusResponse struct {
	Version     string `json:"version"`
	Model       string `json:"model"`
	Uptime      string `json:"uptime"`
	Percepts    int    `json:"percepts"`
	Goals       int    `json:"goals"`
	ToolCount   int    `json:"tool_count"`
	QueuedJobs  int    `json:"queued_jobs"`
	RunningJobs int    `json:"running_jobs"`
}

// JobsResponse is the JSON response for listing background jobs.
type JobsResponse struct {
	Jobs []Job `json:"jobs"`
}

// TodayResponse is the JSON response for assistant inbox and schedule state.
type TodayResponse struct {
	Notifications []assistant.Notification `json:"notifications"`
	Today         []assistant.Task         `json:"today"`
	Upcoming      []assistant.Task         `json:"upcoming"`
}

// TasksResponse is the JSON response for assistant tasks.
type TasksResponse struct {
	Tasks []assistant.Task `json:"tasks"`
}

// PreferencesResponse is the JSON response for assistant preferences.
type PreferencesResponse struct {
	Preferences []assistant.Preference `json:"preferences"`
}

// RoutinesResponse is the JSON response for assistant routines.
type RoutinesResponse struct {
	Routines []assistant.Routine `json:"routines"`
}

// CreateTaskRequest is the JSON body for POST /api/assistant/tasks.
type CreateTaskRequest struct {
	Title      string    `json:"title"`
	DueAt      time.Time `json:"due_at"`
	Recurrence string    `json:"recurrence"`
}

// PreferenceRequest is the JSON body for POST /api/assistant/preferences.
type PreferenceRequest struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

// CreateRoutineRequest is the JSON body for POST /api/assistant/routines.
type CreateRoutineRequest struct {
	Title     string `json:"title"`
	Schedule  string `json:"schedule"`
	TimeOfDay string `json:"time_of_day"`
}

// New creates a Nous HTTP server.
func New(addr string, board *blackboard.Blackboard, perceiver *cognitive.Perceiver, assistantStore *assistant.Store, apiKey string) *Server {
	return &Server{
		board:      board,
		perceiver:  perceiver,
		assistant:  assistantStore,
		classifier: &cognitive.FastPathClassifier{},
		jobs:       NewJobManager(),
		addr:       addr,
		apiKey:     apiKey,
	}
}

// SetFastPath configures the fast/medium path responder and conversation
// so simple queries can skip the full cognitive pipeline.
func (s *Server) SetFastPath(responder *cognitive.FastPathResponder, conv *cognitive.Conversation) {
	s.fastPath = responder
	s.conv = conv
}

// Start begins listening for HTTP requests.
func (s *Server) Start(version, model string, toolCount int) error {
	startTime := time.Now()
	var handler http.Handler = s.newMux(version, model, toolCount, startTime)
	handler = AuthMiddleware(s.apiKey, handler)
	s.server = &http.Server{
		Addr:         s.addr,
		Handler:      handler,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 300 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	return s.server.ListenAndServe()
}

func (s *Server) newMux(version, model string, toolCount int, startTime time.Time) *http.ServeMux {
	mux := http.NewServeMux()

	// CORS middleware
	cors := func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			origin := strings.TrimSpace(r.Header.Get("Origin"))
			w.Header().Set("Vary", "Origin")
			if allowedOrigin(origin) {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
				w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
			}
			if r.Method == "OPTIONS" {
				if origin != "" && !allowedOrigin(origin) {
					http.Error(w, "origin not allowed", http.StatusForbidden)
					return
				}
				w.WriteHeader(http.StatusOK)
				return
			}
			next(w, r)
		}
	}

	// POST /api/chat — send a message, get a response
	// Uses a mutex to serialize requests (one active conversation at a time)
	// and per-request answer keys to avoid cross-request interference.
	var chatMu sync.Mutex
	var reqCounter uint64
	submitPrompt := func(message string) (string, int64) {
		chatMu.Lock()
		defer chatMu.Unlock()

		start := time.Now()

		// Fast/medium path: skip perceiver entirely for simple queries.
		if s.fastPath != nil && s.conv != nil {
			path := s.classifier.ClassifyQuery(message)
			if path == cognitive.PathFast || path == cognitive.PathMedium {
				answer, err := s.fastPath.RespondWithPath(s.conv, message, path)
				if err == nil {
					return answer, time.Since(start).Milliseconds()
				}
				// Fall through to full pipeline on error.
			}
		}

		reqCounter++
		answerKey := fmt.Sprintf("last_answer_%d", reqCounter)
		s.board.Set("answer_key", answerKey)

		s.perceiver.Submit(message)
		answer := waitForAnswer(s.board, 300*time.Second, answerKey)
		return answer, time.Since(start).Milliseconds()
	}
	s.jobs.StartWorker(submitPrompt)
	mux.HandleFunc("/api/chat", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}

		if strings.TrimSpace(req.Message) == "" {
			http.Error(w, "empty message", http.StatusBadRequest)
			return
		}

		answer, duration := submitPrompt(req.Message)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{
			Answer:   answer,
			Duration: duration,
		})
	}))

	// POST /api/jobs — queue a background task for the always-on server worker.
	mux.HandleFunc("/api/jobs", cors(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case "POST":
			var req ChatRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, "invalid JSON", http.StatusBadRequest)
				return
			}
			if strings.TrimSpace(req.Message) == "" {
				http.Error(w, "empty message", http.StatusBadRequest)
				return
			}
			job := s.jobs.Submit(req.Message)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusAccepted)
			json.NewEncoder(w).Encode(job)
		case "GET":
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(JobsResponse{Jobs: s.jobs.List()})
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}))

	// GET/DELETE /api/jobs/{id} — inspect or cancel a queued background task.
	mux.HandleFunc("/api/jobs/", cors(func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimPrefix(r.URL.Path, "/api/jobs/")
		if strings.TrimSpace(id) == "" {
			http.NotFound(w, r)
			return
		}

		switch r.Method {
		case "GET":
			job, ok := s.jobs.Get(id)
			if !ok {
				http.NotFound(w, r)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(job)
		case "DELETE":
			if !s.jobs.Cancel(id) {
				http.Error(w, "job not cancelable", http.StatusConflict)
				return
			}
			w.WriteHeader(http.StatusNoContent)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}))

	// GET /api/status — system status
	mux.HandleFunc("/api/status", cors(func(w http.ResponseWriter, r *http.Request) {
		queuedJobs, runningJobs, _, _ := s.jobs.Stats()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(StatusResponse{
			Version:     version,
			Model:       model,
			Uptime:      time.Since(startTime).Truncate(time.Second).String(),
			Percepts:    len(s.board.Percepts()),
			Goals:       len(s.board.ActiveGoals()),
			ToolCount:   toolCount,
			QueuedJobs:  queuedJobs,
			RunningJobs: runningJobs,
		})
	}))

	// GET /api/health — health check
	mux.HandleFunc("/api/health", cors(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))

	// GET /api/assistant/today — unread reminders and upcoming tasks.
	mux.HandleFunc("/api/assistant/today", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if s.assistant == nil {
			http.Error(w, "assistant store unavailable", http.StatusServiceUnavailable)
			return
		}

		now := time.Now()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(TodayResponse{
			Notifications: s.assistant.UnreadNotifications(),
			Today:         s.assistant.Today(now),
			Upcoming:      s.assistant.Upcoming(10, now),
		})
	}))

	// GET/POST /api/assistant/tasks — list and create persistent assistant tasks.
	mux.HandleFunc("/api/assistant/tasks", cors(func(w http.ResponseWriter, r *http.Request) {
		if s.assistant == nil {
			http.Error(w, "assistant store unavailable", http.StatusServiceUnavailable)
			return
		}

		switch r.Method {
		case "GET":
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(TasksResponse{Tasks: s.assistant.PendingTasks()})
		case "POST":
			var req CreateTaskRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, "invalid JSON", http.StatusBadRequest)
				return
			}
			task, err := s.assistant.AddTask(strings.TrimSpace(req.Title), req.DueAt, strings.TrimSpace(req.Recurrence))
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusCreated)
			json.NewEncoder(w).Encode(task)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}))

	// POST /api/assistant/tasks/{id}/done — mark a task completed.
	mux.HandleFunc("/api/assistant/tasks/", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if s.assistant == nil {
			http.Error(w, "assistant store unavailable", http.StatusServiceUnavailable)
			return
		}

		path := strings.TrimPrefix(r.URL.Path, "/api/assistant/tasks/")
		if !strings.HasSuffix(path, "/done") {
			http.NotFound(w, r)
			return
		}
		id := strings.TrimSuffix(path, "/done")
		id = strings.TrimSuffix(id, "/")
		if strings.TrimSpace(id) == "" {
			http.NotFound(w, r)
			return
		}

		task, err := s.assistant.MarkDone(id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(task)
	}))

	// GET/POST /api/assistant/preferences — inspect and update saved preferences.
	mux.HandleFunc("/api/assistant/preferences", cors(func(w http.ResponseWriter, r *http.Request) {
		if s.assistant == nil {
			http.Error(w, "assistant store unavailable", http.StatusServiceUnavailable)
			return
		}

		switch r.Method {
		case "GET":
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(PreferencesResponse{Preferences: s.assistant.Preferences()})
		case "POST":
			var req PreferenceRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, "invalid JSON", http.StatusBadRequest)
				return
			}
			if err := s.assistant.SetPreference(strings.TrimSpace(req.Key), strings.TrimSpace(req.Value)); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			w.WriteHeader(http.StatusNoContent)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}))

	// GET/POST /api/assistant/routines — list and create recurring assistant routines.
	mux.HandleFunc("/api/assistant/routines", cors(func(w http.ResponseWriter, r *http.Request) {
		if s.assistant == nil {
			http.Error(w, "assistant store unavailable", http.StatusServiceUnavailable)
			return
		}

		switch r.Method {
		case "GET":
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(RoutinesResponse{Routines: s.assistant.Routines()})
		case "POST":
			var req CreateRoutineRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, "invalid JSON", http.StatusBadRequest)
				return
			}
			routine, err := s.assistant.AddRoutine(strings.TrimSpace(req.Title), strings.TrimSpace(req.Schedule), strings.TrimSpace(req.TimeOfDay))
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusCreated)
			json.NewEncoder(w).Encode(routine)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}))

	// POST /api/assistant/notifications/read — acknowledge unread assistant reminders.
	mux.HandleFunc("/api/assistant/notifications/read", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if s.assistant == nil {
			http.Error(w, "assistant store unavailable", http.StatusServiceUnavailable)
			return
		}
		if err := s.assistant.MarkNotificationsRead(); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}))

	// GET / — simple web UI placeholder
	mux.HandleFunc("/", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprint(w, webUI)
	}))

	return mux
}

func allowedOrigin(origin string) bool {
	if origin == "" {
		return false
	}
	parsed, err := url.Parse(origin)
	if err != nil {
		return false
	}
	host := strings.ToLower(parsed.Hostname())
	return host == "localhost" || host == "127.0.0.1" || host == "::1"
}

// Stop gracefully shuts down the server.
func (s *Server) Stop(ctx context.Context) error {
	if s.server == nil {
		return nil
	}
	return s.server.Shutdown(ctx)
}

func waitForAnswer(board *blackboard.Blackboard, timeout time.Duration, keys ...string) string {
	deadline := time.After(timeout)
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	// Check per-request key first, then fallback to "last_answer"
	checkKeys := append(keys, "last_answer")

	for {
		select {
		case <-deadline:
			return "(timeout waiting for response)"
		case <-ticker.C:
			for _, key := range checkKeys {
				if answer, ok := board.Get(key); ok {
					board.Delete(key)
					if s, ok := answer.(string); ok {
						return s
					}
					return fmt.Sprintf("%v", answer)
				}
			}
		}
	}
}

// webUI is the web interface — Apple/Notion-inspired, clean and modern.
const webUI = `<!DOCTYPE html>
<html lang="en">
<head>
  <title>Nous</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg: #fafafa; --bg2: #ffffff; --bg3: #f5f5f5; --bg-hover: #f0f0f0;
      --fg: #1a1a1a; --fg2: #6b6b6b; --fg3: #999;
      --accent: #0066ff; --accent-light: #e8f0ff; --accent-hover: #0052cc;
      --green: #34c759; --yellow: #ff9f0a; --red: #ff3b30; --blue: #007aff;
      --border: #e5e5e5; --border2: #ebebeb;
      --radius: 12px; --radius-sm: 8px;
      --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
      --shadow-lg: 0 4px 12px rgba(0,0,0,0.08);
      --font: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', system-ui, sans-serif;
      --mono: 'SF Mono', 'Fira Code', 'Cascadia Code', 'JetBrains Mono', monospace;
      --transition: 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0f0f0f; --bg2: #1a1a1a; --bg3: #242424; --bg-hover: #2a2a2a;
        --fg: #f0f0f0; --fg2: #999; --fg3: #666;
        --accent: #3b82f6; --accent-light: #1e293b; --accent-hover: #60a5fa;
        --green: #4ade80; --yellow: #fbbf24; --red: #f87171; --blue: #60a5fa;
        --border: #2a2a2a; --border2: #333;
        --shadow: 0 1px 3px rgba(0,0,0,0.3);
        --shadow-lg: 0 4px 12px rgba(0,0,0,0.4);
      }
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body { height: 100%; }
    body { font-family: var(--font); background: var(--bg); color: var(--fg); display: flex; flex-direction: column; -webkit-font-smoothing: antialiased; }

    /* Header */
    .header { padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--border); background: var(--bg2); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); }
    .header .brand { display: flex; align-items: center; gap: 10px; }
    .header .logo { width: 32px; height: 32px; background: var(--accent); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: 700; font-size: 14px; letter-spacing: 1px; }
    .header .title { font-size: 16px; font-weight: 600; letter-spacing: -0.01em; }
    .header .status-bar { display: flex; gap: 16px; font-size: 12px; color: var(--fg2); }
    .header .status-bar .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); display: inline-block; margin-right: 4px; animation: pulse 2s ease-in-out infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

    /* Layout */
    .layout { flex: 1; display: flex; min-height: 0; }
    .chat-container { flex: 1; display: flex; flex-direction: column; min-width: 0; }
    .chat { flex: 1; overflow-y: auto; padding: 24px; scroll-behavior: smooth; }
    .chat::-webkit-scrollbar { width: 6px; }
    .chat::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

    /* Messages */
    .msg { max-width: 720px; margin: 0 auto 16px; animation: fadeUp 0.3s var(--transition); }
    @keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    .msg .label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--fg3); margin-bottom: 6px; }
    .msg .bubble { padding: 14px 18px; border-radius: var(--radius); font-size: 14px; line-height: 1.6; }
    .msg.user .label { text-align: right; }
    .msg.user .bubble { background: var(--accent); color: #fff; border-radius: var(--radius) var(--radius) 4px var(--radius); margin-left: 60px; }
    .msg.nous .bubble { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius) var(--radius) var(--radius) 4px; margin-right: 60px; box-shadow: var(--shadow); }

    /* Thinking indicator */
    .thinking { display: flex; gap: 4px; padding: 4px 0; }
    .thinking span { width: 6px; height: 6px; border-radius: 50%; background: var(--fg3); animation: bounce 1.4s ease-in-out infinite; }
    .thinking span:nth-child(2) { animation-delay: 0.2s; }
    .thinking span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce { 0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; } 40% { transform: scale(1); opacity: 1; } }

    /* Input */
    .input-area { padding: 16px 24px 24px; background: var(--bg2); border-top: 1px solid var(--border); }
    .input-wrap { max-width: 720px; margin: 0 auto; display: flex; gap: 8px; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius); padding: 4px; transition: border-color var(--transition), box-shadow var(--transition); }
    .input-wrap:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(59,130,246,0.15); }
    .input-wrap input { flex: 1; padding: 10px 14px; background: transparent; border: none; color: var(--fg); font-family: var(--font); font-size: 15px; outline: none; }
    .input-wrap input::placeholder { color: var(--fg3); }
    .input-wrap button { padding: 8px 18px; border: none; border-radius: var(--radius-sm); font-family: var(--font); font-size: 13px; font-weight: 600; cursor: pointer; transition: all var(--transition); white-space: nowrap; }
    .input-wrap .btn-send { background: var(--accent); color: #fff; }
    .input-wrap .btn-send:hover { background: var(--accent-hover); transform: scale(1.02); }
    .input-wrap .btn-send:active { transform: scale(0.98); }
    .input-wrap .btn-queue { background: transparent; color: var(--fg2); }
    .input-wrap .btn-queue:hover { background: var(--bg-hover); color: var(--fg); }
    .input-wrap button:disabled { opacity: 0.4; cursor: wait; transform: none !important; }

    /* Sidebar */
    .sidebar { width: 320px; border-left: 1px solid var(--border); background: var(--bg2); overflow-y: auto; padding: 20px; }
    .sidebar::-webkit-scrollbar { width: 4px; }
    .sidebar::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
    .sidebar h2 { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--fg3); margin-bottom: 12px; }
    .job { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 12px 14px; margin-bottom: 8px; transition: all var(--transition); }
    .job:hover { border-color: var(--border2); box-shadow: var(--shadow); }
    .job .meta { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
    .job .id { font-size: 11px; color: var(--fg3); font-family: var(--mono); }
    .job .pill { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; padding: 2px 8px; border-radius: 10px; }
    .job .pill.queued { background: #fef3c7; color: #92400e; }
    .job .pill.running { background: #dbeafe; color: #1e40af; }
    .job .pill.completed { background: #d1fae5; color: #065f46; }
    .job .pill.failed, .job .pill.canceled { background: #fee2e2; color: #991b1b; }
    @media (prefers-color-scheme: dark) {
      .job .pill.queued { background: #422006; color: #fbbf24; }
      .job .pill.running { background: #1e3a5f; color: #60a5fa; }
      .job .pill.completed { background: #064e3b; color: #4ade80; }
      .job .pill.failed, .job .pill.canceled { background: #450a0a; color: #f87171; }
    }
    .job .message { font-size: 13px; margin-bottom: 4px; line-height: 1.4; }
    .job .result { font-size: 12px; color: var(--fg2); line-height: 1.4; white-space: pre-wrap; word-break: break-word; }

    /* Empty state */
    .empty { text-align: center; padding: 60px 24px; color: var(--fg3); }
    .empty .icon { font-size: 48px; margin-bottom: 16px; opacity: 0.3; }
    .empty p { font-size: 14px; line-height: 1.6; }

    /* Responsive */
    @media (max-width: 768px) {
      .sidebar { display: none; }
      .header .status-bar { display: none; }
      .msg.user .bubble { margin-left: 20px; }
      .msg.nous .bubble { margin-right: 20px; }
    }
  </style>
</head>
<body>
  <div class="header">
    <div class="brand">
      <div class="logo">N</div>
      <span class="title">Nous</span>
    </div>
    <div class="status-bar" id="status">
      <span><span class="dot"></span>connecting...</span>
    </div>
  </div>
  <div class="layout">
    <div class="chat-container">
      <div class="chat" id="chat">
        <div class="empty">
          <div class="icon">&#x2728;</div>
          <p><strong>Your personal AI, running locally.</strong><br>Everything stays on your machine. Ask anything.</p>
        </div>
      </div>
      <div class="input-area">
        <div class="input-wrap">
          <input type="text" id="input" placeholder="Message Nous..." autofocus>
          <button class="btn-send" id="send" onclick="send()">Send</button>
          <button class="btn-queue" id="queue" onclick="queueJob()">Queue</button>
        </div>
      </div>
    </div>
    <aside class="sidebar">
      <h2>Background Jobs</h2>
      <div id="jobs">
        <div class="job"><div class="result" style="color:var(--fg3);text-align:center;padding:8px;">No jobs yet</div></div>
      </div>
    </aside>
  </div>
  <script>
    const chat = document.getElementById('chat');
    const jobs = document.getElementById('jobs');
    const input = document.getElementById('input');
    const btn = document.getElementById('send');
    const queueBtn = document.getElementById('queue');
    let firstMsg = true;

    let apiKey = localStorage.getItem('nous_api_key') || '';
    function authHeaders(extra) {
      const h = Object.assign({'Content-Type': 'application/json'}, extra || {});
      if (apiKey) h['Authorization'] = 'Bearer ' + apiKey;
      return h;
    }
    async function authFetch(url, opts) {
      opts = opts || {};
      opts.headers = authHeaders(opts.headers);
      const res = await fetch(url, opts);
      if (res.status === 401) {
        const key = prompt('API key required:');
        if (key) { apiKey = key; localStorage.setItem('nous_api_key', key); opts.headers = authHeaders(); return fetch(url, opts); }
      }
      return res;
    }

    input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) send(); });

    async function send() {
      const msg = input.value.trim();
      if (!msg) return;
      if (firstMsg) { chat.innerHTML = ''; firstMsg = false; }
      input.value = '';
      btn.disabled = true; queueBtn.disabled = true;
      addMsg(msg, 'user');
      const loading = addMsg('<div class="thinking"><span></span><span></span><span></span></div>', 'nous');
      try {
        const res = await authFetch('/api/chat', { method: 'POST', body: JSON.stringify({message: msg}) });
        const data = await res.json();
        loading.querySelector('.bubble').innerHTML = escapeHtml(data.answer || '').replace(/\n/g, '<br>');
      } catch (e) {
        loading.querySelector('.bubble').innerHTML = '<span style="color:var(--red)">Error: ' + escapeHtml(e.message) + '</span>';
      }
      btn.disabled = false; queueBtn.disabled = false;
      input.focus();
    }

    async function queueJob() {
      const msg = input.value.trim();
      if (!msg) return;
      if (firstMsg) { chat.innerHTML = ''; firstMsg = false; }
      input.value = '';
      btn.disabled = true; queueBtn.disabled = true;
      addMsg(msg, 'user');
      addMsg('Queued for background execution.', 'nous');
      try { await authFetch('/api/jobs', { method: 'POST', body: JSON.stringify({message: msg}) }); await refreshJobs(); }
      catch (e) { addMsg('Queue error: ' + e.message, 'nous'); }
      btn.disabled = false; queueBtn.disabled = false;
      input.focus();
    }

    function addMsg(html, cls) {
      const wrap = document.createElement('div');
      wrap.className = 'msg ' + cls;
      const label = cls === 'user' ? 'You' : 'Nous';
      wrap.innerHTML = '<div class="label">' + label + '</div><div class="bubble">' + html + '</div>';
      chat.appendChild(wrap);
      chat.scrollTop = chat.scrollHeight;
      return wrap;
    }

    async function refreshJobs() {
      try {
        const res = await authFetch('/api/jobs');
        const data = await res.json();
        jobs.innerHTML = '';
        if (!data.jobs || data.jobs.length === 0) {
          jobs.innerHTML = '<div class="job"><div class="result" style="color:var(--fg3);text-align:center;padding:8px;">No jobs yet</div></div>';
          return;
        }
        for (const job of data.jobs) {
          const card = document.createElement('div');
          card.className = 'job';
          const preview = (job.result || job.error || '').slice(0, 200);
          card.innerHTML =
            '<div class="meta"><span class="id">' + escapeHtml(job.id).slice(0,8) + '</span><span class="pill ' + escapeHtml(job.status) + '">' + escapeHtml(job.status) + '</span></div>' +
            '<div class="message">' + escapeHtml(job.message) + '</div>' +
            (preview ? '<div class="result">' + escapeHtml(preview) + '</div>' : '');
          jobs.appendChild(card);
        }
      } catch (e) { jobs.innerHTML = '<div class="job"><div class="result">Unable to load jobs.</div></div>'; }
    }

    function escapeHtml(t) { return (t||'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;'); }

    authFetch('/api/status').then(r=>r.json()).then(s=>{
      document.getElementById('status').innerHTML =
        '<span><span class="dot"></span>' + escapeHtml(s.model) + '</span>' +
        '<span>' + s.tool_count + ' tools</span>' +
        '<span>v' + escapeHtml(s.version) + '</span>';
    }).catch(()=>{
      document.getElementById('status').innerHTML = '<span><span class="dot" style="background:var(--red)"></span>offline</span>';
    });
    refreshJobs();
    setInterval(refreshJobs, 5000);
  </script>
</body>
</html>`
