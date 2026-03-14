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
func New(addr string, board *blackboard.Blackboard, perceiver *cognitive.Perceiver, assistantStore *assistant.Store) *Server {
	return &Server{
		board:      board,
		perceiver:  perceiver,
		assistant:  assistantStore,
		classifier: &cognitive.FastPathClassifier{},
		jobs:       NewJobManager(),
		addr:       addr,
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
	s.server = &http.Server{
		Addr:         s.addr,
		Handler:      s.newMux(version, model, toolCount, startTime),
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

// webUI is a minimal web interface for the HTTP API.
const webUI = `<!DOCTYPE html>
<html>
<head>
  <title>Nous</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
		body { font-family: 'Courier New', monospace; background: #0a0a0a; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
    .header { padding: 20px; text-align: center; border-bottom: 1px solid #333; }
    .header h1 { font-size: 24px; color: #00ff88; }
    .header p { font-size: 12px; color: #666; margin-top: 4px; }
		.layout { flex: 1; display: grid; grid-template-columns: minmax(0, 2fr) minmax(320px, 1fr); min-height: 0; }
		.chat { overflow-y: auto; padding: 20px; border-right: 1px solid #222; }
		.jobs { overflow-y: auto; padding: 20px; background: #0d0d0d; }
		.jobs h2 { font-size: 14px; margin-bottom: 10px; color: #9bd3b0; }
		.job { border: 1px solid #222; border-radius: 8px; padding: 12px; margin-bottom: 10px; background: #121212; }
		.job .meta { font-size: 11px; color: #777; margin-bottom: 6px; display: flex; justify-content: space-between; gap: 8px; }
		.job .status { font-weight: bold; text-transform: uppercase; }
		.job .status.queued { color: #e7c15a; }
		.job .status.running { color: #7cc7ff; }
		.job .status.completed { color: #55d68a; }
		.job .status.failed, .job .status.canceled { color: #ff7a7a; }
		.job .message { font-size: 13px; margin-bottom: 8px; }
		.job .result { font-size: 12px; color: #bbb; white-space: pre-wrap; }
    .msg { margin: 12px 0; padding: 12px 16px; border-radius: 8px; max-width: 80%; }
    .msg.user { background: #1a1a2e; margin-left: auto; text-align: right; }
    .msg.nous { background: #162016; border-left: 3px solid #00ff88; }
    .input-area { padding: 16px; border-top: 1px solid #333; display: flex; gap: 12px; }
    .input-area input { flex: 1; padding: 12px; background: #111; border: 1px solid #333; color: #e0e0e0; border-radius: 6px; font-family: inherit; font-size: 14px; }
    .input-area input:focus { outline: none; border-color: #00ff88; }
    .input-area button { padding: 12px 24px; background: #00ff88; color: #000; border: none; border-radius: 6px; cursor: pointer; font-family: inherit; font-weight: bold; }
    .input-area button:hover { background: #00cc6a; }
		.input-area button.secondary { background: #1d3526; color: #9feab8; border: 1px solid #2e6b45; }
		.input-area button.secondary:hover { background: #274b35; }
    .input-area button:disabled { background: #333; color: #666; cursor: wait; }
    .spinner { display: inline-block; width: 12px; height: 12px; border: 2px solid #00ff88; border-top: 2px solid transparent; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 8px; }
    @keyframes spin { to { transform: rotate(360deg); } }
		@media (max-width: 960px) {
			.layout { grid-template-columns: 1fr; }
			.chat { border-right: none; border-bottom: 1px solid #222; }
		}
  </style>
</head>
<body>
  <div class="header">
    <h1>N O U S</h1>
    <p>Native Orchestration of Unified Streams</p>
  </div>
	<div class="layout">
		<div class="chat" id="chat"></div>
		<aside class="jobs">
			<h2>Background jobs</h2>
			<div id="jobs"></div>
		</aside>
	</div>
  <div class="input-area">
    <input type="text" id="input" placeholder="Ask Nous anything..." autofocus>
    <button id="send" onclick="send()">Send</button>
		<button id="queue" class="secondary" onclick="queueJob()">Queue</button>
  </div>
  <script>
    const chat = document.getElementById('chat');
		const jobs = document.getElementById('jobs');
    const input = document.getElementById('input');
    const btn = document.getElementById('send');
		const queueBtn = document.getElementById('queue');

    input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

    async function send() {
      const msg = input.value.trim();
      if (!msg) return;
      input.value = '';
      btn.disabled = true;
			queueBtn.disabled = true;

      addMsg(msg, 'user');
      const loading = addMsg('<span class="spinner"></span>thinking...', 'nous');

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({message: msg})
        });
        const data = await res.json();
        loading.innerHTML = data.answer.replace(/\n/g, '<br>');
      } catch (e) {
        loading.innerHTML = 'Error: ' + e.message;
      }
      btn.disabled = false;
			queueBtn.disabled = false;
      input.focus();
    }

		async function queueJob() {
			const msg = input.value.trim();
			if (!msg) return;
			input.value = '';
			btn.disabled = true;
			queueBtn.disabled = true;

			addMsg(msg, 'user');
			addMsg('queued for background execution', 'nous');

			try {
				await fetch('/api/jobs', {
					method: 'POST',
					headers: {'Content-Type': 'application/json'},
					body: JSON.stringify({message: msg})
				});
				await refreshJobs();
			} catch (e) {
				addMsg('Queue error: ' + e.message, 'nous');
			}

			btn.disabled = false;
			queueBtn.disabled = false;
			input.focus();
		}

    function addMsg(text, cls) {
      const div = document.createElement('div');
      div.className = 'msg ' + cls;
      div.innerHTML = text;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
      return div;
    }

		async function refreshJobs() {
			try {
				const res = await fetch('/api/jobs');
				const data = await res.json();
				jobs.innerHTML = '';

				if (!data.jobs || data.jobs.length === 0) {
					jobs.innerHTML = '<div class="job"><div class="result">No background jobs yet.</div></div>';
					return;
				}

				for (const job of data.jobs) {
					const card = document.createElement('div');
					card.className = 'job';
					const preview = (job.result || job.error || '').slice(0, 180);
					card.innerHTML =
						'<div class="meta">' +
							'<span>' + escapeHtml(job.id) + '</span>' +
							'<span class="status ' + escapeHtml(job.status) + '">' + escapeHtml(job.status) + '</span>' +
						'</div>' +
						'<div class="message">' + escapeHtml(job.message) + '</div>' +
						'<div class="result">' + escapeHtml(preview || 'Waiting for output...') + '</div>';
					jobs.appendChild(card);
				}
			} catch (e) {
				jobs.innerHTML = '<div class="job"><div class="result">Unable to load jobs.</div></div>';
			}
		}

		function escapeHtml(text) {
			return text
				.replaceAll('&', '&amp;')
				.replaceAll('<', '&lt;')
				.replaceAll('>', '&gt;');
		}

    fetch('/api/status').then(r=>r.json()).then(s=>{
      document.querySelector('.header p').textContent =
				s.version + ' | ' + s.model + ' | ' + s.tool_count + ' tools | ' + s.running_jobs + ' running | ' + s.queued_jobs + ' queued | up ' + s.uptime;
    });
		refreshJobs();
		setInterval(refreshJobs, 5000);
  </script>
</body>
</html>`
