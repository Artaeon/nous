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
	"github.com/artaeon/nous/internal/memory"
	"github.com/artaeon/nous/internal/tools"
	"github.com/artaeon/nous/internal/training"
)

// Server exposes Nous as an HTTP API for remote access.
// This enables deployment on a server where clients connect via HTTP.
type Server struct {
	board      *blackboard.Blackboard
	perceiver  *cognitive.Perceiver
	assistant  *assistant.Store
	fastPath   *cognitive.FastPathResponder
	conv       *cognitive.Conversation
	classifier *cognitive.FastPathClassifier
	jobs       *JobManager
	addr       string
	apiKey     string
	server     *http.Server

	// Extended data sources for dashboard/memory endpoints
	workingMem  *memory.WorkingMemory
	longTermMem *memory.LongTermMemory
	episodicMem *memory.EpisodicMemory
	toolReg     *tools.Registry
	collector   *training.Collector
	sessions    *cognitive.SessionStore
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

// SetDataSources connects memory, tools, training, and session subsystems
// for the dashboard and memory API endpoints.
func (s *Server) SetDataSources(wm *memory.WorkingMemory, ltm *memory.LongTermMemory, em *memory.EpisodicMemory, tr *tools.Registry, col *training.Collector, sess *cognitive.SessionStore) {
	s.workingMem = wm
	s.longTermMem = ltm
	s.episodicMem = em
	s.toolReg = tr
	s.collector = col
	s.sessions = sess
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

	// POST /api/chat/stream — streaming chat via SSE
	mux.HandleFunc("/api/chat/stream", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		message := strings.TrimSpace(req.Message)
		if message == "" {
			http.Error(w, "empty message", http.StatusBadRequest)
			return
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming not supported", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		chatMu.Lock()
		defer chatMu.Unlock()

		start := time.Now()

		// Try streaming fast/medium path
		if s.fastPath != nil && s.conv != nil {
			path := s.classifier.ClassifyQuery(message)
			if path == cognitive.PathFast || path == cognitive.PathMedium {
				_, err := s.fastPath.RespondStreamWithPathFull(s.conv, message, path, func(token string, done bool) {
					if done {
						// Crystal/greeting hits send full text with done=true — emit text first
						if token != "" {
							tokenJSON, _ := json.Marshal(token)
							fmt.Fprintf(w, "data: {\"t\":%s,\"d\":false}\n\n", tokenJSON)
							flusher.Flush()
						}
						ms := time.Since(start).Milliseconds()
						fmt.Fprintf(w, "data: {\"t\":\"\",\"d\":true,\"ms\":%d}\n\n", ms)
					} else {
						tokenJSON, _ := json.Marshal(token)
						fmt.Fprintf(w, "data: {\"t\":%s,\"d\":false}\n\n", tokenJSON)
					}
					flusher.Flush()
				})
				if err == nil {
					return
				}
			}
		}

		// Full pipeline fallback — non-streaming, send complete answer as single event
		reqCounter++
		answerKey := fmt.Sprintf("last_answer_%d", reqCounter)
		s.board.Set("answer_key", answerKey)
		s.perceiver.Submit(message)
		answer := waitForAnswer(s.board, 300*time.Second, answerKey)

		answerJSON, _ := json.Marshal(answer)
		ms := time.Since(start).Milliseconds()
		fmt.Fprintf(w, "data: {\"t\":%s,\"d\":false}\n\n", answerJSON)
		flusher.Flush()
		fmt.Fprintf(w, "data: {\"t\":\"\",\"d\":true,\"ms\":%d}\n\n", ms)
		flusher.Flush()
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

	// GET /api/dashboard — comprehensive system overview
	mux.HandleFunc("/api/dashboard", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" { http.Error(w, "method not allowed", http.StatusMethodNotAllowed); return }
		dash := map[string]interface{}{
			"version": version, "model": model, "tool_count": toolCount,
			"uptime": time.Since(startTime).Round(time.Second).String(),
		}
		if s.workingMem != nil { dash["working_memory_size"] = s.workingMem.Size() }
		if s.longTermMem != nil { dash["longterm_memory_size"] = s.longTermMem.Size() }
		if s.episodicMem != nil { dash["episodes_total"] = s.episodicMem.Size(); dash["success_rate"] = s.episodicMem.SuccessRate() }
		if s.collector != nil { dash["training_pairs"] = s.collector.Size(); dash["quality_distribution"] = s.collector.QualityDistribution() }
		if s.conv != nil { dash["conversation_messages"] = len(s.conv.Messages()) }
		if s.assistant != nil {
			dash["pending_tasks"] = len(s.assistant.PendingTasks())
			dash["unread_notifications"] = len(s.assistant.UnreadNotifications())
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(dash)
	}))

	// GET /api/memory — working memory items
	mux.HandleFunc("/api/memory", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" { http.Error(w, "method not allowed", http.StatusMethodNotAllowed); return }
		var items []map[string]interface{}
		if s.workingMem != nil {
			for _, slot := range s.workingMem.MostRelevant(20) {
				items = append(items, map[string]interface{}{"key": slot.Key, "value": slot.Value, "relevance": slot.Relevance})
			}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"items": items})
	}))

	// GET /api/longterm — long-term memory entries
	mux.HandleFunc("/api/longterm", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" { http.Error(w, "method not allowed", http.StatusMethodNotAllowed); return }
		var entries []map[string]interface{}
		if s.longTermMem != nil {
			for _, e := range s.longTermMem.All() {
				entries = append(entries, map[string]interface{}{"key": e.Key, "value": e.Value, "category": e.Category, "access_count": e.AccessCount})
			}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"entries": entries})
	}))

	// GET /api/episodes — episodic memory
	mux.HandleFunc("/api/episodes", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" { http.Error(w, "method not allowed", http.StatusMethodNotAllowed); return }
		resp := map[string]interface{}{"total": 0, "success_rate": 0.0, "episodes": []interface{}{}}
		if s.episodicMem != nil {
			resp["total"] = s.episodicMem.Size()
			resp["success_rate"] = s.episodicMem.SuccessRate()
			var eps []map[string]interface{}
			for _, ep := range s.episodicMem.Recent(20) {
				eps = append(eps, map[string]interface{}{"timestamp": ep.Timestamp, "input": ep.Input, "output": ep.Output, "success": ep.Success, "duration_ms": ep.Duration})
			}
			resp["episodes"] = eps
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))

	// GET /api/tools — tool catalog
	mux.HandleFunc("/api/tools", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" { http.Error(w, "method not allowed", http.StatusMethodNotAllowed); return }
		var toolList []map[string]string
		if s.toolReg != nil {
			for _, t := range s.toolReg.List() {
				toolList = append(toolList, map[string]string{"name": t.Name, "description": t.Description})
			}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"tools": toolList})
	}))

	// GET /api/training — training data stats
	mux.HandleFunc("/api/training", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" { http.Error(w, "method not allowed", http.StatusMethodNotAllowed); return }
		resp := map[string]interface{}{"pair_count": 0}
		if s.collector != nil { resp["pair_count"] = s.collector.Size(); resp["quality_distribution"] = s.collector.QualityDistribution() }
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))

	// GET /api/sessions — session list
	mux.HandleFunc("/api/sessions", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" { http.Error(w, "method not allowed", http.StatusMethodNotAllowed); return }
		var sessionList []map[string]interface{}
		if s.sessions != nil {
			if list, err := s.sessions.List(); err == nil {
				for _, sess := range list {
					sessionList = append(sessionList, map[string]interface{}{"id": sess.ID, "name": sess.Name, "message_count": len(sess.Messages), "updated_at": sess.UpdatedAt})
				}
			}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"sessions": sessionList})
	}))

	// GET /api/conversation — current conversation messages
	mux.HandleFunc("/api/conversation", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" { http.Error(w, "method not allowed", http.StatusMethodNotAllowed); return }
		var msgs []map[string]string
		if s.conv != nil {
			for _, m := range s.conv.Messages() {
				msgs = append(msgs, map[string]string{"role": m.Role, "content": m.Content})
			}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"messages": msgs})
	}))

	// GET / — web UI
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

// webUI is defined in webui.go
// (moved to separate file for maintainability)

// webUI HTML is in webui.go

const _legacyUI_unused = `<!DOCTYPE html>
<html lang="en">
<head>
<title>Nous</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--bg4:#30363d;--fg:#e6edf3;--fg2:#8b949e;--fg3:#484f58;--accent:#58a6ff;--accent2:#1f6feb;--green:#3fb950;--yellow:#d29922;--red:#f85149;--cyan:#79c0ff;--magenta:#d2a8ff;--border:#30363d;--font:-apple-system,BlinkMacSystemFont,'SF Pro Text','Segoe UI',system-ui,sans-serif;--mono:'SF Mono','Fira Code','JetBrains Mono','Cascadia Code',Consolas,monospace;--t:0.15s ease}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden}
body{font-family:var(--mono);background:var(--bg);color:var(--fg);font-size:13px;-webkit-font-smoothing:antialiased}

/* Login */
.login{position:fixed;inset:0;background:var(--bg);display:flex;align-items:center;justify-content:center;z-index:100;animation:fadeIn .3s ease}
.login.hidden{display:none}
.login-box{width:360px;text-align:center}
.login-box h1{font-size:28px;font-weight:300;letter-spacing:8px;color:var(--fg);margin-bottom:4px}
.login-box p{color:var(--fg2);font-size:12px;margin-bottom:32px;font-family:var(--font)}
.login-box input{width:100%;padding:12px 16px;background:var(--bg2);border:1px solid var(--border);color:var(--fg);border-radius:8px;font-family:var(--mono);font-size:14px;text-align:center;outline:none;transition:border var(--t)}
.login-box input:focus{border-color:var(--accent)}
.login-box input::placeholder{color:var(--fg3)}
.login-box button{width:100%;margin-top:12px;padding:10px;background:var(--accent2);color:#fff;border:none;border-radius:8px;font-family:var(--font);font-size:14px;font-weight:600;cursor:pointer;transition:background var(--t)}
.login-box button:hover{background:var(--accent)}
.login-box .skip{margin-top:12px;font-size:11px;color:var(--fg3);cursor:pointer;font-family:var(--font)}
.login-box .skip:hover{color:var(--fg2)}
.login-err{color:var(--red);font-size:12px;margin-top:8px;min-height:18px;font-family:var(--font)}

/* App shell */
.app{display:flex;height:100vh;flex-direction:column}
.app.hidden{display:none}

/* Top bar */
.topbar{display:flex;align-items:center;justify-content:space-between;padding:0 16px;height:40px;background:var(--bg2);border-bottom:1px solid var(--border);flex-shrink:0}
.topbar .left{display:flex;align-items:center;gap:12px}
.topbar .logo{color:var(--green);font-weight:700;font-size:13px;letter-spacing:2px}
.topbar .sep{color:var(--fg3)}
.topbar .model{color:var(--fg2);font-size:12px}
.topbar .right{display:flex;gap:16px;font-size:11px;color:var(--fg3)}
.topbar .dot{width:6px;height:6px;border-radius:50%;background:var(--green);display:inline-block;margin-right:4px;animation:pulse 2s ease infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.topbar .tab{padding:4px 10px;border-radius:4px;cursor:pointer;transition:all var(--t);color:var(--fg2)}
.topbar .tab:hover,.topbar .tab.active{background:var(--bg3);color:var(--fg)}

/* Main layout */
.main{flex:1;display:flex;min-height:0}

/* Sidebar */
.side{width:260px;border-right:1px solid var(--border);background:var(--bg2);display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}
.side-section{padding:12px;border-bottom:1px solid var(--border)}
.side-section h3{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--fg3);margin-bottom:8px;font-weight:600}
.side-scroll{flex:1;overflow-y:auto;padding:8px 12px}
.side-scroll::-webkit-scrollbar{width:4px}
.side-scroll::-webkit-scrollbar-thumb{background:var(--bg4);border-radius:2px}
.task-item{display:flex;align-items:flex-start;gap:8px;padding:6px 8px;border-radius:6px;margin-bottom:2px;cursor:pointer;transition:background var(--t);font-size:12px}
.task-item:hover{background:var(--bg3)}
.task-item .check{width:14px;height:14px;border:1.5px solid var(--fg3);border-radius:50%;flex-shrink:0;margin-top:1px;cursor:pointer;transition:all var(--t)}
.task-item .check:hover{border-color:var(--green);background:rgba(63,185,80,.15)}
.task-item.done .check{background:var(--green);border-color:var(--green)}
.task-item.done .text{text-decoration:line-through;color:var(--fg3)}
.task-item .text{flex:1;line-height:1.4}
.task-item .due{font-size:10px;color:var(--fg3);white-space:nowrap}
.task-item .due.overdue{color:var(--red)}
.side-empty{color:var(--fg3);font-size:11px;text-align:center;padding:16px 0}
.job-card{padding:8px 10px;border:1px solid var(--border);border-radius:6px;margin-bottom:6px;background:var(--bg)}
.job-card .jmeta{display:flex;justify-content:space-between;font-size:10px;color:var(--fg3);margin-bottom:4px}
.pill{font-size:9px;font-weight:700;text-transform:uppercase;padding:1px 6px;border-radius:8px;letter-spacing:.03em}
.pill.queued{background:#3d2e00;color:var(--yellow)}.pill.running{background:#0c2d6b;color:var(--cyan)}.pill.completed{background:#0d3117;color:var(--green)}.pill.failed,.pill.canceled{background:#3c1111;color:var(--red)}
.job-card .jmsg{font-size:11px;line-height:1.3}

/* Chat area */
.chat-area{flex:1;display:flex;flex-direction:column;min-width:0}
.output{flex:1;overflow-y:auto;padding:16px 20px;font-size:13px;line-height:1.65}
.output::-webkit-scrollbar{width:6px}
.output::-webkit-scrollbar-thumb{background:var(--bg4);border-radius:3px}

/* Terminal-style messages */
.line{margin-bottom:12px;animation:fadeUp .2s ease;max-width:900px}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
.line .prompt{color:var(--green);font-weight:600}
.line .cmd{color:var(--fg)}
.line .out{color:var(--fg2);white-space:pre-wrap;word-break:break-word;margin-top:4px;padding-left:2px;border-left:2px solid var(--bg4);padding:4px 0 4px 12px}
.line .out.err{color:var(--red);border-left-color:var(--red)}
.line .time{color:var(--fg3);font-size:11px;margin-top:2px}
.line .sys{color:var(--cyan);font-style:italic}
.thinking-dots{display:inline-flex;gap:3px;margin-left:4px}
.thinking-dots span{width:5px;height:5px;border-radius:50%;background:var(--fg3);animation:bounce 1.4s ease infinite}
.thinking-dots span:nth-child(2){animation-delay:.15s}
.thinking-dots span:nth-child(3){animation-delay:.3s}
@keyframes bounce{0%,80%,100%{transform:scale(.5);opacity:.3}40%{transform:scale(1);opacity:1}}

/* Welcome */
.welcome{text-align:center;padding:80px 20px;color:var(--fg3)}
.welcome h2{font-size:18px;font-weight:300;letter-spacing:4px;color:var(--fg2);margin-bottom:8px;font-family:var(--font)}
.welcome p{font-size:12px;line-height:1.8;max-width:400px;margin:0 auto;font-family:var(--font)}
.welcome .cmds{margin-top:24px;display:flex;flex-wrap:wrap;gap:6px;justify-content:center}
.welcome .cmds span{padding:4px 10px;border:1px solid var(--border);border-radius:4px;font-size:11px;color:var(--fg2);cursor:pointer;transition:all var(--t)}
.welcome .cmds span:hover{border-color:var(--accent);color:var(--accent);background:rgba(88,166,255,.08)}

/* Input */
.input-bar{padding:8px 16px 12px;background:var(--bg2);border-top:1px solid var(--border);display:flex;gap:8px;flex-shrink:0}
.input-bar input{flex:1;padding:8px 12px;background:var(--bg);border:1px solid var(--border);color:var(--fg);border-radius:6px;font-family:var(--mono);font-size:13px;outline:none;transition:border var(--t)}
.input-bar input:focus{border-color:var(--accent)}
.input-bar input::placeholder{color:var(--fg3)}
.input-bar button{padding:6px 14px;border:none;border-radius:6px;font-family:var(--mono);font-size:12px;font-weight:600;cursor:pointer;transition:all var(--t)}
.input-bar .btn-go{background:var(--accent2);color:#fff}
.input-bar .btn-go:hover{background:var(--accent)}
.input-bar .btn-q{background:var(--bg3);color:var(--fg2)}
.input-bar .btn-q:hover{background:var(--bg4);color:var(--fg)}
.input-bar button:disabled{opacity:.3;cursor:wait}

/* Bottom bar */
.bottombar{height:24px;background:var(--accent2);display:flex;align-items:center;padding:0 12px;font-size:11px;color:rgba(255,255,255,.85);gap:16px;flex-shrink:0;font-family:var(--font)}
.bottombar .spacer{flex:1}

@media(max-width:768px){.side{display:none}.topbar .right{display:none}}
</style>
</head>
<body>

<!-- Login -->
<div class="login" id="login">
<div class="login-box">
  <h1>NOUS</h1>
  <p>Your personal AI assistant</p>
  <input type="password" id="keyInput" placeholder="API Key" autofocus>
  <button onclick="doLogin()">Connect</button>
  <div class="skip" onclick="skipLogin()">Connect without key (local mode)</div>
  <div class="login-err" id="loginErr"></div>
</div>
</div>

<!-- App -->
<div class="app hidden" id="app">
<div class="topbar">
  <div class="left">
    <span class="logo">NOUS</span>
    <span class="sep">|</span>
    <span class="model" id="modelInfo">connecting...</span>
  </div>
  <div class="right" id="topRight">
    <span><span class="dot"></span>online</span>
  </div>
</div>
<div class="main">
  <div class="side">
    <div class="side-section">
      <h3>Tasks</h3>
      <div id="taskList"><div class="side-empty">No tasks</div></div>
    </div>
    <div class="side-section" style="border-bottom:none">
      <h3>Background Jobs</h3>
    </div>
    <div class="side-scroll" id="jobList">
      <div class="side-empty">No jobs</div>
    </div>
  </div>
  <div class="chat-area">
    <div class="output" id="output">
      <div class="welcome">
        <h2>Welcome</h2>
        <p>Fully local AI. Everything private. Type a message or try a command.</p>
        <div class="cmds" id="quickCmds">
          <span onclick="runCmd('/briefing')">/briefing</span>
          <span onclick="runCmd('/today')">/today</span>
          <span onclick="runCmd('/now')">/now</span>
          <span onclick="runCmd('/tasks')">/tasks</span>
          <span onclick="runCmd('/compass')">/compass</span>
          <span onclick="runCmd('/dashboard')">/dashboard</span>
          <span onclick="runCmd('/help')">/help</span>
        </div>
      </div>
    </div>
    <div class="input-bar">
      <input type="text" id="chatInput" placeholder="nous >" disabled>
      <button class="btn-go" id="btnGo" onclick="handleInput()" disabled>Run</button>
      <button class="btn-q" id="btnQ" onclick="handleQueue()" disabled>Queue</button>
    </div>
  </div>
</div>
<div class="bottombar">
  <span id="bbVersion">v0.9.0</span>
  <span id="bbModel">-</span>
  <span id="bbTools">-</span>
  <span class="spacer"></span>
  <span id="bbTasks">0 tasks</span>
  <span id="bbJobs">0 jobs</span>
  <span id="bbUptime">-</span>
</div>
</div>

<script>
const $ = s => document.getElementById(s);
let apiKey = localStorage.getItem('nous_api_key') || '';
let started = false;

// Auth
function hdr(extra){const h=Object.assign({'Content-Type':'application/json'},extra||{});if(apiKey)h['Authorization']='Bearer '+apiKey;return h}
async function af(url,opts){opts=opts||{};opts.headers=hdr(opts.headers);return fetch(url,opts)}

// Login
$('keyInput').addEventListener('keydown',e=>{if(e.key==='Enter')doLogin()});
async function doLogin(){
  apiKey=$('keyInput').value.trim();
  if(!apiKey){$('loginErr').textContent='Please enter an API key';return}
  try{
    const r=await af('/api/status');
    if(r.status===401){$('loginErr').textContent='Invalid API key';return}
    localStorage.setItem('nous_api_key',apiKey);
    enterApp();
  }catch(e){$('loginErr').textContent='Cannot connect: '+e.message}
}
function skipLogin(){apiKey='';localStorage.removeItem('nous_api_key');enterApp()}

// Check if already authed
(async()=>{
  try{
    const r=await af('/api/health');
    if(r.ok){
      const s=await af('/api/status');
      if(s.ok){enterApp();return}
    }
  }catch(e){}
  $('login').classList.remove('hidden');
})();

function enterApp(){
  $('login').classList.add('hidden');
  $('app').classList.remove('hidden');
  $('chatInput').disabled=false;$('btnGo').disabled=false;$('btnQ').disabled=false;
  $('chatInput').focus();
  loadStatus();loadTasks();loadJobs();
  setInterval(loadJobs,5000);setInterval(loadTasks,10000);setInterval(loadStatus,15000);
}

// Status
async function loadStatus(){
  try{
    const r=await af('/api/status');const s=await r.json();
    $('modelInfo').textContent=s.model+' | '+s.tool_count+' tools';
    $('bbVersion').textContent='v'+s.version;
    $('bbModel').textContent=s.model;
    $('bbTools').textContent=s.tool_count+' tools';
    $('bbUptime').textContent=s.uptime;
    $('bbJobs').textContent=(s.running_jobs+s.queued_jobs)+' jobs';
  }catch(e){$('modelInfo').textContent='offline'}
}

// Tasks
async function loadTasks(){
  try{
    const r=await af('/api/assistant/tasks');const d=await r.json();
    const tl=$('taskList');tl.innerHTML='';
    if(!d.tasks||d.tasks.length===0){tl.innerHTML='<div class="side-empty">No tasks</div>';$('bbTasks').textContent='0 tasks';return}
    $('bbTasks').textContent=d.tasks.length+' tasks';
    d.tasks.forEach(t=>{
      const el=document.createElement('div');
      el.className='task-item'+(t.status==='done'?' done':'');
      const due=t.due_at?new Date(t.due_at):null;
      const overdue=due&&due<new Date()&&t.status!=='done';
      el.innerHTML='<div class="check" onclick="event.stopPropagation();doneTask(\''+esc(t.id)+'\')"></div>'+
        '<span class="text">'+esc(t.title)+'</span>'+
        (due?'<span class="due'+(overdue?' overdue':'')+'">'+fmtDate(due)+'</span>':'');
      tl.appendChild(el);
    });
  }catch(e){}
}
async function doneTask(id){
  await af('/api/assistant/tasks/'+id+'/done',{method:'POST'});
  loadTasks();
  addSys('Task completed.');
}

// Jobs
async function loadJobs(){
  try{
    const r=await af('/api/jobs');const d=await r.json();
    const jl=$('jobList');jl.innerHTML='';
    if(!d.jobs||d.jobs.length===0){jl.innerHTML='<div class="side-empty">No jobs</div>';return}
    d.jobs.forEach(j=>{
      const c=document.createElement('div');c.className='job-card';
      c.innerHTML='<div class="jmeta"><span>'+esc(j.id).slice(0,8)+'</span><span class="pill '+esc(j.status)+'">'+esc(j.status)+'</span></div>'+
        '<div class="jmsg">'+esc(j.message)+'</div>';
      jl.appendChild(c);
    });
  }catch(e){}
}

// Input handling
$('chatInput').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey)handleInput()});

function runCmd(cmd){$('chatInput').value=cmd;handleInput()}

async function handleInput(){
  const msg=$('chatInput').value.trim();if(!msg)return;
  $('chatInput').value='';
  clearWelcome();

  // Local slash commands via API
  if(msg.startsWith('/')){
    addPrompt(msg);
    const handled=await handleSlash(msg);
    if(handled)return;
  }

  // Regular chat
  addPrompt(msg);
  disable(true);
  const el=addThinking();
  try{
    const r=await af('/api/chat',{method:'POST',body:JSON.stringify({message:msg})});
    const d=await r.json();
    el.querySelector('.out').textContent=d.answer||'(no response)';
    if(d.duration_ms)el.querySelector('.time').textContent=d.duration_ms+'ms';
  }catch(e){el.querySelector('.out').className='out err';el.querySelector('.out').textContent='Error: '+e.message}
  disable(false);
}

async function handleQueue(){
  const msg=$('chatInput').value.trim();if(!msg)return;
  $('chatInput').value='';clearWelcome();
  addPrompt(msg);
  try{await af('/api/jobs',{method:'POST',body:JSON.stringify({message:msg})});addSys('Queued for background execution.');loadJobs();}
  catch(e){addOut('Queue error: '+e.message,true)}
  $('chatInput').focus();
}

// Slash command handlers (client-side via REST API)
async function handleSlash(cmd){
  const parts=cmd.split(/\s+/);const c=parts[0].toLowerCase();
  try{
    if(c==='/tasks'){const r=await af('/api/assistant/tasks');const d=await r.json();
      if(!d.tasks||!d.tasks.length){addOut('No pending tasks.');return true}
      addOut(d.tasks.map(t=>(t.status==='done'?'[x]':'[ ]')+' '+t.title+(t.due_at?' ('+fmtDate(new Date(t.due_at))+')':'')+' #'+t.id).join('\n'));return true}
    if(c==='/today'){const r=await af('/api/assistant/today');const d=await r.json();
      let o='';
      if(d.notifications&&d.notifications.length)o+='Notifications:\n'+d.notifications.map(n=>'  '+n.message).join('\n')+'\n\n';
      if(d.today&&d.today.length)o+='Today:\n'+d.today.map(t=>'  '+t.title+(t.due_at?' @ '+new Date(t.due_at).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}):'')).join('\n')+'\n\n';
      if(d.upcoming&&d.upcoming.length)o+='Upcoming:\n'+d.upcoming.map(t=>'  '+t.title+(t.due_at?' - '+fmtDate(new Date(t.due_at)):'')).join('\n');
      addOut(o||'All clear. No tasks or notifications.');return true}
    if(c==='/prefs'){const r=await af('/api/assistant/preferences');const d=await r.json();
      if(!d.preferences||!Object.keys(d.preferences).length){addOut('No preferences set.');return true}
      addOut(Object.entries(d.preferences).map(([k,v])=>k+': '+v).join('\n'));return true}
    if(c==='/pref'&&parts.length>=3){await af('/api/assistant/preferences',{method:'POST',body:JSON.stringify({key:parts[1],value:parts.slice(2).join(' ')})});addSys('Preference saved: '+parts[1]);return true}
    if(c==='/routines'){const r=await af('/api/assistant/routines');const d=await r.json();
      if(!d.routines||!d.routines.length){addOut('No routines configured.');return true}
      addOut(d.routines.map(r=>(r.enabled?'ON ':'OFF ')+r.title+' ('+r.schedule+' @ '+r.time_of_day+')').join('\n'));return true}
    if(c==='/remind'&&parts.length>=2){await af('/api/assistant/tasks',{method:'POST',body:JSON.stringify({title:parts.slice(1).join(' ')})});addSys('Reminder created.');loadTasks();return true}
    if(c==='/done'&&parts[1]){await af('/api/assistant/tasks/'+parts[1]+'/done',{method:'POST'});addSys('Task completed.');loadTasks();return true}
    if(c==='/status'){const r=await af('/api/status');const d=await r.json();
      addOut('Version:  '+d.version+'\nModel:    '+d.model+'\nTools:    '+d.tool_count+'\nUptime:   '+d.uptime+'\nPercepts: '+d.percepts+'\nGoals:    '+d.goals+'\nJobs:     '+d.running_jobs+' running, '+d.queued_jobs+' queued');return true}
  }catch(e){addOut('Command error: '+e.message,true);return true}

  // Unhandled slash commands — send through chat API as regular message
  return false;
}

// Output helpers
function clearWelcome(){const w=document.querySelector('.welcome');if(w)w.remove()}
function addPrompt(text){
  const d=document.createElement('div');d.className='line';
  d.innerHTML='<span class="prompt">nous &gt;</span> <span class="cmd">'+esc(text)+'</span>';
  $('output').appendChild(d);$('output').scrollTop=$('output').scrollHeight;
}
function addThinking(){
  const d=document.createElement('div');d.className='line';
  d.innerHTML='<div class="out"><span class="thinking-dots"><span></span><span></span><span></span></span></div><div class="time"></div>';
  $('output').appendChild(d);$('output').scrollTop=$('output').scrollHeight;return d;
}
function addOut(text,isErr){
  const d=document.createElement('div');d.className='line';
  d.innerHTML='<div class="out'+(isErr?' err':'')+'">'+esc(text)+'</div>';
  $('output').appendChild(d);$('output').scrollTop=$('output').scrollHeight;
}
function addSys(text){
  const d=document.createElement('div');d.className='line';
  d.innerHTML='<div class="sys">'+esc(text)+'</div>';
  $('output').appendChild(d);$('output').scrollTop=$('output').scrollHeight;
}
function disable(v){$('btnGo').disabled=v;$('btnQ').disabled=v;$('chatInput').disabled=v;if(!v)$('chatInput').focus()}
function esc(t){return(t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function fmtDate(d){const now=new Date();const diff=d-now;const days=Math.ceil(diff/864e5);
  if(days===0)return'today';if(days===1)return'tomorrow';if(days===-1)return'yesterday';
  if(days<-1)return Math.abs(days)+'d ago';if(days<7)return'in '+days+'d';
  return d.toLocaleDateString([],{month:'short',day:'numeric'})}
</script>
</body>
</html>`
