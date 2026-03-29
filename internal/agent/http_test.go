package agent

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// TestHTTPEndpoints tests the agent API endpoints directly using httptest.
// This avoids needing the full Nous server startup (neural classifier, etc.).
func TestHTTPEndpoints(t *testing.T) {
	reg := mockRegistry()
	config := AgentConfig{
		Workspace:    t.TempDir(),
		MaxToolCalls: 50,
		MaxRetries:   1,
	}
	a := NewAgent(reg, config)
	defer a.Stop()

	mux := http.NewServeMux()
	registerAgentEndpoints(mux, a)

	// 1. GET /api/agent/status — idle
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("GET", "/api/agent/status", nil))
	if rec.Code != 200 {
		t.Fatalf("GET status: %d", rec.Code)
	}
	var status AgentStatus
	json.NewDecoder(rec.Body).Decode(&status)
	if status.Running {
		t.Error("agent should not be running initially")
	}

	// 2. POST /api/agent/start — start a goal
	body := `{"goal": "Research Go programming"}`
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("POST", "/api/agent/start", strings.NewReader(body)))
	if rec.Code != 200 {
		t.Fatalf("POST start: %d — %s", rec.Code, rec.Body.String())
	}

	// 3. GET /api/agent/status — should be running
	time.Sleep(50 * time.Millisecond)
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("GET", "/api/agent/status", nil))
	json.NewDecoder(rec.Body).Decode(&status)
	// May already be finished with mock tools, so just check it responded
	if rec.Code != 200 {
		t.Fatalf("GET status while running: %d", rec.Code)
	}

	// 4. POST /api/agent/start — double start should fail
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("POST", "/api/agent/start", strings.NewReader(body)))
	// Either conflict (409) if still running, or 200 if already finished
	if rec.Code != 200 && rec.Code != 409 {
		t.Fatalf("POST double start: unexpected %d", rec.Code)
	}

	// 5. Wait for completion
	deadline := time.After(10 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("agent did not finish within 10s")
		default:
		}
		rec = httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest("GET", "/api/agent/status", nil))
		json.NewDecoder(rec.Body).Decode(&status)
		if !status.Running {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	// 6. GET /api/agent/report — should have a final report
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("GET", "/api/agent/report", nil))
	if rec.Code != 200 {
		t.Fatalf("GET report: %d", rec.Code)
	}
	var reportResp map[string]string
	json.NewDecoder(rec.Body).Decode(&reportResp)
	if !strings.Contains(reportResp["report"], "[COMPLETE]") {
		t.Errorf("report should contain [COMPLETE], got: %s", reportResp["report"])
	}

	// 7. POST /api/agent/stop — stop when already done should be safe
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("POST", "/api/agent/stop", nil))
	if rec.Code != 200 {
		t.Fatalf("POST stop: %d", rec.Code)
	}

	// 8. POST /api/agent/input — input when not paused
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("POST", "/api/agent/input", strings.NewReader(`{"input":"hello"}`)))
	if rec.Code != 200 {
		t.Fatalf("POST input: %d", rec.Code)
	}

	// 9. Method checks — wrong methods should 405
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("GET", "/api/agent/start", nil))
	if rec.Code != 405 {
		t.Errorf("GET on /start should be 405, got %d", rec.Code)
	}
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("POST", "/api/agent/status", nil))
	if rec.Code != 405 {
		t.Errorf("POST on /status should be 405, got %d", rec.Code)
	}
}

// registerAgentEndpoints registers the agent HTTP handlers on a mux.
// This mirrors the exact logic from internal/server/server.go but
// without requiring the full Server struct.
func registerAgentEndpoints(mux *http.ServeMux, a *Agent) {
	mux.HandleFunc("/api/agent/start", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req struct{ Goal string `json:"goal"` }
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		if err := a.Start(req.Goal); err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusConflict)
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "started", "goal": req.Goal})
	})

	mux.HandleFunc("/api/agent/input", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req struct{ Input string `json:"input"` }
		json.NewDecoder(r.Body).Decode(&req)
		a.Resume(req.Input)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "resumed"})
	})

	mux.HandleFunc("/api/agent/status", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(a.Status())
	})

	mux.HandleFunc("/api/agent/stop", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		a.Stop()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "stopped"})
	})

	mux.HandleFunc("/api/agent/report", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"report": a.Report()})
	})
}
