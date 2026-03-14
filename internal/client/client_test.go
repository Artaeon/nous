package client

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestChat(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/api/chat" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{Answer: "hello back", Duration: 42})
	}))
	defer srv.Close()

	c := New(srv.URL, "test-key")
	answer, err := c.Chat("hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if answer != "hello back" {
		t.Errorf("expected 'hello back', got %q", answer)
	}
}

func TestChatSendsAuthHeader(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ChatResponse{Answer: "ok"})
	}))
	defer srv.Close()

	c := New(srv.URL, "secret-123")
	_, _ = c.Chat("test")
	if gotAuth != "Bearer secret-123" {
		t.Errorf("expected 'Bearer secret-123', got %q", gotAuth)
	}
}

func TestStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/status" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(StatusResponse{
			Version:   "0.6.0",
			Model:     "qwen2.5:1.5b",
			Uptime:    "5m0s",
			ToolCount: 10,
		})
	}))
	defer srv.Close()

	c := New(srv.URL, "")
	status, err := c.Status()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if status.Version != "0.6.0" {
		t.Errorf("expected version 0.6.0, got %s", status.Version)
	}
	if status.ToolCount != 10 {
		t.Errorf("expected 10 tools, got %d", status.ToolCount)
	}
}

func TestHealth(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/health" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer srv.Close()

	c := New(srv.URL, "")
	if err := c.Health(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestHealthServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("boom"))
	}))
	defer srv.Close()

	c := New(srv.URL, "")
	err := c.Health()
	if err == nil {
		t.Fatal("expected error for 500 response")
	}
}

func TestListHands(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/hands" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(HandsResponse{
			Hands: []HandInfo{
				{Name: "digest", Description: "Daily digest", Enabled: true},
			},
		})
	}))
	defer srv.Close()

	c := New(srv.URL, "key")
	hands, err := c.ListHands()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(hands) != 1 {
		t.Fatalf("expected 1 hand, got %d", len(hands))
	}
	if hands[0].Name != "digest" {
		t.Errorf("expected hand name 'digest', got %q", hands[0].Name)
	}
}

func TestRunHand(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)
		json.NewEncoder(w).Encode(map[string]string{"status": "triggered"})
	}))
	defer srv.Close()

	c := New(srv.URL, "")
	if err := c.RunHand("digest"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotPath != "/api/hands/digest/run" {
		t.Errorf("expected path /api/hands/digest/run, got %s", gotPath)
	}
}

func TestActivateHand(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	c := New(srv.URL, "")
	if err := c.ActivateHand("research"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotPath != "/api/hands/research/activate" {
		t.Errorf("expected path /api/hands/research/activate, got %s", gotPath)
	}
}

func TestToday(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/assistant/today" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"notifications":[],"today":[],"upcoming":[]}`))
	}))
	defer srv.Close()

	c := New(srv.URL, "")
	today, err := c.Today()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(today.Notifications) != "[]" {
		t.Errorf("expected empty notifications, got %s", string(today.Notifications))
	}
}

func TestCreateTask(t *testing.T) {
	var gotMethod, gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		w.Write([]byte(`{"id":"t1","title":"test task"}`))
	}))
	defer srv.Close()

	c := New(srv.URL, "key")
	if err := c.CreateTask("test task", "2026-03-15T09:00:00Z"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotMethod != "POST" {
		t.Errorf("expected POST, got %s", gotMethod)
	}
	if gotPath != "/api/assistant/tasks" {
		t.Errorf("expected path /api/assistant/tasks, got %s", gotPath)
	}
}

func TestNoAuthHeaderWhenKeyEmpty(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer srv.Close()

	c := New(srv.URL, "")
	_ = c.Health()
	if gotAuth != "" {
		t.Errorf("expected no auth header, got %q", gotAuth)
	}
}
