package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/artaeon/nous/internal/assistant"
	"github.com/artaeon/nous/internal/blackboard"
)

func TestWaitForAnswerPrefersRequestSpecificKey(t *testing.T) {
	board := blackboard.New()
	board.Set("last_answer", "fallback")
	board.Set("job-1", "specific")

	got := waitForAnswer(board, 100*time.Millisecond, "job-1")
	if got != "specific" {
		t.Fatalf("waitForAnswer() = %q, want %q", got, "specific")
	}

	if _, ok := board.Get("job-1"); ok {
		t.Fatal("expected request-specific key to be deleted after read")
	}
	if fallback, ok := board.Get("last_answer"); !ok || fallback.(string) != "fallback" {
		t.Fatal("expected fallback answer to remain untouched")
	}
}

func TestWaitForAnswerFallsBackToLastAnswer(t *testing.T) {
	board := blackboard.New()
	board.Set("last_answer", "fallback")

	got := waitForAnswer(board, 100*time.Millisecond, "missing-key")
	if got != "fallback" {
		t.Fatalf("waitForAnswer() = %q, want %q", got, "fallback")
	}
}

func TestWaitForAnswerTimesOut(t *testing.T) {
	board := blackboard.New()
	got := waitForAnswer(board, 20*time.Millisecond, "missing")
	if got != "(timeout waiting for response)" {
		t.Fatalf("waitForAnswer() = %q", got)
	}
}

func TestWebUIContainsBackgroundJobsControls(t *testing.T) {
	checks := []string{"Background jobs", "/api/jobs", "Queue", "refreshJobs"}
	for _, check := range checks {
		if !strings.Contains(webUI, check) {
			t.Fatalf("webUI should contain %q", check)
		}
	}
}

func TestCORSRejectsDisallowedOrigin(t *testing.T) {
	srv := New(":0", blackboard.New(), nil, assistant.NewStore(t.TempDir()))
	mux := srv.newMux("0.6.0", "test-model", 0, time.Now())

	req := httptest.NewRequest(http.MethodOptions, "/api/health", nil)
	req.Header.Set("Origin", "https://evil.example")
	res := httptest.NewRecorder()
	mux.ServeHTTP(res, req)

	if res.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want 403", res.Code)
	}
	if got := res.Header().Get("Access-Control-Allow-Origin"); got != "" {
		t.Fatalf("expected no CORS allow header, got %q", got)
	}
}

func TestCORSAllowsLocalOrigin(t *testing.T) {
	srv := New(":0", blackboard.New(), nil, assistant.NewStore(t.TempDir()))
	mux := srv.newMux("0.6.0", "test-model", 0, time.Now())

	req := httptest.NewRequest(http.MethodOptions, "/api/health", nil)
	req.Header.Set("Origin", "http://localhost:3333")
	res := httptest.NewRecorder()
	mux.ServeHTTP(res, req)

	if res.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", res.Code)
	}
	if got := res.Header().Get("Access-Control-Allow-Origin"); got != "http://localhost:3333" {
		t.Fatalf("allow origin = %q, want localhost origin", got)
	}
}

func TestAssistantTodayEndpointReturnsNotificationsAndTasks(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	now := time.Now()
	_, _ = store.AddTask("Call mom", now.Add(-time.Minute), "")
	_, _ = store.AddTask("Pay rent", now.Add(2*time.Hour), "")
	_, _ = store.TriggerDue(now)

	srv := New(":0", blackboard.New(), nil, store)
	mux := srv.newMux("0.6.0", "test-model", 0, now)

	req := httptest.NewRequest(http.MethodGet, "/api/assistant/today", nil)
	res := httptest.NewRecorder()
	mux.ServeHTTP(res, req)

	if res.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", res.Code)
	}
	var body TodayResponse
	if err := json.NewDecoder(res.Body).Decode(&body); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(body.Notifications) != 1 {
		t.Fatalf("notifications = %d, want 1", len(body.Notifications))
	}
	if len(body.Upcoming) < 1 {
		t.Fatalf("upcoming = %d, want at least 1", len(body.Upcoming))
	}
}

func TestAssistantTaskLifecycleEndpoints(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	srv := New(":0", blackboard.New(), nil, store)
	mux := srv.newMux("0.6.0", "test-model", 0, time.Now())

	createReq := httptest.NewRequest(http.MethodPost, "/api/assistant/tasks", strings.NewReader(`{"title":"Stretch","due_at":"2026-03-12T09:00:00Z","recurrence":"daily"}`))
	createReq.Header.Set("Content-Type", "application/json")
	createRes := httptest.NewRecorder()
	mux.ServeHTTP(createRes, createReq)
	if createRes.Code != http.StatusCreated {
		t.Fatalf("create status = %d, want 201", createRes.Code)
	}
	var created assistant.Task
	if err := json.NewDecoder(createRes.Body).Decode(&created); err != nil {
		t.Fatalf("decode created task: %v", err)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/api/assistant/tasks", nil)
	listRes := httptest.NewRecorder()
	mux.ServeHTTP(listRes, listReq)
	if listRes.Code != http.StatusOK {
		t.Fatalf("list status = %d, want 200", listRes.Code)
	}
	var listed TasksResponse
	if err := json.NewDecoder(listRes.Body).Decode(&listed); err != nil {
		t.Fatalf("decode task list: %v", err)
	}
	if len(listed.Tasks) != 1 || listed.Tasks[0].ID != created.ID {
		t.Fatalf("unexpected task list: %+v", listed.Tasks)
	}

	doneReq := httptest.NewRequest(http.MethodPost, "/api/assistant/tasks/"+created.ID+"/done", nil)
	doneRes := httptest.NewRecorder()
	mux.ServeHTTP(doneRes, doneReq)
	if doneRes.Code != http.StatusOK {
		t.Fatalf("done status = %d, want 200", doneRes.Code)
	}
}

func TestAssistantPreferenceAndReadEndpoints(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	_, _ = store.AddTask("Hydrate", time.Now().Add(-time.Minute), "")
	_, _ = store.TriggerDue(time.Now())
	srv := New(":0", blackboard.New(), nil, store)
	mux := srv.newMux("0.6.0", "test-model", 0, time.Now())

	prefReq := httptest.NewRequest(http.MethodPost, "/api/assistant/preferences", strings.NewReader(`{"key":"language","value":"de"}`))
	prefRes := httptest.NewRecorder()
	mux.ServeHTTP(prefRes, prefReq)
	if prefRes.Code != http.StatusNoContent {
		t.Fatalf("preference status = %d, want 204", prefRes.Code)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/api/assistant/preferences", nil)
	listRes := httptest.NewRecorder()
	mux.ServeHTTP(listRes, listReq)
	if listRes.Code != http.StatusOK {
		t.Fatalf("preferences list status = %d, want 200", listRes.Code)
	}
	var prefs PreferencesResponse
	if err := json.NewDecoder(listRes.Body).Decode(&prefs); err != nil {
		t.Fatalf("decode preferences: %v", err)
	}
	if len(prefs.Preferences) != 1 || prefs.Preferences[0].Value != "de" {
		t.Fatalf("unexpected preferences: %+v", prefs.Preferences)
	}

	readReq := httptest.NewRequest(http.MethodPost, "/api/assistant/notifications/read", nil)
	readRes := httptest.NewRecorder()
	mux.ServeHTTP(readRes, readReq)
	if readRes.Code != http.StatusNoContent {
		t.Fatalf("read status = %d, want 204", readRes.Code)
	}
	if len(srv.assistant.UnreadNotifications()) != 0 {
		t.Fatal("expected notifications to be marked read")
	}
}

func TestAssistantRoutinesEndpointCreatesAndListsRoutines(t *testing.T) {
	store := assistant.NewStore(t.TempDir())
	srv := New(":0", blackboard.New(), nil, store)
	mux := srv.newMux("0.6.0", "test-model", 0, time.Now())

	createReq := httptest.NewRequest(http.MethodPost, "/api/assistant/routines", strings.NewReader(`{"title":"Morning review","schedule":"daily","time_of_day":"08:30"}`))
	createReq.Header.Set("Content-Type", "application/json")
	createRes := httptest.NewRecorder()
	mux.ServeHTTP(createRes, createReq)
	if createRes.Code != http.StatusCreated {
		t.Fatalf("create routine status = %d, want 201", createRes.Code)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/api/assistant/routines", nil)
	listRes := httptest.NewRecorder()
	mux.ServeHTTP(listRes, listReq)
	if listRes.Code != http.StatusOK {
		t.Fatalf("list routines status = %d, want 200", listRes.Code)
	}
	var body RoutinesResponse
	if err := json.NewDecoder(listRes.Body).Decode(&body); err != nil {
		t.Fatalf("decode routines: %v", err)
	}
	if len(body.Routines) != 1 || body.Routines[0].Title != "Morning review" {
		t.Fatalf("unexpected routines: %+v", body.Routines)
	}
}
