package cognitive

import (
	"testing"
	"time"

	"github.com/artaeon/nous/internal/ollama"
)

func TestNewSessionStore(t *testing.T) {
	dir := t.TempDir()
	store := NewSessionStore(dir)
	if store == nil {
		t.Fatal("NewSessionStore returned nil")
	}
}

func TestSaveAndLoad(t *testing.T) {
	dir := t.TempDir()
	store := NewSessionStore(dir)

	session := &Session{
		ID:    "test-123",
		Name:  "Test Session",
		Model: "qwen2.5:1.5b",
		Messages: []ollama.Message{
			{Role: "system", Content: "You are helpful."},
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there!"},
		},
		CreatedAt: time.Now(),
		Metadata:  map[string]string{"project": "nous"},
	}

	// Save
	if err := store.Save(session); err != nil {
		t.Fatalf("failed to save session: %v", err)
	}

	// Load
	loaded, err := store.Load("test-123")
	if err != nil {
		t.Fatalf("failed to load session: %v", err)
	}

	if loaded.ID != "test-123" {
		t.Errorf("expected ID 'test-123', got %q", loaded.ID)
	}
	if loaded.Name != "Test Session" {
		t.Errorf("expected name 'Test Session', got %q", loaded.Name)
	}
	if loaded.Model != "qwen2.5:1.5b" {
		t.Errorf("expected model 'qwen2.5:1.5b', got %q", loaded.Model)
	}
	if len(loaded.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(loaded.Messages))
	}
	if loaded.Messages[1].Content != "Hello" {
		t.Errorf("expected second message 'Hello', got %q", loaded.Messages[1].Content)
	}
	if loaded.Metadata["project"] != "nous" {
		t.Errorf("expected metadata project='nous', got %q", loaded.Metadata["project"])
	}
}

func TestSaveSetsUpdatedAt(t *testing.T) {
	dir := t.TempDir()
	store := NewSessionStore(dir)

	session := &Session{
		ID:        "ts-test",
		CreatedAt: time.Now().Add(-time.Hour),
	}

	before := time.Now()
	store.Save(session)
	after := time.Now()

	loaded, _ := store.Load("ts-test")
	if loaded.UpdatedAt.Before(before) || loaded.UpdatedAt.After(after) {
		t.Error("expected UpdatedAt to be set to approximately now")
	}
}

func TestLoadNonExistent(t *testing.T) {
	dir := t.TempDir()
	store := NewSessionStore(dir)

	_, err := store.Load("nonexistent")
	if err == nil {
		t.Fatal("expected error loading nonexistent session")
	}
}

func TestListSessions(t *testing.T) {
	dir := t.TempDir()
	store := NewSessionStore(dir)

	// Save multiple sessions with different update times
	s1 := &Session{ID: "s1", Name: "First", CreatedAt: time.Now()}
	s2 := &Session{ID: "s2", Name: "Second", CreatedAt: time.Now()}
	s3 := &Session{ID: "s3", Name: "Third", CreatedAt: time.Now()}

	store.Save(s1)
	// Small sleep to ensure different UpdatedAt timestamps
	time.Sleep(10 * time.Millisecond)
	store.Save(s2)
	time.Sleep(10 * time.Millisecond)
	store.Save(s3)

	sessions, err := store.List()
	if err != nil {
		t.Fatalf("failed to list sessions: %v", err)
	}

	if len(sessions) != 3 {
		t.Fatalf("expected 3 sessions, got %d", len(sessions))
	}

	// Should be sorted by most recent first
	if sessions[0].ID != "s3" {
		t.Errorf("expected most recent session first (s3), got %q", sessions[0].ID)
	}
	if sessions[2].ID != "s1" {
		t.Errorf("expected oldest session last (s1), got %q", sessions[2].ID)
	}
}

func TestListSessionsEmpty(t *testing.T) {
	dir := t.TempDir()
	store := NewSessionStore(dir)

	sessions, err := store.List()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(sessions) != 0 {
		t.Errorf("expected 0 sessions, got %d", len(sessions))
	}
}

func TestDeleteSession(t *testing.T) {
	dir := t.TempDir()
	store := NewSessionStore(dir)

	session := &Session{ID: "to-delete", Name: "Temp"}
	store.Save(session)

	// Verify it exists
	_, err := store.Load("to-delete")
	if err != nil {
		t.Fatal("expected session to exist before delete")
	}

	// Delete it
	if err := store.Delete("to-delete"); err != nil {
		t.Fatalf("failed to delete session: %v", err)
	}

	// Verify it's gone
	_, err = store.Load("to-delete")
	if err == nil {
		t.Fatal("expected error loading deleted session")
	}
}

func TestDeleteNonExistent(t *testing.T) {
	dir := t.TempDir()
	store := NewSessionStore(dir)

	err := store.Delete("nonexistent")
	if err == nil {
		t.Fatal("expected error deleting nonexistent session")
	}
}

func TestGenerateSessionID(t *testing.T) {
	id1 := GenerateSessionID()
	if id1 == "" {
		t.Fatal("expected non-empty session ID")
	}

	// Generate another and ensure they're different
	time.Sleep(2 * time.Millisecond)
	id2 := GenerateSessionID()
	if id1 == id2 {
		t.Error("expected different session IDs for different timestamps")
	}
}

func TestSaveAndLoadPreservesMessageOrder(t *testing.T) {
	dir := t.TempDir()
	store := NewSessionStore(dir)

	session := &Session{
		ID: "msg-order",
		Messages: []ollama.Message{
			{Role: "system", Content: "sys"},
			{Role: "user", Content: "q1"},
			{Role: "assistant", Content: "a1"},
			{Role: "user", Content: "q2"},
			{Role: "assistant", Content: "a2"},
		},
	}

	store.Save(session)
	loaded, _ := store.Load("msg-order")

	for i, msg := range session.Messages {
		if loaded.Messages[i].Role != msg.Role {
			t.Errorf("msg[%d] role mismatch: %q != %q", i, loaded.Messages[i].Role, msg.Role)
		}
		if loaded.Messages[i].Content != msg.Content {
			t.Errorf("msg[%d] content mismatch: %q != %q", i, loaded.Messages[i].Content, msg.Content)
		}
	}
}
