package cognitive

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/artaeon/nous/internal/ollama"
)

func TestSaveAndLoadConversation(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "test_conv.json")

	conv := NewConversation(20)
	conv.System("You are helpful.")
	conv.User("Hello there")
	conv.Assistant("Hi! How can I help?")

	if err := SaveConversation(fpath, conv); err != nil {
		t.Fatalf("SaveConversation failed: %v", err)
	}

	loaded, err := LoadConversation(fpath)
	if err != nil {
		t.Fatalf("LoadConversation failed: %v", err)
	}

	origMsgs := conv.Messages()
	loadedMsgs := loaded.Messages()
	if len(loadedMsgs) != len(origMsgs) {
		t.Fatalf("message count mismatch: got %d, want %d", len(loadedMsgs), len(origMsgs))
	}

	for i, m := range origMsgs {
		if loadedMsgs[i].Role != m.Role {
			t.Errorf("message %d role: got %q, want %q", i, loadedMsgs[i].Role, m.Role)
		}
		if loadedMsgs[i].Content != m.Content {
			t.Errorf("message %d content: got %q, want %q", i, loadedMsgs[i].Content, m.Content)
		}
	}
}

func TestSaveConversationNil(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "nil_conv.json")
	err := SaveConversation(fpath, nil)
	if err == nil {
		t.Fatal("expected error for nil conversation")
	}
}

func TestSaveConversationCreatesDirectory(t *testing.T) {
	dir := t.TempDir()
	nested := filepath.Join(dir, "a", "b", "c")
	fpath := filepath.Join(nested, "conv.json")

	conv := NewConversation(10)
	conv.User("test")
	if err := SaveConversation(fpath, conv); err != nil {
		t.Fatalf("SaveConversation should create directories: %v", err)
	}

	if _, err := os.Stat(fpath); os.IsNotExist(err) {
		t.Fatal("file was not created")
	}
}

func TestLoadConversationMissing(t *testing.T) {
	_, err := LoadConversation("/nonexistent/path.json")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestLoadConversationBadJSON(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "bad.json")
	os.WriteFile(fpath, []byte("not json"), 0o600)

	_, err := LoadConversation(fpath)
	if err == nil {
		t.Fatal("expected error for bad JSON")
	}
}

func TestListConversations(t *testing.T) {
	dir := t.TempDir()

	// Save two conversations
	conv1 := NewConversation(10)
	conv1.User("First conversation")
	SaveConversation(filepath.Join(dir, "conv_001.json"), conv1)

	conv2 := NewConversation(10)
	conv2.User("Second conversation")
	SaveConversation(filepath.Join(dir, "conv_002.json"), conv2)

	// Add a non-JSON file that should be ignored
	os.WriteFile(filepath.Join(dir, "readme.txt"), []byte("ignore me"), 0o600)

	metas, err := ListConversations(dir)
	if err != nil {
		t.Fatalf("ListConversations failed: %v", err)
	}

	if len(metas) != 2 {
		t.Fatalf("expected 2 conversations, got %d", len(metas))
	}

	// Verify they have titles derived from first user message
	foundFirst, foundSecond := false, false
	for _, m := range metas {
		if m.Title == "First conversation" {
			foundFirst = true
		}
		if m.Title == "Second conversation" {
			foundSecond = true
		}
	}
	if !foundFirst || !foundSecond {
		t.Errorf("expected titles from user messages, got %v", metas)
	}
}

func TestListConversationsEmptyDir(t *testing.T) {
	dir := t.TempDir()
	metas, err := ListConversations(dir)
	if err != nil {
		t.Fatalf("ListConversations failed: %v", err)
	}
	if len(metas) != 0 {
		t.Fatalf("expected 0 conversations, got %d", len(metas))
	}
}

func TestListConversationsMissingDir(t *testing.T) {
	metas, err := ListConversations("/nonexistent/dir")
	if err != nil {
		t.Fatalf("expected nil error for missing dir, got: %v", err)
	}
	if metas != nil {
		t.Fatalf("expected nil metas, got %v", metas)
	}
}

func TestDeleteConversation(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "to_delete.json")

	conv := NewConversation(10)
	conv.User("ephemeral")
	SaveConversation(fpath, conv)

	if err := DeleteConversation(fpath); err != nil {
		t.Fatalf("DeleteConversation failed: %v", err)
	}

	if _, err := os.Stat(fpath); !os.IsNotExist(err) {
		t.Fatal("file should have been deleted")
	}
}

func TestDeleteConversationMissing(t *testing.T) {
	err := DeleteConversation("/nonexistent/conv.json")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestTitleFromMessages(t *testing.T) {
	tests := []struct {
		name     string
		msgs     []ollama.Message
		expected string
	}{
		{
			name:     "empty",
			msgs:     nil,
			expected: "Untitled conversation",
		},
		{
			name: "system only",
			msgs: []ollama.Message{
				{Role: "system", Content: "You are helpful."},
			},
			expected: "Untitled conversation",
		},
		{
			name: "short user message",
			msgs: []ollama.Message{
				{Role: "system", Content: "system"},
				{Role: "user", Content: "Hello world"},
			},
			expected: "Hello world",
		},
		{
			name: "long user message truncated",
			msgs: []ollama.Message{
				{Role: "user", Content: "This is a very long message that definitely exceeds fifty characters in total length"},
			},
			expected: "This is a very long message that definitely exceed",
		},
		{
			name: "skip summary prefix",
			msgs: []ollama.Message{
				{Role: "user", Content: "[Previous conversation summary]\nsome text"},
				{Role: "user", Content: "Real question"},
			},
			expected: "Real question",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := titleFromMessages(tt.msgs)
			if got != tt.expected {
				t.Errorf("titleFromMessages() = %q, want %q", got, tt.expected)
			}
		})
	}
}

func TestSetMessages(t *testing.T) {
	conv := NewConversation(10)
	msgs := []ollama.Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	}
	conv.SetMessages(msgs)

	got := conv.Messages()
	if len(got) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(got))
	}

	// Verify it's a copy, not the same slice
	msgs[0].Content = "changed"
	if got[0].Content == "changed" {
		t.Error("SetMessages should copy, not alias the input slice")
	}
}

func TestConversationFilePath(t *testing.T) {
	got := ConversationFilePath("/tmp/convos", "conv_123")
	want := filepath.Join("/tmp/convos", "conv_123.json")
	if got != want {
		t.Errorf("ConversationFilePath = %q, want %q", got, want)
	}
}

func TestSavePreservesMetaOnUpdate(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "update.json")

	conv := NewConversation(10)
	conv.User("Original question")
	if err := SaveConversation(fpath, conv); err != nil {
		t.Fatalf("first save failed: %v", err)
	}

	// Load meta to get the ID
	metas, _ := ListConversations(dir)
	if len(metas) != 1 {
		t.Fatalf("expected 1 meta, got %d", len(metas))
	}
	origID := metas[0].ID
	origCreated := metas[0].CreatedAt

	// Update and save again
	conv.Assistant("Here is an answer")
	if err := SaveConversation(fpath, conv); err != nil {
		t.Fatalf("second save failed: %v", err)
	}

	metas2, _ := ListConversations(dir)
	if len(metas2) != 1 {
		t.Fatalf("expected 1 meta after update, got %d", len(metas2))
	}

	if metas2[0].ID != origID {
		t.Error("ID should be preserved across saves")
	}
	if !metas2[0].CreatedAt.Equal(origCreated) {
		t.Error("CreatedAt should be preserved across saves")
	}
	if metas2[0].MessageCount != 2 {
		t.Errorf("expected 2 messages, got %d", metas2[0].MessageCount)
	}
}
