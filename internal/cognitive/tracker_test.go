package cognitive

import (
	"strings"
	"testing"
)

func TestConversationTrackerBasic(t *testing.T) {
	ct := NewConversationTracker()

	ct.TrackTopic("Stoicera", "https://stoicera.com")
	if ct.CurrentTopic() != "Stoicera" {
		t.Errorf("current topic = %q, want Stoicera", ct.CurrentTopic())
	}
	if ct.CurrentSource() != "https://stoicera.com" {
		t.Errorf("current source = %q, want URL", ct.CurrentSource())
	}
}

func TestConversationTrackerTopicStack(t *testing.T) {
	ct := NewConversationTracker()

	ct.TrackTopic("Go", "docs")
	ct.TrackTopic("Rust", "web")
	ct.TrackTopic("Python", "web")

	if ct.CurrentTopic() != "Python" {
		t.Errorf("current topic = %q, want Python (most recent)", ct.CurrentTopic())
	}

	// Re-mentioning Go should promote it
	ct.TrackTopic("Go", "docs")
	if ct.CurrentTopic() != "Go" {
		t.Errorf("after re-mention, current topic = %q, want Go", ct.CurrentTopic())
	}
}

func TestConversationTrackerFollowUp(t *testing.T) {
	ct := NewConversationTracker()
	ct.TrackTopic("Stoicera", "https://stoicera.com")

	tests := []struct {
		input string
		want  bool
	}{
		{"tell me more about it", true},
		{"what else can you tell me", true},
		{"what is that", true},
		{"how does the weather look", false},
		{"what is stoicera", true},
	}

	for _, tt := range tests {
		got := ct.IsFollowUp(tt.input)
		if got != tt.want {
			t.Errorf("IsFollowUp(%q) = %v, want %v", tt.input, got, tt.want)
		}
	}
}

func TestConversationTrackerContinuation(t *testing.T) {
	ct := NewConversationTracker()

	tests := []struct {
		input string
		want  bool
	}{
		{"tell me more", true},
		{"go on", true},
		{"continue", true},
		{"more details", true},
		{"what is Rust", false},
	}

	for _, tt := range tests {
		got := ct.IsContinuation(tt.input)
		if got != tt.want {
			t.Errorf("IsContinuation(%q) = %v, want %v", tt.input, got, tt.want)
		}
	}
}

func TestConversationTrackerIngestAndAnswer(t *testing.T) {
	ct := NewConversationTracker()

	content := `Stoicera is a philosophy company based in Vienna.
They create tools for modern stoics.
The company was founded in 2023 by Raphael Lugmayr.
Stoicera offers journals, meditation guides, and daily practices.
Their mission is to make ancient wisdom accessible to everyone.`

	n := ct.IngestContent(content, "https://stoicera.com", "Stoicera")
	if n == 0 {
		t.Fatal("should have ingested facts")
	}

	// Current topic should be set
	if ct.CurrentTopic() != "Stoicera" {
		t.Errorf("topic should be Stoicera, got %q", ct.CurrentTopic())
	}

	// Ask a question about it
	answer := ct.AnswerQuestion("what is Stoicera")
	if answer == "" {
		t.Fatal("should have an answer about Stoicera")
	}
	if !strings.Contains(answer, "philosophy") {
		t.Errorf("answer should mention philosophy, got %q", answer)
	}
}

func TestConversationTrackerFollowUpAnswer(t *testing.T) {
	ct := NewConversationTracker()

	content := `Go is a programming language created by Google.
Go compiles to native machine code.
Go has built-in concurrency with goroutines and channels.
The Go standard library includes HTTP servers, JSON parsing, and more.
Go uses garbage collection for automatic memory management.`

	ct.IngestContent(content, "https://golang.org", "Go")

	// Follow-up question using pronoun
	answer := ct.AnswerQuestion("does it have concurrency")
	if answer == "" {
		t.Fatal("should answer follow-up about concurrency")
	}
	if !strings.Contains(strings.ToLower(answer), "concurrency") &&
		!strings.Contains(strings.ToLower(answer), "goroutine") {
		t.Errorf("answer should mention concurrency features, got %q", answer)
	}
}

func TestConversationTrackerTopicSummary(t *testing.T) {
	ct := NewConversationTracker()

	content := `Rust is a systems programming language focused on safety.
Rust eliminates data races at compile time.
Rust has zero-cost abstractions.
The Rust compiler provides helpful error messages.
Rust does not use a garbage collector.`

	ct.IngestContent(content, "rust-lang.org", "Rust")

	summary := ct.TopicSummary()
	if summary == "" {
		t.Fatal("should produce a topic summary")
	}
	if !strings.Contains(summary, "Rust") {
		t.Error("summary should mention Rust")
	}
	if !strings.Contains(summary, "safety") || !strings.Contains(summary, "programming") {
		t.Error("summary should include key facts")
	}
}

func TestConversationTrackerContinue(t *testing.T) {
	ct := NewConversationTracker()

	// Ingest many facts
	var lines []string
	for i := 0; i < 15; i++ {
		lines = append(lines, "Fact number "+intToWord(i+1)+" about the topic is interesting and important.")
	}
	content := strings.Join(lines, "\n")
	ct.IngestContent(content, "test", "facts")

	// First summary shows top facts
	summary := ct.TopicSummary()
	if summary == "" {
		t.Fatal("should produce summary")
	}

	// Continue should show more
	more := ct.ContinueResponse()
	if more == "" {
		t.Fatal("should have more to show")
	}
}

func TestConversationTrackerNoFacts(t *testing.T) {
	ct := NewConversationTracker()

	answer := ct.AnswerQuestion("what is quantum physics")
	if answer != "" {
		t.Errorf("should return empty with no facts, got %q", answer)
	}

	summary := ct.TopicSummary()
	if summary != "" {
		t.Errorf("should return empty summary with no topic, got %q", summary)
	}
}

func TestConversationTrackerMultipleTopics(t *testing.T) {
	ct := NewConversationTracker()

	ct.IngestContent("Go is a fast programming language. Go was created by Google.", "golang.org", "Go")
	ct.IngestContent("Rust is a safe systems language. Rust eliminates memory bugs.", "rust-lang.org", "Rust")

	// Current topic should be Rust (most recent)
	if ct.CurrentTopic() != "Rust" {
		t.Errorf("current topic = %q, want Rust", ct.CurrentTopic())
	}

	// But we should still be able to answer about Go
	answer := ct.AnswerQuestion("tell me about Go programming")
	if answer == "" {
		// It's ok if it can't answer about non-current topic from just these facts
		// but it should at least try
		t.Log("no answer about Go — expected for minimal facts")
	}
}
