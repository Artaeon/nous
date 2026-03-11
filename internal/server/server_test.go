package server

import (
	"strings"
	"testing"
	"time"

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