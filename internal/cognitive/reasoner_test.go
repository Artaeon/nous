package cognitive

import (
	"testing"

	"github.com/artaeon/nous/internal/blackboard"
)

func TestPublishAnswerSetsDefaultAndRequestSpecificKeys(t *testing.T) {
	board := blackboard.New()
	board.Set("answer_key", "last_answer_42")

	r := &Reasoner{Base: Base{Board: board}}
	r.publishAnswer("done")

	if got, ok := board.Get("last_answer"); !ok || got.(string) != "done" {
		t.Fatalf("expected default answer key to be populated, got %v %t", got, ok)
	}
	if got, ok := board.Get("last_answer_42"); !ok || got.(string) != "done" {
		t.Fatalf("expected request-specific answer key to be populated, got %v %t", got, ok)
	}
}

func TestPublishAnswerWithoutRequestSpecificKey(t *testing.T) {
	board := blackboard.New()
	r := &Reasoner{Base: Base{Board: board}}

	r.publishAnswer("fallback only")

	if got, ok := board.Get("last_answer"); !ok || got.(string) != "fallback only" {
		t.Fatalf("expected default answer key, got %v %t", got, ok)
	}
}
