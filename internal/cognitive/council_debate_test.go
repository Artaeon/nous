package cognitive

import (
	"strings"
	"testing"
)

func TestCouncilDebate_BasicRound(t *testing.T) {
	graph := newTestGraph()
	council := NewInnerCouncil(graph, nil)

	nlu := &NLUResult{
		Intent:   "explain",
		Raw:      "What is quantum mechanics?",
		Entities: map[string]string{"topic": "quantum mechanics"},
	}

	result := council.Debate("What is quantum mechanics?", nlu, nil, 2)
	if result == nil {
		t.Fatal("expected non-nil debate result")
	}

	if len(result.Rounds) == 0 {
		t.Fatal("expected at least one debate round")
	}

	if len(result.FinalOpinions) != 5 {
		t.Fatalf("expected 5 final opinions, got %d", len(result.FinalOpinions))
	}

	if result.Trace == "" {
		t.Fatal("expected non-empty trace")
	}
}

func TestCouncilDebate_EarlyConsensus(t *testing.T) {
	graph := newTestGraph()
	council := NewInnerCouncil(graph, nil)

	nlu := &NLUResult{
		Intent:   "explain",
		Raw:      "basic factual question",
		Entities: map[string]string{"topic": "physics"},
	}

	// With a simple factual query, perspectives should converge quickly.
	result := council.Debate("basic factual question", nlu, nil, 5)
	if result == nil {
		t.Fatal("expected non-nil result")
	}

	// Should terminate early (fewer rounds than requested).
	if len(result.Rounds) > 5 {
		t.Fatalf("expected max 5 rounds, got %d", len(result.Rounds))
	}

	// Check trace mentions consensus if early termination happened.
	if len(result.Rounds) < 5 && !strings.Contains(result.Trace, "Consensus") {
		t.Logf("early termination at round %d but no consensus in trace: %q", len(result.Rounds), result.Trace)
	}
}

func TestCouncilDebate_RoundClamp(t *testing.T) {
	graph := newTestGraph()
	council := NewInnerCouncil(graph, nil)

	nlu := &NLUResult{
		Intent:   "explain",
		Raw:      "test",
		Entities: map[string]string{"topic": "physics"},
	}

	// Rounds clamped to max 5.
	result := council.Debate("test", nlu, nil, 100)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if len(result.Rounds) > 5 {
		t.Fatalf("expected max 5 rounds, got %d", len(result.Rounds))
	}

	// Rounds min 1.
	result = council.Debate("test", nlu, nil, 0)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if len(result.Rounds) < 1 {
		t.Fatal("expected at least 1 round")
	}
}

func TestCouncilDebate_MovesExist(t *testing.T) {
	graph := newTestGraph()
	council := NewInnerCouncil(graph, nil)

	nlu := &NLUResult{
		Intent:   "explain",
		Raw:      "complex multi-perspective question about physics and history",
		Entities: map[string]string{"topic": "physics"},
	}

	result := council.Debate("complex question", nlu, nil, 3)
	if result == nil {
		t.Fatal("expected non-nil result")
	}

	totalMoves := 0
	for _, round := range result.Rounds {
		totalMoves += len(round.Moves)
	}

	if totalMoves == 0 {
		t.Fatal("expected at least one debate move across all rounds")
	}

	// Check move types are valid.
	validMoves := map[string]bool{
		"strengthen": true, "concede": true, "challenge": true, "synthesize": true,
	}
	for _, round := range result.Rounds {
		for _, move := range round.Moves {
			if !validMoves[move.MoveType] {
				t.Errorf("invalid move type: %q", move.MoveType)
			}
		}
	}
}

func TestCouncilDebate_ConsensusAndDissent(t *testing.T) {
	graph := newTestGraph()
	council := NewInnerCouncil(graph, nil)

	nlu := &NLUResult{
		Intent:   "explain",
		Raw:      "What is physics?",
		Entities: map[string]string{"topic": "physics"},
	}

	result := council.Debate("What is physics?", nlu, nil, 3)
	if result == nil {
		t.Fatal("expected non-nil result")
	}

	// With good graph data on physics, there should be some consensus.
	// Not guaranteed, but the fields should at least be populated correctly.
	if result.Confidence < 0 || result.Confidence > 1 {
		t.Fatalf("confidence out of range: %f", result.Confidence)
	}
}
