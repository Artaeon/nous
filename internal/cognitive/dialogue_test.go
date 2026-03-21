package cognitive

import (
	"strings"
	"testing"
)

func TestDialogueNewManager(t *testing.T) {
	dm := NewDialogueManager()
	if dm.CurrentState != StateGreeting {
		t.Errorf("expected initial state StateGreeting, got %v", dm.CurrentState)
	}
	if dm.TurnCount != 0 {
		t.Errorf("expected TurnCount 0, got %d", dm.TurnCount)
	}
	if dm.Entities == nil {
		t.Error("expected Entities map to be initialized")
	}
}

func TestDialogueGreetingToExploration(t *testing.T) {
	dm := NewDialogueManager()

	// First turn: greeting
	nlu := &NLUResult{Action: "respond", Raw: "hello", Entities: map[string]string{}}
	ctx := dm.ProcessTurn(nlu, nil)
	if ctx.State != StateGreeting {
		t.Errorf("expected StateGreeting after 'hello', got %v", ctx.State)
	}

	// Second turn: a question — should transition to exploration
	nlu2 := &NLUResult{Action: "respond", Raw: "what is Go?", Entities: map[string]string{"topic": "Go"}}
	ctx2 := dm.ProcessTurn(nlu2, nil)
	if ctx2.State != StateExploration {
		t.Errorf("expected StateExploration after question, got %v", ctx2.State)
	}
}

func TestDialogueExplorationToDeepDive(t *testing.T) {
	dm := NewDialogueManager()
	dm.CurrentState = StateExploration

	// Same topic for 2 turns should trigger deep dive
	nlu1 := &NLUResult{Action: "respond", Raw: "tell me about Go", Entities: map[string]string{"topic": "Go"}}
	dm.ProcessTurn(nlu1, nil)

	nlu2 := &NLUResult{Action: "respond", Raw: "more about Go", Entities: map[string]string{"topic": "Go"}}
	ctx := dm.ProcessTurn(nlu2, nil)
	if ctx.State != StateDeepDive {
		t.Errorf("expected StateDeepDive after 2 turns on same topic, got %v", ctx.State)
	}
}

func TestDialogueTopicStackPushPop(t *testing.T) {
	dm := NewDialogueManager()

	dm.PushTopic("Go")
	dm.PushTopic("concurrency")
	dm.PushTopic("goroutines")

	if dm.CurrentTopic() != "goroutines" {
		t.Errorf("expected top topic 'goroutines', got %q", dm.CurrentTopic())
	}

	popped := dm.PopTopic()
	if popped != "goroutines" {
		t.Errorf("expected popped 'goroutines', got %q", popped)
	}
	if dm.CurrentTopic() != "concurrency" {
		t.Errorf("expected top topic 'concurrency', got %q", dm.CurrentTopic())
	}

	// Pop empty stack
	dm.PopTopic()
	dm.PopTopic()
	popped = dm.PopTopic()
	if popped != "" {
		t.Errorf("expected empty string from empty stack, got %q", popped)
	}
}

func TestDialogueReferenceResolution(t *testing.T) {
	dm := NewDialogueManager()
	dm.PushTopic("quantum computing")

	resolved := dm.ResolveReference("tell me more about it")
	if !strings.Contains(resolved, "quantum computing") {
		t.Errorf("expected 'it' resolved to 'quantum computing', got %q", resolved)
	}

	resolved2 := dm.ResolveReference("what is that used for")
	if !strings.Contains(resolved2, "quantum computing") {
		t.Errorf("expected 'that' resolved to 'quantum computing', got %q", resolved2)
	}

	// No topic — should return input unchanged
	dm2 := NewDialogueManager()
	input := "tell me about it"
	if dm2.ResolveReference(input) != input {
		t.Errorf("expected unchanged input when no topic set")
	}
}

func TestDialogueFollowUpGeneration(t *testing.T) {
	dm := NewDialogueManager()
	followUps := dm.SuggestFollowUps("machine learning", []string{"type: supervised", "framework: TensorFlow"})

	if len(followUps) == 0 {
		t.Error("expected non-empty follow-ups")
	}
	if len(followUps) > 3 {
		t.Errorf("expected at most 3 follow-ups, got %d", len(followUps))
	}

	// Each follow-up should mention the topic
	for _, fu := range followUps {
		if !strings.Contains(fu, "machine learning") {
			t.Errorf("follow-up should mention topic, got %q", fu)
		}
	}

	// Empty topic should return nil
	empty := dm.SuggestFollowUps("", nil)
	if empty != nil {
		t.Errorf("expected nil follow-ups for empty topic, got %v", empty)
	}
}

func TestDialogueAmbiguityDetection(t *testing.T) {
	dm := NewDialogueManager()

	// Short ambiguous input with multiple candidates
	q := dm.DetectAmbiguity("go", []string{"Go programming", "Go game"})
	if q == "" {
		t.Error("expected clarification question for ambiguous input")
	}
	if dm.CurrentState != StateClarifying {
		t.Errorf("expected StateClarifying after ambiguity, got %v", dm.CurrentState)
	}

	// Unambiguous input — single candidate
	q2 := dm.DetectAmbiguity("Go programming language", []string{"Go programming"})
	if q2 != "" {
		t.Errorf("expected no clarification for single candidate, got %q", q2)
	}
}

func TestDialogueFarewellDetection(t *testing.T) {
	dm := NewDialogueManager()
	dm.CurrentState = StateExploration

	nlu := &NLUResult{Action: "respond", Raw: "goodbye", Entities: map[string]string{}}
	ctx := dm.ProcessTurn(nlu, nil)
	if ctx.State != StateFarewell {
		t.Errorf("expected StateFarewell, got %v", ctx.State)
	}
}

func TestDialogueTransitionPhrases(t *testing.T) {
	dm := NewDialogueManager()

	// Greeting state
	dm.CurrentState = StateGreeting
	phrase := dm.GetTransitionPhrase()
	if phrase != "Welcome!" {
		t.Errorf("expected 'Welcome!' for greeting, got %q", phrase)
	}

	// Deep dive with topic
	dm.CurrentState = StateDeepDive
	dm.PushTopic("neural networks")
	phrase = dm.GetTransitionPhrase()
	if !strings.Contains(phrase, "neural networks") {
		t.Errorf("expected deep dive phrase to mention topic, got %q", phrase)
	}

	// Farewell
	dm.CurrentState = StateFarewell
	phrase = dm.GetTransitionPhrase()
	if phrase != "Until next time!" {
		t.Errorf("expected farewell phrase, got %q", phrase)
	}
}

func TestDialogueMultipleTopicSwitches(t *testing.T) {
	dm := NewDialogueManager()
	dm.CurrentState = StateExploration

	topics := []string{"Go", "Rust", "Python", "Java"}
	for _, topic := range topics {
		nlu := &NLUResult{Action: "respond", Raw: "tell me about " + topic, Entities: map[string]string{"topic": topic}}
		dm.ProcessTurn(nlu, nil)
	}

	// Should have all topics on the stack
	if len(dm.TopicStack) != len(topics) {
		t.Errorf("expected %d topics on stack, got %d", len(topics), len(dm.TopicStack))
	}
	if dm.CurrentTopic() != "Java" {
		t.Errorf("expected current topic 'Java', got %q", dm.CurrentTopic())
	}
}

func TestDialogueProcessTurnAdvancesState(t *testing.T) {
	dm := NewDialogueManager()

	// Turn 1: greeting
	ctx := dm.ProcessTurn(&NLUResult{Action: "respond", Raw: "hi", Entities: map[string]string{}}, nil)
	if ctx.State != StateGreeting {
		t.Errorf("turn 1: expected StateGreeting, got %v", ctx.State)
	}

	// Turn 2: actionable (calculator)
	ctx = dm.ProcessTurn(&NLUResult{Action: "calculate", Raw: "what is 2+2", Entities: map[string]string{"expr": "2+2"}}, nil)
	if ctx.State != StateActionable {
		t.Errorf("turn 2: expected StateActionable, got %v", ctx.State)
	}

	// Turn 3: reflection
	ctx = dm.ProcessTurn(&NLUResult{Action: "respond", Raw: "what do you think about AI ethics", Entities: map[string]string{"topic": "AI ethics"}}, nil)
	if ctx.State != StateReflection {
		t.Errorf("turn 3: expected StateReflection, got %v", ctx.State)
	}

	// Turn count should be 3
	if dm.TurnCount != 3 {
		t.Errorf("expected TurnCount 3, got %d", dm.TurnCount)
	}
}

func TestDialogueEmptyInput(t *testing.T) {
	dm := NewDialogueManager()

	// Nil NLU should not panic
	ctx := dm.ProcessTurn(nil, nil)
	if ctx == nil {
		t.Fatal("expected non-nil context for nil NLU")
	}
	if ctx.State != StateGreeting {
		t.Errorf("expected state unchanged (StateGreeting), got %v", ctx.State)
	}

	// Empty raw input
	ctx = dm.ProcessTurn(&NLUResult{Action: "", Raw: "", Entities: map[string]string{}}, nil)
	if ctx == nil {
		t.Fatal("expected non-nil context for empty input")
	}
}

func TestDialogueDuplicateTopicNotStacked(t *testing.T) {
	dm := NewDialogueManager()
	dm.PushTopic("Go")
	dm.PushTopic("Go")
	dm.PushTopic("Go")

	if len(dm.TopicStack) != 1 {
		t.Errorf("expected 1 topic (deduped), got %d", len(dm.TopicStack))
	}
}

func TestDialogueStateString(t *testing.T) {
	tests := []struct {
		state DialogueState
		want  string
	}{
		{StateGreeting, "greeting"},
		{StateExploration, "exploration"},
		{StateDeepDive, "deep_dive"},
		{StateClarifying, "clarifying"},
		{StateActionable, "actionable"},
		{StateReflection, "reflection"},
		{StateFarewell, "farewell"},
		{DialogueState(99), "unknown"},
	}
	for _, tt := range tests {
		if got := tt.state.String(); got != tt.want {
			t.Errorf("DialogueState(%d).String() = %q, want %q", tt.state, got, tt.want)
		}
	}
}
