package cognitive

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// ConversationState Tests
// -----------------------------------------------------------------------

func TestConversationState_Update(t *testing.T) {
	cs := NewConversationState()

	nlu := &NLUResult{
		Intent:   "explain",
		Action:   "respond",
		Entities: map[string]string{"topic": "Go concurrency"},
		Raw:      "explain Go concurrency",
	}

	cs.Update("explain Go concurrency", nlu, "Go concurrency uses goroutines...")

	if cs.TurnCount != 1 {
		t.Errorf("expected TurnCount=1, got %d", cs.TurnCount)
	}
	if cs.ActiveTopic != "Go concurrency" {
		t.Errorf("expected ActiveTopic='Go concurrency', got %q", cs.ActiveTopic)
	}
	if cs.LastIntent != "explain" {
		t.Errorf("expected LastIntent='explain', got %q", cs.LastIntent)
	}
	if cs.MentionedEntities["topic"] != "Go concurrency" {
		t.Errorf("expected entity topic='Go concurrency', got %q", cs.MentionedEntities["topic"])
	}

	// Second turn
	nlu2 := &NLUResult{
		Intent:   "explain",
		Action:   "respond",
		Entities: map[string]string{"topic": "channels"},
		Raw:      "what about channels?",
	}
	cs.Update("what about channels?", nlu2, "Channels are typed conduits...")

	if cs.TurnCount != 2 {
		t.Errorf("expected TurnCount=2, got %d", cs.TurnCount)
	}
	if cs.ActiveTopic != "channels" {
		t.Errorf("expected ActiveTopic='channels', got %q", cs.ActiveTopic)
	}
}

func TestConversationState_TopicStack(t *testing.T) {
	cs := NewConversationState()

	cs.PushTopic("Go")
	cs.PushTopic("concurrency")
	cs.PushTopic("channels")

	if len(cs.TopicStack) != 3 {
		t.Fatalf("expected 3 topics, got %d", len(cs.TopicStack))
	}

	// Most recent first
	if cs.TopicStack[0] != "channels" {
		t.Errorf("expected TopicStack[0]='channels', got %q", cs.TopicStack[0])
	}
	if cs.TopicStack[1] != "concurrency" {
		t.Errorf("expected TopicStack[1]='concurrency', got %q", cs.TopicStack[1])
	}
	if cs.TopicStack[2] != "Go" {
		t.Errorf("expected TopicStack[2]='Go', got %q", cs.TopicStack[2])
	}

	// Empty topic should be ignored
	cs.PushTopic("")
	if len(cs.TopicStack) != 3 {
		t.Errorf("empty topic should not be pushed, got %d topics", len(cs.TopicStack))
	}
}

func TestConversationState_EntityTracking(t *testing.T) {
	cs := NewConversationState()

	cs.TrackEntity("city", "Paris")
	cs.TrackEntity("language", "French")
	cs.TrackEntity("person", "Marie")

	if cs.MentionedEntities["city"] != "Paris" {
		t.Errorf("expected city=Paris, got %q", cs.MentionedEntities["city"])
	}
	if cs.MentionedEntities["language"] != "French" {
		t.Errorf("expected language=French, got %q", cs.MentionedEntities["language"])
	}

	// Overwrite with newer value
	cs.TrackEntity("city", "Lyon")
	if cs.MentionedEntities["city"] != "Lyon" {
		t.Errorf("expected city=Lyon after update, got %q", cs.MentionedEntities["city"])
	}

	// Empty entity should be ignored
	cs.TrackEntity("", "value")
	cs.TrackEntity("key", "")
	if _, ok := cs.MentionedEntities[""]; ok {
		t.Error("empty key should not be tracked")
	}
}

func TestConversationState_ReferenceResolution(t *testing.T) {
	cs := NewConversationState()

	// Set up context
	cs.ActiveTopic = "Python"
	cs.TrackEntity("person", "Guido")
	cs.TrackEntity("city", "Amsterdam")
	cs.updateCoreferences("")

	tests := []struct {
		ref  string
		want string
	}{
		{"it", "Python"},
		{"this", "Python"},
		{"that", "Python"},
		{"there", "Amsterdam"},
		// Person pronouns
		{"they", "Guido"},
		{"them", "Guido"},
		// Unknown reference returns itself
		{"foobar", "foobar"},
	}

	for _, tt := range tests {
		got := cs.ResolveReference(tt.ref)
		if got != tt.want {
			t.Errorf("ResolveReference(%q) = %q, want %q", tt.ref, got, tt.want)
		}
	}

	// Test "the former" / "the latter" with topic stack
	cs.TopicStack = []string{"Rust", "Go"}
	cs.updateCoreferences("")

	if got := cs.ResolveReference("the latter"); got != "Rust" {
		t.Errorf("ResolveReference('the latter') = %q, want 'Rust'", got)
	}
	if got := cs.ResolveReference("the former"); got != "Go" {
		t.Errorf("ResolveReference('the former') = %q, want 'Go'", got)
	}
}

func TestConversationState_SlotFilling(t *testing.T) {
	cs := NewConversationState()

	// Add unresolved slots
	cs.UnresolvedSlots["city"] = "destination city"
	cs.UnresolvedSlots["language"] = "programming language"

	if !cs.NeedsSlot("city") {
		t.Error("expected city slot to be needed")
	}
	if !cs.NeedsSlot("language") {
		t.Error("expected language slot to be needed")
	}
	if cs.NeedsSlot("unknown") {
		t.Error("unknown slot should not be needed")
	}

	// Fill from entities
	cs.FillUnresolvedSlots("I want to learn python", map[string]string{"language": "python"})

	if cs.NeedsSlot("language") {
		t.Error("language slot should be filled after entity match")
	}
	if cs.MentionedEntities["language"] != "python" {
		t.Errorf("expected entity language='python', got %q", cs.MentionedEntities["language"])
	}

	// City slot still needed
	if !cs.NeedsSlot("city") {
		t.Error("city slot should still be needed")
	}

	// Fill manually
	cs.SetSlot("city", "Tokyo")
	if cs.NeedsSlot("city") {
		t.Error("city slot should be filled after SetSlot")
	}
}

func TestConversationState_MultiTurn(t *testing.T) {
	cs := NewConversationState()

	// Turn 1: Initial question
	nlu1 := &NLUResult{
		Intent:   "explain",
		Action:   "respond",
		Entities: map[string]string{"topic": "machine learning"},
		Raw:      "what is machine learning?",
	}
	cs.Update("what is machine learning?", nlu1, "Machine learning is a subset of AI...")

	if cs.TurnCount != 1 {
		t.Errorf("turn 1: expected TurnCount=1, got %d", cs.TurnCount)
	}
	if cs.ActiveTopic != "machine learning" {
		t.Errorf("turn 1: expected topic='machine learning', got %q", cs.ActiveTopic)
	}

	// Turn 2: Follow-up
	nlu2 := &NLUResult{
		Intent:   "explain",
		Action:   "respond",
		Entities: map[string]string{"topic": "neural networks"},
		Raw:      "tell me more about neural networks",
	}
	cs.Update("tell me more about neural networks", nlu2, "Neural networks are...")

	if cs.TurnCount != 2 {
		t.Errorf("turn 2: expected TurnCount=2, got %d", cs.TurnCount)
	}
	if cs.ActiveTopic != "neural networks" {
		t.Errorf("turn 2: expected topic='neural networks', got %q", cs.ActiveTopic)
	}
	if cs.FollowUpCount != 1 {
		t.Errorf("turn 2: expected FollowUpCount=1, got %d", cs.FollowUpCount)
	}

	// Topic stack should have both topics
	if len(cs.TopicStack) < 2 {
		t.Fatalf("expected at least 2 topics in stack, got %d", len(cs.TopicStack))
	}
	if cs.TopicStack[0] != "neural networks" {
		t.Errorf("expected TopicStack[0]='neural networks', got %q", cs.TopicStack[0])
	}
	if cs.TopicStack[1] != "machine learning" {
		t.Errorf("expected TopicStack[1]='machine learning', got %q", cs.TopicStack[1])
	}

	// Turn 3: Clarification
	nlu3 := &NLUResult{
		Intent:   "clarify",
		Action:   "respond",
		Entities: map[string]string{},
		Raw:      "what do you mean by that?",
	}
	cs.Update("what do you mean by that?", nlu3, "Let me clarify...")

	if cs.ClarificationCount != 1 {
		t.Errorf("turn 3: expected ClarificationCount=1, got %d", cs.ClarificationCount)
	}

	// Turn 4: Correction
	nlu4 := &NLUResult{
		Intent:   "correct",
		Action:   "respond",
		Entities: map[string]string{},
		Raw:      "no, that's not right. I meant supervised learning.",
	}
	cs.Update("no, that's not right. I meant supervised learning.", nlu4, "I see, supervised learning...")

	if cs.CorrectionCount != 1 {
		t.Errorf("turn 4: expected CorrectionCount=1, got %d", cs.CorrectionCount)
	}

	// Verify context output
	ctx := cs.ActiveContext()
	if !strings.Contains(ctx, "topic:") {
		t.Error("ActiveContext should contain topic")
	}
	if !strings.Contains(ctx, "turns: 4") {
		t.Errorf("ActiveContext should show 4 turns, got: %s", ctx)
	}
}

func TestConversationState_ActiveContext(t *testing.T) {
	cs := NewConversationState()

	// Empty state
	ctx := cs.ActiveContext()
	if ctx != "no active context" {
		t.Errorf("empty state context should be 'no active context', got %q", ctx)
	}

	// With some state
	cs.ActiveTopic = "Go"
	cs.UserObjective = "learn Go"
	cs.TurnCount = 3
	cs.TrackEntity("language", "Go")

	ctx = cs.ActiveContext()
	if !strings.Contains(ctx, "topic: Go") {
		t.Errorf("context should contain topic, got %q", ctx)
	}
	if !strings.Contains(ctx, "objective: learn Go") {
		t.Errorf("context should contain objective, got %q", ctx)
	}
	if !strings.Contains(ctx, "turns: 3") {
		t.Errorf("context should contain turns, got %q", ctx)
	}
}

func TestConversationState_RecordAssumption(t *testing.T) {
	cs := NewConversationState()

	cs.RecordAssumption("user is a beginner")
	cs.RecordAssumption("question is about Go, not the game")

	if len(cs.Assumptions) != 2 {
		t.Errorf("expected 2 assumptions, got %d", len(cs.Assumptions))
	}

	// Empty assumption should be ignored
	cs.RecordAssumption("")
	if len(cs.Assumptions) != 2 {
		t.Error("empty assumption should not be recorded")
	}
}

func TestConversationState_FillUnresolvedSlots_Keywords(t *testing.T) {
	cs := NewConversationState()

	// Test programming language detection
	cs.UnresolvedSlots["language"] = "which programming language"
	cs.FillUnresolvedSlots("I want to learn rust programming", map[string]string{})

	if cs.NeedsSlot("language") {
		t.Error("language slot should have been filled with 'rust'")
	}
}

// -----------------------------------------------------------------------
// FollowUpResolver Tests
// -----------------------------------------------------------------------

func TestFollowUpResolver_Deeper(t *testing.T) {
	fr := NewFollowUpResolver()
	state := NewConversationState()
	state.ActiveTopic = "Go concurrency"
	state.TurnCount = 1

	tests := []string{
		"tell me more",
		"go deeper",
		"elaborate",
		"more detail please",
		"expand on that",
		"keep going",
	}

	for _, input := range tests {
		result := fr.Resolve(input, state)
		if !result.IsFollowUp {
			t.Errorf("'%s' should be detected as follow-up", input)
		}
		if result.Type != FollowUpDeeper {
			t.Errorf("'%s' should be FollowUpDeeper, got %v", input, result.Type)
		}
		if !strings.Contains(result.ResolvedQuery, "Go concurrency") {
			t.Errorf("'%s' resolved query should contain prior topic, got %q", input, result.ResolvedQuery)
		}
		if len(result.CarryOver) == 0 {
			t.Errorf("'%s' should carry over prior topic", input)
		}
	}
}

func TestFollowUpResolver_Pivot(t *testing.T) {
	fr := NewFollowUpResolver()
	state := NewConversationState()
	state.ActiveTopic = "Python"
	state.TurnCount = 1

	tests := []struct {
		input     string
		newEntity string
	}{
		{"what about JavaScript?", "javascript"},
		{"how about Rust?", "rust"},
		{"and TypeScript?", "typescript"},
	}

	for _, tt := range tests {
		result := fr.Resolve(tt.input, state)
		if !result.IsFollowUp {
			t.Errorf("'%s' should be detected as follow-up", tt.input)
		}
		if result.Type != FollowUpPivot {
			t.Errorf("'%s' should be FollowUpPivot, got %v", tt.input, result.Type)
		}
		if !strings.EqualFold(result.NewEntity, tt.newEntity) {
			t.Errorf("'%s' new entity should be %q, got %q", tt.input, tt.newEntity, result.NewEntity)
		}
		if !strings.Contains(result.ResolvedQuery, "Python") {
			t.Errorf("'%s' resolved query should reference prior topic 'Python', got %q", tt.input, result.ResolvedQuery)
		}
	}
}

func TestFollowUpResolver_Compare(t *testing.T) {
	fr := NewFollowUpResolver()
	state := NewConversationState()
	state.ActiveTopic = "Python"
	state.TurnCount = 1

	tests := []struct {
		input     string
		newEntity string
	}{
		{"compare with Rust", "Rust"},
		{"how does it differ from Java?", "Java"},
		{"versus Go?", "Go"},
	}

	for _, tt := range tests {
		result := fr.Resolve(tt.input, state)
		if !result.IsFollowUp {
			t.Errorf("'%s' should be detected as follow-up", tt.input)
		}
		if result.Type != FollowUpCompare {
			t.Errorf("'%s' should be FollowUpCompare, got %v", tt.input, result.Type)
		}
		if !strings.EqualFold(result.NewEntity, tt.newEntity) {
			t.Errorf("'%s' new entity should be %q, got %q", tt.input, tt.newEntity, result.NewEntity)
		}
		if !strings.Contains(result.ResolvedQuery, "Python") {
			t.Errorf("'%s' resolved query should contain prior topic, got %q", tt.input, result.ResolvedQuery)
		}
		if !strings.Contains(strings.ToLower(result.ResolvedQuery), "compare") {
			t.Errorf("'%s' resolved query should contain 'compare', got %q", tt.input, result.ResolvedQuery)
		}
	}
}

func TestFollowUpResolver_Clarify(t *testing.T) {
	fr := NewFollowUpResolver()
	state := NewConversationState()
	state.ActiveTopic = "quantum computing"
	state.TurnCount = 1

	tests := []struct {
		input         string
		expectTerm    bool
		expectedTerm  string
	}{
		{"what do you mean by superposition?", true, "superposition"},
		{"can you clarify that?", false, ""},
		{"i don't understand", false, ""},
	}

	for _, tt := range tests {
		result := fr.Resolve(tt.input, state)
		if !result.IsFollowUp {
			t.Errorf("'%s' should be detected as follow-up", tt.input)
		}
		if result.Type != FollowUpClarify {
			t.Errorf("'%s' should be FollowUpClarify, got %v", tt.input, result.Type)
		}
		if tt.expectTerm {
			if !strings.Contains(strings.ToLower(result.ResolvedQuery), strings.ToLower(tt.expectedTerm)) {
				t.Errorf("'%s' resolved query should contain %q, got %q", tt.input, tt.expectedTerm, result.ResolvedQuery)
			}
		}
		if !strings.Contains(result.ResolvedQuery, "quantum computing") {
			t.Errorf("'%s' resolved query should reference prior topic, got %q", tt.input, result.ResolvedQuery)
		}
	}
}

func TestFollowUpResolver_Narrow(t *testing.T) {
	fr := NewFollowUpResolver()
	state := NewConversationState()
	state.ActiveTopic = "machine learning"
	state.TurnCount = 1

	tests := []string{
		"specifically the training part",
		"focus on supervised learning",
		"just the backpropagation part",
		"only about neural networks",
	}

	for _, input := range tests {
		result := fr.Resolve(input, state)
		if !result.IsFollowUp {
			t.Errorf("'%s' should be detected as follow-up", input)
		}
		if result.Type != FollowUpNarrow {
			t.Errorf("'%s' should be FollowUpNarrow, got %v", input, result.Type)
		}
		if !strings.Contains(result.ResolvedQuery, "machine learning") {
			t.Errorf("'%s' resolved query should reference prior topic, got %q", input, result.ResolvedQuery)
		}
	}
}

func TestFollowUpResolver_Broaden(t *testing.T) {
	fr := NewFollowUpResolver()
	state := NewConversationState()
	state.ActiveTopic = "goroutines"
	state.PushTopic("Go")
	state.PushTopic("goroutines")
	state.TurnCount = 2

	tests := []string{
		"more generally",
		"big picture please",
		"stepping back for a moment",
		"zoom out",
	}

	for _, input := range tests {
		result := fr.Resolve(input, state)
		if !result.IsFollowUp {
			t.Errorf("'%s' should be detected as follow-up", input)
		}
		if result.Type != FollowUpBroaden {
			t.Errorf("'%s' should be FollowUpBroaden, got %v", input, result.Type)
		}
		if !strings.Contains(result.ResolvedQuery, "goroutines") {
			t.Errorf("'%s' resolved query should contain prior topic, got %q", input, result.ResolvedQuery)
		}
		if !strings.Contains(result.ResolvedQuery, "broader") {
			t.Errorf("'%s' resolved query should indicate broadening, got %q", input, result.ResolvedQuery)
		}
	}
}

func TestFollowUpResolver_Challenge(t *testing.T) {
	fr := NewFollowUpResolver()
	state := NewConversationState()
	state.ActiveTopic = "global warming"
	state.RecordAssumption("CO2 is the primary driver")
	state.TurnCount = 1

	tests := []string{
		"but what about solar cycles?",
		"that's not right",
		"i disagree with that assessment",
		"however, what about natural variation?",
	}

	for _, input := range tests {
		result := fr.Resolve(input, state)
		if !result.IsFollowUp {
			t.Errorf("'%s' should be detected as follow-up", input)
		}
		if result.Type != FollowUpChallenge {
			t.Errorf("'%s' should be FollowUpChallenge, got %v", input, result.Type)
		}
		if !strings.Contains(result.ResolvedQuery, "global warming") {
			t.Errorf("'%s' resolved query should reference prior topic, got %q", input, result.ResolvedQuery)
		}
	}
}

func TestFollowUpResolver_Apply(t *testing.T) {
	fr := NewFollowUpResolver()
	state := NewConversationState()
	state.ActiveTopic = "design patterns"
	state.TurnCount = 1

	tests := []struct {
		input       string
		application string
	}{
		{"how would that work for a web server?", "a web server"},
		{"what if I have multiple databases?", "have multiple databases"},
		{"how does it apply to microservices?", "microservices"},
	}

	for _, tt := range tests {
		result := fr.Resolve(tt.input, state)
		if !result.IsFollowUp {
			t.Errorf("'%s' should be detected as follow-up", tt.input)
		}
		if result.Type != FollowUpApply {
			t.Errorf("'%s' should be FollowUpApply, got %v", tt.input, result.Type)
		}
		if !strings.Contains(result.ResolvedQuery, "design patterns") {
			t.Errorf("'%s' resolved query should reference prior topic, got %q", tt.input, result.ResolvedQuery)
		}
		if result.NewEntity == "" {
			t.Errorf("'%s' should extract an application entity", tt.input)
		}
	}
}

func TestFollowUpResolver_NotFollowUp(t *testing.T) {
	fr := NewFollowUpResolver()
	state := NewConversationState()
	state.ActiveTopic = "Python"
	state.TurnCount = 1

	standaloneQueries := []string{
		"what is the capital of France?",
		"explain quantum mechanics",
		"write me a poem about the ocean",
		"how do I make pasta?",
		"tell me about the history of Rome",
	}

	for _, input := range standaloneQueries {
		result := fr.Resolve(input, state)
		if result.IsFollowUp {
			t.Errorf("'%s' should NOT be detected as follow-up (got type %v)", input, result.Type)
		}
		if result.Type != FollowUpNone {
			t.Errorf("'%s' should be FollowUpNone, got %v", input, result.Type)
		}
		if result.ResolvedQuery != input {
			t.Errorf("'%s' resolved query should be unchanged, got %q", input, result.ResolvedQuery)
		}
	}
}

func TestFollowUpResolver_NoContext(t *testing.T) {
	fr := NewFollowUpResolver()
	state := NewConversationState()
	// No prior turns, no active topic

	result := fr.Resolve("tell me more", state)
	// With no context, even "tell me more" shouldn't match as a follow-up
	if result.IsFollowUp {
		t.Error("'tell me more' with no context should not be a follow-up")
	}
}

func TestFollowUpResolver_WithState(t *testing.T) {
	fr := NewFollowUpResolver()
	state := NewConversationState()

	// Simulate a multi-turn conversation
	nlu1 := &NLUResult{
		Intent:   "explain",
		Action:   "respond",
		Entities: map[string]string{"topic": "Rust"},
		Raw:      "tell me about Rust",
	}
	state.Update("tell me about Rust", nlu1, "Rust is a systems programming language...")

	// Follow-up: compare
	result := fr.Resolve("compare with Go", state)
	if result.Type != FollowUpCompare {
		t.Errorf("expected FollowUpCompare, got %v", result.Type)
	}
	if result.PriorTopic != "Rust" {
		t.Errorf("expected prior topic 'Rust', got %q", result.PriorTopic)
	}
	if !strings.Contains(result.ResolvedQuery, "Rust") {
		t.Errorf("resolved query should contain 'Rust', got %q", result.ResolvedQuery)
	}
	if !strings.Contains(strings.ToLower(result.ResolvedQuery), "go") {
		t.Errorf("resolved query should contain 'Go', got %q", result.ResolvedQuery)
	}

	// Now update state with this comparison
	nlu2 := &NLUResult{
		Intent:   "compare",
		Action:   "respond",
		Entities: map[string]string{"topic": "Go"},
		Raw:      "compare with Go",
	}
	state.Update("compare with Go", nlu2, "Comparing Rust and Go...")

	// Follow-up: deeper on the comparison
	result2 := fr.Resolve("elaborate on that", state)
	if result2.Type != FollowUpDeeper {
		t.Errorf("expected FollowUpDeeper, got %v", result2.Type)
	}
	if !result2.IsFollowUp {
		t.Error("elaborate should be a follow-up")
	}
}

func TestFollowUpType_String(t *testing.T) {
	tests := []struct {
		ft   FollowUpType
		want string
	}{
		{FollowUpNone, "none"},
		{FollowUpDeeper, "deeper"},
		{FollowUpPivot, "pivot"},
		{FollowUpCompare, "compare"},
		{FollowUpClarify, "clarify"},
		{FollowUpNarrow, "narrow"},
		{FollowUpBroaden, "broaden"},
		{FollowUpChallenge, "challenge"},
		{FollowUpApply, "apply"},
	}

	for _, tt := range tests {
		if got := tt.ft.String(); got != tt.want {
			t.Errorf("FollowUpType(%d).String() = %q, want %q", tt.ft, got, tt.want)
		}
	}
}

// -----------------------------------------------------------------------
// PreferenceModel Tests
// -----------------------------------------------------------------------

func TestPreferenceModel_ObserveTurn(t *testing.T) {
	pm := NewPreferenceModel()

	// Observe a normal turn
	pm.ObserveTurn("explain Go concurrency to me in detail", "Go uses goroutines...", false, false, false)

	if pm.TurnsSampled != 1 {
		t.Errorf("expected TurnsSampled=1, got %d", pm.TurnsSampled)
	}
	if pm.AverageQueryLength == 0 {
		t.Error("AverageQueryLength should be > 0 after a turn")
	}

	// Observe a clarification turn
	pm.ObserveTurn("what do you mean?", "Let me clarify...", false, true, false)

	if pm.ClarificationRate == 0 {
		t.Error("ClarificationRate should be > 0 after a clarification")
	}

	// Observe a correction turn
	pm.ObserveTurn("no that's wrong, I meant X", "I see, let me correct...", false, false, true)

	if pm.CorrectionRate == 0 {
		t.Error("CorrectionRate should be > 0 after a correction")
	}

	// Observe a follow-up turn
	pm.ObserveTurn("tell me more about that", "Certainly...", true, false, false)

	if pm.FollowUpRate == 0 {
		t.Error("FollowUpRate should be > 0 after a follow-up")
	}
}

func TestPreferenceModel_InferPreferences(t *testing.T) {
	pm := NewPreferenceModel()

	// Simulate a user who writes very short queries
	for i := 0; i < 10; i++ {
		pm.ObserveTurn("explain go", "Go is...", false, false, false)
	}

	if pm.Verbosity >= 0.5 {
		t.Errorf("short queries should result in lower verbosity, got %.2f", pm.Verbosity)
	}

	// Reset and simulate verbose user
	pm2 := NewPreferenceModel()
	for i := 0; i < 10; i++ {
		pm2.ObserveTurn(
			"I would like a comprehensive and thorough explanation of how Go handles concurrent programming with goroutines and channels",
			"Go uses goroutines...",
			false, false, false,
		)
	}

	if pm2.Verbosity <= 0.5 {
		t.Errorf("long queries should result in higher verbosity, got %.2f", pm2.Verbosity)
	}

	// Simulate user who asks for lots of clarifications
	pm3 := NewPreferenceModel()
	for i := 0; i < 10; i++ {
		pm3.ObserveTurn("explain that again please", "Let me rephrase...", false, true, false)
	}

	if pm3.Verbosity <= 0.5 {
		t.Errorf("frequent clarifications should increase verbosity, got %.2f", pm3.Verbosity)
	}
}

func TestPreferenceModel_InferPreferences_Technical(t *testing.T) {
	pm := NewPreferenceModel()

	// Simulate a technical user
	technicalQueries := []string{
		"what's the algorithm complexity of hash table lookup?",
		"explain goroutine scheduling and the runtime's work-stealing mechanism",
		"how does the mutex implementation handle contention?",
		"what's the throughput of grpc versus rest api?",
		"explain the concurrency model and how channels work with goroutines",
	}

	for _, q := range technicalQueries {
		pm.ObserveTurn(q, "response...", false, false, false)
	}

	if pm.TechnicalDepth <= 0.5 {
		t.Errorf("technical queries should raise TechnicalDepth, got %.2f", pm.TechnicalDepth)
	}

	// Simulate a non-technical user
	pm2 := NewPreferenceModel()
	simpleQueries := []string{
		"keep it simple, what is a computer?",
		"explain in simple terms please",
		"in layman's terms what does this mean?",
		"explain like i'm five",
		"for a beginner, what is programming?",
	}

	for _, q := range simpleQueries {
		pm2.ObserveTurn(q, "response...", false, false, false)
	}

	if pm2.TechnicalDepth >= 0.5 {
		t.Errorf("simple requests should lower TechnicalDepth, got %.2f", pm2.TechnicalDepth)
	}
}

func TestPreferenceModel_ExplicitPreference(t *testing.T) {
	pm := NewPreferenceModel()

	pm.SetExplicit("verbosity", "terse")
	if pm.Verbosity > 0.2 {
		t.Errorf("explicit terse verbosity should be low, got %.2f", pm.Verbosity)
	}

	pm.SetExplicit("verbosity", "verbose")
	if pm.Verbosity < 0.8 {
		t.Errorf("explicit verbose verbosity should be high, got %.2f", pm.Verbosity)
	}

	pm.SetExplicit("tone", "casual")
	if pm.TonePref != ToneCasual {
		t.Errorf("expected ToneCasual, got %v", pm.TonePref)
	}

	pm.SetExplicit("format", "bullets")
	if pm.FormattingPref != "bullets" {
		t.Errorf("expected format='bullets', got %q", pm.FormattingPref)
	}

	pm.SetExplicit("technical_depth", "expert")
	if pm.TechnicalDepth < 0.8 {
		t.Errorf("explicit expert depth should be high, got %.2f", pm.TechnicalDepth)
	}

	pm.SetExplicit("examples", "lots")
	if pm.ExamplePref < 0.8 {
		t.Errorf("explicit lots of examples should be high, got %.2f", pm.ExamplePref)
	}

	// Verify preference is stored
	if _, ok := pm.Preferences["verbosity"]; !ok {
		t.Error("preference 'verbosity' should be stored")
	}
	if pm.Preferences["verbosity"].Source != "explicit" {
		t.Error("preference source should be 'explicit'")
	}

	// Reinforcing a preference should increment SeenCount.
	// At this point "verbosity" was set twice above (terse, verbose),
	// so this third call should make SeenCount=3.
	pm.SetExplicit("verbosity", "verbose")
	if pm.Preferences["verbosity"].SeenCount != 3 {
		t.Errorf("expected SeenCount=3, got %d", pm.Preferences["verbosity"].SeenCount)
	}
}

func TestPreferenceModel_ApplyToParams(t *testing.T) {
	pm := NewPreferenceModel()
	pm.TonePref = ToneWarm
	pm.TechnicalDepth = 0.1

	params := &TaskParams{
		Topic: "Go",
		Tone:  ToneNeutral, // default — should be overridden
	}

	result := pm.ApplyToParams(params)

	if result.Tone != ToneWarm {
		t.Errorf("expected Tone=ToneWarm, got %v", result.Tone)
	}
	if result.Audience != "beginners" {
		t.Errorf("low technical depth should set audience to 'beginners', got %q", result.Audience)
	}

	// Test that explicit tone is preserved
	pm2 := NewPreferenceModel()
	pm2.TonePref = ToneCasual
	params2 := &TaskParams{
		Topic: "Rust",
		Tone:  ToneDirect, // explicitly set — should be preserved
	}
	result2 := pm2.ApplyToParams(params2)
	if result2.Tone != ToneDirect {
		t.Errorf("explicit tone should be preserved, got %v", result2.Tone)
	}

	// Test expert audience
	pm3 := NewPreferenceModel()
	pm3.TechnicalDepth = 0.9
	params3 := &TaskParams{Topic: "ML"}
	result3 := pm3.ApplyToParams(params3)
	if result3.Audience != "experts" {
		t.Errorf("high technical depth should set audience to 'experts', got %q", result3.Audience)
	}

	// Nil safety
	result4 := pm.ApplyToParams(nil)
	if result4 != nil {
		t.Error("ApplyToParams(nil) should return nil")
	}
}

func TestPreferenceModel_GenerationHints(t *testing.T) {
	// Terse, simple user
	pm := NewPreferenceModel()
	pm.Verbosity = 0.2
	pm.TechnicalDepth = 0.2
	pm.ExamplePref = 0.3
	pm.FormattingPref = "prose"
	pm.TonePref = ToneCasual
	pm.RiskTolerance = 0.2

	hints := pm.GenerationHints()

	if hints.TargetLength != "short" {
		t.Errorf("low verbosity should give short target, got %q", hints.TargetLength)
	}
	if hints.UseExamples {
		t.Error("low example pref should not use examples")
	}
	if hints.UseBulletPoints {
		t.Error("prose format should not use bullet points")
	}
	if hints.TechnicalLevel != "simple" {
		t.Errorf("low depth should give simple level, got %q", hints.TechnicalLevel)
	}
	if hints.Tone != ToneCasual {
		t.Errorf("expected ToneCasual, got %v", hints.Tone)
	}
	if hints.IncludeWarnings != true {
		t.Error("low risk tolerance should include warnings")
	}

	// Verbose, expert user
	pm2 := NewPreferenceModel()
	pm2.Verbosity = 0.9
	pm2.TechnicalDepth = 0.9
	pm2.ExamplePref = 0.8
	pm2.FormattingPref = "bullets"
	pm2.TonePref = ToneDirect
	pm2.RiskTolerance = 0.8

	hints2 := pm2.GenerationHints()

	if hints2.TargetLength != "long" {
		t.Errorf("high verbosity should give long target, got %q", hints2.TargetLength)
	}
	if !hints2.UseExamples {
		t.Error("high example pref should use examples")
	}
	if !hints2.UseBulletPoints {
		t.Error("bullets format should use bullet points")
	}
	if hints2.TechnicalLevel != "advanced" {
		t.Errorf("high depth should give advanced level, got %q", hints2.TechnicalLevel)
	}
	if hints2.IncludeWarnings {
		t.Error("high risk tolerance should not include warnings")
	}
	if hints2.IncludeRecap != true {
		t.Error("high verbosity should include recap")
	}

	// Medium / balanced
	pm3 := NewPreferenceModel()
	hints3 := pm3.GenerationHints()
	if hints3.TargetLength != "medium" {
		t.Errorf("balanced verbosity should give medium target, got %q", hints3.TargetLength)
	}
	if hints3.TechnicalLevel != "intermediate" {
		t.Errorf("balanced depth should give intermediate level, got %q", hints3.TechnicalLevel)
	}
}

func TestPreferenceModel_ClarificationRateAffectsRecap(t *testing.T) {
	pm := NewPreferenceModel()
	pm.ClarificationRate = 0.3
	pm.Verbosity = 0.4 // normally wouldn't include recap

	hints := pm.GenerationHints()
	if !hints.IncludeRecap {
		t.Error("high clarification rate should include recap regardless of verbosity")
	}
}

func TestPreferenceModel_RiskSignals(t *testing.T) {
	pm := NewPreferenceModel()

	// "pros and cons" signals risk awareness
	for i := 0; i < 5; i++ {
		pm.ObserveTurn("what are the pros and cons of microservices?", "response...", false, false, false)
	}

	if pm.RiskTolerance >= 0.5 {
		t.Errorf("pros-and-cons user should have lower risk tolerance, got %.2f", pm.RiskTolerance)
	}
}

// -----------------------------------------------------------------------
// Benchmarks
// -----------------------------------------------------------------------

func BenchmarkConversationStateUpdate(b *testing.B) {
	cs := NewConversationState()
	nlu := &NLUResult{
		Intent:   "explain",
		Action:   "respond",
		Entities: map[string]string{"topic": "Go concurrency"},
		Raw:      "explain Go concurrency in detail please",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cs.Update("explain Go concurrency in detail please", nlu, "Go concurrency uses goroutines and channels for lightweight threading...")
	}
}

func BenchmarkFollowUpResolve(b *testing.B) {
	fr := NewFollowUpResolver()
	state := NewConversationState()
	state.ActiveTopic = "Go concurrency"
	state.TurnCount = 3
	state.PushTopic("Go")
	state.PushTopic("Go concurrency")

	inputs := []string{
		"tell me more",
		"what about channels?",
		"compare with Rust",
		"what do you mean by that?",
		"specifically the scheduling part",
		"more generally speaking",
		"but what about deadlocks?",
		"how would that work for my web server?",
		"what is the capital of France?", // not a follow-up
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fr.Resolve(inputs[i%len(inputs)], state)
	}
}

func BenchmarkPreferenceInference(b *testing.B) {
	pm := NewPreferenceModel()

	// Pre-populate some data
	for i := 0; i < 20; i++ {
		pm.ObserveTurn("explain the algorithm complexity of this approach", "response...", i%3 == 0, i%5 == 0, i%7 == 0)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pm.InferPreferences()
	}
}
