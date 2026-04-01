package cognitive

import (
	"math"
	"os"
	"path/filepath"
	"regexp"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Pattern extraction
// ---------------------------------------------------------------------------

func TestExtractPattern_Basic(t *testing.T) {
	entities := map[string]string{"location": "Paris"}
	p := extractPattern("what is the weather in Paris", entities, "weather")
	if p == nil {
		t.Fatal("expected non-nil pattern")
	}
	if p.Template != "what is the weather in {location}" {
		t.Errorf("template = %q, want %q", p.Template, "what is the weather in {location}")
	}
	if len(p.Slots) != 1 || p.Slots[0] != "location" {
		t.Errorf("slots = %v, want [location]", p.Slots)
	}
	if p.Intent != "weather" {
		t.Errorf("intent = %q, want %q", p.Intent, "weather")
	}
	if p.compiled == nil {
		t.Fatal("regex should be compiled")
	}
}

func TestExtractPattern_MultipleSlots(t *testing.T) {
	entities := map[string]string{
		"source": "english",
		"target": "french",
	}
	p := extractPattern("translate hello from english to french", entities, "translate")
	if p == nil {
		t.Fatal("expected non-nil pattern")
	}
	if len(p.Slots) != 2 {
		t.Fatalf("slots count = %d, want 2", len(p.Slots))
	}
	// Slots should be in order of appearance.
	if p.Slots[0] != "source" {
		t.Errorf("slot[0] = %q, want %q", p.Slots[0], "source")
	}
	if p.Slots[1] != "target" {
		t.Errorf("slot[1] = %q, want %q", p.Slots[1], "target")
	}
}

func TestExtractPattern_NoEntities(t *testing.T) {
	p := extractPattern("hello world", map[string]string{}, "greeting")
	if p != nil {
		t.Error("expected nil pattern when no entities")
	}
}

func TestExtractPattern_EntityNotInInput(t *testing.T) {
	entities := map[string]string{"city": "London"}
	p := extractPattern("what is the weather in Paris", entities, "weather")
	if p != nil {
		t.Error("expected nil pattern when entity value not found in input")
	}
}

func TestExtractPattern_CaseInsensitive(t *testing.T) {
	entities := map[string]string{"topic": "Quantum Mechanics"}
	p := extractPattern("What is Quantum Mechanics", entities, "question")
	if p == nil {
		t.Fatal("expected non-nil pattern")
	}
	if p.Template != "what is {topic}" {
		t.Errorf("template = %q, want %q", p.Template, "what is {topic}")
	}
}

func TestExtractPattern_ConfidenceRange(t *testing.T) {
	entities := map[string]string{"topic": "Go"}
	p := extractPattern("what is Go", entities, "question")
	if p == nil {
		t.Fatal("expected non-nil pattern")
	}
	if p.Confidence <= 0 || p.Confidence > 1.0 {
		t.Errorf("confidence = %f, want in (0, 1]", p.Confidence)
	}
}

// ---------------------------------------------------------------------------
// Template extraction
// ---------------------------------------------------------------------------

func TestExtractTemplate_Basic(t *testing.T) {
	entities := map[string]string{"location": "Paris"}
	tmpl := extractTemplate("The weather in Paris is currently sunny.", entities)
	want := "The weather in {location} is currently sunny."
	if tmpl != want {
		t.Errorf("template = %q, want %q", tmpl, want)
	}
}

func TestExtractTemplate_MultipleOccurrences(t *testing.T) {
	entities := map[string]string{"name": "Go"}
	tmpl := extractTemplate("Go is great. I love Go.", entities)
	want := "{name} is great. I love {name}."
	if tmpl != want {
		t.Errorf("template = %q, want %q", tmpl, want)
	}
}

func TestExtractTemplate_NoEntities(t *testing.T) {
	tmpl := extractTemplate("Just a plain response.", map[string]string{})
	if tmpl != "Just a plain response." {
		t.Errorf("template should be unchanged, got %q", tmpl)
	}
}

func TestExtractTemplate_CaseInsensitive(t *testing.T) {
	entities := map[string]string{"city": "paris"}
	tmpl := extractTemplate("Paris is the capital of France.", entities)
	want := "{city} is the capital of France."
	if tmpl != want {
		t.Errorf("template = %q, want %q", tmpl, want)
	}
}

// ---------------------------------------------------------------------------
// Match + Execute round-trip
// ---------------------------------------------------------------------------

func TestMatchExecuteRoundTrip(t *testing.T) {
	cc := NewCognitiveCompiler("")

	entities := map[string]string{"location": "Paris"}
	h := cc.Compile(
		"what is the weather in Paris",
		"The weather in Paris is currently sunny.",
		"weather",
		entities,
	)
	if h == nil {
		t.Fatal("Compile returned nil")
	}

	// Match a similar query.
	matched, slots := cc.Match("what is the weather in London")
	if matched == nil {
		t.Fatal("expected a match")
	}
	if slots["location"] != "london" {
		t.Errorf("slot location = %q, want %q", slots["location"], "london")
	}

	// Execute with the extracted slots.
	result := cc.Execute(matched, slots)
	want := "The weather in london is currently sunny."
	if result != want {
		t.Errorf("result = %q, want %q", result, want)
	}
}

func TestMatchExecuteRoundTrip_MultiSlot(t *testing.T) {
	cc := NewCognitiveCompiler("")

	entities := map[string]string{
		"source": "english",
		"target": "french",
	}
	h := cc.Compile(
		"translate hello from english to french",
		"Translating from english to french: bonjour",
		"translate",
		entities,
	)
	if h == nil {
		t.Fatal("Compile returned nil")
	}

	matched, slots := cc.Match("translate hello from spanish to german")
	if matched == nil {
		t.Fatal("expected a match")
	}
	result := cc.Execute(matched, slots)
	if result == "" {
		t.Error("Execute returned empty string")
	}
}

func TestMatch_NoMatch(t *testing.T) {
	cc := NewCognitiveCompiler("")

	entities := map[string]string{"location": "Paris"}
	cc.Compile(
		"what is the weather in Paris",
		"The weather in Paris is currently sunny.",
		"weather",
		entities,
	)

	// Completely different query should not match.
	matched, _ := cc.Match("how do I compile Go code")
	if matched != nil {
		t.Error("expected no match for unrelated query")
	}
}

func TestMatch_EmptyInput(t *testing.T) {
	cc := NewCognitiveCompiler("")
	matched, _ := cc.Match("")
	if matched != nil {
		t.Error("expected nil for empty input")
	}
}

func TestExecute_FallbackSlot(t *testing.T) {
	cc := NewCognitiveCompiler("")
	handler := &CompiledHandler{
		Template: "Hello {name}, welcome to {place}.",
		Slots: []SlotDef{
			{Name: "name", Type: "entity", Fallback: "friend"},
			{Name: "place", Type: "entity", Fallback: "the world"},
		},
	}
	// Only provide one slot; the other uses fallback.
	result := cc.Execute(handler, map[string]string{"name": "Alice"})
	want := "Hello Alice, welcome to the world."
	if result != want {
		t.Errorf("result = %q, want %q", result, want)
	}
}

func TestExecute_NilHandler(t *testing.T) {
	cc := NewCognitiveCompiler("")
	result := cc.Execute(nil, nil)
	if result != "" {
		t.Error("expected empty string for nil handler")
	}
}

// ---------------------------------------------------------------------------
// Observe feedback loop
// ---------------------------------------------------------------------------

func TestObserve_Accepted(t *testing.T) {
	cc := NewCognitiveCompiler("")

	entities := map[string]string{"topic": "rust"}
	h := cc.Compile("what is rust", "Rust is a systems programming language.", "question", entities)
	if h == nil {
		t.Fatal("Compile returned nil")
	}

	initialQ := h.Quality
	cc.Observe(h, true)
	if h.Quality <= initialQ {
		t.Errorf("quality should increase on acceptance: was %f, now %f", initialQ, h.Quality)
	}
}

func TestObserve_Rejected(t *testing.T) {
	cc := NewCognitiveCompiler("")

	entities := map[string]string{"topic": "rust"}
	h := cc.Compile("what is rust", "Rust is a systems programming language.", "question", entities)
	if h == nil {
		t.Fatal("Compile returned nil")
	}

	initialQ := h.Quality
	cc.Observe(h, false)
	if h.Quality >= initialQ {
		t.Errorf("quality should decrease on rejection: was %f, now %f", initialQ, h.Quality)
	}
}

func TestObserve_RepeatedRejectionPrunes(t *testing.T) {
	cc := NewCognitiveCompiler("")

	entities := map[string]string{"topic": "rust"}
	h := cc.Compile("what is rust", "Rust is a systems programming language.", "question", entities)
	if h == nil {
		t.Fatal("Compile returned nil")
	}

	// Reject many times until quality drops below threshold.
	for i := 0; i < 20; i++ {
		cc.Observe(h, false)
	}

	// Handler should be pruned.
	if len(cc.Handlers) != 0 {
		t.Errorf("handler should be pruned after repeated rejections, got %d handlers", len(cc.Handlers))
	}
}

func TestObserve_NilHandler(t *testing.T) {
	cc := NewCognitiveCompiler("")
	// Should not panic.
	cc.Observe(nil, true)
	cc.Observe(nil, false)
}

// ---------------------------------------------------------------------------
// Prune stale handlers
// ---------------------------------------------------------------------------

func TestPrune_LowQuality(t *testing.T) {
	cc := NewCognitiveCompiler("")

	now := time.Now()
	cc.Handlers = []*CompiledHandler{
		{ID: "good", Quality: 0.8, Uses: 5, LastUsed: now},
		{ID: "bad", Quality: 0.2, Uses: 1, LastUsed: now},
	}

	cc.Prune()

	if len(cc.Handlers) != 1 {
		t.Fatalf("expected 1 handler after prune, got %d", len(cc.Handlers))
	}
	if cc.Handlers[0].ID != "good" {
		t.Errorf("kept handler ID = %q, want %q", cc.Handlers[0].ID, "good")
	}
}

func TestPrune_StaleUnused(t *testing.T) {
	cc := NewCognitiveCompiler("")

	old := time.Now().Add(-60 * 24 * time.Hour)
	now := time.Now()
	cc.Handlers = []*CompiledHandler{
		{ID: "fresh", Quality: 0.6, Uses: 0, LastUsed: now},
		{ID: "stale", Quality: 0.6, Uses: 0, LastUsed: old},
	}

	cc.Prune()

	if len(cc.Handlers) != 1 {
		t.Fatalf("expected 1 handler after prune, got %d", len(cc.Handlers))
	}
	if cc.Handlers[0].ID != "fresh" {
		t.Errorf("kept handler ID = %q, want %q", cc.Handlers[0].ID, "fresh")
	}
}

func TestPrune_StaleUsedLowQuality(t *testing.T) {
	cc := NewCognitiveCompiler("")

	old := time.Now().Add(-60 * 24 * time.Hour)
	now := time.Now()
	cc.Handlers = []*CompiledHandler{
		{ID: "active", Quality: 0.7, Uses: 10, LastUsed: now},
		{ID: "stale_low", Quality: 0.4, Uses: 3, LastUsed: old},
		{ID: "stale_ok", Quality: 0.6, Uses: 3, LastUsed: old},
	}

	cc.Prune()

	// stale_low (quality<0.5 and stale) should be pruned, stale_ok kept.
	if len(cc.Handlers) != 2 {
		t.Fatalf("expected 2 handlers after prune, got %d", len(cc.Handlers))
	}
	ids := map[string]bool{}
	for _, h := range cc.Handlers {
		ids[h.ID] = true
	}
	if ids["stale_low"] {
		t.Error("stale_low should have been pruned")
	}
}

func TestPrune_Empty(t *testing.T) {
	cc := NewCognitiveCompiler("")
	// Should not panic on empty.
	cc.Prune()
	if len(cc.Handlers) != 0 {
		t.Error("expected 0 handlers")
	}
}

// ---------------------------------------------------------------------------
// Save/Load persistence
// ---------------------------------------------------------------------------

func TestCompilerSaveLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "compiler.json")

	cc := NewCognitiveCompiler(path)

	entities := map[string]string{"topic": "rust"}
	h := cc.Compile("what is rust", "Rust is a systems programming language.", "question", entities)
	if h == nil {
		t.Fatal("Compile returned nil")
	}

	// Execute to bump usage count.
	cc.Execute(h, map[string]string{"topic": "rust"})

	// Force save and verify file exists.
	if err := cc.Save(); err != nil {
		t.Fatalf("Save error: %v", err)
	}
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Fatal("save file does not exist")
	}

	// Load into a fresh compiler.
	cc2 := NewCognitiveCompiler(path)
	if len(cc2.Handlers) != 1 {
		t.Fatalf("loaded handlers = %d, want 1", len(cc2.Handlers))
	}
	loaded := cc2.Handlers[0]
	if loaded.ID != h.ID {
		t.Errorf("loaded ID = %q, want %q", loaded.ID, h.ID)
	}
	if loaded.Pattern == nil {
		t.Fatal("loaded pattern is nil")
	}
	if loaded.Pattern.compiled == nil {
		t.Fatal("loaded regex was not recompiled")
	}
	if loaded.Uses != h.Uses {
		t.Errorf("loaded Uses = %d, want %d", loaded.Uses, h.Uses)
	}

	// Verify the reloaded handler can still match.
	matched, slots := cc2.Match("what is go")
	if matched == nil {
		t.Fatal("expected match after reload")
	}
	if slots["topic"] != "go" {
		t.Errorf("slot topic = %q, want %q", slots["topic"], "go")
	}
}

func TestCompilerSaveLoad_EmptyPath(t *testing.T) {
	cc := NewCognitiveCompiler("")
	if err := cc.Save(); err != nil {
		t.Errorf("Save with empty path should succeed, got: %v", err)
	}
	if err := cc.Load(); err != nil {
		t.Errorf("Load with empty path should succeed, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

func TestStats_Empty(t *testing.T) {
	cc := NewCognitiveCompiler("")
	stats := cc.Stats()
	if stats.TotalHandlers != 0 {
		t.Errorf("TotalHandlers = %d, want 0", stats.TotalHandlers)
	}
}

func TestStats_WithHandlers(t *testing.T) {
	cc := NewCognitiveCompiler("")

	now := time.Now()
	old := now.Add(-60 * 24 * time.Hour)
	cc.Handlers = []*CompiledHandler{
		{ID: "a", Quality: 0.9, Uses: 10, LastUsed: now},
		{ID: "b", Quality: 0.5, Uses: 0, LastUsed: old},
		{ID: "c", Quality: 0.8, Uses: 5, LastUsed: now},
	}

	stats := cc.Stats()
	if stats.TotalHandlers != 3 {
		t.Errorf("TotalHandlers = %d, want 3", stats.TotalHandlers)
	}
	if stats.TotalExecutions != 15 {
		t.Errorf("TotalExecutions = %d, want 15", stats.TotalExecutions)
	}
	if stats.HighQuality != 2 {
		t.Errorf("HighQuality = %d, want 2", stats.HighQuality)
	}
	if stats.Stale != 1 {
		t.Errorf("Stale = %d, want 1", stats.Stale)
	}
	// 2 out of 3 handlers have Uses > 0.
	wantHitRate := 2.0 / 3.0
	if math.Abs(stats.HitRate-wantHitRate) > 0.01 {
		t.Errorf("HitRate = %f, want ~%f", stats.HitRate, wantHitRate)
	}
	wantAvgQ := (0.9 + 0.5 + 0.8) / 3.0
	if math.Abs(stats.AvgQuality-wantAvgQ) > 0.01 {
		t.Errorf("AvgQuality = %f, want ~%f", stats.AvgQuality, wantAvgQ)
	}
}

// ---------------------------------------------------------------------------
// Compile edge cases
// ---------------------------------------------------------------------------

func TestCompile_EmptyInput(t *testing.T) {
	cc := NewCognitiveCompiler("")
	h := cc.Compile("", "response", "intent", map[string]string{"a": "b"})
	if h != nil {
		t.Error("expected nil for empty input")
	}
}

func TestCompile_EmptyResponse(t *testing.T) {
	cc := NewCognitiveCompiler("")
	h := cc.Compile("what is Go", "", "question", map[string]string{"topic": "Go"})
	if h != nil {
		t.Error("expected nil for empty response")
	}
}

func TestCompile_Deduplication(t *testing.T) {
	cc := NewCognitiveCompiler("")

	entities := map[string]string{"topic": "rust"}
	cc.Compile("what is rust", "Rust is great.", "question", entities)
	cc.Compile("what is python", "Python is great.", "question", map[string]string{"topic": "python"})

	// Both should produce the same pattern template "what is {topic}",
	// so the second should replace the first.
	if len(cc.Handlers) != 1 {
		t.Errorf("expected 1 handler (deduplicated), got %d", len(cc.Handlers))
	}
}

func TestCompile_KnowledgeLookupSlotType(t *testing.T) {
	cc := NewCognitiveCompiler("")

	entities := map[string]string{"knowledge": "physics"}
	h := cc.Compile("tell me about physics", "Physics is the study of matter.", "question", entities)
	if h == nil {
		t.Fatal("expected non-nil handler")
	}
	found := false
	for _, sd := range h.Slots {
		if sd.Name == "knowledge" && sd.Type == "knowledge_lookup" {
			found = true
		}
	}
	if !found {
		t.Error("expected slot type 'knowledge_lookup' for 'knowledge' entity")
	}
}

// ---------------------------------------------------------------------------
// Regex pattern matching correctness
// ---------------------------------------------------------------------------

func TestBuildPatternRegex(t *testing.T) {
	tests := []struct {
		template string
		slots    []string
		input    string
		match    bool
	}{
		{
			template: "what is {topic}",
			slots:    []string{"topic"},
			input:    "what is gravity",
			match:    true,
		},
		{
			template: "translate {text} from {source} to {target}",
			slots:    []string{"text", "source", "target"},
			input:    "translate hello from english to french",
			match:    true,
		},
		{
			template: "what is {topic}",
			slots:    []string{"topic"},
			input:    "how is gravity",
			match:    false,
		},
	}

	for _, tt := range tests {
		regexStr := buildPatternRegex(tt.template, tt.slots)
		re, err := regexp.Compile(regexStr)
		if err != nil {
			t.Errorf("regex compile error for %q: %v", tt.template, err)
			continue
		}
		got := re.MatchString(tt.input)
		if got != tt.match {
			t.Errorf("pattern %q, input %q: match = %v, want %v (regex: %s)", tt.template, tt.input, got, tt.match, regexStr)
		}
	}
}

// ---------------------------------------------------------------------------
// Concurrent access
// ---------------------------------------------------------------------------

func TestConcurrentAccess(t *testing.T) {
	cc := NewCognitiveCompiler("")

	entities := map[string]string{"topic": "concurrency"}
	cc.Compile(
		"what is concurrency",
		"Concurrency is doing many things at once.",
		"question",
		entities,
	)

	done := make(chan struct{})
	// Hammer Match and Execute from multiple goroutines.
	for i := 0; i < 10; i++ {
		go func() {
			defer func() { done <- struct{}{} }()
			for j := 0; j < 50; j++ {
				h, slots := cc.Match("what is parallelism")
				if h != nil {
					cc.Execute(h, slots)
				}
			}
		}()
	}
	for i := 0; i < 10; i++ {
		<-done
	}
}
