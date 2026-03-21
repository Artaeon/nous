package cognitive

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestDailyBriefingBasic(t *testing.T) {
	prg := &PersonalResponseGenerator{}

	toolResults := map[string]string{
		"weather": "Sunny, 22°C in Vienna",
		"todos":   "- Buy groceries\n- Call dentist",
	}

	briefing := prg.DailyBriefing(toolResults)
	if briefing == "" {
		t.Fatal("expected non-empty briefing")
	}
	if !strings.Contains(briefing, "Sunny") {
		t.Error("briefing should include weather")
	}
	if !strings.Contains(briefing, "Buy groceries") {
		t.Error("briefing should include tasks")
	}
}

func TestDailyBriefingWithName(t *testing.T) {
	dir := t.TempDir()
	growth := NewPersonalGrowth(filepath.Join(dir, "growth.json"))
	growth.LearnFact("my name is Raphael", "identity")

	prg := &PersonalResponseGenerator{Growth: growth}
	briefing := prg.DailyBriefing(map[string]string{})

	if !strings.Contains(briefing, "Raphael") {
		t.Errorf("briefing should include user name, got %q", briefing)
	}
}

func TestDailyBriefingEmpty(t *testing.T) {
	prg := &PersonalResponseGenerator{}
	briefing := prg.DailyBriefing(map[string]string{})

	// Should still produce a greeting even with no data
	if briefing == "" {
		t.Fatal("briefing should not be empty even without tool data")
	}
	if !strings.Contains(briefing, "!") {
		t.Error("briefing should contain a greeting")
	}
}

func TestPersonalizeResponseConcise(t *testing.T) {
	dir := t.TempDir()
	growth := NewPersonalGrowth(filepath.Join(dir, "growth.json"))
	// Feed a short query to trigger concise preference detection
	growth.RecordInteraction("hi")

	prg := &PersonalResponseGenerator{Growth: growth}

	// Long response with many sentences (must be >300 chars to trigger trim)
	long := "First sentence about the topic that is reasonably detailed and informative for the reader. " +
		"Second sentence with more details about the specifics of this particular subject matter. " +
		"Third sentence explaining further context that a reader might find useful to understand. " +
		"Fourth sentence nobody really needs but it adds unnecessary padding to the response. " +
		"Fifth sentence that's clearly excessive and should be removed. Sixth sentence too much."

	result := prg.PersonalizeResponse(long)
	sentences := splitSentences(result)
	if len(sentences) > 4 {
		t.Errorf("concise mode should trim to ~3 sentences, got %d", len(sentences))
	}
}

func TestPersonalizeResponsePassthrough(t *testing.T) {
	prg := &PersonalResponseGenerator{} // no Growth

	input := "Some response text."
	result := prg.PersonalizeResponse(input)
	if result != input {
		t.Errorf("without Growth, should pass through unchanged, got %q", result)
	}
}

func TestEnrichWithContext(t *testing.T) {
	dir := t.TempDir()
	growth := NewPersonalGrowth(filepath.Join(dir, "growth.json"))
	growth.LearnFact("I use Go for backend development", "interest")

	prg := &PersonalResponseGenerator{Growth: growth}

	response := "Go is a statically typed language."
	enriched := prg.EnrichWithContext(response, "Go")

	if !strings.Contains(enriched, "Go is a statically typed") {
		t.Error("enriched should retain original response")
	}
	if !strings.Contains(enriched, "backend development") {
		t.Error("enriched should add personal context about Go")
	}
}

func TestEnrichWithContextNoMatch(t *testing.T) {
	dir := t.TempDir()
	growth := NewPersonalGrowth(filepath.Join(dir, "growth.json"))
	growth.LearnFact("I like cooking", "interest")

	prg := &PersonalResponseGenerator{Growth: growth}

	response := "Rust is a systems language."
	enriched := prg.EnrichWithContext(response, "Rust")

	// No personal facts about Rust, so should be unchanged
	if enriched != response {
		t.Errorf("should not enrich when no relevant facts, got %q", enriched)
	}
}

func TestSmartLLMPrompt(t *testing.T) {
	dir := t.TempDir()
	growth := NewPersonalGrowth(filepath.Join(dir, "growth.json"))
	growth.LearnFact("I'm interested in philosophy", "interest")

	tracker := NewConversationTracker()
	tracker.IngestContent(
		"Stoicism is a school of philosophy. It was founded in Athens by Zeno.",
		"https://example.com", "stoicism",
	)

	prg := &PersonalResponseGenerator{
		Growth:  growth,
		Tracker: tracker,
	}

	prompt := prg.SmartLLMPrompt("what is stoicism", "")
	if !strings.Contains(prompt, "Known facts") {
		t.Error("smart prompt should include known facts from tracker")
	}
	if !strings.Contains(prompt, "[Question]") {
		t.Error("smart prompt should include the question section")
	}
}

func TestSmartLLMPromptWithRawData(t *testing.T) {
	prg := &PersonalResponseGenerator{}

	prompt := prg.SmartLLMPrompt("explain Go", "Go is fast and concurrent")
	if !strings.Contains(prompt, "[Data]") {
		t.Error("smart prompt should include raw data section")
	}
	if !strings.Contains(prompt, "Go is fast and concurrent") {
		t.Error("smart prompt should include the raw data content")
	}
}

func TestNLUDailyBriefing(t *testing.T) {
	nlu := NewNLU()
	tests := []struct {
		input string
		want  string
	}{
		{"good morning", "daily_briefing"},
		{"daily briefing", "daily_briefing"},
		{"my day", "daily_briefing"},
		{"brief me", "daily_briefing"},
		{"morning briefing", "daily_briefing"},
		{"start my day", "daily_briefing"},
	}

	for _, tt := range tests {
		r := nlu.Understand(tt.input)
		if r.Intent != tt.want {
			t.Errorf("NLU(%q) intent = %q, want %q", tt.input, r.Intent, tt.want)
		}
	}
}

func TestDailyBriefingAction(t *testing.T) {
	ar := NewActionRouter()
	nlu := &NLUResult{
		Intent: "daily_briefing",
		Action: "daily_briefing",
		Raw:    "good morning",
	}

	result := ar.Execute(nlu, nil)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if result.DirectResponse == "" {
		t.Error("daily briefing should produce a direct response")
	}
	if result.Source != "briefing" {
		t.Errorf("expected source 'briefing', got %q", result.Source)
	}
}
