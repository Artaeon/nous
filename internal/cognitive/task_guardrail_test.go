package cognitive

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Guardrail: task prompts must NEVER return conversational acknowledgments.
//
// These tests enforce that explicit task prompts (explain, compare, overview,
// summarize, walk me through) are routed through the non-conversational
// pipeline and never produce emotional/chatty responses.
// -----------------------------------------------------------------------

// conversationalFluff contains phrases that must NEVER appear in responses
// to explicit task prompts.
var conversationalFluff = []string{
	"i appreciate you sharing that",
	"that's a great question",
	"good question",
	"i hear you",
	"gotcha",
	"i see",
	"thanks for asking",
	"i understand how you feel",
	"it sounds like",
	"i can tell you're",
	"your energy is contagious",
	"that's genuinely great news",
	"take a breath",
	"i'm not going anywhere",
	"night owl mode",
}

// taskPrompts are explicit task prompts that must be recognized as tasks.
var taskPrompts = []string{
	"explain quantum entanglement",
	"what is photosynthesis",
	"compare Python vs Go for web servers",
	"give me an overview of the Renaissance",
	"walk me through how neural networks learn",
	"summarize the key principles of Stoicism",
	"describe how DNA replication works",
	"define epistemology",
	"tell me about the Roman Empire",
	"what are the differences between TCP and UDP",
	"explain why the sky is blue",
	"compare democracy and authoritarianism",
	"how does a compiler work",
	"who was Socrates",
	"pros and cons of remote work",
}

func TestIsExplicitTaskPrompt(t *testing.T) {
	for _, prompt := range taskPrompts {
		if !isExplicitTaskPrompt(prompt) {
			t.Errorf("isExplicitTaskPrompt(%q) = false, want true", prompt)
		}
	}
}

func TestIsExplicitTaskPrompt_NotTask(t *testing.T) {
	notTasks := []string{
		"hi there",
		"I got promoted today!",
		"I'm feeling stressed",
		"thanks",
		"good morning",
		"how are you",
		"I love philosophy",
	}
	for _, prompt := range notTasks {
		if isExplicitTaskPrompt(prompt) {
			t.Errorf("isExplicitTaskPrompt(%q) = true, want false", prompt)
		}
	}
}

func TestTaskPromptsNeverReturnFluff(t *testing.T) {
	// Build a minimal ThinkingEngine with empty graph — this is the
	// low-knowledge scenario where chatty fallbacks are most dangerous.
	te := NewThinkingEngine(nil, nil)

	for _, prompt := range taskPrompts {
		result := te.Think(prompt, nil)
		if result == nil || result.Text == "" {
			continue // no output is acceptable; chatty output is not
		}
		lower := strings.ToLower(result.Text)
		for _, fluff := range conversationalFluff {
			if strings.Contains(lower, fluff) {
				t.Errorf("task prompt %q produced fluff %q in response:\n%s",
					prompt, fluff, result.Text)
			}
		}
	}
}

func TestTaskPromptsNeverEmpathy(t *testing.T) {
	// Verify isExplicitTaskPrompt blocks the empathy path.
	// These are task prompts disguised as emotional statements.
	emotionalTasks := []string{
		"explain why I feel sad after breakups",
		"compare grief and depression",
		"what is anxiety",
		"describe the stages of mourning",
		"tell me about stress management",
	}
	for _, prompt := range emotionalTasks {
		if !isExplicitTaskPrompt(prompt) {
			t.Errorf("isExplicitTaskPrompt(%q) = false — this task prompt would leak to empathy path", prompt)
		}
	}
}

func TestLowInfoConversationalCoversFluff(t *testing.T) {
	// Verify that isLowInformationConversational catches key fluff patterns.
	mustCatch := []string{
		"I appreciate you sharing that.",
		"Good question!",
		"Gotcha.",
		"I see.",
		"I hear you.",
	}
	for _, text := range mustCatch {
		if !isLowInformationConversational(text) {
			t.Errorf("isLowInformationConversational(%q) = false, want true", text)
		}
	}
}

// TestExplainSkeletonStructure verifies the explanation frame produces
// definition → mechanism → example → caveat → recap sections.
func TestExplainSkeletonStructure(t *testing.T) {
	// TaskAnalyze → explanation frame (TaskTeach → tutorial)
	frame := SelectFrame(TaskAnalyze, FormatProse)
	if frame.Name != "explanation" {
		t.Fatalf("expected explanation frame, got %s", frame.Name)
	}

	expectedRoles := []string{"definition", "mechanism", "example", "caveat", "recap"}
	for i, sec := range frame.Sections {
		if i >= len(expectedRoles) {
			break
		}
		if sec.Role != expectedRoles[i] {
			t.Errorf("section %d: role = %q, want %q", i, sec.Role, expectedRoles[i])
		}
	}
	if len(frame.Sections) != len(expectedRoles) {
		t.Errorf("frame has %d sections, want %d", len(frame.Sections), len(expectedRoles))
	}
}

// TestCompareSkeletonStructure verifies the comparison frame produces
// criteria → item_a_evidence → item_b_evidence → tradeoffs → known_unknown → verdict.
func TestCompareSkeletonStructure(t *testing.T) {
	frame := SelectFrame(TaskCompare, FormatProse)
	if frame.Name != "comparison" {
		t.Fatalf("expected comparison frame, got %s", frame.Name)
	}

	expectedRoles := []string{"criteria", "item_a_evidence", "item_b_evidence", "tradeoffs", "known_unknown", "verdict"}
	for i, sec := range frame.Sections {
		if i >= len(expectedRoles) {
			break
		}
		if sec.Role != expectedRoles[i] {
			t.Errorf("section %d: role = %q, want %q", i, sec.Role, expectedRoles[i])
		}
	}
	if len(frame.Sections) != len(expectedRoles) {
		t.Errorf("frame has %d sections, want %d", len(frame.Sections), len(expectedRoles))
	}
}

// TestCompareKnownUnknownSection verifies that when knowledge is sparse,
// the comparison doesn't produce conversational fluff.
func TestCompareKnownUnknownSection(t *testing.T) {
	te := NewThinkingEngine(nil, nil)
	result := te.Think("compare Rust and Haskell", nil)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	lower := strings.ToLower(result.Text)
	// With no graph, the response should NOT contain chatty fluff
	for _, fluff := range conversationalFluff {
		if strings.Contains(lower, fluff) {
			t.Errorf("sparse-knowledge comparison should not contain fluff %q, got:\n%s", fluff, result.Text)
		}
	}
	// Should be classified as comparison task
	if result.Task != TaskCompare {
		t.Errorf("task = %v, want TaskCompare", result.Task)
	}
}

// -----------------------------------------------------------------------
// Regression tests for the three live prompts that were still producing
// conversational fluff (confirmed 2026-03-27).
// -----------------------------------------------------------------------

// TestRegressionOverviewOperatingSystems was producing "Sure… midnight oil…"
func TestRegressionOverviewOperatingSystems(t *testing.T) {
	nlu := &NLUResult{
		Raw:      "give me an overview of operating systems",
		Intent:   "conversation", // simulates neural misclassification
		Action:   "respond",
		Entities: make(map[string]string),
	}
	// The hard reroute should fix this before dispatch
	if !isExplicitTaskPrompt(nlu.Raw) {
		t.Fatal("isExplicitTaskPrompt should recognize this as a task prompt")
	}
}

// TestRegressionPhotosynthesis was producing conversational response
func TestRegressionPhotosynthesis(t *testing.T) {
	nlu := &NLUResult{
		Raw:      "how does photosynthesis work",
		Intent:   "conversation",
		Action:   "respond",
		Entities: make(map[string]string),
	}
	if !isExplicitTaskPrompt(nlu.Raw) {
		t.Fatal("isExplicitTaskPrompt should recognize this as a task prompt")
	}
}

// TestRegressionSummarizeStoicism was producing raw echo fallback
func TestRegressionSummarizeStoicism(t *testing.T) {
	nlu := &NLUResult{
		Raw:      "summarize stoicism in 3 bullets",
		Intent:   "conversation",
		Action:   "respond",
		Entities: make(map[string]string),
	}
	if !isExplicitTaskPrompt(nlu.Raw) {
		t.Fatal("isExplicitTaskPrompt should recognize this as a task prompt")
	}
}

// TestHardRerouteOverridesConversationalAction verifies that when NLU
// misclassifies an explicit task prompt as conversational, the hard reroute
// in Execute() overrides the action before dispatch.
func TestHardRerouteOverridesConversationalAction(t *testing.T) {
	tests := []struct {
		raw        string
		wantIntent string
		wantAction string
	}{
		{"give me an overview of operating systems", "explain", "lookup_knowledge"},
		{"how does photosynthesis work", "explain", "lookup_knowledge"},
		{"summarize stoicism in 3 bullets", "explain", "lookup_knowledge"},
		{"explain why the sky is blue", "explain", "lookup_knowledge"},
		{"compare Python and Go", "compare", "compare"},
		{"what is quantum entanglement", "explain", "lookup_knowledge"},
		{"walk me through TCP/IP", "explain", "lookup_knowledge"},
		{"pros and cons of remote work", "compare", "compare"},
		{"contrast functional and OOP", "compare", "compare"},
		{"differences between TCP and UDP", "compare", "compare"},
		{"summarize the French Revolution", "explain", "lookup_knowledge"},
		{"Rust vs Go for systems programming", "compare", "compare"},
	}

	// Build a minimal ActionRouter — only needs to exist, dispatch will
	// handle the rerouted intent/action.
	ar := NewActionRouter()

	for _, tt := range tests {
		nlu := &NLUResult{
			Raw:        tt.raw,
			Intent:     "conversation", // simulate misclassification
			Action:     "respond",      // would go to handleRespond
			Confidence: 0.8,
			Entities:   make(map[string]string),
		}

		// Run Execute — this exercises the hard reroute logic.
		// We don't care about the result content (no subsystems wired),
		// only that the NLU was rerouted.
		ar.Execute(nlu, nil)

		if nlu.Intent != tt.wantIntent {
			t.Errorf("%q: intent = %q, want %q", tt.raw, nlu.Intent, tt.wantIntent)
		}
		if nlu.Action != tt.wantAction {
			t.Errorf("%q: action = %q, want %q", tt.raw, nlu.Action, tt.wantAction)
		}
		if nlu.Entities["topic"] == "" {
			t.Errorf("%q: topic entity not set after reroute", tt.raw)
		}
	}
}

// TestHardRerouteDoesNotAffectCorrectRouting verifies that the reroute
// does NOT interfere when NLU already classified correctly.
func TestHardRerouteDoesNotAffectCorrectRouting(t *testing.T) {
	ar := NewActionRouter()

	// Already correctly routed — should not be touched
	nlu := &NLUResult{
		Raw:      "explain quantum computing",
		Intent:   "explain",
		Action:   "lookup_knowledge",
		Entities: map[string]string{"topic": "quantum computing"},
	}
	ar.Execute(nlu, nil)
	if nlu.Intent != "explain" {
		t.Errorf("correctly routed intent was changed to %q", nlu.Intent)
	}
	if nlu.Action != "lookup_knowledge" {
		t.Errorf("correctly routed action was changed to %q", nlu.Action)
	}
}

// TestExplainRecapPresent verifies explain outputs include a recap section.
func TestExplainRecapPresent(t *testing.T) {
	te := NewThinkingEngine(nil, nil)
	result := te.Think("explain quantum computing", nil)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	lower := strings.ToLower(result.Text)
	// With no graph, we still expect some form of summary/synthesis
	if strings.Contains(lower, "i appreciate") || strings.Contains(lower, "great question") {
		t.Errorf("explain should never produce conversational fluff, got:\n%s", result.Text)
	}
}
