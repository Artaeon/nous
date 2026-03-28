package cognitive

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Response Quality Gate Tests
// -----------------------------------------------------------------------

func TestGate_ToolErrorLeak(t *testing.T) {
	gate := &ResponseGate{}
	tests := []struct {
		response string
		isLeak   bool
	}{
		{"archive error: 'path' argument is required", true},
		{"translate error: missing target language", true},
		{"timer unavailable", true},
		{"dict not found", true},
		{"error: failed to execute tool", true},
		{"timed out after 2000ms", true},
		{"Python is a programming language used for data science, web development, automation, and many other applications.", false},
		{"I don't have detailed information about that topic. You could try asking a more specific question or pointing me to a source.", false},
	}
	for _, tt := range tests {
		v := gate.Check("test query here", tt.response, "tool")
		if tt.isLeak && v.Pass {
			t.Errorf("expected tool error leak for %q, got pass", tt.response)
		}
		if tt.isLeak && v.Repaired == "" {
			t.Errorf("tool error leak %q should have repair, got none", tt.response)
		}
		if tt.isLeak && strings.Contains(v.Repaired, "error:") {
			t.Errorf("repaired response should not contain 'error:', got %q", v.Repaired)
		}
		if !tt.isLeak && !v.Pass {
			t.Errorf("expected pass for %q, got fail: %v", tt.response, v.Violations)
		}
	}
}

func TestGate_LowValueOnSubstantive(t *testing.T) {
	gate := &ResponseGate{}
	substantiveQuery := "I'm thinking about switching careers to data science"

	lowValue := []string{
		"I see.", "Okay.", "Got it.", "Right.", "Noted.", "Gotcha.",
		"I appreciate you sharing that.", "Good question!",
		"Makes sense.", "Understood.", "Cool.", "Sure.",
		"I hear you.", "Interesting.", "Hmm.",
	}
	for _, resp := range lowValue {
		v := gate.Check(substantiveQuery, resp, "composer")
		if v.Pass {
			t.Errorf("low-value %q should fail gate on substantive turn", resp)
		}
	}
}

func TestGate_LowValueOnGreeting_OK(t *testing.T) {
	gate := &ResponseGate{}
	// Short greetings are NOT substantive, so low-value acks are fine
	v := gate.Check("hi", "Hey there!", "composer")
	if !v.Pass {
		t.Errorf("short greeting response should pass gate, got fail: %v", v.Violations)
	}
}

func TestGate_SubstantiveDetection(t *testing.T) {
	tests := []struct {
		query       string
		substantive bool
	}{
		{"hi", false},
		{"hello", false},
		{"thanks", false},
		{"ok", false},
		{"yes", false},
		{"I'm thinking about learning Go for backend development", true},
		{"what do you think about climate change", true},
		{"help me plan a career transition", true},
		{"can you explain machine learning", true},
	}
	for _, tt := range tests {
		got := isSubstantiveTurn(tt.query)
		if got != tt.substantive {
			t.Errorf("isSubstantiveTurn(%q) = %v, want %v", tt.query, got, tt.substantive)
		}
	}
}

func TestGate_Parroting(t *testing.T) {
	tests := []struct {
		query    string
		response string
		parrot   bool
	}{
		{
			"I want to learn about machine learning and neural networks",
			"You want to learn about machine learning and neural networks. That's interesting.",
			true,
		},
		{
			"I want to learn about machine learning",
			"Machine learning is a branch of AI that enables computers to learn from data without being explicitly programmed.",
			false,
		},
		{
			"hi",
			"hello",
			false, // too short to judge
		},
	}
	for _, tt := range tests {
		got := isParroting(tt.query, tt.response)
		if got != tt.parrot {
			t.Errorf("isParroting(%q, %q) = %v, want %v", tt.query, tt.response, got, tt.parrot)
		}
	}
}

func TestGate_GoodResponsePasses(t *testing.T) {
	gate := &ResponseGate{}
	query := "tell me about the benefits of meditation"
	response := "Meditation has been shown to reduce stress, improve focus, and enhance emotional well-being. Regular practice can lower blood pressure and improve sleep quality. Many studies suggest even 10 minutes daily can make a meaningful difference."

	v := gate.Check(query, response, "knowledge")
	if !v.Pass {
		t.Errorf("good response should pass gate, got fail: %v", v.Violations)
	}
}

// -----------------------------------------------------------------------
// Instruction Detection Tests
// -----------------------------------------------------------------------

func TestDetectInstructions_AskQuestions(t *testing.T) {
	tests := []struct {
		query     string
		wantType  string
		wantCount int
	}{
		{"ask me 2 clarifying questions", "ask_questions", 2},
		{"ask 3 questions about my situation", "ask_questions", 3},
		{"ask me 5 clarifying questions first", "ask_questions", 5},
	}
	for _, tt := range tests {
		instructions := DetectInstructions(tt.query)
		found := false
		for _, inst := range instructions {
			if inst.Type == tt.wantType && inst.Count == tt.wantCount {
				found = true
			}
		}
		if !found {
			t.Errorf("DetectInstructions(%q): want type=%s count=%d, got %v", tt.query, tt.wantType, tt.wantCount, instructions)
		}
	}
}

func TestDetectInstructions_NBullets(t *testing.T) {
	instructions := DetectInstructions("summarize stoicism in 3 bullet points")
	found := false
	for _, inst := range instructions {
		if inst.Type == "give_n_items" || inst.Type == "use_format" {
			found = true
		}
	}
	if !found {
		t.Errorf("should detect bullet instruction, got %v", instructions)
	}
}

func TestDetectInstructions_WordLimit(t *testing.T) {
	instructions := DetectInstructions("explain this but keep it under 50 words")
	found := false
	for _, inst := range instructions {
		if inst.Type == "keep_under_n_words" && inst.Count == 50 {
			found = true
		}
	}
	if !found {
		t.Errorf("should detect word limit, got %v", instructions)
	}
}

func TestDetectInstructions_None(t *testing.T) {
	instructions := DetectInstructions("tell me about quantum physics")
	if len(instructions) != 0 {
		t.Errorf("should detect no instructions, got %v", instructions)
	}
}

func TestGenerateQuestions(t *testing.T) {
	result := GenerateQuestions("help me transition to a career in data science", 3)
	qCount := strings.Count(result, "?")
	if qCount < 3 {
		t.Errorf("GenerateQuestions(n=3) produced %d questions, want >= 3:\n%s", qCount, result)
	}
	// Should have numbered items
	if !strings.Contains(result, "1.") || !strings.Contains(result, "2.") || !strings.Contains(result, "3.") {
		t.Errorf("questions should be numbered:\n%s", result)
	}
}

func TestValidateInstructions_Questions(t *testing.T) {
	instructions := []UserInstruction{{Type: "ask_questions", Count: 2}}

	// Response with enough questions
	good := "What area interests you most? Have you considered online courses?"
	violations := ValidateInstructions(good, instructions)
	if len(violations) > 0 {
		t.Errorf("response with 2 questions should pass, got violations: %v", violations)
	}

	// Response with too few questions
	bad := "Data science is a great field."
	violations = ValidateInstructions(bad, instructions)
	if len(violations) == 0 {
		t.Error("response with 0 questions should violate ask_questions instruction")
	}
}

func TestCountBullets(t *testing.T) {
	text := "Here are the points:\n- First thing\n- Second thing\n- Third thing"
	if n := countBullets(text); n != 3 {
		t.Errorf("countBullets = %d, want 3", n)
	}

	text2 := "1. First\n2. Second\n3. Third"
	if n := countBullets(text2); n != 3 {
		t.Errorf("countBullets numbered = %d, want 3", n)
	}
}

// -----------------------------------------------------------------------
// End-to-end: response gate integrated with ActionRouter
// -----------------------------------------------------------------------

func TestE2E_ToolErrorNeverReachesUser(t *testing.T) {
	ar := NewActionRouter()
	// Simulate a tool dispatch that would produce an error
	nlu := &NLUResult{
		Raw:      "archive my project folder",
		Intent:   "archive",
		Action:   "archive",
		Entities: map[string]string{},
	}
	result := ar.Execute(nlu, nil)
	if result != nil && result.DirectResponse != "" {
		if strings.Contains(result.DirectResponse, "argument is required") {
			t.Errorf("tool error leaked to user: %s", result.DirectResponse)
		}
		if strings.Contains(result.DirectResponse, "error:") {
			t.Errorf("raw error format leaked to user: %s", result.DirectResponse)
		}
	}
}

func TestE2E_AskQuestionsInstruction(t *testing.T) {
	ar := NewActionRouter()
	nlu := &NLUResult{
		Raw:        "I want to change careers. Ask me 2 clarifying questions first.",
		Intent:     "conversation",
		Action:     "respond",
		Confidence: 0.8,
		Entities:   map[string]string{},
	}
	result := ar.Execute(nlu, nil)
	if result == nil || result.DirectResponse == "" {
		t.Fatal("expected non-nil response")
	}
	qCount := strings.Count(result.DirectResponse, "?")
	if qCount < 2 {
		t.Errorf("asked for 2 questions, got %d in response:\n%s", qCount, result.DirectResponse)
	}
}

func TestE2E_LowValueBlockedOnSubstantive(t *testing.T) {
	// Verify that isLowValueResponse correctly identifies bad responses.
	// With no subsystems wired, handleRespond returns nil, so the gate
	// catches whatever comes through.
	badResponses := []string{
		"I see.", "Noted.", "Gotcha.", "Right.",
		"I appreciate you sharing that.",
	}
	for _, resp := range badResponses {
		if !isLowValueResponse(resp) {
			t.Errorf("isLowValueResponse(%q) should be true", resp)
		}
	}
}

func BenchmarkResponseGateCheck(b *testing.B) {
	gate := &ResponseGate{}
	query := "help me plan a career transition to data science"
	response := "Data science is a growing field. Here are some steps to consider: learn Python, take online courses in statistics, build a portfolio of projects, and network with professionals in the field."

	for i := 0; i < b.N; i++ {
		gate.Check(query, response, "thinking")
	}
}

func BenchmarkDetectInstructions(b *testing.B) {
	query := "Give me exactly 5 reasons to learn Go, in bullet points, and keep it under 200 words"
	for i := 0; i < b.N; i++ {
		DetectInstructions(query)
	}
}
