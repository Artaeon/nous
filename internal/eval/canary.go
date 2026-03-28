package eval

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"
)

// CanaryPrompt is one test case in the canary set, derived from real failures.
type CanaryPrompt struct {
	ID           string   `json:"id"`
	Query        string   `json:"query"`
	ExpectedType string   `json:"expected_type"` // "explain", "compare", "factual_qa", "plan", "tool", "summarize", "style", "multi_turn"
	MustRoute    string   `json:"must_route"`    // expected action: "lookup_knowledge", "compare", "schedule", etc.
	MustNotMatch []string `json:"must_not_match"` // patterns that indicate failure
	MustMatch    []string `json:"must_match"`     // patterns that indicate success
	FailureType  string   `json:"failure_type"`   // what went wrong originally: "filler", "wrong_route", "chatty_fallback", "low_info"
}

// CanarySet is the fixed 50-prompt test set for weekly regression testing.
type CanarySet struct {
	Prompts []CanaryPrompt `json:"prompts"`
	Version string         `json:"version"`
	Created time.Time      `json:"created"`
}

// GenerateCanarySet produces the 50-prompt canary set based on real failure patterns.
func GenerateCanarySet() *CanarySet {
	cs := &CanarySet{
		Version: "1.0",
		Created: time.Now(),
	}

	// Category 1: Task prompts that must NOT route to conversational (15 prompts)
	// Failure type: wrong_route — these were misrouted to empathy/greeting/conversation
	taskRouting := []CanaryPrompt{
		{Query: "explain how photosynthesis works", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"i appreciate", "great question", "gotcha"}, FailureType: "wrong_route"},
		{Query: "what is quantum entanglement", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"how are you", "night owl"}, FailureType: "wrong_route"},
		{Query: "compare Python and Go", ExpectedType: "compare", MustRoute: "compare",
			MustNotMatch: []string{"i hear you"}, FailureType: "wrong_route"},
		{Query: "walk me through TCP/IP", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"good question", "i see"}, FailureType: "wrong_route"},
		{Query: "summarize the French Revolution", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"your energy"}, FailureType: "wrong_route"},
		{Query: "give me an overview of machine learning", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"take a breath"}, FailureType: "wrong_route"},
		{Query: "describe how vaccines work", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"i understand how you feel"}, FailureType: "wrong_route"},
		{Query: "tell me about the Roman Empire", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"gotcha", "i see"}, FailureType: "wrong_route"},
		{Query: "explain why the sky is blue", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"interesting"}, FailureType: "wrong_route"},
		{Query: "what are the differences between TCP and UDP", ExpectedType: "compare", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"i appreciate"}, FailureType: "wrong_route"},
		{Query: "define epistemology", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"sure!", "absolutely!"}, FailureType: "wrong_route"},
		{Query: "pros and cons of remote work", ExpectedType: "compare", MustRoute: "compare",
			MustNotMatch: []string{"i hear you"}, FailureType: "wrong_route"},
		{Query: "how does a compiler work", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"that's great"}, FailureType: "wrong_route"},
		{Query: "compare democracy and authoritarianism", ExpectedType: "compare", MustRoute: "compare",
			MustNotMatch: []string{"i appreciate you sharing"}, FailureType: "wrong_route"},
		{Query: "explain the theory of relativity", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"how are you"}, FailureType: "wrong_route"},
	}

	// Category 2: Filler responses on task prompts (10 prompts)
	// Failure type: filler — system produced "As an AI..." or vague hedging
	noFiller := []CanaryPrompt{
		{Query: "explain blockchain technology", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"as an ai", "i think", "well,", "basically"}, FailureType: "filler"},
		{Query: "what is natural selection", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"in my opinion", "i believe", "i'd say"}, FailureType: "filler"},
		{Query: "compare React and Vue", ExpectedType: "compare", MustRoute: "compare",
			MustNotMatch: []string{"that's a great question", "let me think"}, FailureType: "filler"},
		{Query: "describe the water cycle", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"you know,", "so,", "i mean,"}, FailureType: "filler"},
		{Query: "explain how gravity works", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"well,", "basically,"}, FailureType: "filler"},
		{Query: "what is the Turing test", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"as an ai", "i think"}, FailureType: "filler"},
		{Query: "compare SQL and NoSQL databases", ExpectedType: "compare", MustRoute: "compare",
			MustNotMatch: []string{"that's interesting", "good question"}, FailureType: "filler"},
		{Query: "explain how HTTPS encryption works", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"i appreciate", "thanks for asking"}, FailureType: "filler"},
		{Query: "what is the difference between a virus and bacteria", ExpectedType: "compare", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"so,", "well,"}, FailureType: "filler"},
		{Query: "describe the process of mitosis", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"basically,", "i mean,"}, FailureType: "filler"},
	}

	// Category 3: Chatty fallback on sparse knowledge (10 prompts)
	// Failure type: chatty_fallback — should give structured uncertainty, not "I see" / "Interesting"
	structuredFallback := []CanaryPrompt{
		{Query: "explain how CRISPR gene editing works", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"gotcha", "interesting", "cool"}, MustMatch: []string{"knowledge", "don't have", "learn"},
			FailureType: "chatty_fallback"},
		{Query: "compare Rust and Zig for systems programming", ExpectedType: "compare", MustRoute: "compare",
			MustNotMatch: []string{"i see", "sure"}, FailureType: "chatty_fallback"},
		{Query: "what is homomorphic encryption", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"hmm", "wow"}, FailureType: "chatty_fallback"},
		{Query: "explain zero-knowledge proofs", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"alright", "noted"}, FailureType: "chatty_fallback"},
		{Query: "compare WebAssembly and JavaScript for performance", ExpectedType: "compare", MustRoute: "compare",
			MustNotMatch: []string{"understood", "clear"}, FailureType: "chatty_fallback"},
		{Query: "describe how quantum computers handle errors", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"thank you for telling me"}, FailureType: "chatty_fallback"},
		{Query: "explain the P vs NP problem", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"i appreciate"}, FailureType: "chatty_fallback"},
		{Query: "what is topological quantum computing", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"gotcha"}, FailureType: "chatty_fallback"},
		{Query: "compare different consensus algorithms in distributed systems", ExpectedType: "compare", MustRoute: "compare",
			MustNotMatch: []string{"i hear you"}, FailureType: "chatty_fallback"},
		{Query: "explain how transformers work in machine learning", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"good question"}, FailureType: "chatty_fallback"},
	}

	// Category 4: Compare/explain usefulness (10 prompts)
	// Failure type: low_info — response was too short or didn't have structure
	useful := []CanaryPrompt{
		{Query: "explain DNA replication step by step", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"i appreciate"}, FailureType: "low_info"},
		{Query: "compare capitalism and socialism", ExpectedType: "compare", MustRoute: "compare",
			MustNotMatch: []string{"gotcha"}, FailureType: "low_info"},
		{Query: "explain how the internet works", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"interesting"}, FailureType: "low_info"},
		{Query: "compare electric cars and hydrogen cars", ExpectedType: "compare", MustRoute: "compare",
			MustNotMatch: []string{"sure"}, FailureType: "low_info"},
		{Query: "explain the greenhouse effect", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"cool"}, FailureType: "low_info"},
		{Query: "compare functional and object-oriented programming", ExpectedType: "compare", MustRoute: "compare",
			MustNotMatch: []string{"hmm"}, FailureType: "low_info"},
		{Query: "explain how mRNA vaccines work", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"wow"}, FailureType: "low_info"},
		{Query: "compare renewable and fossil fuel energy sources", ExpectedType: "compare", MustRoute: "compare",
			MustNotMatch: []string{"alright"}, FailureType: "low_info"},
		{Query: "explain the halting problem", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"noted"}, FailureType: "low_info"},
		{Query: "compare agile and waterfall methodologies", ExpectedType: "compare", MustRoute: "compare",
			MustNotMatch: []string{"understood"}, FailureType: "low_info"},
	}

	// Category 5: Edge cases — emotional-looking task prompts (5 prompts)
	// Failure type: wrong_route — these have emotional words but ARE task prompts
	emotionalTasks := []CanaryPrompt{
		{Query: "explain why I feel sad after breakups", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"i'm here for you", "that must be hard"}, FailureType: "wrong_route"},
		{Query: "compare grief and clinical depression", ExpectedType: "compare", MustRoute: "compare",
			MustNotMatch: []string{"i hear you", "take care"}, FailureType: "wrong_route"},
		{Query: "what is anxiety from a neuroscience perspective", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"i understand how you feel"}, FailureType: "wrong_route"},
		{Query: "describe the stages of grief", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"i'm sorry to hear"}, FailureType: "wrong_route"},
		{Query: "explain stress management techniques", ExpectedType: "explain", MustRoute: "lookup_knowledge",
			MustNotMatch: []string{"take a breath"}, FailureType: "wrong_route"},
	}

	id := 1
	for _, group := range [][]CanaryPrompt{taskRouting, noFiller, structuredFallback, useful, emotionalTasks} {
		for _, p := range group {
			p.ID = fmt.Sprintf("canary_%03d", id)
			id++
			cs.Prompts = append(cs.Prompts, p)
		}
	}

	return cs
}

// SaveCanarySet writes the canary set to a JSON file.
func SaveCanarySet(cs *CanarySet, path string) error {
	data, err := json.MarshalIndent(cs, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// LoadCanarySet reads a canary set from a JSON file.
func LoadCanarySet(path string) (*CanarySet, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cs CanarySet
	if err := json.Unmarshal(data, &cs); err != nil {
		return nil, err
	}
	return &cs, nil
}

// CanaryResult captures results from running the canary set.
type CanaryResult struct {
	TotalPrompts      int
	CorrectlyRouted   int
	FillerDetected    int
	UsefulResponses   int
	FailedPrompts     []CanaryFailure
	RoutingAccuracy   float64 // % task prompts routed correctly
	FillerRate        float64 // % filler responses
	UsefulRate        float64 // % compare/explain judged useful
}

// CanaryFailure records one canary prompt failure.
type CanaryFailure struct {
	Prompt    CanaryPrompt
	Response  string
	Reason    string // "wrong_route", "filler_detected", "not_useful", "fluff_detected"
}

// ValidateCanaryResponse checks a response against a canary prompt's criteria.
func ValidateCanaryResponse(prompt CanaryPrompt, response string, routedAction string) *CanaryFailure {
	lower := strings.ToLower(response)

	// Check routing
	if prompt.MustRoute != "" && routedAction != prompt.MustRoute {
		// Allow lookup_knowledge as acceptable for compare prompts that route there
		if !(prompt.MustRoute == "compare" && routedAction == "lookup_knowledge") {
			return &CanaryFailure{
				Prompt:   prompt,
				Response: response,
				Reason:   fmt.Sprintf("wrong_route: got %q, want %q", routedAction, prompt.MustRoute),
			}
		}
	}

	// Check must-not-match patterns
	for _, pattern := range prompt.MustNotMatch {
		if strings.Contains(lower, strings.ToLower(pattern)) {
			return &CanaryFailure{
				Prompt:   prompt,
				Response: response,
				Reason:   fmt.Sprintf("fluff_detected: contains %q", pattern),
			}
		}
	}

	// Check must-match patterns (if any)
	if len(prompt.MustMatch) > 0 {
		anyMatch := false
		for _, pattern := range prompt.MustMatch {
			if strings.Contains(lower, strings.ToLower(pattern)) {
				anyMatch = true
				break
			}
		}
		if !anyMatch {
			return &CanaryFailure{
				Prompt:   prompt,
				Response: response,
				Reason:   fmt.Sprintf("missing_expected: none of %v found", prompt.MustMatch),
			}
		}
	}

	return nil // pass
}

// MergeGateConfig defines merge-blocking quality gates.
type MergeGateConfig struct {
	MaxFillerRate        float64 // max filler rate for task prompts (0.0 = zero tolerance)
	MinIntentAccuracy    float64 // min intent routing accuracy (0.90)
	MinUsefulFallback    float64 // min rate of useful fallback responses (0.70)
	MinCanaryPassRate    float64 // min canary set pass rate (0.90)
}

// DefaultMergeGateConfig returns the standard merge gate thresholds.
func DefaultMergeGateConfig() *MergeGateConfig {
	return &MergeGateConfig{
		MaxFillerRate:     0.0,
		MinIntentAccuracy: 0.90,
		MinUsefulFallback: 0.70,
		MinCanaryPassRate: 0.90,
	}
}

// CheckMergeGates validates canary results against merge gate thresholds.
func CheckMergeGates(result *CanaryResult, config *MergeGateConfig) []string {
	var failures []string

	if result.FillerRate > config.MaxFillerRate {
		failures = append(failures, fmt.Sprintf(
			"BLOCKED: filler rate %.1f%% exceeds maximum %.1f%%",
			result.FillerRate*100, config.MaxFillerRate*100))
	}
	if result.RoutingAccuracy < config.MinIntentAccuracy {
		failures = append(failures, fmt.Sprintf(
			"BLOCKED: intent accuracy %.1f%% below minimum %.1f%%",
			result.RoutingAccuracy*100, config.MinIntentAccuracy*100))
	}
	if result.UsefulRate < config.MinUsefulFallback {
		failures = append(failures, fmt.Sprintf(
			"BLOCKED: useful fallback rate %.1f%% below minimum %.1f%%",
			result.UsefulRate*100, config.MinUsefulFallback*100))
	}

	passRate := 1.0
	if result.TotalPrompts > 0 {
		passRate = float64(result.TotalPrompts-len(result.FailedPrompts)) / float64(result.TotalPrompts)
	}
	if passRate < config.MinCanaryPassRate {
		failures = append(failures, fmt.Sprintf(
			"BLOCKED: canary pass rate %.1f%% below minimum %.1f%%",
			passRate*100, config.MinCanaryPassRate*100))
	}

	return failures
}

// -----------------------------------------------------------------------
// Three KPIs: routing accuracy, filler rate, compare/explain usefulness
// -----------------------------------------------------------------------

// TaskKPIs tracks the three key performance indicators the user specified.
type TaskKPIs struct {
	// KPI 1: % task prompts routed correctly
	TaskPromptsTotal   int
	TaskPromptsCorrect int

	// KPI 2: % filler responses
	ResponsesTotal int
	FillerCount    int

	// KPI 3: % compare/explain judged useful
	ExplainCompareTotal  int
	ExplainCompareUseful int
}

// RecordRouting logs a routing decision.
func (k *TaskKPIs) RecordRouting(correct bool) {
	k.TaskPromptsTotal++
	if correct {
		k.TaskPromptsCorrect++
	}
}

// RecordFiller logs whether a response contained filler.
func (k *TaskKPIs) RecordFiller(hasFiller bool) {
	k.ResponsesTotal++
	if hasFiller {
		k.FillerCount++
	}
}

// RecordUsefulness logs whether an explain/compare response was useful.
func (k *TaskKPIs) RecordUsefulness(useful bool) {
	k.ExplainCompareTotal++
	if useful {
		k.ExplainCompareUseful++
	}
}

// RoutingAccuracy returns KPI 1.
func (k *TaskKPIs) RoutingAccuracy() float64 {
	if k.TaskPromptsTotal == 0 {
		return 1.0
	}
	return float64(k.TaskPromptsCorrect) / float64(k.TaskPromptsTotal)
}

// FillerRate returns KPI 2.
func (k *TaskKPIs) FillerRate() float64 {
	if k.ResponsesTotal == 0 {
		return 0.0
	}
	return float64(k.FillerCount) / float64(k.ResponsesTotal)
}

// UsefulRate returns KPI 3.
func (k *TaskKPIs) UsefulRate() float64 {
	if k.ExplainCompareTotal == 0 {
		return 1.0
	}
	return float64(k.ExplainCompareUseful) / float64(k.ExplainCompareTotal)
}

// Report returns a concise KPI summary.
func (k *TaskKPIs) Report() string {
	return fmt.Sprintf(
		"KPI 1 — Routing accuracy: %.1f%% (%d/%d)\n"+
			"KPI 2 — Filler rate: %.1f%% (%d/%d)\n"+
			"KPI 3 — Explain/Compare useful: %.1f%% (%d/%d)",
		k.RoutingAccuracy()*100, k.TaskPromptsCorrect, k.TaskPromptsTotal,
		k.FillerRate()*100, k.FillerCount, k.ResponsesTotal,
		k.UsefulRate()*100, k.ExplainCompareUseful, k.ExplainCompareTotal,
	)
}
