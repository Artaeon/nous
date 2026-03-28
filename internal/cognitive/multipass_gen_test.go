package cognitive

import (
	"strings"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func makeTestThinkingEngine() *ThinkingEngine {
	graph := NewCognitiveGraph("")
	composer := NewComposer(graph, nil, nil, nil)
	te := NewThinkingEngine(graph, composer)

	// Seed the graph with some facts for testing
	graph.AddEdge("go", "programming language", RelIsA, "test")
	graph.AddEdge("go", "systems software", RelUsedFor, "test")
	graph.AddEdge("go", "2009", RelFoundedIn, "test")
	graph.AddEdge("go", "concurrency", RelHas, "test")
	graph.AddEdge("go", "google", RelCreatedBy, "test")

	graph.AddEdge("python", "programming language", RelIsA, "test")
	graph.AddEdge("python", "data science", RelUsedFor, "test")
	graph.AddEdge("python", "1991", RelFoundedIn, "test")

	graph.AddEdge("rust", "programming language", RelIsA, "test")
	graph.AddEdge("rust", "memory safety", RelHas, "test")
	graph.AddEdge("rust", "systems programming", RelUsedFor, "test")

	return te
}

func makeSamplePlan() *ContentPlan {
	return &ContentPlan{
		Topic:  "go",
		Thesis: "Go is a programming language designed for simplicity and concurrency.",
		Claims: []PlanClaim{
			{Text: "Go is a programming language.", Priority: 100},
			{Text: "Go is used for systems software.", Priority: 90},
			{Text: "Go was founded in 2009.", Priority: 50},
			{Text: "Go has concurrency.", Priority: 80},
		},
	}
}

// ---------------------------------------------------------------------------
// TestMultiPassGenerate_ExplainTask
// ---------------------------------------------------------------------------

func TestMultiPassGenerate_ExplainTask(t *testing.T) {
	te := makeTestThinkingEngine()

	config := DefaultMultiPassConfig()
	params := &TaskParams{Topic: "go", Keywords: []string{"go"}}

	result := te.MultiPassGenerate("explain go", TaskTeach, params, config)
	if result == nil {
		t.Fatal("MultiPassGenerate returned nil")
	}

	// Should have at least plan + draft passes
	if result.TotalPasses < 2 {
		t.Errorf("expected at least 2 passes, got %d", result.TotalPasses)
	}

	// First pass should be plan
	if len(result.Passes) > 0 && result.Passes[0].Pass != PassPlan {
		t.Errorf("expected first pass to be plan, got %s", passName(result.Passes[0].Pass))
	}

	// Second pass should be draft
	if len(result.Passes) > 1 && result.Passes[1].Pass != PassDraft {
		t.Errorf("expected second pass to be draft, got %s", passName(result.Passes[1].Pass))
	}

	// Verify plan was built
	if result.Passes[0].Plan == nil {
		t.Error("expected plan pass to produce a plan")
	}

	// Verify final text is not empty
	if strings.TrimSpace(result.FinalText) == "" {
		t.Error("expected non-empty final text")
	}

	// Verify score was computed
	if result.FinalScore.Total == 0 {
		t.Error("expected non-zero final score")
	}

	// With verification enabled, should have verify pass
	hasVerify := false
	for _, p := range result.Passes {
		if p.Pass == PassVerify {
			hasVerify = true
			break
		}
	}
	if config.EnableVerification && !hasVerify {
		t.Error("expected verification pass when EnableVerification is true")
	}
}

// ---------------------------------------------------------------------------
// TestMultiPassGenerate_CompareTask
// ---------------------------------------------------------------------------

func TestMultiPassGenerate_CompareTask(t *testing.T) {
	te := makeTestThinkingEngine()

	params := &TaskParams{
		Topic:    "go vs python",
		ItemA:    "go",
		ItemB:    "python",
		Keywords: []string{"go", "python"},
	}
	config := DefaultMultiPassConfig()

	result := te.MultiPassGenerate("compare go and python", TaskCompare, params, config)
	if result == nil {
		t.Fatal("MultiPassGenerate returned nil")
	}

	if result.TotalPasses < 2 {
		t.Errorf("expected at least 2 passes for compare task, got %d", result.TotalPasses)
	}

	// Compare tasks should require a plan
	if result.Passes[0].Plan == nil {
		t.Error("compare task should produce a content plan")
	}

	if strings.TrimSpace(result.FinalText) == "" {
		t.Error("expected non-empty final text for compare task")
	}
}

// ---------------------------------------------------------------------------
// TestMultiPassGenerate_SimpleTask
// ---------------------------------------------------------------------------

func TestMultiPassGenerate_SimpleTask(t *testing.T) {
	te := makeTestThinkingEngine()

	params := &TaskParams{Topic: "hello"}
	config := &MultiPassConfig{
		MaxRefineIterations: 0,
		MinQualityScore:     0.3,
		RequirePlan:         false,
		EnableVerification:  false,
	}

	result := te.MultiPassGenerate("hello", TaskConverse, params, config)
	if result == nil {
		t.Fatal("MultiPassGenerate returned nil")
	}

	// With verification disabled, should only have plan + draft
	if result.TotalPasses > 2 {
		t.Errorf("expected at most 2 passes for simple task with no verification, got %d", result.TotalPasses)
	}
}

// ---------------------------------------------------------------------------
// TestVerifyPass
// ---------------------------------------------------------------------------

func TestVerifyPass(t *testing.T) {
	fd := NewFillerDetector()

	t.Run("missing_claims", func(t *testing.T) {
		plan := makeSamplePlan()
		// Draft that's missing some claims
		draft := "Go is a programming language. It is quite popular."
		issues := verifyDraft(draft, plan, fd)

		hasMissing := false
		for _, issue := range issues {
			if issue.Type == "missing_claim" {
				hasMissing = true
				break
			}
		}
		if !hasMissing {
			t.Error("expected missing_claim issues when draft omits plan claims")
		}
	})

	t.Run("contradictions", func(t *testing.T) {
		plan := makeSamplePlan()
		draft := "Go is fast. Go is not fast. Go is a programming language."
		issues := verifyDraft(draft, plan, fd)

		hasContradiction := false
		for _, issue := range issues {
			if issue.Type == "contradiction" {
				hasContradiction = true
				break
			}
		}
		if !hasContradiction {
			t.Error("expected contradiction issues")
		}
	})

	t.Run("filler", func(t *testing.T) {
		plan := makeSamplePlan()
		draft := "That's a great question! As an AI, I think Go is interesting. Go is a programming language."
		issues := verifyDraft(draft, plan, fd)

		hasFiller := false
		for _, issue := range issues {
			if issue.Type == "filler" {
				hasFiller = true
				break
			}
		}
		if !hasFiller {
			t.Error("expected filler issues")
		}
	})

	t.Run("repetition", func(t *testing.T) {
		plan := makeSamplePlan()
		draft := "Go is a programming language. Go is a programming language. Go is a programming language."
		issues := verifyDraft(draft, plan, fd)

		hasRepetition := false
		for _, issue := range issues {
			if issue.Type == "repetition" {
				hasRepetition = true
				break
			}
		}
		if !hasRepetition {
			t.Error("expected repetition issues for duplicate sentences")
		}
	})

	t.Run("off_topic", func(t *testing.T) {
		plan := &ContentPlan{
			Topic:  "quantum computing",
			Thesis: "Quantum computing is revolutionary.",
			Claims: []PlanClaim{{Text: "Quantum computing uses qubits."}},
		}
		// Draft that never mentions the topic
		draft := "The weather today is nice. Birds are flying. Trees are green."
		issues := verifyDraft(draft, plan, fd)

		hasOffTopic := false
		for _, issue := range issues {
			if issue.Type == "off_topic" {
				hasOffTopic = true
				break
			}
		}
		if !hasOffTopic {
			t.Error("expected off_topic issue when draft does not mention topic")
		}
	})

	t.Run("low_specificity", func(t *testing.T) {
		plan := makeSamplePlan()
		// Draft full of vague words
		draft := "Go is very really basically a thing. It does stuff and things. It is sort of kind of a lot."
		issues := verifyDraft(draft, plan, fd)

		hasLowSpec := false
		for _, issue := range issues {
			if issue.Type == "low_specificity" {
				hasLowSpec = true
				break
			}
		}
		if !hasLowSpec {
			t.Error("expected low_specificity issue for vague-heavy draft")
		}
	})
}

// ---------------------------------------------------------------------------
// TestRefinePass
// ---------------------------------------------------------------------------

func TestRefinePass(t *testing.T) {
	fd := NewFillerDetector()

	t.Run("inserts_missing_claims", func(t *testing.T) {
		plan := makeSamplePlan()
		issues := []QualityIssue{
			{Type: "missing_claim", Description: "Plan claim not covered: Go has concurrency.", Severity: "major"},
		}
		draft := "Go is a programming language. It was created in 2009."
		refined := refineDraft(draft, plan, issues, nil, fd)

		if !strings.Contains(strings.ToLower(refined), "concurrency") {
			t.Errorf("expected refined text to contain missing claim about concurrency, got %q", refined)
		}
	})

	t.Run("removes_repetition", func(t *testing.T) {
		plan := makeSamplePlan()
		issues := []QualityIssue{
			{Type: "repetition", Description: "Sentences 1 and 2 are highly similar", Severity: "major", Location: 1},
		}
		draft := "Go is a programming language. Go is a programming language. It has concurrency."
		refined := refineDraft(draft, plan, issues, nil, fd)

		sentences := splitSentences(refined)
		// Should have fewer sentences than original
		if len(sentences) >= 3 {
			t.Errorf("expected repetitive sentence to be removed, got %d sentences: %q", len(sentences), refined)
		}
	})

	t.Run("adds_topic_anchor", func(t *testing.T) {
		plan := &ContentPlan{Topic: "rust", Thesis: "Rust is safe."}
		issues := []QualityIssue{
			{Type: "off_topic", Description: "Response does not mention the topic: rust", Severity: "major"},
		}
		draft := "Memory safety is important in modern systems."
		refined := refineDraft(draft, plan, issues, nil, fd)

		if !strings.Contains(strings.ToLower(refined), "rust") {
			t.Errorf("expected refined text to anchor to topic 'rust', got %q", refined)
		}
	})
}

// ---------------------------------------------------------------------------
// TestContentPlanEnforcer_Explain
// ---------------------------------------------------------------------------

func TestContentPlanEnforcer_Explain(t *testing.T) {
	plan := &ContentPlan{
		Topic:  "go",
		Thesis: "Go is a programming language.",
		Claims: []PlanClaim{
			{Text: "Go is a programming language.", Evidence: []edgeFact{{Subject: "go", Relation: RelIsA, Object: "programming language"}}, Priority: 100},
			{Text: "Go is used for systems software.", Evidence: []edgeFact{{Subject: "go", Relation: RelUsedFor, Object: "systems software"}}, Priority: 90},
			{Text: "Go was founded in 2009.", Evidence: []edgeFact{{Subject: "go", Relation: RelFoundedIn, Object: "2009"}}, Priority: 50},
		},
	}
	template := GetPlanTemplate(TaskTeach)
	enforced := EnforceContentPlan(plan, template, "go")

	if enforced == nil {
		t.Fatal("EnforceContentPlan returned nil")
	}

	// Thesis should be set
	if enforced.Thesis == "" {
		t.Error("expected thesis to be set")
	}

	// Should have at least MinClaims claims
	if len(enforced.Claims) < template.MinClaims {
		t.Errorf("expected at least %d claims, got %d", template.MinClaims, len(enforced.Claims))
	}

	// Evidence should be present for supported claims
	supportedCount := 0
	for _, c := range enforced.Claims {
		if c.Supported {
			supportedCount++
		}
	}
	if supportedCount == 0 {
		t.Error("expected at least one supported claim with evidence")
	}

	// Uncertainty should be generated
	if enforced.Uncertainty == "" {
		t.Error("expected uncertainty statement for explain task")
	}

	// Recap should be generated
	if enforced.Recap == "" {
		t.Error("expected recap for explain task")
	}

	// Check completeness
	if !enforced.Complete {
		t.Errorf("expected complete plan, missing: %v", enforced.Missing)
	}
}

// ---------------------------------------------------------------------------
// TestContentPlanEnforcer_Compare
// ---------------------------------------------------------------------------

func TestContentPlanEnforcer_Compare(t *testing.T) {
	plan := &ContentPlan{
		Topic:  "go vs python",
		Thesis: "Go and Python serve different niches.",
		Claims: []PlanClaim{
			{Text: "Go is used for systems software.", Priority: 90},
			{Text: "Python is used for data science.", Priority: 90},
		},
	}
	template := GetPlanTemplate(TaskCompare)
	enforced := EnforceContentPlan(plan, template, "go vs python")

	if enforced == nil {
		t.Fatal("EnforceContentPlan returned nil")
	}

	if len(enforced.Claims) < template.MinClaims {
		t.Errorf("expected at least %d claims for compare, got %d", template.MinClaims, len(enforced.Claims))
	}

	if enforced.Uncertainty == "" {
		t.Error("expected uncertainty statement for compare task")
	}

	if enforced.Recap == "" {
		t.Error("expected recap for compare task")
	}
}

// ---------------------------------------------------------------------------
// TestGetPlanTemplate
// ---------------------------------------------------------------------------

func TestGetPlanTemplate(t *testing.T) {
	tasks := []ThinkTask{
		TaskTeach, TaskCompare, TaskAnalyze, TaskPlan, TaskDebate,
		TaskAdvise, TaskConverse, TaskCompose, TaskBrainstorm,
		TaskSummarize, TaskCreate, TaskReflect,
	}

	for _, task := range tasks {
		tmpl := GetPlanTemplate(task)
		if tmpl == nil {
			t.Errorf("GetPlanTemplate(%s) returned nil", taskName(task))
			continue
		}
		if len(tmpl.RequiredParts) == 0 {
			t.Errorf("GetPlanTemplate(%s) has no required parts", taskName(task))
		}
		if tmpl.MinClaims < 1 {
			t.Errorf("GetPlanTemplate(%s) has MinClaims < 1: %d", taskName(task), tmpl.MinClaims)
		}
		if tmpl.MaxClaims < tmpl.MinClaims {
			t.Errorf("GetPlanTemplate(%s) has MaxClaims < MinClaims", taskName(task))
		}
	}
}

// ---------------------------------------------------------------------------
// TestDeterministicReranker
// ---------------------------------------------------------------------------

func TestDeterministicReranker(t *testing.T) {
	t.Run("coverage_signal", func(t *testing.T) {
		plan := makeSamplePlan()
		high := "Go is a programming language. Go is used for systems software. Go was founded in 2009. Go has concurrency."
		low := "Go is interesting."

		scoreHigh := scoreCoverageSignal(high, plan)
		scoreLow := scoreCoverageSignal(low, plan)

		if scoreHigh <= scoreLow {
			t.Errorf("expected high-coverage text to score higher: high=%.2f low=%.2f", scoreHigh, scoreLow)
		}
		if scoreHigh < 0.8 {
			t.Errorf("expected high-coverage score > 0.8, got %.2f", scoreHigh)
		}
	})

	t.Run("contradiction_signal", func(t *testing.T) {
		clean := "Go is fast. Go is efficient. Go is concurrent."
		contradictory := "Go is fast. Go is not fast. Go is concurrent."

		scoreClean := scoreContradictionSignal(clean)
		scoreContra := scoreContradictionSignal(contradictory)

		if scoreContra >= scoreClean {
			t.Errorf("expected contradictory text to score lower: clean=%.2f contradictory=%.2f", scoreClean, scoreContra)
		}
		if scoreClean != 1.0 {
			t.Errorf("expected perfect contradiction score for clean text, got %.2f", scoreClean)
		}
	})

	t.Run("specificity_signal", func(t *testing.T) {
		specific := "Go compiles to native machine code with garbage collection and concurrent goroutines."
		vague := "Go is very really basically things and stuff. It does a lot of various stuff."

		scoreSpecific := scoreSpecificitySignal(specific)
		scoreVague := scoreSpecificitySignal(vague)

		if scoreVague >= scoreSpecific {
			t.Errorf("expected vague text to score lower: specific=%.2f vague=%.2f", scoreSpecific, scoreVague)
		}
	})

	t.Run("repetition_signal", func(t *testing.T) {
		varied := "Go is a compiled language. It offers concurrency through goroutines. The standard library is comprehensive."
		repetitive := "Go is a compiled language. Go is a compiled language. Go is a compiled language."

		scoreVaried := scoreRepetitionSignal(varied)
		scoreRep := scoreRepetitionSignal(repetitive)

		if scoreRep >= scoreVaried {
			t.Errorf("expected repetitive text to score lower: varied=%.2f repetitive=%.2f", scoreVaried, scoreRep)
		}
	})

	t.Run("user_goal_signal", func(t *testing.T) {
		query := "explain how Go handles concurrency"
		relevant := "Go handles concurrency using goroutines and channels. These primitives make concurrent programming straightforward."
		irrelevant := "The weather is nice today. Birds are singing in the trees."

		scoreRel := scoreUserGoalFit(relevant, query)
		scoreIrr := scoreUserGoalFit(irrelevant, query)

		if scoreIrr >= scoreRel {
			t.Errorf("expected relevant text to score higher: relevant=%.2f irrelevant=%.2f", scoreRel, scoreIrr)
		}
	})
}

// ---------------------------------------------------------------------------
// TestRerankerIntegration
// ---------------------------------------------------------------------------

func TestRerankerIntegration(t *testing.T) {
	reranker := NewDeterministicReranker()
	plan := makeSamplePlan()
	query := "explain go"

	candidates := []string{
		"Go is interesting.",
		"Go is a programming language. Go is used for systems software. Go was founded in 2009. Go has concurrency features that make it ideal for modern systems.",
		"The weather is nice.",
	}

	result := reranker.Rerank(candidates, plan, query)
	if result == nil {
		t.Fatal("Rerank returned nil")
	}

	if result.BestIndex != 1 {
		t.Errorf("expected candidate 1 (detailed Go text) to win, got index %d (score=%.3f)", result.BestIndex, result.BestScore)
	}

	// Verify all candidates have scores
	for i, scores := range result.Scores {
		if len(scores) != 5 {
			t.Errorf("expected 5 signals for candidate %d, got %d", i, len(scores))
		}
	}

	// Verify total scores match
	for i, total := range result.TotalScores {
		expectedTotal := 0.0
		for _, sig := range result.Scores[i] {
			expectedTotal += sig.Score * sig.Weight
		}
		diff := total - expectedTotal
		if diff < -0.001 || diff > 0.001 {
			t.Errorf("total score mismatch for candidate %d: stored=%.4f computed=%.4f", i, total, expectedTotal)
		}
	}
}

// ---------------------------------------------------------------------------
// TestFillerDetector_AllTypes
// ---------------------------------------------------------------------------

func TestFillerDetector_AllTypes(t *testing.T) {
	fd := NewFillerDetector()

	tests := []struct {
		name     string
		text     string
		fillerType string
	}{
		{"ai_prefix", "As an AI, I cannot have personal experiences.", "ai_prefix"},
		{"meta_comment", "That's a great question! Let me explain.", "meta_comment"},
		{"hedge_well", "Well, it depends on context.", "hedge"},
		{"hedge_so", "So, let me think about that.", "hedge"},
		{"hedge_basically", "Basically, it works like this.", "hedge"},
		{"vague_things", "There are things to consider.", "vague"},
		{"vague_stuff", "It involves various stuff.", "vague"},
		{"vague_very", "It is very important.", "vague"},
		{"vague_really", "It is really good.", "vague"},
		{"ack_sure", "Sure!", "meta_comment"},
		{"ack_absolutely", "Absolutely!", "meta_comment"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			instances := fd.DetectFiller(tc.text)
			if len(instances) == 0 {
				t.Errorf("expected filler detection for %q, got none", tc.text)
				return
			}

			found := false
			for _, inst := range instances {
				if inst.Type == tc.fillerType {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("expected filler type %q in %q, got types: %v", tc.fillerType, tc.text, fillerTypes(instances))
			}
		})
	}
}

func fillerTypes(instances []FillerInstance) []string {
	var types []string
	for _, inst := range instances {
		types = append(types, inst.Type)
	}
	return types
}

// ---------------------------------------------------------------------------
// TestFillerRemoval
// ---------------------------------------------------------------------------

func TestFillerRemoval(t *testing.T) {
	fd := NewFillerDetector()

	tests := []struct {
		name     string
		input    string
		mustNotContain []string
	}{
		{
			"remove_ai_prefix",
			"As an AI, I think this is interesting. Go is a programming language.",
			[]string{"As an AI"},
		},
		{
			"remove_meta_comment",
			"That's a great question! Go is a statically typed language.",
			[]string{"great question"},
		},
		{
			"remove_hedge",
			"Well, Go is a programming language. It has concurrency.",
			[]string{"Well,"},
		},
		{
			"remove_repetitive_ack",
			"Sure! Go is a compiled language.",
			[]string{"Sure!"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cleaned := fd.RemoveFiller(tc.input)
			if strings.TrimSpace(cleaned) == "" {
				t.Error("filler removal produced empty string")
				return
			}
			for _, banned := range tc.mustNotContain {
				if strings.Contains(cleaned, banned) {
					t.Errorf("expected %q to be removed, got %q", banned, cleaned)
				}
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestNoFillerPolicy
// ---------------------------------------------------------------------------

func TestNoFillerPolicy(t *testing.T) {
	fd := NewFillerDetector()

	// Task prompt with filler should be cleaned
	fillerText := "As an AI, I think this is interesting. Well, basically, Go is a programming language."
	cleaned, changed := fd.EnforcePolicy(fillerText, true)

	if !changed {
		t.Error("expected policy to detect changes for filler-heavy text")
	}

	// Cleaned text should not have AI prefix
	if strings.Contains(cleaned, "As an AI") {
		t.Error("expected AI prefix to be removed")
	}

	// Clean text should pass without changes
	cleanText := "Go is a statically typed programming language developed at Google."
	_, changed = fd.EnforcePolicy(cleanText, true)
	if changed {
		t.Error("expected clean text to pass policy without changes")
	}
}

// ---------------------------------------------------------------------------
// TestStructuredUncertainty
// ---------------------------------------------------------------------------

func TestStructuredUncertainty(t *testing.T) {
	tests := []struct {
		name       string
		text       string
		isStructured bool
	}{
		{
			"good_structured",
			"I don't have enough information to fully answer this. Here's what I know: Go was created at Google. What's uncertain: the exact decision-making process behind Go's design.",
			true,
		},
		{
			"good_evidence_based",
			"The evidence suggests that Go is faster for concurrent workloads, but it's unclear whether this holds for all use cases.",
			true,
		},
		{
			"bad_vague_hedging",
			"I think maybe it's probably something like that, I guess.",
			false,
		},
		{
			"empty",
			"",
			false,
		},
		{
			"bad_no_structured_marker",
			"Go is pretty good I think.",
			false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := IsStructuredUncertainty(tc.text)
			if got != tc.isStructured {
				t.Errorf("IsStructuredUncertainty(%q) = %v, want %v", tc.text, got, tc.isStructured)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestDefaultMultiPassConfig
// ---------------------------------------------------------------------------

func TestDefaultMultiPassConfig(t *testing.T) {
	config := DefaultMultiPassConfig()
	if config == nil {
		t.Fatal("DefaultMultiPassConfig returned nil")
	}
	if config.MaxRefineIterations != 2 {
		t.Errorf("expected MaxRefineIterations=2, got %d", config.MaxRefineIterations)
	}
	if config.MinQualityScore != 0.65 {
		t.Errorf("expected MinQualityScore=0.65, got %f", config.MinQualityScore)
	}
	if !config.RequirePlan {
		t.Error("expected RequirePlan=true")
	}
	if !config.EnableVerification {
		t.Error("expected EnableVerification=true")
	}
}

// ---------------------------------------------------------------------------
// TestNewDeterministicReranker
// ---------------------------------------------------------------------------

func TestNewDeterministicReranker(t *testing.T) {
	r := NewDeterministicReranker()
	if r == nil {
		t.Fatal("NewDeterministicReranker returned nil")
	}
	if len(r.Weights) != 5 {
		t.Errorf("expected 5 weights, got %d", len(r.Weights))
	}

	// Weights should sum close to 1.0
	total := 0.0
	for _, w := range r.Weights {
		total += w
	}
	if total < 0.99 || total > 1.01 {
		t.Errorf("expected weights to sum to ~1.0, got %.2f", total)
	}
}

// ---------------------------------------------------------------------------
// TestRerankEmptyCandidates
// ---------------------------------------------------------------------------

func TestRerankEmptyCandidates(t *testing.T) {
	r := NewDeterministicReranker()
	result := r.Rerank(nil, nil, "")
	if result == nil {
		t.Fatal("Rerank returned nil for empty input")
	}
	if result.BestIndex != -1 {
		t.Errorf("expected BestIndex=-1 for empty candidates, got %d", result.BestIndex)
	}
}

// ---------------------------------------------------------------------------
// TestEnforceContentPlanNilInputs
// ---------------------------------------------------------------------------

func TestEnforceContentPlanNilInputs(t *testing.T) {
	// Should handle nil plan gracefully
	enforced := EnforceContentPlan(nil, nil, "testing")
	if enforced == nil {
		t.Fatal("EnforceContentPlan returned nil for nil plan")
	}
	if enforced.Thesis == "" {
		t.Error("expected generated thesis for nil plan")
	}
	if enforced.ContentPlan == nil {
		t.Error("expected non-nil ContentPlan")
	}
}

// ---------------------------------------------------------------------------
// TestGenerateUncertaintyStatement
// ---------------------------------------------------------------------------

func TestGenerateUncertaintyStatement(t *testing.T) {
	t.Run("empty_topic", func(t *testing.T) {
		s := GenerateUncertaintyStatement("", nil)
		if s == "" {
			t.Error("expected non-empty uncertainty for empty topic")
		}
	})

	t.Run("no_evidence", func(t *testing.T) {
		claims := []PlanClaim{{Text: "Claim without evidence."}}
		s := GenerateUncertaintyStatement("go", claims)
		if s == "" {
			t.Error("expected non-empty uncertainty")
		}
		if !strings.Contains(s, "go") {
			t.Error("expected uncertainty to mention topic")
		}
	})

	t.Run("with_evidence", func(t *testing.T) {
		claims := []PlanClaim{{Text: "Go is fast.", Evidence: []edgeFact{{Subject: "go", Relation: RelIsA, Object: "fast"}}}}
		s := GenerateUncertaintyStatement("go", claims)
		if s == "" {
			t.Error("expected non-empty uncertainty")
		}
	})
}

// ---------------------------------------------------------------------------
// TestGenerateRecap
// ---------------------------------------------------------------------------

func TestGenerateRecap(t *testing.T) {
	t.Run("with_claims", func(t *testing.T) {
		claims := []PlanClaim{
			{Text: "Go is a programming language."},
			{Text: "Go has concurrency features."},
		}
		recap := GenerateRecap("Go is designed for simplicity.", claims)
		if recap == "" {
			t.Error("expected non-empty recap")
		}
		if !hasSentenceEnding(recap) {
			t.Error("expected recap to end with punctuation")
		}
	})

	t.Run("empty_inputs", func(t *testing.T) {
		recap := GenerateRecap("", nil)
		if recap == "" {
			t.Error("expected non-empty recap even with empty inputs")
		}
	})
}

// ---------------------------------------------------------------------------
// TestSentenceOverlap
// ---------------------------------------------------------------------------

func TestSentenceOverlap(t *testing.T) {
	identical := sentenceOverlap("Go is a programming language.", "Go is a programming language.")
	if identical < 0.9 {
		t.Errorf("expected high overlap for identical sentences, got %.2f", identical)
	}

	different := sentenceOverlap("Go is a programming language.", "Rust has memory safety guarantees.")
	if different > 0.3 {
		t.Errorf("expected low overlap for different sentences, got %.2f", different)
	}
}

// ---------------------------------------------------------------------------
// TestExtractComparisonItems
// ---------------------------------------------------------------------------

func TestExtractComparisonItems(t *testing.T) {
	tests := []struct {
		query string
		wantLen int
	}{
		{"compare Go vs Python", 2},
		{"compare Go and Python", 2},
		{"difference between Go and Python", 2},
		{"what is Go", 0},
	}

	for _, tc := range tests {
		items := extractComparisonItems(tc.query)
		if len(items) != tc.wantLen {
			t.Errorf("extractComparisonItems(%q) = %v (len=%d), want len %d", tc.query, items, len(items), tc.wantLen)
		}
	}
}

// ---------------------------------------------------------------------------
// TestFillerDetectorPatterns
// ---------------------------------------------------------------------------

func TestFillerDetectorPatterns(t *testing.T) {
	fd := NewFillerDetector()

	// Ensure patterns compile (already done in constructor, but verify)
	if len(fd.Patterns) == 0 {
		t.Error("expected non-empty pattern list")
	}
	if len(fd.Prefixes) == 0 {
		t.Error("expected non-empty prefix list")
	}
	if len(fd.Hedges) == 0 {
		t.Error("expected non-empty hedge list")
	}
	if len(fd.VagueWords) == 0 {
		t.Error("expected non-empty vague words list")
	}
}

// ---------------------------------------------------------------------------
// TestMultiPassResult_Accepted
// ---------------------------------------------------------------------------

func TestMultiPassResult_Accepted(t *testing.T) {
	te := makeTestThinkingEngine()

	// Use a very low threshold to ensure acceptance
	config := &MultiPassConfig{
		MaxRefineIterations: 1,
		MinQualityScore:     0.01,
		RequirePlan:         true,
		EnableVerification:  true,
	}
	params := &TaskParams{Topic: "go"}

	result := te.MultiPassGenerate("explain go", TaskTeach, params, config)
	if result == nil {
		t.Fatal("MultiPassGenerate returned nil")
	}

	if !result.Accepted {
		t.Errorf("expected result to be accepted with low threshold, score=%.3f", result.FinalScore.Total)
	}
}

// ---------------------------------------------------------------------------
// BenchmarkMultiPassGenerate
// ---------------------------------------------------------------------------

func BenchmarkMultiPassGenerate(b *testing.B) {
	te := makeTestThinkingEngine()
	config := DefaultMultiPassConfig()
	params := &TaskParams{Topic: "go", Keywords: []string{"go"}}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		te.MultiPassGenerate("explain go", TaskTeach, params, config)
	}
}

// ---------------------------------------------------------------------------
// BenchmarkReranker
// ---------------------------------------------------------------------------

func BenchmarkReranker(b *testing.B) {
	reranker := NewDeterministicReranker()
	plan := makeSamplePlan()
	candidates := []string{
		"Go is interesting.",
		"Go is a programming language. Go is used for systems software. Go was founded in 2009.",
		"The weather is nice.",
		"Go has concurrency features. It was created at Google for building scalable infrastructure.",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reranker.Rerank(candidates, plan, "explain go")
	}
}

// ---------------------------------------------------------------------------
// BenchmarkFillerDetector
// ---------------------------------------------------------------------------

func BenchmarkFillerDetector(b *testing.B) {
	fd := NewFillerDetector()
	text := "Well, that's a great question! As an AI, I think Go is very interesting. " +
		"Basically, it is a programming language. It does stuff with things. " +
		"Sort of like other languages but really very different."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fd.DetectFiller(text)
	}
}

// ---------------------------------------------------------------------------
// TestPassDuration
// ---------------------------------------------------------------------------

func TestPassDuration(t *testing.T) {
	te := makeTestThinkingEngine()
	config := DefaultMultiPassConfig()
	params := &TaskParams{Topic: "go"}

	result := te.MultiPassGenerate("explain go", TaskTeach, params, config)
	if result == nil {
		t.Fatal("MultiPassGenerate returned nil")
	}

	for _, pass := range result.Passes {
		if pass.Duration < 0 {
			t.Errorf("pass %s has negative duration", passName(pass.Pass))
		}
		if pass.Duration > 10*time.Second {
			t.Errorf("pass %s took unreasonably long: %v", passName(pass.Pass), pass.Duration)
		}
	}
}

// ---------------------------------------------------------------------------
// TestRerankerContentWords
// ---------------------------------------------------------------------------

func TestRerankerContentWords(t *testing.T) {
	words := rerankerContentWords("Go is a statically typed programming language")
	if len(words) == 0 {
		t.Fatal("expected content words from non-trivial text")
	}

	// Stop words should be filtered out
	for _, w := range words {
		if w == "is" || w == "a" {
			t.Errorf("stop word %q should be filtered", w)
		}
	}

	// Content words should remain
	found := false
	for _, w := range words {
		if w == "statically" || w == "typed" || w == "programming" || w == "language" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected content words like 'statically', 'typed', etc. in %v", words)
	}
}

// ---------------------------------------------------------------------------
// TestDetectRerankerContradiction
// ---------------------------------------------------------------------------

func TestDetectRerankerContradiction(t *testing.T) {
	tests := []struct {
		name string
		a    string
		b    string
		want bool
	}{
		{"direct_negation", "go is fast", "go is not fast", true},
		{"no_contradiction", "go is fast", "go is concurrent", false},
		{"negation_removal_match", "go is not slow", "go is slow", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := detectRerankerContradiction(tc.a, tc.b)
			if got != tc.want {
				t.Errorf("detectRerankerContradiction(%q, %q) = %v, want %v", tc.a, tc.b, got, tc.want)
			}
		})
	}
}
