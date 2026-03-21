package cognitive

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Advanced Reasoning Tests — validates the four new cognitive systems:
//
//   1. General chain-of-thought decomposer
//   2. Causal/counterfactual reasoning
//   3. Goal planning
//   4. Honest fallback (no more confusing bridges)
// -----------------------------------------------------------------------

func setupReasoningTest(t *testing.T) (*ReasoningEngine, *GraphCausalReasoner, *GoalPlanner, *CognitiveGraph) {
	t.Helper()

	dir := t.TempDir()
	graph := NewCognitiveGraph(filepath.Join(dir, "graph.json"))
	semantic := NewSemanticEngine()

	// Load packages
	packDir := filepath.Join("..", "..", "packages")
	if _, err := os.Stat(packDir); err != nil {
		t.Skip("packages directory not found")
	}
	causal := NewCausalEngine()
	patterns := NewPatternDetector()
	composer := NewComposer(graph, semantic, causal, patterns)
	loader := NewPackageLoader(graph, composer.Generative, composer, packDir)
	loader.LoadAll()

	reasoner := NewReasoningEngine(graph, semantic)
	causalReasoner := NewGraphCausalReasoner(graph)
	planner := NewGoalPlanner(graph, semantic)

	return reasoner, causalReasoner, planner, graph
}

// -----------------------------------------------------------------------
// Test 1: General Chain-of-Thought Decomposer
// -----------------------------------------------------------------------

func TestGeneralDecomposer(t *testing.T) {
	reasoner, _, _, _ := setupReasoningTest(t)

	tests := []struct {
		question    string
		expectSteps bool
		expectType  string // "comparison", "aggregation", "causal", "multi-hop", "single"
	}{
		// Comparisons
		{"How is Stoicism different from Epicureanism?", true, "comparison"},
		{"Compare Python and Go", true, "comparison"},
		{"What's the difference between DNA and RNA?", true, "comparison"},
		{"Python vs Linux", true, "comparison"},

		// Aggregation
		{"What do all Greek philosophers have in common?", true, "aggregation"},
		{"How are programming languages similar?", true, "aggregation"},

		// Causal
		{"Why does evolution happen?", true, "causal"},
		{"What causes genetic mutation?", true, "causal"},

		// Conditional
		{"What would happen if DNA didn't exist?", true, "conditional"},
		{"What if Stoicism was never founded?", true, "conditional"},

		// Multi-hop (possessive chains)
		{"Who is the founder of Stoicism?", true, "multi-hop"},
		{"Where was the creator of Python from?", true, "multi-hop"},

		// Single entity (should still decompose)
		{"Tell me about quantum mechanics", true, "single"},
		{"What is evolution?", true, "single"},
	}

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  GENERAL CHAIN-OF-THOUGHT DECOMPOSER")
	fmt.Println(strings.Repeat("=", 70))

	decomposed := 0
	for _, tc := range tests {
		chain := reasoner.Reason(tc.question)

		hasResult := chain != nil && (chain.Answer != "" || len(chain.Steps) > 0)
		status := "PASS"
		if tc.expectSteps && !hasResult {
			status = "FAIL"
		} else if hasResult {
			decomposed++
		}

		fmt.Printf("  [%s] %-50s", status, tc.question)
		if chain != nil {
			fmt.Printf(" (%d steps)", len(chain.Steps))
			if chain.Answer != "" {
				answer := chain.Answer
				if len(answer) > 60 {
					answer = answer[:60] + "..."
				}
				fmt.Printf(" → %s", answer)
			}
		}
		fmt.Println()

		if chain != nil && chain.Trace != "" {
			for _, line := range strings.Split(chain.Trace, "\n") {
				if line != "" {
					fmt.Printf("    %s\n", line)
				}
			}
		}

		if tc.expectSteps && !hasResult {
			t.Errorf("expected decomposition for %q but got none", tc.question)
		}
	}

	fmt.Printf("\n  Successfully decomposed: %d/%d\n", decomposed, len(tests))
}

// -----------------------------------------------------------------------
// Test 2: Causal/Counterfactual Reasoning
// -----------------------------------------------------------------------

func TestCausalCounterfactual(t *testing.T) {
	_, causalReasoner, _, _ := setupReasoningTest(t)

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  CAUSAL / COUNTERFACTUAL REASONING")
	fmt.Println(strings.Repeat("=", 70))

	// WhatIf tests
	whatIfTests := []string{
		"Stoicism",
		"DNA",
		"Python",
		"evolution",
		"quantum mechanics",
	}

	fmt.Println("\n--- What-If Analysis ---")
	for _, hypothesis := range whatIfTests {
		result := causalReasoner.WhatIf(hypothesis)
		if result != nil && len(result.Effects) > 0 {
			fmt.Printf("  What if '%s'? → %d effects found\n", hypothesis, len(result.Effects))
			for i, e := range result.Effects {
				if i >= 3 {
					fmt.Printf("    ... and %d more\n", len(result.Effects)-3)
					break
				}
				fmt.Printf("    → %s (depth: %d, confidence: %.0f%%)\n", e.Entity, e.Depth, e.Confidence*100)
			}
		} else {
			fmt.Printf("  What if '%s'? → no causal effects found (no causes edges)\n", hypothesis)
		}
	}

	// WhatIfRemoved tests
	fmt.Println("\n--- What-If-Removed Analysis ---")
	removalTests := []string{
		"Stoicism",
		"Python",
		"Albert Einstein",
	}

	for _, entity := range removalTests {
		result := causalReasoner.WhatIfRemoved(entity)
		if result != nil && len(result.Effects) > 0 {
			fmt.Printf("  Without '%s'? → %d dependents affected\n", entity, len(result.Effects))
			for i, e := range result.Effects {
				if i >= 3 {
					break
				}
				fmt.Printf("    → %s\n", e.Entity)
			}
			// Test answer composition
			answer := causalReasoner.ComposeCounterfactualAnswer(entity, result, true)
			preview := answer
			if len(preview) > 120 {
				preview = preview[:120] + "..."
			}
			fmt.Printf("    Response: %s\n", preview)
		} else {
			fmt.Printf("  Without '%s'? → no sole dependents found\n", entity)
		}
	}
}

// -----------------------------------------------------------------------
// Test 3: Goal Planning
// -----------------------------------------------------------------------

func TestGoalPlanning(t *testing.T) {
	_, _, planner, _ := setupReasoningTest(t)

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  GOAL PLANNING")
	fmt.Println(strings.Repeat("=", 70))

	goals := []string{
		"learn Python",
		"learn Stoicism",
		"learn about DNA",
		"study quantum mechanics",
		"get into philosophy",
		"master Linux",
		"learn about evolution",
		"get started with Go",
	}

	plansGenerated := 0
	for _, goal := range goals {
		plan := planner.PlanFor(goal)
		if plan != nil && len(plan.Steps) > 0 {
			plansGenerated++
			fmt.Printf("\n  Goal: %s → %d steps\n", goal, len(plan.Steps))
			for _, step := range plan.Steps {
				fmt.Printf("    %d. %s — %s\n", step.Order, step.Action, step.Reason)
			}
		} else {
			fmt.Printf("\n  Goal: %s → no plan generated (topic not in graph)\n", goal)
		}
	}

	fmt.Printf("\n  Plans generated: %d/%d\n", plansGenerated, len(goals))

	if plansGenerated == 0 {
		t.Error("expected at least some goal plans to be generated")
	}

	// Test planning question detection
	planningQuestions := []struct {
		query  string
		expect bool
	}{
		{"How do I learn Python?", true},
		{"How to get started with Go?", true},
		{"Steps to master Stoicism", true},
		{"Tell me about Python", false},
		{"What is DNA?", false},
		{"Teach me about quantum mechanics", true},
	}

	fmt.Println("\n--- Planning Question Detection ---")
	for _, pq := range planningQuestions {
		detected := IsPlanningQuestion(pq.query)
		status := "OK"
		if detected != pq.expect {
			status = "FAIL"
			t.Errorf("IsPlanningQuestion(%q) = %v, want %v", pq.query, detected, pq.expect)
		}
		fmt.Printf("  [%s] %-45s detected=%v\n", status, pq.query, detected)
	}
}

// -----------------------------------------------------------------------
// Test 4: Honest Fallback (no confusing bridges)
// -----------------------------------------------------------------------

func TestHonestFallback(t *testing.T) {
	dir := t.TempDir()
	router, composer, learning, _ := setupTrainedPipeline(t, dir)

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  HONEST FALLBACK TEST")
	fmt.Println(strings.Repeat("=", 70))

	// Ask about things Nous doesn't know
	unknownQueries := []struct {
		query       string
		shouldHave  []string // at least one of these should appear
		shouldNotHave []string // none of these should appear
	}{
		{
			"Tell me about Zibblyworp",
			[]string{"don't have", "haven't learned", "isn't in", "don't know"},
			[]string{"going back to", "earlier you mentioned", "brought up"},
		},
		{
			"What is a quantum flangewidget?",
			nil, // any response is fine as long as it's not a confusing bridge
			[]string{"going back to", "earlier you mentioned"},
		},
		{
			"Who was Professor McFakerson?",
			nil,
			[]string{"going back to", "earlier you mentioned"},
		},
	}

	// First do some known queries to build up history
	chat(router, learning, composer, "Tell me about Stoicism")
	chat(router, learning, composer, "What is DNA?")

	fmt.Println("\n--- Unknown Query Responses (after building history) ---")
	confusingBridges := 0
	for _, uq := range unknownQueries {
		resp := chat(router, learning, composer, uq.query)
		lower := strings.ToLower(resp)

		preview := resp
		if len(preview) > 120 {
			preview = preview[:120] + "..."
		}
		fmt.Printf("  Q: %s\n  A: %s\n", uq.query, preview)

		// Check for confusing bridges
		for _, bad := range uq.shouldNotHave {
			if strings.Contains(lower, bad) {
				confusingBridges++
				fmt.Printf("  [BAD] Contains confusing bridge: '%s'\n", bad)
			}
		}

		// Check for honest fallback
		if len(uq.shouldHave) > 0 {
			found := false
			for _, good := range uq.shouldHave {
				if strings.Contains(lower, good) {
					found = true
					break
				}
			}
			if found {
				fmt.Println("  [GOOD] Honest fallback detected")
			}
		}
		fmt.Println()
	}

	if confusingBridges > 0 {
		t.Errorf("got %d confusing bridge responses for unknown queries", confusingBridges)
	}
}

// -----------------------------------------------------------------------
// Test 5: Full Pipeline with New Systems
// -----------------------------------------------------------------------

func TestFullPipelineWithReasoning(t *testing.T) {
	dir := t.TempDir()
	graph := NewCognitiveGraph(filepath.Join(dir, "graph.json"))
	semantic := NewSemanticEngine()
	causal := NewCausalEngine()
	patterns := NewPatternDetector()
	composer := NewComposer(graph, semantic, causal, patterns)
	learning := NewLearningEngine(graph, composer, dir)

	packDir := filepath.Join("..", "..", "packages")
	if _, err := os.Stat(packDir); err != nil {
		t.Skip("packages directory not found")
	}
	loader := NewPackageLoader(graph, composer.Generative, composer, packDir)
	loader.LoadAll()

	// Wire up all new systems
	router := NewActionRouter()
	router.CogGraph = graph
	router.Composer = composer
	router.Semantic = semantic
	router.Causal = causal
	router.Patterns = patterns
	router.Reasoner = NewReasoningEngine(graph, semantic)
	router.GoalPlanner = NewGoalPlanner(graph, semantic)
	router.CausalReasoner = NewGraphCausalReasoner(graph)

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  FULL PIPELINE WITH ALL REASONING SYSTEMS")
	fmt.Println(strings.Repeat("=", 70))

	queries := []struct {
		msg      string
		category string
	}{
		// Knowledge
		{"Tell me about Stoicism", "knowledge"},
		{"What is DNA?", "knowledge"},

		// Planning
		{"How do I learn Python?", "planning"},
		{"How to get started with Stoicism?", "planning"},

		// Comparison (via reasoning)
		{"Compare Python and Go", "comparison"},

		// Counterfactual
		{"What would happen without Stoicism?", "counterfactual"},

		// Multi-hop reasoning
		{"Who founded Stoicism?", "multi-hop"},
		{"What is Python used for?", "multi-hop"},

		// Unknown (honest fallback)
		{"Tell me about Zibblyworp", "unknown"},

		// Personal
		{"I'm feeling stressed today", "personal"},
		{"Good morning!", "greeting"},
		{"Goodbye!", "farewell"},
	}

	for _, q := range queries {
		resp := chat(router, learning, composer, q.msg)
		preview := resp
		if len(preview) > 100 {
			preview = preview[:100] + "..."
		}
		fmt.Printf("  [%-13s] %s\n                → %s\n", q.category, q.msg, preview)
	}
}
