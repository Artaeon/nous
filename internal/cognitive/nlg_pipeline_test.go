package cognitive

import (
	"strings"
	"testing"
)

func makeThinkingEngineForNLGTests() *ThinkingEngine {
	graph := NewCognitiveGraph("")
	composer := NewComposer(graph, nil, nil, nil)
	return NewThinkingEngine(graph, composer)
}

func TestBuildContentPlanRanksClaims(t *testing.T) {
	te := makeThinkingEngineForNLGTests()

	te.graph.AddEdge("go", "programming language", RelIsA, "test")
	te.graph.AddEdge("go", "systems software", RelUsedFor, "test")
	te.graph.AddEdge("go", "2009", RelFoundedIn, "test")

	plan := te.BuildContentPlan("explain go", TaskTeach, &TaskParams{Topic: "go"})
	if plan == nil {
		t.Fatal("expected non-nil content plan")
	}
	if strings.TrimSpace(plan.Thesis) == "" {
		t.Fatal("expected thesis in content plan")
	}
	if len(plan.Claims) == 0 {
		t.Fatal("expected ranked claims in content plan")
	}

	if !strings.Contains(strings.ToLower(plan.Claims[0].Text), "is a") {
		t.Fatalf("expected highest-priority claim to be type/identity, got %q", plan.Claims[0].Text)
	}
}

func TestSelectBestCandidatePrefersCoverage(t *testing.T) {
	te := makeThinkingEngineForNLGTests()

	plan := &ContentPlan{
		Topic:  "go",
		Thesis: "Go is a programming language.",
		Claims: []PlanClaim{
			{Text: "Go is used for systems software."},
			{Text: "Go was founded in 2009."},
		},
	}

	lowCoverage := "Go is interesting and useful."
	highCoverage := "Go is a programming language. First, go is used for systems software. Finally, go was founded in 2009."

	best := te.SelectBestCandidate(plan, []string{lowCoverage, highCoverage})
	if best != highCoverage {
		t.Fatalf("expected high-coverage candidate to win, got %q", best)
	}
}

func TestBuildPlanCandidatesIncludesBaselineAndVariants(t *testing.T) {
	te := makeThinkingEngineForNLGTests()

	plan := &ContentPlan{
		Topic:  "stoicism",
		Thesis: "Stoicism is a school of philosophy.",
		Claims: []PlanClaim{
			{Text: "Stoicism emphasizes virtue."},
			{Text: "Stoicism teaches emotional discipline."},
		},
	}

	baseline := "Stoicism is an old philosophy."
	candidates := te.BuildPlanCandidates(plan, &summaryFrame, baseline)

	if len(candidates) < 2 {
		t.Fatalf("expected multiple candidates, got %d", len(candidates))
	}
	if candidates[0] != baseline {
		t.Fatalf("expected baseline candidate first, got %q", candidates[0])
	}
}

func TestScoreCandidatePenalizesContradiction(t *testing.T) {
	te := makeThinkingEngineForNLGTests()

	plan := &ContentPlan{
		Topic:  "go",
		Thesis: "Go is a programming language.",
		Claims: []PlanClaim{{Text: "Go is a programming language."}},
	}

	consistent := "Go is a programming language. It is used for systems software."
	contradictory := "Go is a programming language. It is not go is a programming language."

	sConsistent := te.ScoreCandidate(plan, consistent)
	sContradictory := te.ScoreCandidate(plan, contradictory)

	if sContradictory.Consistency >= sConsistent.Consistency {
		t.Fatalf("expected contradictory candidate to score lower consistency: contradictory=%.2f consistent=%.2f", sContradictory.Consistency, sConsistent.Consistency)
	}
	if sContradictory.Total >= sConsistent.Total {
		t.Fatalf("expected contradictory candidate total score to be lower: contradictory=%.2f consistent=%.2f", sContradictory.Total, sConsistent.Total)
	}
}