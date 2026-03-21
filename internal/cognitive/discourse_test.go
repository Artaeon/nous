package cognitive

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDiscoursePlanner(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	dp := NewDiscoursePlanner(rng)

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  DISCOURSE PLANNER")
	fmt.Println(strings.Repeat("=", 70))

	// Test schema selection with different fact profiles

	// Rich fact set — should pick explanatory or narrative
	richFacts := []edgeFact{
		{Subject: "Python", Relation: RelIsA, Object: "programming language"},
		{Subject: "Python", Relation: RelCreatedBy, Object: "Guido van Rossum"},
		{Subject: "Python", Relation: RelFoundedIn, Object: "1991"},
		{Subject: "Python", Relation: RelUsedFor, Object: "data science"},
		{Subject: "Python", Relation: RelUsedFor, Object: "web development"},
		{Subject: "Python", Relation: RelHas, Object: "readable syntax"},
		{Subject: "Python", Relation: RelHas, Object: "extensive libraries"},
		{Subject: "Python", Relation: RelDescribedAs, Object: "beginner-friendly"},
	}

	fmt.Println("\n--- Schema Selection ---")

	// Run selection 10 times to see distribution
	schemaCounts := make(map[string]int)
	for i := 0; i < 10; i++ {
		schema := dp.SelectSchema(richFacts, RespFactual)
		schemaCounts[schema.Name]++
	}
	fmt.Println("  Rich facts (8 facts, all types):")
	for name, count := range schemaCounts {
		fmt.Printf("    %s: %d/10\n", name, count)
	}

	// Feature-heavy facts
	featureFacts := []edgeFact{
		{Subject: "Go", Relation: RelIsA, Object: "programming language"},
		{Subject: "Go", Relation: RelHas, Object: "goroutines"},
		{Subject: "Go", Relation: RelHas, Object: "channels"},
		{Subject: "Go", Relation: RelHas, Object: "fast compilation"},
		{Subject: "Go", Relation: RelHas, Object: "garbage collection"},
	}

	schema := dp.SelectSchema(featureFacts, RespFactual)
	fmt.Printf("  Feature-heavy (5 facts, mostly has): %s\n", schema.Name)

	// Brief facts
	briefFacts := []edgeFact{
		{Subject: "Go", Relation: RelIsA, Object: "programming language"},
	}
	schema = dp.SelectSchema(briefFacts, RespFactual)
	fmt.Printf("  Brief (1 fact): %s\n", schema.Name)
	if schema.Name != "brief" {
		t.Errorf("expected brief schema for 1 fact, got %s", schema.Name)
	}

	// Explain response type
	schema = dp.SelectSchema(richFacts, RespExplain)
	fmt.Printf("  Explain response type: %s\n", schema.Name)
	if schema.Name != "explanatory" {
		t.Errorf("expected explanatory schema for RespExplain, got %s", schema.Name)
	}

	// Test plan generation
	fmt.Println("\n--- Plan Generation ---")

	plan := dp.PlanFromFacts("Python", richFacts, RespFactual)
	if plan == nil {
		t.Fatal("expected plan, got nil")
	}
	fmt.Printf("  Schema: %s\n", plan.Schema)
	fmt.Printf("  Sections: %d\n", len(plan.Sections))
	for i, s := range plan.Sections {
		fmt.Printf("    %d. [%s] %s (facts: %d, connector: %q)\n",
			i+1, sectionRoleName(s.Role), s.Goal, len(s.Facts), s.Connector)
	}

	if len(plan.Sections) < 2 {
		t.Errorf("expected at least 2 sections, got %d", len(plan.Sections))
	}

	// Verify sections have appropriate content
	hasHook := false
	hasClose := false
	for _, s := range plan.Sections {
		if s.Role == SectionHook {
			hasHook = true
		}
		if s.Role == SectionClose {
			hasClose = true
		}
	}
	if !hasHook {
		t.Error("plan should have a HOOK section")
	}
	if !hasClose {
		t.Error("plan should have a CLOSE section")
	}

	// Test trace formatting
	trace := FormatPlan(plan)
	fmt.Printf("\n  Plan trace:\n")
	for _, line := range strings.Split(trace, "\n") {
		if line != "" {
			fmt.Printf("    %s\n", line)
		}
	}
}

func TestDiscourseIntegration(t *testing.T) {
	dir := t.TempDir()
	graph := NewCognitiveGraph(filepath.Join(dir, "graph.json"))
	semantic := NewSemanticEngine()
	causal := NewCausalEngine()
	patterns := NewPatternDetector()
	composer := NewComposer(graph, semantic, causal, patterns)

	packDir := filepath.Join("..", "..", "packages")
	if _, err := os.Stat(packDir); err != nil {
		t.Skip("packages directory not found")
	}
	loader := NewPackageLoader(graph, composer.Generative, composer, packDir)
	loader.LoadAll()

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  DISCOURSE-PLANNED vs UNPLANNED TEXT")
	fmt.Println(strings.Repeat("=", 70))

	topics := []string{"Python", "Stoicism", "DNA", "evolution", "Linux"}

	for _, topic := range topics {
		facts, _ := composer.gatherFacts("Tell me about " + topic)
		if len(facts) == 0 {
			fmt.Printf("\n  %s: no facts found\n", topic)
			continue
		}

		// Generate with discourse plan
		plan := composer.Discourse.PlanFromFacts(topic, facts, RespFactual)
		planned := ""
		if plan != nil {
			planned = composer.Generative.ComposeWithPlan(plan)
		}

		// Generate without plan (original method)
		unplanned := composer.Generative.ComposeCreativeText(topic, facts)

		fmt.Printf("\n  === %s (%d facts) ===\n", topic, len(facts))
		if plan != nil {
			fmt.Printf("  Schema: %s (%d sections)\n", plan.Schema, len(plan.Sections))
		}

		plannedPreview := planned
		if len(plannedPreview) > 200 {
			plannedPreview = plannedPreview[:200] + "..."
		}
		unplannedPreview := unplanned
		if len(unplannedPreview) > 200 {
			unplannedPreview = unplannedPreview[:200] + "..."
		}

		fmt.Printf("  PLANNED:   %s\n", plannedPreview)
		fmt.Printf("  UNPLANNED: %s\n", unplannedPreview)

		if planned == "" {
			t.Errorf("discourse-planned text for %s was empty", topic)
		}
		if len(planned) < 50 {
			t.Errorf("discourse-planned text for %s too short: %d chars", topic, len(planned))
		}
	}
}

func TestDiscourseSchemaVariety(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	dp := NewDiscoursePlanner(rng)

	richFacts := []edgeFact{
		{Subject: "X", Relation: RelIsA, Object: "thing"},
		{Subject: "X", Relation: RelCreatedBy, Object: "someone"},
		{Subject: "X", Relation: RelFoundedIn, Object: "2000"},
		{Subject: "X", Relation: RelUsedFor, Object: "something"},
		{Subject: "X", Relation: RelHas, Object: "feature A"},
		{Subject: "X", Relation: RelHas, Object: "feature B"},
		{Subject: "X", Relation: RelRelatedTo, Object: "Y"},
	}

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  SCHEMA VARIETY TEST")
	fmt.Println(strings.Repeat("=", 70))

	// Generate 20 plans and verify variety
	schemas := make(map[string]int)
	for i := 0; i < 20; i++ {
		plan := dp.PlanFromFacts("X", richFacts, RespFactual)
		schemas[plan.Schema]++
	}

	fmt.Println("  Schema distribution over 20 plans:")
	for name, count := range schemas {
		fmt.Printf("    %-20s %d\n", name, count)
	}

	if len(schemas) < 2 {
		t.Errorf("expected at least 2 different schemas, got %d", len(schemas))
	}
}
