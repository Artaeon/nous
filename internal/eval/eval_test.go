package eval

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// --- Scorecard Tests ---

func TestDefaultScorecards(t *testing.T) {
	cards := DefaultScorecards()
	if len(cards) != 8 {
		t.Fatalf("expected 8 scorecards, got %d", len(cards))
	}

	expectedCaps := map[string]bool{
		"IntentRouting":    false,
		"FactualQA":       false,
		"DeepExplain":     false,
		"CompareTradeoff": false,
		"MultiTurnContext": false,
		"Planning":        false,
		"ToolUseAccuracy": false,
		"StyleControl":    false,
	}

	for _, card := range cards {
		if _, ok := expectedCaps[card.Capability]; !ok {
			t.Errorf("unexpected capability: %s", card.Capability)
		}
		expectedCaps[card.Capability] = true

		if card.Weight <= 0 {
			t.Errorf("capability %s has non-positive weight: %f", card.Capability, card.Weight)
		}
		if card.Description == "" {
			t.Errorf("capability %s has empty description", card.Capability)
		}
		if len(card.Metrics) == 0 {
			t.Errorf("capability %s has no metrics", card.Capability)
		}

		// Verify metric weights sum close to 1.0.
		var weightSum float64
		for _, m := range card.Metrics {
			if m.Name == "" {
				t.Errorf("capability %s has a metric with empty name", card.Capability)
			}
			if m.Weight <= 0 || m.Weight > 1 {
				t.Errorf("capability %s metric %s has invalid weight: %f", card.Capability, m.Name, m.Weight)
			}
			if m.Threshold < 0 || m.Threshold > 1 {
				t.Errorf("capability %s metric %s has invalid threshold: %f", card.Capability, m.Name, m.Threshold)
			}
			weightSum += m.Weight
		}
		if weightSum < 0.99 || weightSum > 1.01 {
			t.Errorf("capability %s metric weights sum to %f, expected ~1.0", card.Capability, weightSum)
		}
	}

	for cap, found := range expectedCaps {
		if !found {
			t.Errorf("missing capability: %s", cap)
		}
	}
}

func TestScorecardEvaluate(t *testing.T) {
	cards := DefaultScorecards()
	card := &cards[0] // IntentRouting

	t.Run("all_passing", func(t *testing.T) {
		scores := map[string]float64{
			"classification_accuracy":  0.95,
			"confidence_calibration":   0.90,
			"ambiguity_handling":       0.85,
			"fallback_appropriateness": 0.90,
		}
		result := Evaluate(card, scores)
		if result.Capability != "IntentRouting" {
			t.Errorf("expected IntentRouting, got %s", result.Capability)
		}
		if !result.Pass {
			t.Error("expected pass with all scores above threshold")
		}
		if result.Total < 0.85 {
			t.Errorf("expected total >= 0.85, got %f", result.Total)
		}
	})

	t.Run("one_failing", func(t *testing.T) {
		scores := map[string]float64{
			"classification_accuracy":  0.95,
			"confidence_calibration":   0.50, // below 0.75 threshold
			"ambiguity_handling":       0.85,
			"fallback_appropriateness": 0.90,
		}
		result := Evaluate(card, scores)
		if result.Pass {
			t.Error("expected fail when one metric is below threshold")
		}
	})

	t.Run("missing_metric", func(t *testing.T) {
		scores := map[string]float64{
			"classification_accuracy": 0.95,
			// Missing other metrics.
		}
		result := Evaluate(card, scores)
		if result.Pass {
			t.Error("expected fail when metrics are missing (scored as 0)")
		}
		if result.Scores["confidence_calibration"] != 0 {
			t.Error("missing metric should score 0")
		}
	})

	t.Run("clamping", func(t *testing.T) {
		scores := map[string]float64{
			"classification_accuracy":  1.5, // should clamp to 1.0
			"confidence_calibration":   -0.3, // should clamp to 0.0
			"ambiguity_handling":       0.85,
			"fallback_appropriateness": 0.90,
		}
		result := Evaluate(card, scores)
		if result.Scores["classification_accuracy"] != 1.0 {
			t.Errorf("expected clamped to 1.0, got %f", result.Scores["classification_accuracy"])
		}
		if result.Scores["confidence_calibration"] != 0.0 {
			t.Errorf("expected clamped to 0.0, got %f", result.Scores["confidence_calibration"])
		}
	})

	t.Run("perfect_scores", func(t *testing.T) {
		scores := map[string]float64{
			"classification_accuracy":  1.0,
			"confidence_calibration":   1.0,
			"ambiguity_handling":       1.0,
			"fallback_appropriateness": 1.0,
		}
		result := Evaluate(card, scores)
		if !result.Pass {
			t.Error("perfect scores should pass")
		}
		if result.Total != 1.0 {
			t.Errorf("perfect scores should give total 1.0, got %f", result.Total)
		}
	})
}

// --- EvalSet Tests ---

func TestGenerateEvalSet(t *testing.T) {
	es := GenerateEvalSet()

	if es.Version == "" {
		t.Error("eval set should have a version")
	}
	if es.Created.IsZero() {
		t.Error("eval set should have a creation time")
	}
	if len(es.Prompts) < 1000 {
		t.Fatalf("expected at least 1000 prompts, got %d", len(es.Prompts))
	}

	// Check balance across capabilities.
	capCounts := make(map[string]int)
	for _, p := range es.Prompts {
		capCounts[p.Capability]++
	}

	expectedCaps := []string{
		"IntentRouting", "FactualQA", "DeepExplain", "CompareTradeoff",
		"MultiTurnContext", "Planning", "ToolUseAccuracy", "StyleControl",
	}

	for _, cap := range expectedCaps {
		count := capCounts[cap]
		if count < 125 {
			t.Errorf("capability %s has only %d prompts, expected >= 125", cap, count)
		}
	}

	// Verify all prompts have required fields.
	for i, p := range es.Prompts {
		if p.ID == "" {
			t.Errorf("prompt %d has empty ID", i)
		}
		if p.Capability == "" {
			t.Errorf("prompt %d (%s) has empty capability", i, p.ID)
		}
		if p.Query == "" {
			t.Errorf("prompt %d (%s) has empty query", i, p.ID)
		}
		if p.GoldAnswer == "" {
			t.Errorf("prompt %d (%s) has empty gold answer", i, p.ID)
		}
		if len(p.Rubric) == 0 {
			t.Errorf("prompt %d (%s) has no rubric items", i, p.ID)
		}
		if p.Difficulty != "easy" && p.Difficulty != "medium" && p.Difficulty != "hard" {
			t.Errorf("prompt %d (%s) has invalid difficulty: %s", i, p.ID, p.Difficulty)
		}
	}

	// Check difficulty distribution (should be roughly balanced).
	diffCounts := make(map[string]int)
	for _, p := range es.Prompts {
		diffCounts[p.Difficulty]++
	}
	for _, diff := range []string{"easy", "medium", "hard"} {
		if diffCounts[diff] == 0 {
			t.Errorf("no prompts with difficulty %s", diff)
		}
	}

	// Check for unique IDs.
	seen := make(map[string]bool)
	for _, p := range es.Prompts {
		if seen[p.ID] {
			t.Errorf("duplicate prompt ID: %s", p.ID)
		}
		seen[p.ID] = true
	}
}

func TestEvalSetPersistence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_evalset.json")

	// Generate and save.
	original := GenerateEvalSet()
	if err := SaveEvalSet(original, path); err != nil {
		t.Fatalf("save failed: %v", err)
	}

	// Verify file exists.
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("file not found after save: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("saved file is empty")
	}

	// Load and compare.
	loaded, err := LoadEvalSet(path)
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}

	if loaded.Version != original.Version {
		t.Errorf("version mismatch: %s vs %s", loaded.Version, original.Version)
	}
	if len(loaded.Prompts) != len(original.Prompts) {
		t.Fatalf("prompt count mismatch: %d vs %d", len(loaded.Prompts), len(original.Prompts))
	}

	// Spot-check a few prompts.
	for i := 0; i < 10 && i < len(original.Prompts); i++ {
		if loaded.Prompts[i].ID != original.Prompts[i].ID {
			t.Errorf("prompt %d ID mismatch: %s vs %s", i, loaded.Prompts[i].ID, original.Prompts[i].ID)
		}
		if loaded.Prompts[i].Query != original.Prompts[i].Query {
			t.Errorf("prompt %d query mismatch", i)
		}
		if loaded.Prompts[i].GoldAnswer != original.Prompts[i].GoldAnswer {
			t.Errorf("prompt %d gold answer mismatch", i)
		}
	}

	// Test loading from nonexistent path.
	_, err = LoadEvalSet(filepath.Join(dir, "nonexistent.json"))
	if err == nil {
		t.Error("expected error loading nonexistent file")
	}
}

// --- Gate Tests ---

func TestDefaultGates(t *testing.T) {
	gates := DefaultGates()

	if gates.Correctness != 0.85 {
		t.Errorf("expected correctness 0.85, got %f", gates.Correctness)
	}
	if gates.HallucinationRate != 0.05 {
		t.Errorf("expected hallucination rate 0.05, got %f", gates.HallucinationRate)
	}
	if gates.Helpfulness != 0.80 {
		t.Errorf("expected helpfulness 0.80, got %f", gates.Helpfulness)
	}
	if gates.Coherence != 0.85 {
		t.Errorf("expected coherence 0.85, got %f", gates.Coherence)
	}
	if gates.LatencyMs != 500 {
		t.Errorf("expected latency 500ms, got %d", gates.LatencyMs)
	}
	if gates.FailureQuality != 0.70 {
		t.Errorf("expected failure quality 0.70, got %f", gates.FailureQuality)
	}
}

func TestCheckCorrectness(t *testing.T) {
	t.Run("high_overlap", func(t *testing.T) {
		gold := "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen."
		response := "Photosynthesis is a process where green plants use sunlight, water, and carbon dioxide to produce glucose and oxygen."
		result := CheckCorrectness(response, gold)
		if result.Score < 0.7 {
			t.Errorf("expected high correctness score for matching response, got %f", result.Score)
		}
	})

	t.Run("low_overlap", func(t *testing.T) {
		gold := "Photosynthesis is the process by which green plants convert sunlight into glucose."
		response := "The weather in Tokyo is sunny and warm today with temperatures around 25 degrees."
		result := CheckCorrectness(response, gold)
		if result.Score > 0.3 {
			t.Errorf("expected low correctness score for unrelated response, got %f", result.Score)
		}
	})

	t.Run("negation_penalty", func(t *testing.T) {
		gold := "Venus has a thick atmosphere composed of carbon dioxide."
		response := "Venus does not have a thick atmosphere. It has no carbon dioxide."
		result := CheckCorrectness(response, gold)
		// The negation should lower the score.
		gold2 := "Venus has a thick atmosphere composed of carbon dioxide."
		response2 := "Venus has a thick atmosphere composed of carbon dioxide."
		result2 := CheckCorrectness(response2, gold2)
		if result.Score >= result2.Score {
			t.Errorf("negated response should score lower: negated=%f, correct=%f", result.Score, result2.Score)
		}
	})

	t.Run("empty_gold", func(t *testing.T) {
		result := CheckCorrectness("any response", "")
		if result.Score != 1.0 {
			t.Errorf("empty gold should give score 1.0, got %f", result.Score)
		}
	})
}

func TestCheckHallucination(t *testing.T) {
	t.Run("grounded_response", func(t *testing.T) {
		sources := []string{
			"The Eiffel Tower is located in Paris, France. It was built in 1889 and stands 330 meters tall.",
			"The tower was designed by Gustave Eiffel's engineering company for the 1889 World's Fair.",
		}
		response := "The Eiffel Tower is in Paris, France. It was built in 1889 and is 330 meters tall. It was designed by Gustave Eiffel's company."
		result := CheckHallucination(response, sources)
		if result.Score < 0.8 {
			t.Errorf("grounded response should have high score, got %f (%s)", result.Score, result.Details)
		}
	})

	t.Run("ungrounded_response", func(t *testing.T) {
		sources := []string{
			"The Eiffel Tower is located in Paris, France.",
		}
		response := "The Eiffel Tower was originally purple and was built by Napoleon Bonaparte in 1750 as a military watchtower during the Crimean War."
		result := CheckHallucination(response, sources)
		if result.Score > 0.5 {
			t.Errorf("ungrounded response should have low score, got %f (%s)", result.Score, result.Details)
		}
	})

	t.Run("no_sources_hedged", func(t *testing.T) {
		response := "Approximately 70% of the Earth's surface is covered by water. Research suggests that ocean temperatures have generally been rising over recent decades."
		result := CheckHallucination(response, nil)
		if result.Score < 0.7 {
			t.Errorf("hedged response without sources should score reasonably, got %f", result.Score)
		}
	})

	t.Run("no_sources_overconfident", func(t *testing.T) {
		response := "It is absolutely certain and proven fact that exactly 100% of all scientists always agree on everything."
		result := CheckHallucination(response, nil)
		if result.Score > 0.8 {
			t.Errorf("overconfident response should score lower, got %f", result.Score)
		}
	})

	t.Run("empty_response", func(t *testing.T) {
		result := CheckHallucination("", []string{"source"})
		if result.Score < 0.9 {
			t.Errorf("empty response should not hallucinate, got %f", result.Score)
		}
	})
}

func TestCheckHelpfulness(t *testing.T) {
	t.Run("helpful_response", func(t *testing.T) {
		query := "what is photosynthesis"
		response := "Photosynthesis is the process by which green plants and some organisms convert light energy, usually from the Sun, into chemical energy in the form of glucose. This process uses water and carbon dioxide as inputs and produces oxygen as a byproduct."
		result := CheckHelpfulness(response, query)
		if result.Score < 0.6 {
			t.Errorf("helpful response should score well, got %f (%s)", result.Score, result.Details)
		}
	})

	t.Run("unhelpful_response", func(t *testing.T) {
		query := "what is photosynthesis"
		response := "That's interesting."
		result := CheckHelpfulness(response, query)
		if result.Score > 0.5 {
			t.Errorf("unhelpful response should score low, got %f", result.Score)
		}
	})

	t.Run("refusal_response", func(t *testing.T) {
		query := "what is photosynthesis"
		response := "I cannot answer that question. As an AI, I don't have the ability to provide information on scientific topics."
		result := CheckHelpfulness(response, query)
		if result.Score > 0.7 {
			t.Errorf("refusal should be penalized, got %f", result.Score)
		}
	})

	t.Run("empty_response", func(t *testing.T) {
		result := CheckHelpfulness("", "any query")
		if result.Score != 0.0 {
			t.Errorf("empty response should score 0, got %f", result.Score)
		}
	})
}

func TestCheckCoherence(t *testing.T) {
	t.Run("coherent_text", func(t *testing.T) {
		response := "The water cycle is a continuous process that circulates water through the environment. First, water evaporates from oceans and lakes due to solar heating. Then, the water vapor rises and condenses into clouds. Finally, precipitation occurs as rain or snow, returning water to the surface. This cycle is essential for sustaining life on Earth."
		result := CheckCoherence(response)
		if result.Score < 0.5 {
			t.Errorf("coherent text should score well, got %f (%s)", result.Score, result.Details)
		}
	})

	t.Run("incoherent_text", func(t *testing.T) {
		response := "cat cat cat dog dog dog fish fish fish bird bird bird cat cat cat dog dog dog"
		result := CheckCoherence(response)
		if result.Score > 0.6 {
			t.Errorf("incoherent repetitive text should score low, got %f (%s)", result.Score, result.Details)
		}
	})

	t.Run("empty_text", func(t *testing.T) {
		result := CheckCoherence("")
		if result.Score != 0.0 {
			t.Errorf("empty text should score 0, got %f", result.Score)
		}
		if result.Pass {
			t.Error("empty text should not pass")
		}
	})

	t.Run("repeated_sentences", func(t *testing.T) {
		response := "This is a sentence. This is a sentence. This is a sentence. This is a sentence."
		result := CheckCoherence(response)
		if result.Score > 0.7 {
			t.Errorf("repeated sentences should be penalized, got %f", result.Score)
		}
	})

	t.Run("short_text", func(t *testing.T) {
		response := "ok"
		result := CheckCoherence(response)
		if result.Score > 0.5 {
			t.Errorf("very short text should score low, got %f", result.Score)
		}
	})
}

func TestCheckLatency(t *testing.T) {
	t.Run("within_budget", func(t *testing.T) {
		result := CheckLatency(200 * time.Millisecond)
		if !result.Pass {
			t.Error("200ms should pass 500ms budget")
		}
		if result.Score < 0.5 {
			t.Errorf("200ms should have good score, got %f", result.Score)
		}
	})

	t.Run("over_budget", func(t *testing.T) {
		result := CheckLatency(600 * time.Millisecond)
		if result.Pass {
			t.Error("600ms should fail 500ms budget")
		}
	})

	t.Run("way_over_budget", func(t *testing.T) {
		result := CheckLatency(2000 * time.Millisecond)
		if result.Pass {
			t.Error("2000ms should fail 500ms budget")
		}
		if result.Score > 0.1 {
			t.Errorf("2000ms should have very low score, got %f", result.Score)
		}
	})

	t.Run("zero_latency", func(t *testing.T) {
		result := CheckLatency(0)
		if !result.Pass {
			t.Error("0ms should pass")
		}
		if result.Score != 1.0 {
			t.Errorf("0ms should have perfect score, got %f", result.Score)
		}
	})

	t.Run("exact_budget", func(t *testing.T) {
		result := CheckLatency(500 * time.Millisecond)
		if !result.Pass {
			t.Error("exactly 500ms should pass")
		}
	})
}

func TestCheckFailureQuality(t *testing.T) {
	t.Run("not_failure", func(t *testing.T) {
		result := CheckFailureQuality("Here is the information you requested about photosynthesis.", false)
		if !result.Pass {
			t.Error("non-failure should always pass")
		}
		if result.Score != 1.0 {
			t.Errorf("non-failure should score 1.0, got %f", result.Score)
		}
	})

	t.Run("good_failure", func(t *testing.T) {
		response := "I'm not sure about the exact details of that topic. However, I can tell you that it's generally related to quantum physics. You might want to try consulting a textbook or academic resource for more precise information."
		result := CheckFailureQuality(response, true)
		if result.Score < 0.5 {
			t.Errorf("good failure response should score reasonably, got %f (%s)", result.Score, result.Details)
		}
	})

	t.Run("poor_failure", func(t *testing.T) {
		response := "No."
		result := CheckFailureQuality(response, true)
		if result.Score > 0.5 {
			t.Errorf("terse failure should score low, got %f", result.Score)
		}
	})
}

func TestRunAllGates(t *testing.T) {
	gates := DefaultGates()
	query := "what is photosynthesis"
	gold := "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen."
	response := "Photosynthesis is a biological process where plants use sunlight to convert water and carbon dioxide into glucose and oxygen. This process occurs primarily in the leaves using chlorophyll."
	sources := []string{
		"Photosynthesis is the process by which green plants convert light energy into chemical energy.",
		"Plants use chlorophyll in their leaves to absorb sunlight for photosynthesis.",
	}
	elapsed := 200 * time.Millisecond

	results := RunAllGates(gates, response, query, gold, sources, elapsed)

	if len(results) != 6 {
		t.Fatalf("expected 6 gate results, got %d", len(results))
	}

	// Check that all expected gates are present.
	gateNames := map[string]bool{
		"correctness":     false,
		"hallucination":   false,
		"helpfulness":     false,
		"coherence":       false,
		"latency":         false,
		"failure_quality": false,
	}
	for _, r := range results {
		if _, ok := gateNames[r.Gate]; !ok {
			t.Errorf("unexpected gate: %s", r.Gate)
		}
		gateNames[r.Gate] = true
		if r.Details == "" {
			t.Errorf("gate %s has empty details", r.Gate)
		}
		if r.Score < 0 || r.Score > 1 {
			t.Errorf("gate %s has score out of range: %f", r.Gate, r.Score)
		}
	}
	for gate, found := range gateNames {
		if !found {
			t.Errorf("missing gate: %s", gate)
		}
	}

	// Latency should pass (200ms < 500ms).
	for _, r := range results {
		if r.Gate == "latency" && !r.Pass {
			t.Error("latency gate should pass at 200ms")
		}
	}
}

func TestRunAllGatesWithFailure(t *testing.T) {
	gates := DefaultGates()
	query := "what is the meaning of life"
	gold := "This is a philosophical question with many perspectives."
	response := "I cannot answer that question. As an AI, I'm unable to provide opinions on philosophical matters."
	elapsed := 100 * time.Millisecond

	results := RunAllGates(gates, response, query, gold, nil, elapsed)
	if len(results) != 6 {
		t.Fatalf("expected 6 results, got %d", len(results))
	}

	// Failure quality gate should be present and evaluated.
	for _, r := range results {
		if r.Gate == "failure_quality" {
			if r.Details == "" {
				t.Error("failure quality should have details for failure response")
			}
		}
	}
}

// --- Utility function tests ---

func TestExtractTokens(t *testing.T) {
	tokens := extractTokens("the quick brown fox jumps over a lazy dog")
	// "the", "a" are stopwords; "over" is a stopword-ish but not in our list at 4 chars
	for _, tok := range tokens {
		if len(tok) < 3 {
			t.Errorf("token too short: %q", tok)
		}
		if stopWords[tok] {
			t.Errorf("stopword not filtered: %q", tok)
		}
	}
	if len(tokens) == 0 {
		t.Error("should extract some tokens")
	}
}

func TestSplitSentences(t *testing.T) {
	text := "This is sentence one. This is sentence two! Is this sentence three? Yes it is."
	sentences := splitSentences(text)
	if len(sentences) != 4 {
		t.Errorf("expected 4 sentences, got %d: %v", len(sentences), sentences)
	}
}

func TestBigrams(t *testing.T) {
	tokens := []string{"the", "quick", "brown", "fox"}
	bg := bigrams(tokens)
	if len(bg) != 3 {
		t.Errorf("expected 3 bigrams, got %d", len(bg))
	}
	if bg[0] != "the quick" {
		t.Errorf("unexpected first bigram: %s", bg[0])
	}
}

func TestCleanToken(t *testing.T) {
	cases := []struct {
		input, expected string
	}{
		{"hello", "hello"},
		{"hello.", "hello"},
		{"(hello)", "hello"},
		{"hello's", "hello's"},
		{"123", "123"},
		{",word;", "word"},
	}
	for _, c := range cases {
		got := cleanToken(c.input)
		if got != c.expected {
			t.Errorf("cleanToken(%q) = %q, expected %q", c.input, got, c.expected)
		}
	}
}

// --- Edge case tests ---

func TestCheckCorrectnessEdgeCases(t *testing.T) {
	// Very short gold.
	result := CheckCorrectness("yes", "yes")
	// "yes" is 3 chars but is a stopword, so no meaningful tokens.
	if result.Gate != "correctness" {
		t.Error("gate name should be 'correctness'")
	}

	// Unicode content.
	result = CheckCorrectness("Tokyo is the capital of Japan", "Tokyo is the capital of Japan")
	if result.Score < 0.5 {
		t.Errorf("identical response should score high, got %f", result.Score)
	}
}

func TestCheckHelpfulnessQAAlignment(t *testing.T) {
	t.Run("what_question", func(t *testing.T) {
		result := CheckHelpfulness("Photosynthesis is the process by which plants convert light to energy.", "what is photosynthesis")
		if result.Score < 0.5 {
			t.Errorf("definitional answer to 'what is' should score well, got %f", result.Score)
		}
	})

	t.Run("how_question", func(t *testing.T) {
		result := CheckHelpfulness("You can do this by first installing the software, then configuring the settings.", "how do I set up the application")
		if result.Score < 0.5 {
			t.Errorf("procedural answer to 'how' should score well, got %f", result.Score)
		}
	})

	t.Run("why_question", func(t *testing.T) {
		result := CheckHelpfulness("The sky is blue because of Rayleigh scattering of sunlight in the atmosphere.", "why is the sky blue")
		if result.Score < 0.5 {
			t.Errorf("causal answer to 'why' should score well, got %f", result.Score)
		}
	})
}

func TestGenerateEvalSetTagsPresent(t *testing.T) {
	es := GenerateEvalSet()
	tagsFound := 0
	for _, p := range es.Prompts {
		if len(p.Tags) > 0 {
			tagsFound++
		}
	}
	// At least 90% of prompts should have tags.
	ratio := float64(tagsFound) / float64(len(es.Prompts))
	if ratio < 0.9 {
		t.Errorf("expected >= 90%% of prompts to have tags, got %.1f%%", ratio*100)
	}
}

func TestEvalSetSaveToInvalidPath(t *testing.T) {
	es := &EvalSet{Version: "test", Created: time.Now()}
	err := SaveEvalSet(es, "/nonexistent/directory/file.json")
	if err == nil {
		t.Error("expected error saving to invalid path")
	}
}

func TestCoherenceWithLogicalConnectors(t *testing.T) {
	// Text with connectors should score higher than without.
	withConnectors := "Climate change is a global challenge. Furthermore, it affects agriculture and water resources. However, there are solutions available. For example, renewable energy can reduce emissions. Therefore, collective action is essential."
	withoutConnectors := "Climate is changing. Agriculture changes. Water changes. Energy exists. People exist."

	r1 := CheckCoherence(withConnectors)
	r2 := CheckCoherence(withoutConnectors)
	if r1.Score <= r2.Score {
		t.Errorf("text with connectors should score higher: with=%f without=%f", r1.Score, r2.Score)
	}
}

func TestHallucinationWithPartialSources(t *testing.T) {
	sources := []string{"The Sun is a star at the center of our solar system."}
	response := "The Sun is a star. It is located at the center of our solar system."
	result := CheckHallucination(response, sources)
	if !result.Pass {
		t.Errorf("fully grounded response should pass, score=%f details=%s", result.Score, result.Details)
	}
}

func TestRunAllGatesScoreRange(t *testing.T) {
	gates := DefaultGates()
	results := RunAllGates(gates,
		"This is a test response with some content about the topic.",
		"test query",
		"test gold answer",
		nil,
		100*time.Millisecond,
	)
	for _, r := range results {
		if r.Score < 0 || r.Score > 1 {
			t.Errorf("gate %s score %f out of [0,1] range", r.Gate, r.Score)
		}
		if !strings.Contains(r.Details, "=") && r.Details != "not a failure response" {
			// Most details should contain key=value pairs.
			t.Logf("gate %s has unusual details format: %s", r.Gate, r.Details)
		}
	}
}

// --- Benchmarks ---

func BenchmarkScorecardEvaluate(b *testing.B) {
	cards := DefaultScorecards()
	card := &cards[1] // FactualQA
	scores := map[string]float64{
		"factual_correctness": 0.92,
		"source_grounding":    0.85,
		"completeness":        0.80,
		"conciseness":         0.75,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Evaluate(card, scores)
	}
}

func BenchmarkRunAllGates(b *testing.B) {
	gates := DefaultGates()
	response := "Photosynthesis is a biological process where plants use sunlight to convert water and carbon dioxide into glucose and oxygen. This process occurs primarily in the leaves using chlorophyll. The light reactions take place in the thylakoid membranes while the Calvin cycle occurs in the stroma."
	query := "what is photosynthesis"
	gold := "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen."
	sources := []string{
		"Photosynthesis converts light energy into chemical energy.",
		"Chlorophyll in plant leaves absorbs sunlight for photosynthesis.",
	}
	elapsed := 150 * time.Millisecond

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		RunAllGates(gates, response, query, gold, sources, elapsed)
	}
}

func BenchmarkGenerateEvalSet(b *testing.B) {
	for i := 0; i < b.N; i++ {
		GenerateEvalSet()
	}
}

func BenchmarkCheckCorrectness(b *testing.B) {
	gold := "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen."
	response := "Photosynthesis is a biological process where plants use sunlight to convert water and carbon dioxide into glucose and oxygen."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CheckCorrectness(response, gold)
	}
}

func BenchmarkCheckCoherence(b *testing.B) {
	response := "The water cycle is a continuous process. First, water evaporates from oceans. Then, vapor condenses into clouds. Finally, precipitation returns water to the surface. This cycle sustains life on Earth."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CheckCoherence(response)
	}
}
