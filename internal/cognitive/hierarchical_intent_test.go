package cognitive

import (
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Tests for hierarchical intent classification, confidence calibration,
// slot extraction, and evaluation framework.
// -----------------------------------------------------------------------

func TestClassifyHierarchical(t *testing.T) {
	nlu := NewNLU()

	tests := []struct {
		name           string
		input          string
		wantCoarse     CoarseIntent
		wantSubName    string
		wantAction     string
		wantAbstain    bool
	}{
		// Query intents
		{
			name:        "factual_qa/what_is",
			input:       "what is quantum physics",
			wantCoarse:  CoarseQuery,
			wantSubName: "deep_explain",
			wantAction:  "knowledge_lookup",
		},
		{
			name:        "deep_explain/tell_me_about",
			input:       "tell me about the roman empire",
			wantCoarse:  CoarseQuery,
			wantSubName: "deep_explain",
			wantAction:  "knowledge_lookup",
		},
		{
			name:        "deep_explain/how_does",
			input:       "how does photosynthesis work",
			wantCoarse:  CoarseQuery,
			wantSubName: "deep_explain",
			wantAction:  "knowledge_lookup",
		},
		{
			name:        "compare_tradeoff/vs",
			input:       "python vs golang",
			wantCoarse:  CoarseQuery,
			wantSubName: "compare_tradeoff",
			wantAction:  "knowledge_lookup",
		},
		{
			name:        "compare_tradeoff/difference",
			input:       "what's the difference between RAM and ROM",
			wantCoarse:  CoarseQuery,
			wantSubName: "compare_tradeoff",
			wantAction:  "knowledge_lookup",
		},
		// Task intents
		{
			name:        "compose/write_email",
			input:       "write me an email to my boss about the project deadline",
			wantCoarse:  CoarseTask,
			wantSubName: "compose",
			wantAction:  "text_generation",
		},
		{
			name:        "creative_writing/poem",
			input:       "write me a poem about the ocean",
			wantCoarse:  CoarseTask,
			wantSubName: "creative_writing",
			wantAction:  "text_generation",
		},
		{
			name:        "planning/trip",
			input:       "plan a trip to japan for next month",
			wantCoarse:  CoarseTask,
			wantSubName: "planning",
			wantAction:  "structured_generation",
		},
		// Conversation intents
		{
			name:        "social/greeting",
			input:       "hello there",
			wantCoarse:  CoarseConversation,
			wantSubName: "social",
			wantAction:  "empathy_or_greeting",
		},
		{
			name:        "farewell/goodbye",
			input:       "goodbye see you later",
			wantCoarse:  CoarseConversation,
			wantSubName: "farewell",
			wantAction:  "empathy_or_greeting",
		},
		{
			name:        "acknowledgment/thanks",
			input:       "thanks",
			wantCoarse:  CoarseConversation,
			wantSubName: "acknowledgment",
			wantAction:  "respond",
		},
		// Meta intents
		{
			name:        "meta/identity",
			input:       "who are you",
			wantCoarse:  CoarseMeta,
			wantSubName: "identity",
			wantAction:  "self_describe",
		},
		{
			name:        "meta/capabilities",
			input:       "what can you do for me",
			wantCoarse:  CoarseMeta,
			wantSubName: "capabilities",
			wantAction:  "self_describe",
		},
		// Navigation intents
		{
			name:        "navigation/weather",
			input:       "what's the weather like today",
			wantCoarse:  CoarseNavigation,
			wantSubName: "tool_weather",
			wantAction:  "tool_dispatch",
		},
		{
			name:        "navigation/timer",
			input:       "set a timer for 5 minutes",
			wantCoarse:  CoarseNavigation,
			wantSubName: "tool_timer",
			wantAction:  "tool_dispatch",
		},
		{
			name:        "navigation/calculator",
			input:       "calculate 25 times 4",
			wantCoarse:  CoarseNavigation,
			wantSubName: "tool_calculator",
			wantAction:  "tool_dispatch",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ClassifyHierarchical(tt.input, nlu)

			if result.Coarse != tt.wantCoarse {
				t.Errorf("Coarse = %v (%s), want %v (%s)",
					result.Coarse, coarseIntentName(result.Coarse),
					tt.wantCoarse, coarseIntentName(tt.wantCoarse))
			}
			if result.Sub.Name != tt.wantSubName {
				t.Errorf("Sub.Name = %q, want %q", result.Sub.Name, tt.wantSubName)
			}
			if result.Sub.Action != tt.wantAction {
				t.Errorf("Sub.Action = %q, want %q", result.Sub.Action, tt.wantAction)
			}
			if result.Abstain != tt.wantAbstain {
				t.Errorf("Abstain = %v, want %v", result.Abstain, tt.wantAbstain)
			}

			// Verify hierarchy string is well-formed
			parts := strings.Split(result.Hierarchy, "/")
			if len(parts) != 3 {
				t.Errorf("Hierarchy = %q, want 3 parts separated by /", result.Hierarchy)
			}

			// Verify NLUResult is populated
			if result.NLUResult == nil {
				t.Error("NLUResult is nil")
			}

			// Verify slots are populated
			if result.Slots == nil {
				t.Error("Slots is nil")
			}
		})
	}
}

func TestCoarseClassification(t *testing.T) {
	nlu := NewNLU()

	tests := []struct {
		input string
		want  CoarseIntent
	}{
		// Query
		{"what is the speed of light", CoarseQuery},
		{"who invented the telephone", CoarseQuery},
		{"how does encryption work", CoarseQuery},
		{"why is the sky blue", CoarseQuery},
		{"is Pluto a planet", CoarseQuery},

		// Task
		{"write me a story about a detective", CoarseTask},
		{"help me draft an email", CoarseTask},
		{"create a shopping list", CoarseTask},
		{"summarize this article", CoarseTask},

		// Conversation
		{"hello", CoarseConversation},
		{"bye", CoarseConversation},
		{"thanks", CoarseConversation},

		// Meta
		{"who are you", CoarseMeta},
		{"what is your name", CoarseMeta},
		{"are you an AI", CoarseMeta},

		// Navigation
		{"set a timer for 10 minutes", CoarseNavigation},
		{"what's the weather", CoarseNavigation},
		{"remind me to call mom", CoarseNavigation},
		{"translate hello to spanish", CoarseNavigation},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := ClassifyHierarchical(tt.input, nlu)
			if result.Coarse != tt.want {
				t.Errorf("input=%q: Coarse = %s, want %s",
					tt.input,
					coarseIntentName(result.Coarse),
					coarseIntentName(tt.want))
			}
		})
	}
}

func TestSubIntentClassification(t *testing.T) {
	nlu := NewNLU()

	tests := []struct {
		input   string
		wantSub string
	}{
		// Deep explain
		{"explain quantum entanglement", "deep_explain"},
		{"teach me about photosynthesis", "deep_explain"},
		{"tell me about black holes", "deep_explain"},

		// Compare
		{"python vs javascript", "compare_tradeoff"},
		{"difference between TCP and UDP", "compare_tradeoff"},
		{"compare react and vue", "compare_tradeoff"},

		// Creative writing
		{"write me a poem about spring", "creative_writing"},
		{"compose a haiku about rain", "creative_writing"},

		// Compose (functional)
		{"write me an email to the team", "compose"},

		// Social
		{"hey there", "social"},
		{"hi how are you", "social"},

		// Farewell
		{"goodbye", "farewell"},
		{"see ya later", "farewell"},

		// Identity
		{"who are you", "identity"},
		{"what's your name", "identity"},

		// Capabilities
		{"what can you do", "capabilities"},

		// Navigation tools
		{"set a timer for 5 minutes", "tool_timer"},
		{"what's the weather like", "tool_weather"},
		{"define serendipity", "tool_dict"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := ClassifyHierarchical(tt.input, nlu)
			if result.Sub.Name != tt.wantSub {
				t.Errorf("input=%q: Sub.Name = %q, want %q (coarse=%s, base_intent=%s)",
					tt.input, result.Sub.Name, tt.wantSub,
					coarseIntentName(result.Coarse), result.NLUResult.Intent)
			}
		})
	}
}

func TestConfidenceCalibration(t *testing.T) {
	t.Run("agreement_boosts_confidence", func(t *testing.T) {
		signals := ConfidenceSignals{
			PatternConf:   0.80,
			NeuralConf:    0.75,
			PatternIntent: "explain",
			NeuralIntent:  "explain",
			QueryLength:   5,
			HasEntities:   true,
			SlotsFilled:   1,
			SlotsExpected: 1,
		}
		calibrated := CalibrateConfidence(0.75, signals)
		if calibrated <= 0.75 {
			t.Errorf("Agreement should boost confidence: got %f, want > 0.75", calibrated)
		}
	})

	t.Run("disagreement_penalizes_confidence", func(t *testing.T) {
		signals := ConfidenceSignals{
			PatternConf:   0.60,
			NeuralConf:    0.55,
			PatternIntent: "explain",
			NeuralIntent:  "creative",
			QueryLength:   3,
		}
		calibrated := CalibrateConfidence(0.60, signals)
		if calibrated >= 0.60 {
			t.Errorf("Disagreement should penalize: got %f, want < 0.60", calibrated)
		}
	})

	t.Run("short_query_penalized", func(t *testing.T) {
		signals := ConfidenceSignals{
			QueryLength: 1,
		}
		calibrated := CalibrateConfidence(0.50, signals)
		if calibrated >= 0.50 {
			t.Errorf("Short query should be penalized: got %f, want < 0.50", calibrated)
		}
	})

	t.Run("entities_boost_confidence", func(t *testing.T) {
		signals := ConfidenceSignals{
			HasEntities: true,
			QueryLength: 5,
		}
		calibrated := CalibrateConfidence(0.60, signals)
		if calibrated <= 0.60 {
			t.Errorf("Entities should boost: got %f, want > 0.60", calibrated)
		}
	})

	t.Run("ambiguity_penalizes", func(t *testing.T) {
		signals := ConfidenceSignals{
			IsAmbiguous: true,
			QueryLength: 3,
		}
		calibrated := CalibrateConfidence(0.60, signals)
		if calibrated >= 0.60 {
			t.Errorf("Ambiguity should penalize: got %f, want < 0.60", calibrated)
		}
	})

	t.Run("clamped_to_zero_one", func(t *testing.T) {
		signals := ConfidenceSignals{
			PatternIntent: "a",
			NeuralIntent:  "b",
			IsAmbiguous:   true,
			QueryLength:   1,
		}
		calibrated := CalibrateConfidence(0.10, signals)
		if calibrated < 0 || calibrated > 1 {
			t.Errorf("Calibrated confidence should be in [0,1]: got %f", calibrated)
		}

		// Also test ceiling
		signals2 := ConfidenceSignals{
			PatternConf:   0.99,
			NeuralConf:    0.99,
			PatternIntent: "explain",
			NeuralIntent:  "explain",
			HasEntities:   true,
			QueryLength:   10,
			SlotsFilled:   3,
			SlotsExpected: 3,
		}
		calibrated2 := CalibrateConfidence(0.99, signals2)
		if calibrated2 > 1.0 {
			t.Errorf("Calibrated confidence should be <= 1.0: got %f", calibrated2)
		}
	})

	t.Run("slot_fill_rate_boosts", func(t *testing.T) {
		signals := ConfidenceSignals{
			SlotsFilled:   3,
			SlotsExpected: 3,
			QueryLength:   5,
		}
		calibrated := CalibrateConfidence(0.70, signals)
		if calibrated <= 0.70 {
			t.Errorf("Full slot fill should boost: got %f, want > 0.70", calibrated)
		}
	})
}

func TestAbstention(t *testing.T) {
	config := DefaultCalibration()

	t.Run("low_confidence_abstains", func(t *testing.T) {
		signals := ConfidenceSignals{QueryLength: 3}
		result := ShouldAbstain(0.15, signals, config)
		if !result.ShouldAbstain {
			t.Error("Should abstain at confidence 0.15")
		}
		if result.Reason != "low_confidence" {
			t.Errorf("Reason = %q, want %q", result.Reason, "low_confidence")
		}
		if result.Suggestion == "" {
			t.Error("Suggestion should not be empty when abstaining")
		}
	})

	t.Run("high_confidence_does_not_abstain", func(t *testing.T) {
		signals := ConfidenceSignals{QueryLength: 5}
		result := ShouldAbstain(0.85, signals, config)
		if result.ShouldAbstain {
			t.Error("Should not abstain at confidence 0.85")
		}
	})

	t.Run("ambiguous_low_conf_abstains", func(t *testing.T) {
		signals := ConfidenceSignals{
			IsAmbiguous: true,
			QueryLength: 2,
		}
		result := ShouldAbstain(0.40, signals, config)
		if !result.ShouldAbstain {
			t.Error("Should abstain on ambiguous + low confidence")
		}
		if result.Reason != "ambiguous_intent" {
			t.Errorf("Reason = %q, want %q", result.Reason, "ambiguous_intent")
		}
	})

	t.Run("missing_slots_abstains", func(t *testing.T) {
		signals := ConfidenceSignals{
			SlotsFilled:   0,
			SlotsExpected: 3,
			QueryLength:   3,
		}
		result := ShouldAbstain(0.40, signals, config)
		if !result.ShouldAbstain {
			t.Error("Should abstain when missing critical slots")
		}
		if result.Reason != "missing_critical_slots" {
			t.Errorf("Reason = %q, want %q", result.Reason, "missing_critical_slots")
		}
	})

	t.Run("nil_config_uses_defaults", func(t *testing.T) {
		signals := ConfidenceSignals{QueryLength: 3}
		result := ShouldAbstain(0.15, signals, nil)
		if !result.ShouldAbstain {
			t.Error("Should abstain with nil config at low confidence")
		}
	})
}

func TestSlotExtraction(t *testing.T) {
	t.Run("extracts_topic", func(t *testing.T) {
		slots := ExtractSlots("explain quantum physics", CoarseQuery, "deep_explain")
		if slots.Topic == "" {
			t.Error("Should extract topic from 'explain quantum physics'")
		}
		if !strings.Contains(slots.Topic, "quantum physics") {
			t.Errorf("Topic = %q, want to contain 'quantum physics'", slots.Topic)
		}
	})

	t.Run("extracts_goal", func(t *testing.T) {
		slots := ExtractSlots("I want to learn about machine learning", CoarseTask, "general_task")
		if slots.Goal == "" {
			t.Error("Should extract goal from 'I want to ...'")
		}
		if !strings.Contains(slots.Goal, "learn") {
			t.Errorf("Goal = %q, want to contain 'learn'", slots.Goal)
		}
	})

	t.Run("extracts_budget_constraint", func(t *testing.T) {
		slots := ExtractSlots("find me a laptop under $1000", CoarseTask, "recommendation")
		if len(slots.Constraints) == 0 {
			t.Error("Should extract budget constraint from 'under $1000'")
		}
	})

	t.Run("extracts_time_constraint", func(t *testing.T) {
		slots := ExtractSlots("find a recipe within 30 minutes", CoarseTask, "recommendation")
		if len(slots.Constraints) == 0 {
			t.Error("Should extract time constraint from 'within 30 minutes'")
		}
	})

	t.Run("extracts_comparison_axes", func(t *testing.T) {
		slots := ExtractSlots("compare python and javascript in terms of performance", CoarseQuery, "compare_tradeoff")
		if len(slots.ComparisonAxes) == 0 {
			t.Error("Should extract comparison axis from 'in terms of performance'")
		}
		found := false
		for _, axis := range slots.ComparisonAxes {
			if strings.Contains(axis, "performance") {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("ComparisonAxes = %v, want to contain 'performance'", slots.ComparisonAxes)
		}
	})

	t.Run("extracts_time_horizon", func(t *testing.T) {
		slots := ExtractSlots("plan a trip by next week", CoarseTask, "planning")
		if slots.TimeHorizon == "" {
			t.Error("Should extract time horizon from 'by next week'")
		}
		if !strings.Contains(strings.ToLower(slots.TimeHorizon), "next week") {
			t.Errorf("TimeHorizon = %q, want to contain 'next week'", slots.TimeHorizon)
		}
	})

	t.Run("extracts_person", func(t *testing.T) {
		slots := ExtractSlots("write a letter to my boss about the deadline", CoarseTask, "compose")
		if _, ok := slots.AllSlots["person"]; !ok {
			t.Error("Should extract person 'boss' from 'to my boss'")
		} else if !strings.Contains(slots.AllSlots["person"].Value, "boss") {
			t.Errorf("person = %q, want to contain 'boss'", slots.AllSlots["person"].Value)
		}
	})

	t.Run("extracts_format", func(t *testing.T) {
		slots := ExtractSlots("explain this as a list", CoarseQuery, "deep_explain")
		if _, ok := slots.AllSlots["format"]; !ok {
			t.Error("Should extract format 'list' from 'as a list'")
		}
	})

	t.Run("extracts_tone", func(t *testing.T) {
		slots := ExtractSlots("write this in a formal tone", CoarseTask, "compose")
		if _, ok := slots.AllSlots["tone"]; !ok {
			t.Error("Should extract tone 'formal' from 'in a formal tone'")
		}
	})

	t.Run("extracts_quantity", func(t *testing.T) {
		slots := ExtractSlots("I need to buy 5 kg of flour", CoarseTask, "general_task")
		if _, ok := slots.AllSlots["quantity"]; !ok {
			t.Error("Should extract quantity '5 kg'")
		}
	})

	t.Run("filled_count_accurate", func(t *testing.T) {
		slots := ExtractSlots("explain quantum physics in terms of energy as a list in a formal tone", CoarseQuery, "deep_explain")
		if slots.FilledCount == 0 {
			t.Error("FilledCount should be > 0 for a slot-rich query")
		}
		if slots.FilledCount != len(slots.AllSlots) {
			t.Errorf("FilledCount = %d, len(AllSlots) = %d, should match", slots.FilledCount, len(slots.AllSlots))
		}
	})

	t.Run("empty_input", func(t *testing.T) {
		slots := ExtractSlots("", CoarseQuery, "factual_qa")
		if slots.FilledCount != 0 {
			t.Errorf("Empty input should have 0 filled slots, got %d", slots.FilledCount)
		}
	})

	t.Run("compare_topic_extraction", func(t *testing.T) {
		slots := ExtractSlots("compare python vs javascript", CoarseQuery, "compare_tradeoff")
		if slots.Topic == "" {
			t.Error("Should extract compare topic")
		}
		if !strings.Contains(slots.Topic, "vs") {
			t.Errorf("Compare topic = %q, want to contain 'vs'", slots.Topic)
		}
	})
}

func TestSlotValidation(t *testing.T) {
	tests := []struct {
		name     string
		slot     ExtractedSlot
		wantValid bool
	}{
		{
			name:      "valid_topic",
			slot:      ExtractedSlot{Name: "topic", Value: "quantum physics", Type: IntentSlotTopic, Confidence: 0.8},
			wantValid: true,
		},
		{
			name:      "empty_topic",
			slot:      ExtractedSlot{Name: "topic", Value: "", Type: IntentSlotTopic, Confidence: 0.8},
			wantValid: false,
		},
		{
			name:      "stopword_only_topic",
			slot:      ExtractedSlot{Name: "topic", Value: "the", Type: IntentSlotTopic, Confidence: 0.8},
			wantValid: false,
		},
		{
			name:      "valid_time_horizon",
			slot:      ExtractedSlot{Name: "time", Value: "next week", Type: IntentSlotTimeHorizon, Confidence: 0.8},
			wantValid: true,
		},
		{
			name:      "invalid_time_horizon",
			slot:      ExtractedSlot{Name: "time", Value: "purple elephant", Type: IntentSlotTimeHorizon, Confidence: 0.8},
			wantValid: false,
		},
		{
			name:      "valid_quantity",
			slot:      ExtractedSlot{Name: "qty", Value: "5 kg", Type: IntentSlotQuantity, Confidence: 0.8},
			wantValid: true,
		},
		{
			name:      "invalid_quantity_no_digits",
			slot:      ExtractedSlot{Name: "qty", Value: "some amount", Type: IntentSlotQuantity, Confidence: 0.8},
			wantValid: false,
		},
		{
			name:      "valid_format",
			slot:      ExtractedSlot{Name: "fmt", Value: "list", Type: IntentSlotFormat, Confidence: 0.8},
			wantValid: true,
		},
		{
			name:      "invalid_format",
			slot:      ExtractedSlot{Name: "fmt", Value: "banana", Type: IntentSlotFormat, Confidence: 0.8},
			wantValid: false,
		},
		{
			name:      "valid_tone",
			slot:      ExtractedSlot{Name: "tone", Value: "formal", Type: IntentSlotTone, Confidence: 0.8},
			wantValid: true,
		},
		{
			name:      "invalid_tone",
			slot:      ExtractedSlot{Name: "tone", Value: "purple", Type: IntentSlotTone, Confidence: 0.8},
			wantValid: false,
		},
		{
			name:      "valid_goal",
			slot:      ExtractedSlot{Name: "goal", Value: "learn programming", Type: IntentSlotGoal, Confidence: 0.8},
			wantValid: true,
		},
		{
			name:      "valid_constraint",
			slot:      ExtractedSlot{Name: "constraint", Value: "under $500", Type: IntentSlotConstraint, Confidence: 0.8},
			wantValid: true,
		},
		{
			name:      "valid_person",
			slot:      ExtractedSlot{Name: "person", Value: "mom", Type: IntentSlotPerson, Confidence: 0.8},
			wantValid: true,
		},
		{
			name:      "valid_location",
			slot:      ExtractedSlot{Name: "loc", Value: "new york city", Type: IntentSlotLocation, Confidence: 0.8},
			wantValid: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			slot := tt.slot
			valid := ValidateSlot(&slot)
			if valid != tt.wantValid {
				t.Errorf("ValidateSlot(%+v) = %v, want %v", tt.slot, valid, tt.wantValid)
			}
			if slot.Validated != tt.wantValid {
				t.Errorf("slot.Validated = %v, want %v", slot.Validated, tt.wantValid)
			}
			// Invalid slots should have reduced confidence
			if !tt.wantValid && slot.Confidence >= tt.slot.Confidence {
				t.Errorf("Invalid slot confidence should be reduced: got %f, original %f",
					slot.Confidence, tt.slot.Confidence)
			}
		})
	}
}

func TestConfusionMatrix(t *testing.T) {
	t.Run("accuracy", func(t *testing.T) {
		cm := NewConfusionMatrix([]string{"cat", "dog", "bird"})
		// 10 correct cats, 2 cats misclassified as dogs
		for i := 0; i < 10; i++ {
			cm.Record("cat", "cat")
		}
		for i := 0; i < 2; i++ {
			cm.Record("dog", "cat")
		}
		// 8 correct dogs, 1 dog misclassified as bird
		for i := 0; i < 8; i++ {
			cm.Record("dog", "dog")
		}
		cm.Record("bird", "dog")
		// 5 correct birds
		for i := 0; i < 5; i++ {
			cm.Record("bird", "bird")
		}

		// Total: 26, Correct: 23
		acc := cm.Accuracy()
		expected := 23.0 / 26.0
		if diff := acc - expected; diff > 0.001 || diff < -0.001 {
			t.Errorf("Accuracy = %f, want %f", acc, expected)
		}
	})

	t.Run("precision_recall_f1", func(t *testing.T) {
		cm := NewConfusionMatrix([]string{"positive", "negative"})
		// TP=8, FP=2 (predicted positive but actual negative)
		for i := 0; i < 8; i++ {
			cm.Record("positive", "positive")
		}
		for i := 0; i < 2; i++ {
			cm.Record("positive", "negative")
		}
		// TN=7, FN=3 (predicted negative but actual positive)
		for i := 0; i < 7; i++ {
			cm.Record("negative", "negative")
		}
		for i := 0; i < 3; i++ {
			cm.Record("negative", "positive")
		}

		metrics := cm.PerIntentMetrics()

		// Positive: TP=8, FP=2, FN=3
		posM := metrics["positive"]
		expectedPrec := 8.0 / 10.0 // 0.8
		expectedRec := 8.0 / 11.0  // ~0.727
		if diff := posM.Precision - expectedPrec; diff > 0.01 || diff < -0.01 {
			t.Errorf("positive Precision = %f, want %f", posM.Precision, expectedPrec)
		}
		if diff := posM.Recall - expectedRec; diff > 0.01 || diff < -0.01 {
			t.Errorf("positive Recall = %f, want %f", posM.Recall, expectedRec)
		}
		expectedF1 := 2 * expectedPrec * expectedRec / (expectedPrec + expectedRec)
		if diff := posM.F1 - expectedF1; diff > 0.01 || diff < -0.01 {
			t.Errorf("positive F1 = %f, want %f", posM.F1, expectedF1)
		}

		// Negative: TP=7, FP=3, FN=2
		negM := metrics["negative"]
		negExpPrec := 7.0 / 10.0 // 0.7
		negExpRec := 7.0 / 9.0   // ~0.778
		if diff := negM.Precision - negExpPrec; diff > 0.01 || diff < -0.01 {
			t.Errorf("negative Precision = %f, want %f", negM.Precision, negExpPrec)
		}
		if diff := negM.Recall - negExpRec; diff > 0.01 || diff < -0.01 {
			t.Errorf("negative Recall = %f, want %f", negM.Recall, negExpRec)
		}
	})

	t.Run("macro_f1", func(t *testing.T) {
		cm := NewConfusionMatrix([]string{"a", "b"})
		for i := 0; i < 10; i++ {
			cm.Record("a", "a")
		}
		for i := 0; i < 10; i++ {
			cm.Record("b", "b")
		}
		// Perfect classification: macro F1 should be 1.0
		f1 := cm.MacroF1()
		if f1 != 1.0 {
			t.Errorf("MacroF1 = %f, want 1.0 for perfect classification", f1)
		}
	})

	t.Run("weighted_f1", func(t *testing.T) {
		cm := NewConfusionMatrix([]string{"a", "b"})
		for i := 0; i < 10; i++ {
			cm.Record("a", "a")
		}
		for i := 0; i < 10; i++ {
			cm.Record("b", "b")
		}
		wf1 := cm.WeightedF1()
		if wf1 != 1.0 {
			t.Errorf("WeightedF1 = %f, want 1.0 for perfect classification", wf1)
		}
	})

	t.Run("empty_matrix", func(t *testing.T) {
		cm := NewConfusionMatrix([]string{"a"})
		if cm.Accuracy() != 0 {
			t.Errorf("Empty matrix accuracy should be 0, got %f", cm.Accuracy())
		}
		if cm.MacroF1() != 0 {
			t.Errorf("Empty matrix MacroF1 should be 0, got %f", cm.MacroF1())
		}
	})

	t.Run("dynamic_label_expansion", func(t *testing.T) {
		cm := NewConfusionMatrix([]string{"a", "b"})
		cm.Record("c", "a") // "c" is new
		if len(cm.Labels) != 3 {
			t.Errorf("Labels should have 3 entries after expansion, got %d", len(cm.Labels))
		}
		// Verify matrix dimensions
		for i, row := range cm.Matrix {
			if len(row) != 3 {
				t.Errorf("Row %d has %d columns, want 3", i, len(row))
			}
		}
		if len(cm.Matrix) != 3 {
			t.Errorf("Matrix has %d rows, want 3", len(cm.Matrix))
		}
	})
}

func TestIntentEvalSetGeneration(t *testing.T) {
	examples := GenerateIntentEvalSet()

	if len(examples) == 0 {
		t.Fatal("GenerateIntentEvalSet returned empty set")
	}

	// Check that we have multiple distinct intents
	intentCounts := make(map[string]int)
	for _, ex := range examples {
		intentCounts[ex.Expected]++
		if ex.Input == "" {
			t.Error("Found example with empty input")
		}
		if ex.Expected == "" {
			t.Errorf("Found example with empty expected intent: %q", ex.Input)
		}
	}

	// We should have at least 10 distinct intents
	if len(intentCounts) < 10 {
		t.Errorf("Eval set should cover at least 10 intents, got %d: %v", len(intentCounts), intentCounts)
	}

	// Each intent should have at least 2 examples
	for intent, count := range intentCounts {
		if count < 2 {
			t.Errorf("Intent %q has only %d examples, want at least 2", intent, count)
		}
	}

	// Verify key intents are present
	requiredIntents := []string{
		"greeting", "farewell", "meta", "explain", "creative",
		"compare", "weather", "calculate", "timer",
	}
	for _, ri := range requiredIntents {
		if intentCounts[ri] == 0 {
			t.Errorf("Required intent %q not found in eval set", ri)
		}
	}
}

func TestEvaluateIntentClassifier(t *testing.T) {
	// Simple classifier that always returns "greeting"
	alwaysGreeting := func(input string) string { return "greeting" }
	examples := []IntentEvalExample{
		{Input: "hello", Expected: "greeting"},
		{Input: "hi", Expected: "greeting"},
		{Input: "what is X", Expected: "explain"},
		{Input: "bye", Expected: "farewell"},
	}

	result := EvaluateIntentClassifier(alwaysGreeting, examples)

	// Accuracy should be 2/4 = 0.5
	if diff := result.Accuracy - 0.5; diff > 0.01 || diff < -0.01 {
		t.Errorf("Accuracy = %f, want 0.5", result.Accuracy)
	}

	// Greeting precision should be 2/4 = 0.5 (2 correct out of 4 predicted)
	greetingM := result.PerIntent["greeting"]
	if diff := greetingM.Precision - 0.5; diff > 0.01 || diff < -0.01 {
		t.Errorf("Greeting precision = %f, want 0.5", greetingM.Precision)
	}
	// Greeting recall should be 1.0 (2 out of 2 actual greetings)
	if diff := greetingM.Recall - 1.0; diff > 0.01 || diff < -0.01 {
		t.Errorf("Greeting recall = %f, want 1.0", greetingM.Recall)
	}
}

func BenchmarkClassifyHierarchical(b *testing.B) {
	nlu := NewNLU()
	inputs := []string{
		"what is quantum physics",
		"write me a poem about the ocean",
		"hello there",
		"set a timer for 5 minutes",
		"compare python and javascript",
		"who are you",
		"plan a trip to japan",
		"remind me to call mom tomorrow",
		"thanks",
		"how does photosynthesis work",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		input := inputs[i%len(inputs)]
		ClassifyHierarchical(input, nlu)
	}
}

func BenchmarkSlotExtraction(b *testing.B) {
	inputs := []struct {
		text    string
		coarse  CoarseIntent
		subName string
	}{
		{"explain quantum physics in terms of energy", CoarseQuery, "deep_explain"},
		{"write me an email to my boss in a formal tone", CoarseTask, "compose"},
		{"find a laptop under $1000 with at least 16GB RAM", CoarseTask, "recommendation"},
		{"compare python and javascript for web development", CoarseQuery, "compare_tradeoff"},
		{"plan a trip to japan by next month", CoarseTask, "planning"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		in := inputs[i%len(inputs)]
		ExtractSlots(in.text, in.coarse, in.subName)
	}
}
