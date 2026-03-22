package cognitive

import (
	"fmt"
	"testing"
)

func TestRound2Routing(t *testing.T) {
	nlu := NewNLU()
	inputs := []string{
		"hey nous how are you?",
		"what do you think about artificial intelligence?",
		"can you help me write a shopping list?",
		"remind me to call mom tomorrow",
		"define serendipity",
		"tell me something interesting",
		"who made you?",
		"do you have feelings?",
	}
	for _, input := range inputs {
		r := nlu.Understand(input)
		fmt.Printf("%-55s → intent=%-15s action=%-20s conf=%.2f topic=%q\n",
			input, r.Intent, r.Action, r.Confidence, r.Entities["topic"])
	}
}

func TestNeuralOverrideDebug(t *testing.T) {
	nlu := NewNLU()
	// Train the neural classifier
	nlu.Neural = NewNeuralNLU("")
	nlu.Neural.LoadOrTrain(nlu)

	problematic := []struct {
		input      string
		wantIntent string
		wantAction string
	}{
		{"thanks!", "affirmation", "respond"},
		{"thanks", "affirmation", "respond"},
		{"thank you", "affirmation", "respond"},
		{"thank you so much", "affirmation", "respond"},
		{"thx", "affirmation", "respond"},
		{"sqrt of 144", "calculate", "calculate"},
		{"how do I learn guitar?", "question", "lookup_knowledge"},
		{"who was einstein?", "explain", "lookup_knowledge"},
		{"how do I learn to cook?", "question", "lookup_knowledge"},
		{"I'm feeling a bit tired", "conversation", "respond"},
		{"I will run tomorrow", "conversation", "respond"},
		{"compare python and javascript", "compare", "compare"},
		{"I feel tired", "conversation", "respond"},
		{"tell me a joke about programming", "creative", "creative"},
	}

	for _, tt := range problematic {
		r := nlu.Understand(tt.input)
		neuralResult := nlu.Neural.Classify(tt.input)
		neuralIntent := "none"
		neuralConf := 0.0
		if neuralResult != nil {
			neuralIntent = neuralResult.Intent
			neuralConf = neuralResult.Confidence
		}
		status := "✓"
		if r.Intent != tt.wantIntent || r.Action != tt.wantAction {
			status = "✗"
		}
		fmt.Printf("%s %-35s → neural=%-15s(%.2f) final_intent=%-15s action=%-20s (want: %s/%s)\n",
			status, tt.input, neuralIntent, neuralConf, r.Intent, r.Action, tt.wantIntent, tt.wantAction)
		if r.Intent != tt.wantIntent {
			t.Errorf("Understand(%q): want intent=%s, got %s", tt.input, tt.wantIntent, r.Intent)
		}
		if r.Action != tt.wantAction {
			t.Errorf("Understand(%q): want action=%s, got %s", tt.input, tt.wantAction, r.Action)
		}
	}
}
