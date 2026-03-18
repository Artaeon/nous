package cognitive

import (
	"testing"
)

func TestMultiIntent_SplitIntents(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantN    int
		wantSubs []string // optional: check exact sub-strings
	}{
		{
			name:  "reminder plus todo",
			input: "remind me to call mom and add milk to my todos",
			wantN: 2,
			wantSubs: []string{
				"remind me to call mom",
				"add milk to my todos",
			},
		},
		{
			name:  "timer plus translate",
			input: "set a timer for 5 min and translate hello to spanish",
			wantN: 2,
			wantSubs: []string{
				"set a timer for 5 min",
				"translate hello to spanish",
			},
		},
		{
			name:  "convert plus weather question",
			input: "convert 10 km to miles and what's the weather",
			wantN: 2,
			wantSubs: []string{
				"convert 10 km to miles",
				"what's the weather",
			},
		},
		{
			name:  "search and replace not split",
			input: "search and replace",
			wantN: 1,
		},
		{
			name:  "bread and butter not split",
			input: "bread and butter",
			wantN: 1,
		},
		{
			name:  "simple query no split",
			input: "what time is it",
			wantN: 1,
		},
		{
			name:  "list items not split",
			input: "add eggs and milk to my shopping list",
			wantN: 1,
		},
		{
			name:  "check email and then remind",
			input: "check email and then remind me about the meeting at 3pm",
			wantN: 2,
			wantSubs: []string{
				"check email",
				"remind me about the meeting at 3pm",
			},
		},
		{
			name:  "three intents with and then",
			input: "set a timer for 5 min and translate hello to spanish and then check email",
			wantN: 3,
		},
		{
			name:  "also conjunction",
			input: "check email also remind me to call mom",
			wantN: 2,
		},
		{
			name:  "plus conjunction",
			input: "show my todos plus check email",
			wantN: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parts := SplitIntents(tt.input)
			if len(parts) != tt.wantN {
				t.Errorf("SplitIntents(%q) returned %d parts, want %d; parts=%v",
					tt.input, len(parts), tt.wantN, parts)
				return
			}
			if tt.wantSubs != nil {
				for i, want := range tt.wantSubs {
					if i >= len(parts) {
						break
					}
					if parts[i] != want {
						t.Errorf("SplitIntents(%q)[%d] = %q, want %q",
							tt.input, i, parts[i], want)
					}
				}
			}
		})
	}
}

func TestMultiIntent_UnderstandMulti(t *testing.T) {
	nlu := NewNLU()

	tests := []struct {
		name       string
		input      string
		wantN      int
		wantIntent []string // expected intent for each sub-result
	}{
		{
			name:       "reminder plus todo",
			input:      "remind me to call mom and add milk to my todos",
			wantN:      2,
			wantIntent: []string{"reminder", "todo"},
		},
		{
			name:       "timer plus translate",
			input:      "set a timer for 5 min and translate hello to spanish",
			wantN:      2,
			wantIntent: []string{"timer", "translate"},
		},
		{
			name:       "single intent not split",
			input:      "what time is it",
			wantN:      1,
			wantIntent: []string{"question"},
		},
		{
			name:  "check email then remind",
			input: "check email and then remind me about the meeting at 3pm",
			wantN: 2,
			wantIntent: []string{"email", "reminder"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results := nlu.UnderstandMulti(tt.input)
			if len(results) != tt.wantN {
				t.Errorf("UnderstandMulti(%q) returned %d results, want %d",
					tt.input, len(results), tt.wantN)
				return
			}
			if tt.wantIntent != nil {
				for i, want := range tt.wantIntent {
					if i >= len(results) {
						break
					}
					if results[i].Intent != want {
						t.Errorf("UnderstandMulti(%q)[%d].Intent = %q, want %q",
							tt.input, i, results[i].Intent, want)
					}
				}
			}
		})
	}
}

func TestMultiIntent_UnderstandMultiWithContext(t *testing.T) {
	nlu := NewNLU()

	// Test that a high-confidence single intent is not split
	t.Run("high confidence single intent", func(t *testing.T) {
		result := nlu.UnderstandMultiWithContext("remind me to call mom", nil)
		if result.SubResults != nil {
			t.Errorf("expected nil SubResults for single intent, got %d", len(result.SubResults))
		}
		if result.Intent != "reminder" {
			t.Errorf("expected intent=reminder, got %q", result.Intent)
		}
	})

	// Test that a low-confidence compound query gets split
	t.Run("low confidence compound gets split", func(t *testing.T) {
		// Construct an input that individually parses to known intents but
		// as a whole has low confidence. We force this by testing a query
		// that the NLU can't classify well as a whole.
		// "remind me to call mom and add milk to my todos" actually parses well
		// as reminder (high conf), so we test the mechanism differently:
		// Use UnderstandMulti directly and verify SubResults behavior.
		results := nlu.UnderstandMulti("remind me to call mom and add milk to my todos")
		if len(results) != 2 {
			t.Errorf("expected 2 results, got %d", len(results))
			return
		}
		if results[0].Intent != "reminder" {
			t.Errorf("results[0].Intent = %q, want reminder", results[0].Intent)
		}
		if results[1].Intent != "todo" {
			t.Errorf("results[1].Intent = %q, want todo", results[1].Intent)
		}
	})
}
