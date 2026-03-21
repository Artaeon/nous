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
