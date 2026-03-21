package cognitive

import (
	"strings"
	"testing"

	"github.com/artaeon/nous/internal/tools"
)

// TestNLUToolsEndToEnd tests the full pipeline: natural language → NLU → ActionRouter → response.
// This verifies that NLU correctly routes to the right tool handler AND the handler produces
// a meaningful direct response (no LLM needed).
func TestNLUToolsEndToEnd(t *testing.T) {
	nlu := NewNLU()
	ar := NewActionRouter()
	reg := tools.NewRegistry()
	tools.RegisterCalculatorTools(reg)
	tools.RegisterPasswordTools(reg)
	tools.RegisterBookmarkTools(reg)
	tools.RegisterJournalTools(reg)
	tools.RegisterHabitTools(reg)
	tools.RegisterExpenseTools(reg)
	ar.Tools = reg

	tests := []struct {
		input      string
		wantAction string
		wantDirect bool   // expect DirectResponse (no LLM)
		contains   string // substring in DirectResponse
	}{
		// Calculator (may route through "calculate" or "compute" — both produce DirectResponse)
		{"calculate 5 + 3", "calculate", true, "8"},

		// Password
		{"generate a password", "password", true, "Password:"},
		{"create a passphrase", "password", true, "Passphrase:"},
		{"generate a pin", "password", true, "PIN:"},

		// Journal
		{"dear diary today was amazing", "journal", true, "saved"},
		{"show my journal entries", "journal", true, ""},

		// Habits
		{"create habit exercise", "habit", true, "exercise"},
		{"list my habits", "habit", true, ""},

		// Expenses
		{"spent 15 on coffee", "expense", true, "15"},
		{"expense summary", "expense", true, ""},

		// Bookmarks
		{"list my bookmarks", "bookmark", true, ""},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := nlu.Understand(tt.input)
			if result.Action != tt.wantAction {
				t.Errorf("NLU(%q): action=%q, want %q (intent=%q, conf=%.2f)",
					tt.input, result.Action, tt.wantAction, result.Intent, result.Confidence)
				return
			}

			actionResult := ar.Execute(result, NewConversation(10))
			if tt.wantDirect && actionResult.DirectResponse == "" {
				t.Errorf("NLU(%q) → %s: expected DirectResponse, got Data=%q",
					tt.input, tt.wantAction, actionResult.Data)
				return
			}
			if tt.contains != "" && !strings.Contains(actionResult.DirectResponse, tt.contains) {
				t.Errorf("NLU(%q) → %s: DirectResponse=%q, want to contain %q",
					tt.input, tt.wantAction, actionResult.DirectResponse, tt.contains)
			}
		})
	}
}
