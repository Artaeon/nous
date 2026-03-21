package cognitive

import (
	"strings"
	"testing"

	"github.com/artaeon/nous/internal/tools"
)

func newTestRouterWithTools() *ActionRouter {
	ar := NewActionRouter()
	reg := tools.NewRegistry()
	tools.RegisterCalculatorTools(reg)
	tools.RegisterPasswordTools(reg)
	tools.RegisterBookmarkTools(reg)
	tools.RegisterJournalTools(reg)
	tools.RegisterHabitTools(reg)
	tools.RegisterExpenseTools(reg)
	ar.Tools = reg
	return ar
}

func TestHandleCalculate(t *testing.T) {
	ar := newTestRouterWithTools()

	tests := []struct {
		raw      string
		contains string
	}{
		{"calculate 5+3", "8"},
		{"what is 15 * 3", "45"},
		{"compute sqrt(144)", "12"},
		{"how much is 100/4", "25"},
	}

	for _, tt := range tests {
		nlu := &NLUResult{Action: "calculate", Raw: tt.raw, Entities: map[string]string{}}
		result := ar.Execute(nlu, NewConversation(10))
		if result.DirectResponse == "" {
			t.Errorf("calculate(%q): no DirectResponse, Data=%q", tt.raw, result.Data)
			continue
		}
		if !strings.Contains(result.DirectResponse, tt.contains) {
			t.Errorf("calculate(%q) = %q, want to contain %q", tt.raw, result.DirectResponse, tt.contains)
		}
	}
}

func TestHandlePassword(t *testing.T) {
	ar := newTestRouterWithTools()

	tests := []struct {
		raw     string
		prefix  string
		minLen  int
	}{
		{"generate a password", "Password:", 10},
		{"create a passphrase", "Passphrase:", 5},
		{"generate a pin", "PIN:", 4},
	}

	for _, tt := range tests {
		nlu := &NLUResult{Action: "password", Raw: tt.raw, Entities: map[string]string{}}
		result := ar.Execute(nlu, NewConversation(10))
		if result.DirectResponse == "" {
			t.Errorf("password(%q): no DirectResponse", tt.raw)
			continue
		}
		if !strings.HasPrefix(result.DirectResponse, tt.prefix) {
			t.Errorf("password(%q) = %q, want prefix %q", tt.raw, result.DirectResponse, tt.prefix)
		}
	}
}

func TestHandleJournal(t *testing.T) {
	ar := newTestRouterWithTools()

	// Write entry
	nlu := &NLUResult{Action: "journal", Raw: "dear diary today was a great day", Entities: map[string]string{}}
	result := ar.Execute(nlu, NewConversation(10))
	if result.DirectResponse == "" {
		t.Fatalf("journal write: no DirectResponse, Data=%q", result.Data)
	}
	if !strings.Contains(result.DirectResponse, "saved") {
		t.Errorf("journal write should contain 'saved', got %q", result.DirectResponse)
	}

	// Today's entries
	nlu = &NLUResult{Action: "journal", Raw: "show today's journal", Entities: map[string]string{}}
	result = ar.Execute(nlu, NewConversation(10))
	if result.DirectResponse == "" {
		t.Fatalf("journal today: no DirectResponse")
	}
	if !strings.Contains(result.DirectResponse, "great day") {
		t.Errorf("journal today should contain entry text, got %q", result.DirectResponse)
	}
}

func TestHandleHabit(t *testing.T) {
	ar := newTestRouterWithTools()

	// Create
	nlu := &NLUResult{Action: "habit", Raw: "create habit meditation", Entities: map[string]string{"topic": "meditation"}}
	result := ar.Execute(nlu, NewConversation(10))
	if !strings.Contains(result.DirectResponse, "meditation") {
		t.Errorf("habit create: expected meditation, got %q", result.DirectResponse)
	}

	// List
	nlu = &NLUResult{Action: "habit", Raw: "list my habits", Entities: map[string]string{}}
	result = ar.Execute(nlu, NewConversation(10))
	if !strings.Contains(result.DirectResponse, "meditation") {
		t.Errorf("habit list should include meditation, got %q", result.DirectResponse)
	}

	// Check
	nlu = &NLUResult{Action: "habit", Raw: "check off meditation", Entities: map[string]string{"topic": "meditation"}}
	result = ar.Execute(nlu, NewConversation(10))
	if !strings.Contains(result.DirectResponse, "checked") {
		t.Errorf("habit check: expected 'checked', got %q", result.DirectResponse)
	}
}

func TestHandleExpense(t *testing.T) {
	ar := newTestRouterWithTools()

	// Add expense
	nlu := &NLUResult{Action: "expense", Raw: "spent 25 on groceries", Entities: map[string]string{}}
	result := ar.Execute(nlu, NewConversation(10))
	if result.DirectResponse == "" {
		t.Fatalf("expense add: no DirectResponse, Data=%q", result.Data)
	}
	if !strings.Contains(result.DirectResponse, "25") {
		t.Errorf("expense should contain amount, got %q", result.DirectResponse)
	}

	// Summary
	nlu = &NLUResult{Action: "expense", Raw: "expense summary this month", Entities: map[string]string{}}
	result = ar.Execute(nlu, NewConversation(10))
	if result.DirectResponse == "" {
		t.Fatalf("expense summary: no DirectResponse")
	}
}

func TestHandleBookmark(t *testing.T) {
	ar := newTestRouterWithTools()

	// Save
	nlu := &NLUResult{
		Action:   "bookmark",
		Raw:      "bookmark https://golang.org",
		Entities: map[string]string{"url": "https://golang.org"},
	}
	result := ar.Execute(nlu, NewConversation(10))
	if !strings.Contains(result.DirectResponse, "Saved") && !strings.Contains(result.DirectResponse, "Updated") {
		t.Errorf("bookmark save: expected Saved/Updated, got %q", result.DirectResponse)
	}

	// List
	nlu = &NLUResult{Action: "bookmark", Raw: "list my bookmarks", Entities: map[string]string{}}
	result = ar.Execute(nlu, NewConversation(10))
	if !strings.Contains(result.DirectResponse, "golang.org") {
		t.Errorf("bookmark list should include golang.org, got %q", result.DirectResponse)
	}
}

func TestExtractExpenseDetails(t *testing.T) {
	tests := []struct {
		raw    string
		amount string
		desc   string
	}{
		{"spent 25 on groceries", "25", "groceries"},
		{"coffee 4.50", "4.50", "coffee"},
		{"$12.99 for lunch", "12.99", "lunch"},
		{"bought shoes for 89.99", "89.99", "shoes"},
	}

	for _, tt := range tests {
		amount, desc, _ := extractExpenseDetails(tt.raw)
		if amount != tt.amount {
			t.Errorf("extractExpenseDetails(%q): amount=%q, want %q", tt.raw, amount, tt.amount)
		}
		if desc != tt.desc {
			t.Errorf("extractExpenseDetails(%q): desc=%q, want %q", tt.raw, desc, tt.desc)
		}
	}
}

func TestExtractJournalEntry(t *testing.T) {
	tests := []struct {
		raw  string
		want string
	}{
		{"dear diary today was great", "today was great"},
		{"journal entry: feeling productive", "feeling productive"},
		{"journal: I learned Go today", "I learned Go today"},
	}

	for _, tt := range tests {
		got := extractJournalEntry(tt.raw)
		if got != tt.want {
			t.Errorf("extractJournalEntry(%q) = %q, want %q", tt.raw, got, tt.want)
		}
	}
}

func TestGuessExpenseCategory(t *testing.T) {
	tests := []struct {
		desc string
		want string
	}{
		{"coffee", "food"},
		{"uber ride", "transport"},
		{"netflix", "entertainment"},
		{"gym membership", "health"},
		{"rent", "bills"},
		{"new shoes", "shopping"},
		{"random thing", "other"},
	}

	for _, tt := range tests {
		got := guessExpenseCategory(tt.desc)
		if got != tt.want {
			t.Errorf("guessExpenseCategory(%q) = %q, want %q", tt.desc, got, tt.want)
		}
	}
}
