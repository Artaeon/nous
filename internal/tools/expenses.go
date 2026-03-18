package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Expense represents a single expense entry.
type Expense struct {
	Amount      float64   `json:"amount"`
	Category    string    `json:"category"`
	Description string    `json:"description,omitempty"`
	Timestamp   time.Time `json:"timestamp"`
}

// ExpenseStore manages expenses persisted in a JSON file.
type ExpenseStore struct {
	filePath string
	expenses []Expense
	currency string
}

// NewExpenseStore creates a new ExpenseStore at the default location.
func NewExpenseStore() *ExpenseStore {
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, ".nous")
	os.MkdirAll(dir, 0755)
	return newExpenseStoreAt(filepath.Join(dir, "expenses.json"), "€")
}

// newExpenseStoreAt creates an ExpenseStore at a specific path (for testing).
func newExpenseStoreAt(path, currency string) *ExpenseStore {
	es := &ExpenseStore{filePath: path, currency: currency}
	es.load()
	return es
}

func (es *ExpenseStore) load() {
	data, err := os.ReadFile(es.filePath)
	if err != nil {
		es.expenses = nil
		return
	}
	if err := json.Unmarshal(data, &es.expenses); err != nil {
		es.expenses = nil
	}
}

func (es *ExpenseStore) save() error {
	data, err := json.MarshalIndent(es.expenses, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal expenses: %w", err)
	}
	dir := filepath.Dir(es.filePath)
	tmp, err := os.CreateTemp(dir, ".expenses-*.tmp")
	if err != nil {
		return fmt.Errorf("create temp: %w", err)
	}
	tmpPath := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("write temp: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("close temp: %w", err)
	}
	if err := os.Rename(tmpPath, es.filePath); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("rename: %w", err)
	}
	return nil
}

var validCategories = map[string]bool{
	"food":          true,
	"transport":     true,
	"entertainment": true,
	"bills":         true,
	"health":        true,
	"shopping":      true,
	"other":         true,
}

// Add logs a new expense.
func (es *ExpenseStore) Add(amount float64, category, description string) (*Expense, error) {
	if amount <= 0 {
		return nil, fmt.Errorf("amount must be positive")
	}
	if category == "" {
		category = "other"
	}
	category = strings.ToLower(category)
	if !validCategories[category] {
		return nil, fmt.Errorf("invalid category %q (use: food, transport, entertainment, bills, health, shopping, other)", category)
	}
	expense := Expense{
		Amount:      amount,
		Category:    category,
		Description: description,
		Timestamp:   time.Now(),
	}
	es.expenses = append(es.expenses, expense)
	if err := es.save(); err != nil {
		return nil, err
	}
	return &expense, nil
}

// List shows recent expenses, optionally filtered by category or date range.
func (es *ExpenseStore) List(category, from, to string) (string, error) {
	if len(es.expenses) == 0 {
		return "No expenses recorded.", nil
	}

	var filtered []Expense
	for _, e := range es.expenses {
		if category != "" && !strings.EqualFold(e.Category, category) {
			continue
		}
		dateStr := e.Timestamp.Format("2006-01-02")
		if from != "" && dateStr < from {
			continue
		}
		if to != "" && dateStr > to {
			continue
		}
		filtered = append(filtered, e)
	}

	if len(filtered) == 0 {
		return "No expenses matching filter.", nil
	}

	return es.formatExpenses(filtered), nil
}

// Summary shows spending by category for the given period.
func (es *ExpenseStore) Summary(period string) (string, error) {
	if period == "" {
		period = "month"
	}

	now := time.Now()
	var cutoff string
	switch period {
	case "today":
		cutoff = now.Format("2006-01-02")
	case "week":
		cutoff = now.AddDate(0, 0, -7).Format("2006-01-02")
	case "month":
		cutoff = now.AddDate(0, -1, 0).Format("2006-01-02")
	default:
		return "", fmt.Errorf("invalid period %q (use: today, week, month)", period)
	}

	today := now.Format("2006-01-02")
	categoryTotals := make(map[string]float64)
	grandTotal := 0.0

	for _, e := range es.expenses {
		dateStr := e.Timestamp.Format("2006-01-02")
		if dateStr < cutoff || dateStr > today {
			continue
		}
		categoryTotals[e.Category] += e.Amount
		grandTotal += e.Amount
	}

	if len(categoryTotals) == 0 {
		return fmt.Sprintf("No expenses for period %q.", period), nil
	}

	// Sort categories for consistent output.
	var cats []string
	for c := range categoryTotals {
		cats = append(cats, c)
	}
	sort.Strings(cats)

	var sb strings.Builder
	fmt.Fprintf(&sb, "Expense summary (%s):\n", period)
	for _, c := range cats {
		fmt.Fprintf(&sb, "  %s: %s%.2f\n", c, es.currency, categoryTotals[c])
	}
	fmt.Fprintf(&sb, "  Total: %s%.2f\n", es.currency, grandTotal)

	return sb.String(), nil
}

// DeleteLast removes the last expense entry.
func (es *ExpenseStore) DeleteLast() (string, error) {
	if len(es.expenses) == 0 {
		return "", fmt.Errorf("no expenses to delete")
	}
	removed := es.expenses[len(es.expenses)-1]
	es.expenses = es.expenses[:len(es.expenses)-1]
	if err := es.save(); err != nil {
		return "", err
	}
	return fmt.Sprintf("deleted: %s%.2f %s (%s)", es.currency, removed.Amount, removed.Category, removed.Description), nil
}

// DeleteByIndex removes an expense by index (1-based).
func (es *ExpenseStore) DeleteByIndex(idx int) (string, error) {
	if idx < 1 || idx > len(es.expenses) {
		return "", fmt.Errorf("index %d out of range (1-%d)", idx, len(es.expenses))
	}
	i := idx - 1
	removed := es.expenses[i]
	es.expenses = append(es.expenses[:i], es.expenses[i+1:]...)
	if err := es.save(); err != nil {
		return "", err
	}
	return fmt.Sprintf("deleted: %s%.2f %s (%s)", es.currency, removed.Amount, removed.Category, removed.Description), nil
}

func (es *ExpenseStore) formatExpenses(expenses []Expense) string {
	var sb strings.Builder
	for i, e := range expenses {
		if i > 0 {
			sb.WriteString("\n")
		}
		ts := e.Timestamp.Format("2006-01-02 15:04")
		desc := e.Description
		if desc == "" {
			desc = "-"
		}
		fmt.Fprintf(&sb, "#%d [%s] %s%.2f  %s  %s", i+1, ts, es.currency, e.Amount, e.Category, desc)
	}
	return sb.String()
}

// RegisterExpenseTools adds the expenses tool to the registry.
func RegisterExpenseTools(r *Registry) {
	store := NewExpenseStore()
	r.Register(Tool{
		Name:        "expenses",
		Description: "Track expenses. Args: action (add/list/summary/delete), amount, category, description, period (today/week/month), from, to, index.",
		Execute: func(args map[string]string) (string, error) {
			return toolExpenses(store, args)
		},
	})
}

func toolExpenses(store *ExpenseStore, args map[string]string) (string, error) {
	action := args["action"]
	switch action {
	case "add":
		amountStr := args["amount"]
		if amountStr == "" {
			return "", fmt.Errorf("expenses add requires 'amount'")
		}
		amount, err := strconv.ParseFloat(amountStr, 64)
		if err != nil {
			return "", fmt.Errorf("invalid amount: %s", amountStr)
		}
		category := args["category"]
		description := args["description"]
		expense, err := store.Add(amount, category, description)
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("logged: %s%.2f %s (%s)", store.currency, expense.Amount, expense.Category, expense.Description), nil

	case "list":
		category := args["category"]
		from := args["from"]
		to := args["to"]
		return store.List(category, from, to)

	case "summary":
		period := args["period"]
		return store.Summary(period)

	case "delete":
		idxStr := args["index"]
		if idxStr != "" {
			idx, err := strconv.Atoi(idxStr)
			if err != nil {
				return "", fmt.Errorf("invalid index: %s", idxStr)
			}
			return store.DeleteByIndex(idx)
		}
		return store.DeleteLast()

	default:
		return "", fmt.Errorf("expenses: unknown action %q (use add/list/summary/delete)", action)
	}
}
