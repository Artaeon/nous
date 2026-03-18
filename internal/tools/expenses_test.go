package tools

import (
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestExpenseAddAndList(t *testing.T) {
	path := filepath.Join(t.TempDir(), "expenses.json")
	store := newExpenseStoreAt(path, "€")

	exp, err := store.Add(12.50, "food", "Lunch at cafe")
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	if exp.Amount != 12.50 {
		t.Errorf("amount = %f, want 12.50", exp.Amount)
	}
	if exp.Category != "food" {
		t.Errorf("category = %q, want 'food'", exp.Category)
	}

	list, err := store.List("", "", "")
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if !strings.Contains(list, "€12.50") {
		t.Errorf("list missing formatted amount: %s", list)
	}
	if !strings.Contains(list, "food") {
		t.Errorf("list missing category: %s", list)
	}
	if !strings.Contains(list, "Lunch at cafe") {
		t.Errorf("list missing description: %s", list)
	}
}

func TestExpenseCategoryFiltering(t *testing.T) {
	path := filepath.Join(t.TempDir(), "expenses.json")
	store := newExpenseStoreAt(path, "€")

	store.Add(10.00, "food", "Groceries")
	store.Add(25.00, "transport", "Taxi")
	store.Add(8.50, "food", "Coffee")

	list, err := store.List("food", "", "")
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if !strings.Contains(list, "Groceries") || !strings.Contains(list, "Coffee") {
		t.Errorf("category filter missing food items: %s", list)
	}
	if strings.Contains(list, "Taxi") {
		t.Errorf("category filter should exclude transport: %s", list)
	}
}

func TestExpenseSummaryByPeriod(t *testing.T) {
	path := filepath.Join(t.TempDir(), "expenses.json")
	store := newExpenseStoreAt(path, "€")

	// Add entries for today.
	store.Add(10.00, "food", "Breakfast")
	store.Add(20.00, "food", "Dinner")
	store.Add(15.00, "transport", "Bus")

	// Add an old entry outside the period.
	old := Expense{
		Amount:    100.00,
		Category:  "shopping",
		Timestamp: time.Now().AddDate(0, -2, 0),
	}
	store.expenses = append(store.expenses, old)
	store.save()

	summary, err := store.Summary("month")
	if err != nil {
		t.Fatalf("Summary: %v", err)
	}
	if !strings.Contains(summary, "food: €30.00") {
		t.Errorf("summary missing food total: %s", summary)
	}
	if !strings.Contains(summary, "transport: €15.00") {
		t.Errorf("summary missing transport total: %s", summary)
	}
	if !strings.Contains(summary, "Total: €45.00") {
		t.Errorf("summary missing grand total: %s", summary)
	}
	// Old shopping entry should not be in the month summary.
	if strings.Contains(summary, "shopping") {
		t.Errorf("summary should not include old entries: %s", summary)
	}
}

func TestExpenseDelete(t *testing.T) {
	path := filepath.Join(t.TempDir(), "expenses.json")
	store := newExpenseStoreAt(path, "€")

	store.Add(10.00, "food", "First")
	store.Add(20.00, "transport", "Second")
	store.Add(30.00, "health", "Third")

	// Delete last.
	result, err := store.DeleteLast()
	if err != nil {
		t.Fatalf("DeleteLast: %v", err)
	}
	if !strings.Contains(result, "€30.00") {
		t.Errorf("delete result should show removed item: %s", result)
	}
	if len(store.expenses) != 2 {
		t.Errorf("expected 2 expenses, got %d", len(store.expenses))
	}

	// Delete by index.
	result, err = store.DeleteByIndex(1)
	if err != nil {
		t.Fatalf("DeleteByIndex: %v", err)
	}
	if !strings.Contains(result, "€10.00") {
		t.Errorf("delete by index should show removed item: %s", result)
	}
	if len(store.expenses) != 1 {
		t.Errorf("expected 1 expense, got %d", len(store.expenses))
	}
}

func TestExpenseDeleteEmpty(t *testing.T) {
	path := filepath.Join(t.TempDir(), "expenses.json")
	store := newExpenseStoreAt(path, "€")

	_, err := store.DeleteLast()
	if err == nil {
		t.Error("expected error deleting from empty list")
	}
}

func TestExpenseDeleteOutOfRange(t *testing.T) {
	path := filepath.Join(t.TempDir(), "expenses.json")
	store := newExpenseStoreAt(path, "€")

	store.Add(10.00, "food", "Test")

	_, err := store.DeleteByIndex(5)
	if err == nil {
		t.Error("expected error for out-of-range index")
	}
}

func TestExpenseAmountFormatting(t *testing.T) {
	path := filepath.Join(t.TempDir(), "expenses.json")
	store := newExpenseStoreAt(path, "$")

	store.Add(9.99, "food", "Snack")
	list, err := store.List("", "", "")
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if !strings.Contains(list, "$9.99") {
		t.Errorf("expected dollar formatting: %s", list)
	}
}

func TestExpenseInvalidAmount(t *testing.T) {
	path := filepath.Join(t.TempDir(), "expenses.json")
	store := newExpenseStoreAt(path, "€")

	_, err := store.Add(0, "food", "Zero")
	if err == nil {
		t.Error("expected error for zero amount")
	}

	_, err = store.Add(-5.00, "food", "Negative")
	if err == nil {
		t.Error("expected error for negative amount")
	}
}

func TestExpenseInvalidCategory(t *testing.T) {
	path := filepath.Join(t.TempDir(), "expenses.json")
	store := newExpenseStoreAt(path, "€")

	_, err := store.Add(10.00, "invalid", "Test")
	if err == nil {
		t.Error("expected error for invalid category")
	}
}

func TestExpenseDefaultCategory(t *testing.T) {
	path := filepath.Join(t.TempDir(), "expenses.json")
	store := newExpenseStoreAt(path, "€")

	exp, err := store.Add(5.00, "", "No category")
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	if exp.Category != "other" {
		t.Errorf("expected default category 'other', got %q", exp.Category)
	}
}

func TestExpensePersistence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "expenses.json")

	store1 := newExpenseStoreAt(path, "€")
	store1.Add(42.00, "food", "Persistent")

	store2 := newExpenseStoreAt(path, "€")
	list, err := store2.List("", "", "")
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if !strings.Contains(list, "Persistent") {
		t.Errorf("expense not persisted: %s", list)
	}
}

func TestExpenseEmptyList(t *testing.T) {
	path := filepath.Join(t.TempDir(), "expenses.json")
	store := newExpenseStoreAt(path, "€")

	list, err := store.List("", "", "")
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if list != "No expenses recorded." {
		t.Errorf("expected empty message: %s", list)
	}
}

func TestExpenseSummaryToday(t *testing.T) {
	path := filepath.Join(t.TempDir(), "expenses.json")
	store := newExpenseStoreAt(path, "€")

	store.Add(10.00, "food", "Breakfast")

	// Add a yesterday entry.
	yesterday := Expense{
		Amount:    50.00,
		Category:  "shopping",
		Timestamp: time.Now().AddDate(0, 0, -1),
	}
	store.expenses = append(store.expenses, yesterday)
	store.save()

	summary, err := store.Summary("today")
	if err != nil {
		t.Fatalf("Summary: %v", err)
	}
	if !strings.Contains(summary, "food: €10.00") {
		t.Errorf("today summary missing food: %s", summary)
	}
	if strings.Contains(summary, "shopping") {
		t.Errorf("today summary should not include yesterday: %s", summary)
	}
}
