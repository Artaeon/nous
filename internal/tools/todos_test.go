package tools

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTodoAddAndList(t *testing.T) {
	path := filepath.Join(t.TempDir(), "todos.json")
	store := newTodoStoreAt(path)

	todo := store.AddTodo("Buy milk", "high", []string{"shopping"})
	if todo.ID != 1 {
		t.Errorf("first todo ID = %d, want 1", todo.ID)
	}
	if todo.Text != "Buy milk" {
		t.Errorf("text = %q", todo.Text)
	}
	if todo.Priority != "high" {
		t.Errorf("priority = %q", todo.Priority)
	}
	if todo.Done {
		t.Error("new todo should not be done")
	}

	list, err := store.ListTodos("all")
	if err != nil {
		t.Fatalf("ListTodos: %v", err)
	}
	if !strings.Contains(list, "Buy milk") {
		t.Errorf("list missing todo: %s", list)
	}
	if !strings.Contains(list, "[ ]") {
		t.Errorf("list missing checkbox: %s", list)
	}
	if !strings.Contains(list, "#shopping") {
		t.Errorf("list missing tag: %s", list)
	}
}

func TestTodoComplete(t *testing.T) {
	path := filepath.Join(t.TempDir(), "todos.json")
	store := newTodoStoreAt(path)

	store.AddTodo("Task 1", "medium", nil)
	err := store.CompleteTodo(1)
	if err != nil {
		t.Fatalf("CompleteTodo: %v", err)
	}

	list, _ := store.ListTodos("done")
	if !strings.Contains(list, "[x]") {
		t.Errorf("completed todo missing checkmark: %s", list)
	}

	// Active filter should not show completed.
	active, _ := store.ListTodos("active")
	if strings.Contains(active, "Task 1") {
		t.Errorf("completed todo should not appear in active list")
	}
}

func TestTodoCompleteNotFound(t *testing.T) {
	path := filepath.Join(t.TempDir(), "todos.json")
	store := newTodoStoreAt(path)

	err := store.CompleteTodo(999)
	if err == nil {
		t.Error("expected error completing nonexistent todo")
	}
}

func TestTodoDelete(t *testing.T) {
	path := filepath.Join(t.TempDir(), "todos.json")
	store := newTodoStoreAt(path)

	store.AddTodo("To delete", "low", nil)
	store.AddTodo("To keep", "medium", nil)

	err := store.DeleteTodo(1)
	if err != nil {
		t.Fatalf("DeleteTodo: %v", err)
	}

	if len(store.Todos) != 1 {
		t.Errorf("expected 1 todo, got %d", len(store.Todos))
	}
	if store.Todos[0].Text != "To keep" {
		t.Errorf("wrong todo remaining: %s", store.Todos[0].Text)
	}
}

func TestTodoDeleteNotFound(t *testing.T) {
	path := filepath.Join(t.TempDir(), "todos.json")
	store := newTodoStoreAt(path)

	err := store.DeleteTodo(999)
	if err == nil {
		t.Error("expected error deleting nonexistent todo")
	}
}

func TestTodoListFilters(t *testing.T) {
	path := filepath.Join(t.TempDir(), "todos.json")
	store := newTodoStoreAt(path)

	store.AddTodo("High task", "high", []string{"work"})
	store.AddTodo("Low task", "low", []string{"personal"})
	store.AddTodo("Medium task", "medium", nil)

	// Filter by priority.
	high, _ := store.ListTodos("high")
	if !strings.Contains(high, "High task") || strings.Contains(high, "Low task") {
		t.Errorf("high filter wrong: %s", high)
	}

	// Filter by tag.
	work, _ := store.ListTodos("work")
	if !strings.Contains(work, "High task") || strings.Contains(work, "Low task") {
		t.Errorf("tag filter wrong: %s", work)
	}
}

func TestTodoListEmpty(t *testing.T) {
	path := filepath.Join(t.TempDir(), "todos.json")
	store := newTodoStoreAt(path)

	list, err := store.ListTodos("all")
	if err != nil {
		t.Fatalf("ListTodos: %v", err)
	}
	if list != "No todos." {
		t.Errorf("expected empty message, got: %s", list)
	}
}

func TestParseTodoInput(t *testing.T) {
	tests := []struct {
		input    string
		wantText string
		wantPri  string
		wantTags []string
	}{
		{
			"buy groceries #shopping !high",
			"buy groceries",
			"high",
			[]string{"shopping"},
		},
		{
			"write report #work #urgent !low",
			"write report",
			"low",
			[]string{"work", "urgent"},
		},
		{
			"simple task",
			"simple task",
			"medium",
			nil,
		},
		{
			"#tagged only",
			"only",
			"medium",
			[]string{"tagged"},
		},
	}

	for _, tt := range tests {
		text, pri, tags := ParseTodoInput(tt.input)
		if text != tt.wantText {
			t.Errorf("ParseTodoInput(%q) text = %q, want %q", tt.input, text, tt.wantText)
		}
		if pri != tt.wantPri {
			t.Errorf("ParseTodoInput(%q) priority = %q, want %q", tt.input, pri, tt.wantPri)
		}
		if len(tags) != len(tt.wantTags) {
			t.Errorf("ParseTodoInput(%q) tags = %v, want %v", tt.input, tags, tt.wantTags)
		} else {
			for i, tag := range tags {
				if tag != tt.wantTags[i] {
					t.Errorf("ParseTodoInput(%q) tag[%d] = %q, want %q", tt.input, i, tag, tt.wantTags[i])
				}
			}
		}
	}
}

func TestTodoAutoIncrement(t *testing.T) {
	path := filepath.Join(t.TempDir(), "todos.json")
	store := newTodoStoreAt(path)

	t1 := store.AddTodo("first", "medium", nil)
	t2 := store.AddTodo("second", "medium", nil)
	t3 := store.AddTodo("third", "medium", nil)

	if t1.ID != 1 || t2.ID != 2 || t3.ID != 3 {
		t.Errorf("IDs not auto-incrementing: %d, %d, %d", t1.ID, t2.ID, t3.ID)
	}
}

func TestTodoPersistence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "todos.json")

	// Add todos with first store instance.
	store1 := newTodoStoreAt(path)
	store1.AddTodo("Persistent", "high", []string{"test"})

	// Create new store from same file.
	store2 := newTodoStoreAt(path)
	if len(store2.Todos) != 1 {
		t.Fatalf("persistence: expected 1 todo, got %d", len(store2.Todos))
	}
	if store2.Todos[0].Text != "Persistent" {
		t.Errorf("persistence: text = %q", store2.Todos[0].Text)
	}

	// New todo should get ID 2.
	t2 := store2.AddTodo("Second", "low", nil)
	if t2.ID != 2 {
		t.Errorf("persistence: next ID = %d, want 2", t2.ID)
	}
}

func TestTodoDefaultPriority(t *testing.T) {
	path := filepath.Join(t.TempDir(), "todos.json")
	store := newTodoStoreAt(path)

	todo := store.AddTodo("No priority", "", nil)
	if todo.Priority != "medium" {
		t.Errorf("default priority = %q, want 'medium'", todo.Priority)
	}
}

func TestTodoSaveCreatesFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "todos.json")
	store := newTodoStoreAt(path)
	store.AddTodo("Test", "medium", nil)

	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Error("todos.json file was not created")
	}
}
