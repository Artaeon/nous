package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Todo represents a single task item.
type Todo struct {
	ID        int        `json:"id"`
	Text      string     `json:"text"`
	Done      bool       `json:"done"`
	Priority  string     `json:"priority"`
	CreatedAt time.Time  `json:"created_at"`
	DueDate   *time.Time `json:"due_date,omitempty"`
	Tags      []string   `json:"tags,omitempty"`
}

// TodoStore manages a list of todos persisted in a JSON file.
type TodoStore struct {
	filePath string
	Todos    []Todo `json:"todos"`
	nextID   int
}

// NewTodoStore creates a new TodoStore and loads from the default file.
func NewTodoStore() *TodoStore {
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, ".nous")
	os.MkdirAll(dir, 0755)
	return newTodoStoreAt(filepath.Join(dir, "todos.json"))
}

// newTodoStoreAt creates a TodoStore at a specific path (for testing).
func newTodoStoreAt(path string) *TodoStore {
	ts := &TodoStore{filePath: path}
	ts.load()
	return ts
}

func (ts *TodoStore) load() {
	data, err := os.ReadFile(ts.filePath)
	if err != nil {
		ts.Todos = nil
		ts.nextID = 1
		return
	}
	if err := json.Unmarshal(data, &ts.Todos); err != nil {
		ts.Todos = nil
		ts.nextID = 1
		return
	}
	maxID := 0
	for _, t := range ts.Todos {
		if t.ID > maxID {
			maxID = t.ID
		}
	}
	ts.nextID = maxID + 1
}

func (ts *TodoStore) save() error {
	data, err := json.MarshalIndent(ts.Todos, "", "  ")
	if err != nil {
		return fmt.Errorf("save todos: %w", err)
	}
	return os.WriteFile(ts.filePath, data, 0644)
}

// AddTodo adds a new todo item and saves.
func (ts *TodoStore) AddTodo(text string, priority string, tags []string) *Todo {
	if priority == "" {
		priority = "medium"
	}
	todo := Todo{
		ID:        ts.nextID,
		Text:      text,
		Done:      false,
		Priority:  priority,
		CreatedAt: time.Now(),
		Tags:      tags,
	}
	ts.nextID++
	ts.Todos = append(ts.Todos, todo)
	ts.save()
	return &todo
}

// CompleteTodo marks a todo as done.
func (ts *TodoStore) CompleteTodo(id int) error {
	for i := range ts.Todos {
		if ts.Todos[i].ID == id {
			ts.Todos[i].Done = true
			return ts.save()
		}
	}
	return fmt.Errorf("todo #%d not found", id)
}

// DeleteTodo removes a todo by ID.
func (ts *TodoStore) DeleteTodo(id int) error {
	for i := range ts.Todos {
		if ts.Todos[i].ID == id {
			ts.Todos = append(ts.Todos[:i], ts.Todos[i+1:]...)
			return ts.save()
		}
	}
	return fmt.Errorf("todo #%d not found", id)
}

// ListTodos returns a formatted list of todos.
// Filter can be: "all", "active", "done", "high", "medium", "low", or a tag name.
func (ts *TodoStore) ListTodos(filter string) (string, error) {
	if len(ts.Todos) == 0 {
		return "No todos.", nil
	}

	var sb strings.Builder
	count := 0

	for _, t := range ts.Todos {
		if !matchesFilter(t, filter) {
			continue
		}
		count++

		check := "[ ]"
		if t.Done {
			check = "[x]"
		}

		pri := ""
		if t.Priority == "high" {
			pri = " !high"
		} else if t.Priority == "low" {
			pri = " !low"
		}

		tags := ""
		if len(t.Tags) > 0 {
			tags = " #" + strings.Join(t.Tags, " #")
		}

		fmt.Fprintf(&sb, "%s #%d %s%s%s\n", check, t.ID, t.Text, pri, tags)
	}

	if count == 0 {
		return fmt.Sprintf("No todos matching filter %q.", filter), nil
	}

	return fmt.Sprintf("%d todo(s):\n%s", count, sb.String()), nil
}

func matchesFilter(t Todo, filter string) bool {
	switch strings.ToLower(filter) {
	case "", "all":
		return true
	case "active":
		return !t.Done
	case "done":
		return t.Done
	case "high", "medium", "low":
		return t.Priority == strings.ToLower(filter)
	default:
		// Match by tag name.
		for _, tag := range t.Tags {
			if strings.EqualFold(tag, filter) {
				return true
			}
		}
		return false
	}
}

// ParseTodoInput parses input like "buy groceries #shopping !high"
// into text, priority, and tags.
func ParseTodoInput(input string) (text string, priority string, tags []string) {
	priority = "medium"
	words := strings.Fields(input)
	var textParts []string

	for _, w := range words {
		if strings.HasPrefix(w, "#") && len(w) > 1 {
			tags = append(tags, w[1:])
		} else if strings.HasPrefix(w, "!") && len(w) > 1 {
			priority = strings.ToLower(w[1:])
		} else {
			textParts = append(textParts, w)
		}
	}

	text = strings.Join(textParts, " ")
	return
}

// RegisterTodoTools adds the todos tool to the registry.
func RegisterTodoTools(r *Registry) {
	store := NewTodoStore()
	r.Register(Tool{
		Name:        "todos",
		Description: "Manage a todo list. Args: action (add/complete/delete/list), text, id, filter, priority, tags.",
		Execute: func(args map[string]string) (string, error) {
			return toolTodos(store, args)
		},
	})
}

func toolTodos(store *TodoStore, args map[string]string) (string, error) {
	action := args["action"]
	switch action {
	case "add":
		input := args["text"]
		if input == "" {
			return "", fmt.Errorf("todos add requires 'text'")
		}
		text, priority, tags := ParseTodoInput(input)

		// Allow explicit priority/tags overrides from args.
		if p, ok := args["priority"]; ok && p != "" {
			priority = p
		}
		if t, ok := args["tags"]; ok && t != "" {
			tags = append(tags, strings.Split(t, ",")...)
		}

		todo := store.AddTodo(text, priority, tags)
		return fmt.Sprintf("added todo #%d: %s", todo.ID, todo.Text), nil

	case "complete":
		idStr := args["id"]
		if idStr == "" {
			return "", fmt.Errorf("todos complete requires 'id'")
		}
		id, err := strconv.Atoi(idStr)
		if err != nil {
			return "", fmt.Errorf("invalid todo id: %s", idStr)
		}
		if err := store.CompleteTodo(id); err != nil {
			return "", err
		}
		return fmt.Sprintf("completed todo #%d", id), nil

	case "delete":
		idStr := args["id"]
		if idStr == "" {
			return "", fmt.Errorf("todos delete requires 'id'")
		}
		id, err := strconv.Atoi(idStr)
		if err != nil {
			return "", fmt.Errorf("invalid todo id: %s", idStr)
		}
		if err := store.DeleteTodo(id); err != nil {
			return "", err
		}
		return fmt.Sprintf("deleted todo #%d", id), nil

	case "list":
		filter := args["filter"]
		return store.ListTodos(filter)

	default:
		return "", fmt.Errorf("todos: unknown action %q (use add/complete/delete/list)", action)
	}
}
