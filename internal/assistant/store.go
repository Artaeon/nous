package assistant

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

const (
	TaskPending = "pending"
	TaskDone    = "done"
)

type Task struct {
	ID             string    `json:"id"`
	Title          string    `json:"title"`
	DueAt          time.Time `json:"due_at"`
	Recurrence     string    `json:"recurrence,omitempty"`
	Status         string    `json:"status"`
	CreatedAt      time.Time `json:"created_at"`
	CompletedAt    time.Time `json:"completed_at,omitempty"`
	LastNotifiedAt time.Time `json:"last_notified_at,omitempty"`
}

type Notification struct {
	ID        string    `json:"id"`
	TaskID    string    `json:"task_id"`
	Message   string    `json:"message"`
	CreatedAt time.Time `json:"created_at"`
	Read      bool      `json:"read"`
}

type Preference struct {
	Key       string    `json:"key"`
	Value     string    `json:"value"`
	UpdatedAt time.Time `json:"updated_at"`
}

type State struct {
	Tasks         []Task                  `json:"tasks"`
	Notifications []Notification          `json:"notifications"`
	Preferences   map[string]Preference   `json:"preferences"`
}

type Store struct {
	mu    sync.RWMutex
	path  string
	state State
}

func NewStore(baseDir string) *Store {
	store := &Store{
		path: filepath.Join(baseDir, "assistant.json"),
		state: State{
			Preferences: make(map[string]Preference),
		},
	}
	store.load()
	return store
}

func (s *Store) AddTask(title string, dueAt time.Time, recurrence string) (Task, error) {
	if title == "" {
		return Task{}, fmt.Errorf("task title cannot be empty")
	}
	if dueAt.IsZero() {
		return Task{}, fmt.Errorf("due time cannot be empty")
	}

	now := time.Now()
	task := Task{
		ID:         fmt.Sprintf("tsk-%d", now.UnixNano()),
		Title:      title,
		DueAt:      dueAt,
		Recurrence: recurrence,
		Status:     TaskPending,
		CreatedAt:  now,
	}

	s.mu.Lock()
	s.state.Tasks = append(s.state.Tasks, task)
	s.mu.Unlock()
	return task, s.Save()
}

func (s *Store) Tasks() []Task {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Task, len(s.state.Tasks))
	copy(out, s.state.Tasks)
	sort.Slice(out, func(i, j int) bool {
		return out[i].DueAt.Before(out[j].DueAt)
	})
	return out
}

func (s *Store) PendingTasks() []Task {
	all := s.Tasks()
	out := all[:0]
	for _, task := range all {
		if task.Status == TaskPending {
			out = append(out, task)
		}
	}
	return out
}

func (s *Store) MarkDone(id string) (Task, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.state.Tasks {
		if s.state.Tasks[i].ID == id {
			s.state.Tasks[i].Status = TaskDone
			s.state.Tasks[i].CompletedAt = time.Now()
			task := s.state.Tasks[i]
			return task, s.saveLocked()
		}
	}
	return Task{}, fmt.Errorf("task not found: %s", id)
}

func (s *Store) Upcoming(limit int, from time.Time) []Task {
	pending := s.PendingTasks()
	out := make([]Task, 0, len(pending))
	for _, task := range pending {
		if task.DueAt.After(from) || task.DueAt.Equal(from) {
			out = append(out, task)
		}
	}
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

func (s *Store) Today(now time.Time) []Task {
	pending := s.PendingTasks()
	out := make([]Task, 0)
	y, m, d := now.Date()
	for _, task := range pending {
		ty, tm, td := task.DueAt.Date()
		if y == ty && m == tm && d == td {
			out = append(out, task)
		}
	}
	return out
}

func (s *Store) SetPreference(key, value string) error {
	if key == "" {
		return fmt.Errorf("preference key cannot be empty")
	}
	s.mu.Lock()
	s.state.Preferences[key] = Preference{Key: key, Value: value, UpdatedAt: time.Now()}
	s.mu.Unlock()
	return s.Save()
}

func (s *Store) Preferences() []Preference {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Preference, 0, len(s.state.Preferences))
	for _, pref := range s.state.Preferences {
		out = append(out, pref)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Key < out[j].Key })
	return out
}

func (s *Store) UnreadNotifications() []Notification {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Notification, 0)
	for i := len(s.state.Notifications) - 1; i >= 0; i-- {
		note := s.state.Notifications[i]
		if !note.Read {
			out = append(out, note)
		}
	}
	return out
}

func (s *Store) MarkNotificationsRead() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.state.Notifications {
		s.state.Notifications[i].Read = true
	}
	return s.saveLocked()
}

func (s *Store) TriggerDue(now time.Time) ([]Notification, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var notifications []Notification
	for i := range s.state.Tasks {
		task := &s.state.Tasks[i]
		if task.Status != TaskPending || task.DueAt.After(now) {
			continue
		}
		if !task.LastNotifiedAt.IsZero() && (task.LastNotifiedAt.After(task.DueAt) || task.LastNotifiedAt.Equal(task.DueAt)) {
			continue
		}

		note := Notification{
			ID:        fmt.Sprintf("ntf-%d", now.UnixNano()+int64(i)),
			TaskID:    task.ID,
			Message:   fmt.Sprintf("Reminder: %s", task.Title),
			CreatedAt: now,
		}
		s.state.Notifications = append(s.state.Notifications, note)
		notifications = append(notifications, note)
		task.LastNotifiedAt = now

		switch task.Recurrence {
		case "daily":
			task.DueAt = task.DueAt.Add(24 * time.Hour)
			task.LastNotifiedAt = time.Time{}
		case "weekly":
			task.DueAt = task.DueAt.Add(7 * 24 * time.Hour)
			task.LastNotifiedAt = time.Time{}
		}
	}

	if len(notifications) == 0 {
		return nil, nil
	}
	return notifications, s.saveLocked()
}

func (s *Store) Save() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.saveLocked()
}

func (s *Store) saveLocked() error {
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(s.state, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.path, data, 0o644)
}

func (s *Store) load() {
	data, err := os.ReadFile(s.path)
	if err != nil {
		return
	}
	_ = json.Unmarshal(data, &s.state)
	if s.state.Preferences == nil {
		s.state.Preferences = make(map[string]Preference)
	}
}
