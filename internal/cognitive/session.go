package cognitive

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/artaeon/nous/internal/ollama"
	"github.com/artaeon/nous/internal/safefile"
)

// Session represents a persistable conversation session.
type Session struct {
	ID        string            `json:"id"`
	Name      string            `json:"name"`
	Model     string            `json:"model"`
	Messages  []ollama.Message  `json:"messages"`
	CreatedAt time.Time         `json:"created_at"`
	UpdatedAt time.Time         `json:"updated_at"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

// SessionStore manages persistent sessions on disk.
type SessionStore struct {
	dir string
}

func NewSessionStore(baseDir string) *SessionStore {
	dir := filepath.Join(baseDir, "sessions")
	os.MkdirAll(dir, 0755)
	return &SessionStore{dir: dir}
}

// Save persists a session to disk.
func (s *SessionStore) Save(session *Session) error {
	session.UpdatedAt = time.Now()
	data, err := json.MarshalIndent(session, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal session: %w", err)
	}

	path := filepath.Join(s.dir, session.ID+".json")
	return safefile.WriteAtomic(path, data, 0644)
}

// Load retrieves a session by ID.
func (s *SessionStore) Load(id string) (*Session, error) {
	path := filepath.Join(s.dir, id+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("load session %s: %w", id, err)
	}

	var session Session
	if err := json.Unmarshal(data, &session); err != nil {
		return nil, fmt.Errorf("unmarshal session: %w", err)
	}

	return &session, nil
}

// List returns all saved sessions, sorted by most recent.
func (s *SessionStore) List() ([]Session, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, nil // No sessions dir yet
	}

	var sessions []Session
	for _, e := range entries {
		if filepath.Ext(e.Name()) != ".json" {
			continue
		}

		data, err := os.ReadFile(filepath.Join(s.dir, e.Name()))
		if err != nil {
			continue
		}

		var session Session
		if err := json.Unmarshal(data, &session); err != nil {
			continue
		}

		sessions = append(sessions, session)
	}

	sort.Slice(sessions, func(i, j int) bool {
		return sessions[i].UpdatedAt.After(sessions[j].UpdatedAt)
	})

	return sessions, nil
}

// Delete removes a session.
func (s *SessionStore) Delete(id string) error {
	path := filepath.Join(s.dir, id+".json")
	return os.Remove(path)
}

// GenerateID creates a short unique session ID from timestamp.
func GenerateSessionID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano()/1e6)
}
