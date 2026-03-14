package cognitive

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/artaeon/nous/internal/ollama"
)

// ConversationMeta holds summary information about a saved conversation.
type ConversationMeta struct {
	ID           string    `json:"id"`
	Title        string    `json:"title"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
	MessageCount int       `json:"message_count"`
	Summary      string    `json:"summary"`
}

// savedConversation is the on-disk JSON format for a conversation.
type savedConversation struct {
	Meta        ConversationMeta `json:"meta"`
	Messages    []ollama.Message `json:"messages"`
	MaxMessages int              `json:"max_messages"`
}

// DefaultConversationDir returns ~/.nous/conversations/.
func DefaultConversationDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("cannot determine home directory: %w", err)
	}
	return filepath.Join(home, ".nous", "conversations"), nil
}

// SaveConversation serializes a conversation to a JSON file.
func SaveConversation(fpath string, conv *Conversation) error {
	if conv == nil {
		return fmt.Errorf("conversation is nil")
	}

	msgs := conv.Messages()
	now := time.Now()

	// Try to load existing meta or create new one
	var meta ConversationMeta
	if data, err := os.ReadFile(fpath); err == nil {
		var existing savedConversation
		if json.Unmarshal(data, &existing) == nil {
			meta = existing.Meta
		}
	}

	if meta.ID == "" {
		meta.ID = generateID(now)
		meta.CreatedAt = now
	}
	meta.UpdatedAt = now
	meta.MessageCount = len(msgs)
	meta.Summary = conv.Summary()

	if meta.Title == "" {
		meta.Title = titleFromMessages(msgs)
	}

	saved := savedConversation{
		Meta:        meta,
		Messages:    msgs,
		MaxMessages: conv.maxMessages,
	}

	data, err := json.MarshalIndent(saved, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal conversation: %w", err)
	}

	if err := os.MkdirAll(filepath.Dir(fpath), 0o700); err != nil {
		return fmt.Errorf("create directory: %w", err)
	}

	return os.WriteFile(fpath, data, 0o600)
}

// LoadConversation deserializes a conversation from a JSON file.
func LoadConversation(fpath string) (*Conversation, error) {
	data, err := os.ReadFile(fpath)
	if err != nil {
		return nil, fmt.Errorf("read conversation file: %w", err)
	}

	var saved savedConversation
	if err := json.Unmarshal(data, &saved); err != nil {
		return nil, fmt.Errorf("unmarshal conversation: %w", err)
	}

	maxMsgs := saved.MaxMessages
	if maxMsgs <= 0 {
		maxMsgs = 20 // sensible default
	}

	conv := NewConversation(maxMsgs)
	conv.SetMessages(saved.Messages)
	return conv, nil
}

// ListConversations returns metadata for all conversations in a directory.
func ListConversations(dir string) ([]ConversationMeta, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read conversation directory: %w", err)
	}

	var metas []ConversationMeta
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}

		data, err := os.ReadFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			continue
		}

		var saved savedConversation
		if json.Unmarshal(data, &saved) != nil {
			continue
		}

		metas = append(metas, saved.Meta)
	}

	// Sort by updated time, most recent first
	sort.Slice(metas, func(i, j int) bool {
		return metas[i].UpdatedAt.After(metas[j].UpdatedAt)
	})

	return metas, nil
}

// DeleteConversation removes a saved conversation file.
func DeleteConversation(fpath string) error {
	if err := os.Remove(fpath); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("conversation not found")
		}
		return fmt.Errorf("delete conversation: %w", err)
	}
	return nil
}

// SetMessages replaces the internal message list. Used for loading saved conversations.
func (c *Conversation) SetMessages(msgs []ollama.Message) {
	c.messages = make([]ollama.Message, len(msgs))
	copy(c.messages, msgs)
}

// titleFromMessages generates a title from the first user message (up to 50 chars).
func titleFromMessages(msgs []ollama.Message) string {
	for _, m := range msgs {
		if m.Role == "user" && m.Content != "" {
			title := strings.TrimSpace(m.Content)
			// Remove any OBSERVE or context prefixes
			if strings.HasPrefix(title, "[Previous conversation summary]") {
				continue
			}
			if strings.HasPrefix(title, "[Earlier context]") {
				continue
			}
			if len(title) > 50 {
				title = title[:50]
			}
			return title
		}
	}
	return "Untitled conversation"
}

// generateID creates a unique conversation ID from the current time.
func generateID(t time.Time) string {
	return fmt.Sprintf("conv_%d", t.UnixNano())
}

// ConversationFilePath returns the full file path for a conversation ID in a directory.
func ConversationFilePath(dir, id string) string {
	return filepath.Join(dir, id+".json")
}
