package hands

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/artaeon/nous/internal/safefile"
)

// HandPackage is a portable, shareable representation of a hand configuration.
type HandPackage struct {
	Version     string    `json:"version"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Author      string    `json:"author"`
	Hand        Hand      `json:"hand"`
	Examples    []string  `json:"examples,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
}

// ExportHand wraps a hand in a shareable package.
func ExportHand(hand *Hand, author string) (*HandPackage, error) {
	if hand == nil {
		return nil, fmt.Errorf("hand is nil")
	}
	return &HandPackage{
		Version:     "1",
		Name:        hand.Name,
		Description: hand.Description,
		Author:      author,
		Hand: Hand{
			Name:        hand.Name,
			Description: hand.Description,
			Schedule:    hand.Schedule,
			Enabled:     hand.Enabled,
			Config:      hand.Config,
			Prompt:      hand.Prompt,
		},
		CreatedAt: time.Now(),
	}, nil
}

// ImportHand deserializes a hand package from JSON bytes.
func ImportHand(data []byte) (*Hand, error) {
	var pkg HandPackage
	if err := json.Unmarshal(data, &pkg); err != nil {
		return nil, fmt.Errorf("invalid hand package: %w", err)
	}
	if pkg.Hand.Name == "" {
		return nil, fmt.Errorf("hand package has no name")
	}
	// Reset runtime state
	pkg.Hand.State = HandIdle
	pkg.Hand.LastRun = time.Time{}
	pkg.Hand.LastError = ""
	return &pkg.Hand, nil
}

// ExportToFile writes a hand package to a JSON file.
func ExportToFile(pkg *HandPackage, path string) error {
	data, err := json.MarshalIndent(pkg, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal hand package: %w", err)
	}
	return safefile.WriteAtomic(path, data, 0644)
}

// ImportFromFile reads a hand package from a JSON file.
func ImportFromFile(path string) (*Hand, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read hand package: %w", err)
	}
	return ImportHand(data)
}
