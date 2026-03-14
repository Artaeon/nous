package hands

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// CustomHandConfig is the on-disk JSON format for user-defined custom hands.
type CustomHandConfig struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Schedule    string     `json:"schedule"`
	Enabled     bool       `json:"enabled"`
	Prompt      string     `json:"prompt"`
	Config      HandConfig `json:"config"`
}

// KnownTools is the set of tool names that custom hands may reference.
// It is populated at startup from the tool registry.
var KnownTools map[string]bool

// SetKnownTools populates the allowed tool set from a list of names.
func SetKnownTools(names []string) {
	KnownTools = make(map[string]bool, len(names))
	for _, n := range names {
		KnownTools[n] = true
	}
}

// ValidateCustomHand checks that a CustomHandConfig is well-formed.
func ValidateCustomHand(cfg CustomHandConfig) error {
	if strings.TrimSpace(cfg.Name) == "" {
		return fmt.Errorf("custom hand: name is required")
	}

	// Validate schedule parses (empty schedule means manual-only)
	if cfg.Schedule != "" {
		if _, err := ParseSchedule(cfg.Schedule); err != nil {
			return fmt.Errorf("custom hand %q: invalid schedule: %w", cfg.Name, err)
		}
	}

	// Validate max_steps: 1-20
	if cfg.Config.MaxSteps < 1 || cfg.Config.MaxSteps > 20 {
		return fmt.Errorf("custom hand %q: max_steps must be between 1 and 20, got %d", cfg.Name, cfg.Config.MaxSteps)
	}

	// Validate timeout: 10-600 seconds
	if cfg.Config.Timeout < 10 || cfg.Config.Timeout > 600 {
		return fmt.Errorf("custom hand %q: timeout must be between 10 and 600 seconds, got %d", cfg.Name, cfg.Config.Timeout)
	}

	// Validate tools exist in known set
	if KnownTools != nil {
		for _, t := range cfg.Config.Tools {
			if !KnownTools[t] {
				return fmt.Errorf("custom hand %q: unknown tool %q", cfg.Name, t)
			}
		}
	}

	return nil
}

// LoadCustomHands scans a directory for .json files, parses each into a Hand,
// validates, and returns the list. Invalid files are skipped with errors collected.
func LoadCustomHands(dir string) ([]*Hand, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // directory doesn't exist yet — not an error
		}
		return nil, fmt.Errorf("reading custom hands directory: %w", err)
	}

	var hands []*Hand
	var errs []string
	seen := make(map[string]bool)

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}

		path := filepath.Join(dir, entry.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", entry.Name(), err))
			continue
		}

		var cfg CustomHandConfig
		if err := json.Unmarshal(data, &cfg); err != nil {
			errs = append(errs, fmt.Sprintf("%s: invalid JSON: %v", entry.Name(), err))
			continue
		}

		if err := ValidateCustomHand(cfg); err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", entry.Name(), err))
			continue
		}

		if seen[cfg.Name] {
			errs = append(errs, fmt.Sprintf("%s: duplicate hand name %q", entry.Name(), cfg.Name))
			continue
		}
		seen[cfg.Name] = true

		hands = append(hands, &Hand{
			Name:        cfg.Name,
			Description: cfg.Description,
			Schedule:    cfg.Schedule,
			Enabled:     cfg.Enabled,
			Config:      cfg.Config,
			Prompt:      cfg.Prompt,
		})
	}

	if len(errs) > 0 {
		return hands, fmt.Errorf("custom hand warnings: %s", strings.Join(errs, "; "))
	}
	return hands, nil
}

// CustomHandDir returns the default directory for custom hand configs.
func CustomHandDir(basePath string) string {
	return filepath.Join(basePath, "hands", "custom")
}

// CustomHandTemplate returns a JSON template for a new custom hand.
func CustomHandTemplate(name string) []byte {
	cfg := CustomHandConfig{
		Name:        name,
		Description: "Describe what this hand does",
		Schedule:    "@daily",
		Enabled:     false,
		Prompt:      "Your detailed instructions for the hand...",
		Config: HandConfig{
			MaxSteps:         8,
			Timeout:          120,
			Tools:            []string{"fetch", "read", "write", "grep"},
			RequiresApproval: false,
		},
	}
	data, _ := json.MarshalIndent(cfg, "", "  ")
	return data
}

// CreateCustomHandFile creates a template JSON file for a new custom hand.
func CreateCustomHandFile(dir, name string) (string, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("creating custom hands directory: %w", err)
	}

	path := filepath.Join(dir, name+".json")
	if _, err := os.Stat(path); err == nil {
		return "", fmt.Errorf("custom hand file already exists: %s", path)
	}

	data := CustomHandTemplate(name)
	if err := os.WriteFile(path, data, 0644); err != nil {
		return "", fmt.Errorf("writing custom hand file: %w", err)
	}
	return path, nil
}
