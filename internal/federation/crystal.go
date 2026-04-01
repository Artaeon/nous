package federation

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// SharedCrystal is a response pattern that can be shared between Nous instances.
// It contains NO personal data — only the abstract pattern and response template.
type SharedCrystal struct {
	ID        string    `json:"id"`         // SHA-256 of pattern
	Pattern   string    `json:"pattern"`    // normalized query pattern (e.g., "what is {topic}")
	Response  string    `json:"response"`   // response template with {slot} placeholders
	Intent    string    `json:"intent"`     // NLU intent category
	Quality   float64   `json:"quality"`    // aggregated quality (0.0-1.0)
	Votes     int       `json:"votes"`      // number of instances that validated this
	Source    string    `json:"source"`     // "compiled" or "manual"
	Tags      []string  `json:"tags"`       // categorization tags
	Created   time.Time `json:"created"`
	LastVoted time.Time `json:"last_voted"`
}

// CrystalBundle is a collection of shared crystals for export/import.
type CrystalBundle struct {
	Version  int             `json:"version"`  // format version (1)
	Instance string          `json:"instance"` // anonymous instance ID (random UUID, not identifying)
	Exported time.Time       `json:"exported"`
	Crystals []SharedCrystal `json:"crystals"`
	Checksum string          `json:"checksum"` // SHA-256 of sorted crystal IDs
}

// NewSharedCrystal creates a SharedCrystal with a deterministic SHA-256 ID
// derived from the pattern string.
func NewSharedCrystal(pattern, response, intent string, quality float64) SharedCrystal {
	h := sha256.Sum256([]byte(pattern))
	id := hex.EncodeToString(h[:])

	now := time.Now()
	return SharedCrystal{
		ID:        id,
		Pattern:   pattern,
		Response:  response,
		Intent:    intent,
		Quality:   quality,
		Votes:     1,
		Source:    "compiled",
		Tags:      []string{},
		Created:   now,
		LastVoted: now,
	}
}

// computeChecksum calculates the SHA-256 checksum over sorted crystal IDs.
func computeChecksum(crystals []SharedCrystal) string {
	ids := make([]string, len(crystals))
	for i, c := range crystals {
		ids[i] = c.ID
	}
	sort.Strings(ids)

	h := sha256.New()
	for _, id := range ids {
		h.Write([]byte(id))
	}
	return hex.EncodeToString(h.Sum(nil))
}

// NewCrystalBundle creates a bundle from the given crystals, computing the checksum
// over sorted crystal IDs.
func NewCrystalBundle(instanceID string, crystals []SharedCrystal) *CrystalBundle {
	return &CrystalBundle{
		Version:  1,
		Instance: instanceID,
		Exported: time.Now(),
		Crystals: crystals,
		Checksum: computeChecksum(crystals),
	}
}

// Export writes the bundle to a JSON file at the given path.
// Parent directories are created as needed.
func (b *CrystalBundle) Export(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create directory: %w", err)
	}

	data, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal bundle: %w", err)
	}

	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write bundle: %w", err)
	}
	return nil
}

// ImportBundle reads a CrystalBundle from a JSON file.
func ImportBundle(path string) (*CrystalBundle, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read bundle: %w", err)
	}

	var bundle CrystalBundle
	if err := json.Unmarshal(data, &bundle); err != nil {
		return nil, fmt.Errorf("unmarshal bundle: %w", err)
	}
	return &bundle, nil
}

// Validate checks the bundle for structural integrity:
//   - version must be 1
//   - checksum must match the sorted crystal IDs
//   - no crystal may have an empty pattern
func (b *CrystalBundle) Validate() error {
	if b.Version != 1 {
		return fmt.Errorf("unsupported bundle version: %d", b.Version)
	}

	for i, c := range b.Crystals {
		if c.Pattern == "" {
			return fmt.Errorf("crystal %d has empty pattern", i)
		}
	}

	expected := computeChecksum(b.Crystals)
	if b.Checksum != expected {
		return fmt.Errorf("checksum mismatch: got %s, want %s", b.Checksum, expected)
	}

	return nil
}
