package safefile

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// MaxBackups is the default number of backups to keep per file.
const MaxBackups = 5

// WriteAtomicWithBackup is like WriteAtomic but creates a timestamped backup
// of the existing file before overwriting. Keeps at most maxBackups old copies.
func WriteAtomicWithBackup(path string, data []byte, perm os.FileMode, maxBackups int) error {
	if maxBackups <= 0 {
		maxBackups = MaxBackups
	}

	// Create backup of existing file (if it exists and has content).
	if info, err := os.Stat(path); err == nil && info.Size() > 0 {
		if err := createBackup(path, maxBackups); err != nil {
			// Log but don't fail — the write itself is more important.
			fmt.Fprintf(os.Stderr, "safefile: backup warning: %v\n", err)
		}
	}

	return WriteAtomic(path, data, perm)
}

// createBackup copies the current file to a timestamped backup and prunes old ones.
func createBackup(path string, maxBackups int) error {
	dir := filepath.Dir(path)
	base := filepath.Base(path)
	ext := filepath.Ext(base)
	name := strings.TrimSuffix(base, ext)

	backupDir := filepath.Join(dir, "backups")
	if err := os.MkdirAll(backupDir, 0755); err != nil {
		return fmt.Errorf("create backup dir: %w", err)
	}

	// Read current file.
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read for backup: %w", err)
	}

	// Write backup with timestamp.
	ts := time.Now().Format("20060102-150405")
	backupPath := filepath.Join(backupDir, fmt.Sprintf("%s-%s%s", name, ts, ext))
	if err := os.WriteFile(backupPath, data, 0644); err != nil {
		return fmt.Errorf("write backup: %w", err)
	}

	// Prune old backups — keep only the newest maxBackups.
	pruneBackups(backupDir, name, ext, maxBackups)
	return nil
}

// pruneBackups removes old backup files, keeping only the newest maxKeep.
func pruneBackups(dir, namePrefix, ext string, maxKeep int) {
	pattern := filepath.Join(dir, namePrefix+"-*"+ext)
	matches, err := filepath.Glob(pattern)
	if err != nil || len(matches) <= maxKeep {
		return
	}

	// Sort ascending by name (timestamps sort lexicographically).
	sort.Strings(matches)

	// Remove the oldest ones.
	for _, old := range matches[:len(matches)-maxKeep] {
		os.Remove(old)
	}
}
