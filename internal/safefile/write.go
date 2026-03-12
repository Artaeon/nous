package safefile

import (
	"fmt"
	"os"
	"path/filepath"
)

// WriteAtomic writes data to path atomically: write to a temp file in the
// same directory, fsync, then rename over the target.  This guarantees that
// readers always see either the old content or the new content — never a
// partial write.
func WriteAtomic(path string, data []byte, perm os.FileMode) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("safefile: mkdir %s: %w", dir, err)
	}

	tmp, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return fmt.Errorf("safefile: create temp: %w", err)
	}
	tmpPath := tmp.Name()

	// Clean up on any error path.
	success := false
	defer func() {
		if !success {
			tmp.Close()
			os.Remove(tmpPath)
		}
	}()

	if _, err := tmp.Write(data); err != nil {
		return fmt.Errorf("safefile: write temp: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("safefile: fsync: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("safefile: close temp: %w", err)
	}
	if err := os.Chmod(tmpPath, perm); err != nil {
		return fmt.Errorf("safefile: chmod: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("safefile: rename: %w", err)
	}

	success = true
	return nil
}

// WriteAtomicFunc creates a temp file, calls fn to write content into it,
// then fsyncs and renames atomically.  Use this for streaming writes (JSONL,
// encoders, etc.) where you don't want to buffer the full payload in memory.
func WriteAtomicFunc(path string, perm os.FileMode, fn func(f *os.File) error) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("safefile: mkdir %s: %w", dir, err)
	}

	tmp, err := os.CreateTemp(dir, ".tmp-*")
	if err != nil {
		return fmt.Errorf("safefile: create temp: %w", err)
	}
	tmpPath := tmp.Name()

	success := false
	defer func() {
		if !success {
			tmp.Close()
			os.Remove(tmpPath)
		}
	}()

	if err := fn(tmp); err != nil {
		return err
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("safefile: fsync: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("safefile: close temp: %w", err)
	}
	if err := os.Chmod(tmpPath, perm); err != nil {
		return fmt.Errorf("safefile: chmod: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("safefile: rename: %w", err)
	}

	success = true
	return nil
}
