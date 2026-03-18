package tools

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// DirSize holds a directory path and its total size.
type DirSize struct {
	Path string
	Size int64
}

// FormatDiskSize returns a human-readable size string.
func FormatDiskSize(bytes int64) string {
	switch {
	case bytes >= 1<<30:
		return fmt.Sprintf("%.1fG", float64(bytes)/(1<<30))
	case bytes >= 1<<20:
		return fmt.Sprintf("%.1fM", float64(bytes)/(1<<20))
	case bytes >= 1<<10:
		return fmt.Sprintf("%.1fK", float64(bytes)/(1<<10))
	default:
		return fmt.Sprintf("%dB", bytes)
	}
}

// shouldSkipDir returns true for directories that should be skipped for speed.
func shouldSkipDir(name string) bool {
	skip := map[string]bool{
		".git":         true,
		".cache":       true,
		"node_modules": true,
		".npm":         true,
		".local":       false, // don't skip .local itself
		"__pycache__":  true,
		".venv":        true,
	}
	return skip[name]
}

// GetDiskUsage analyzes disk usage for a directory path.
func GetDiskUsage(rootPath string, topN int, maxDepth int) (string, error) {
	if rootPath == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("diskusage: cannot determine home dir: %w", err)
		}
		rootPath = home
	}

	// Expand ~.
	if strings.HasPrefix(rootPath, "~") {
		home, _ := os.UserHomeDir()
		rootPath = home + rootPath[1:]
	}

	rootPath, err := filepath.Abs(rootPath)
	if err != nil {
		return "", fmt.Errorf("diskusage: invalid path: %w", err)
	}

	info, err := os.Stat(rootPath)
	if err != nil {
		return "", fmt.Errorf("diskusage: path not found: %w", err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("diskusage: %q is not a directory", rootPath)
	}

	if topN <= 0 {
		topN = 10
	}
	if maxDepth <= 0 {
		maxDepth = 3
	}

	sizes := make(map[string]int64)

	filepath.Walk(rootPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // skip inaccessible paths
		}

		// Calculate depth relative to root.
		rel, _ := filepath.Rel(rootPath, path)
		parts := strings.Split(rel, string(filepath.Separator))
		depth := len(parts)

		if info.IsDir() && path != rootPath {
			name := info.Name()
			// Skip hidden/heavy dirs unless they are the root target.
			if shouldSkipDir(name) {
				return filepath.SkipDir
			}
			if depth > maxDepth {
				return filepath.SkipDir
			}
			return nil
		}

		if !info.IsDir() {
			// Attribute size to each ancestor directory up to maxDepth.
			dir := filepath.Dir(path)
			for {
				rel, err := filepath.Rel(rootPath, dir)
				if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
					break
				}

				dirParts := strings.Split(rel, string(filepath.Separator))
				if len(dirParts) <= maxDepth {
					sizes[dir] += info.Size()
				}

				parent := filepath.Dir(dir)
				if parent == dir {
					break
				}
				dir = parent
			}
		}

		return nil
	})

	// Convert to sorted slice.
	var entries []DirSize
	for path, size := range sizes {
		// Only include immediate children of root (depth 1).
		rel, _ := filepath.Rel(rootPath, path)
		parts := strings.Split(rel, string(filepath.Separator))
		if len(parts) == 1 && rel != "." {
			entries = append(entries, DirSize{Path: path, Size: size})
		}
	}

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Size > entries[j].Size
	})

	if len(entries) > topN {
		entries = entries[:topN]
	}

	if len(entries) == 0 {
		return "No directories found.", nil
	}

	return FormatDiskUsage(entries, rootPath), nil
}

// FormatDiskUsage formats the DirSize entries for display.
func FormatDiskUsage(entries []DirSize, rootPath string) string {
	var sb strings.Builder
	home, _ := os.UserHomeDir()

	for _, e := range entries {
		displayPath := e.Path
		if home != "" && strings.HasPrefix(displayPath, home) {
			displayPath = "~" + displayPath[len(home):]
		}
		fmt.Fprintf(&sb, "%8s  %s\n", FormatDiskSize(e.Size), displayPath)
	}
	return strings.TrimRight(sb.String(), "\n")
}

// RegisterDiskUsageTools adds the diskusage tool to the registry.
func RegisterDiskUsageTools(r *Registry) {
	r.Register(Tool{
		Name:        "diskusage",
		Description: "Analyze disk space usage. Args: path (optional, default home dir), top (number of largest items, default 10), depth (max depth, default 3).",
		Execute: func(args map[string]string) (string, error) {
			path := args["path"]
			topN := 10
			maxDepth := 3

			if v, ok := args["top"]; ok {
				if n, err := strconv.Atoi(v); err == nil && n > 0 {
					topN = n
				}
			}
			if v, ok := args["depth"]; ok {
				if n, err := strconv.Atoi(v); err == nil && n > 0 {
					maxDepth = n
				}
			}

			return GetDiskUsage(path, topN, maxDepth)
		},
	})
}
