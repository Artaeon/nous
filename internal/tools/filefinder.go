package tools

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// FileEntry holds metadata for a found file.
type FileEntry struct {
	Path    string
	Size    int64
	ModTime time.Time
}

// FindFiles searches the filesystem based on query parameters.
func FindFiles(query string, searchPath string, maxResults int) (string, error) {
	if maxResults <= 0 {
		maxResults = 20
	}

	name, ext, dir, sortBy := ParseFileQuery(query)

	if searchPath != "" {
		dir = searchPath
	}
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = home
	}

	// Expand ~ in dir.
	if strings.HasPrefix(dir, "~") {
		home, _ := os.UserHomeDir()
		dir = home + dir[1:]
	}

	// Verify dir exists.
	info, err := os.Stat(dir)
	if err != nil || !info.IsDir() {
		return "", fmt.Errorf("directory %q not found", dir)
	}

	var entries []FileEntry

	switch {
	case sortBy == "size":
		// Large files mode.
		filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err != nil || info.IsDir() {
				return skipHidden(info, err)
			}
			if info.Size() > 100*1024*1024 { // > 100MB
				entries = append(entries, FileEntry{Path: path, Size: info.Size(), ModTime: info.ModTime()})
			}
			return nil
		})
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].Size > entries[j].Size
		})

	case sortBy == "mtime":
		// Recent files mode.
		filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err != nil || info.IsDir() {
				return skipHidden(info, err)
			}
			if ext != "" && !strings.HasSuffix(strings.ToLower(info.Name()), "."+ext) {
				return nil
			}
			entries = append(entries, FileEntry{Path: path, Size: info.Size(), ModTime: info.ModTime()})
			return nil
		})
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].ModTime.After(entries[j].ModTime)
		})

	default:
		// Name / extension search.
		filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err != nil || info.IsDir() {
				return skipHidden(info, err)
			}
			match := false
			lowerName := strings.ToLower(info.Name())
			if ext != "" {
				match = strings.HasSuffix(lowerName, "."+ext)
			}
			if name != "" {
				matched, _ := filepath.Match(strings.ToLower(name), lowerName)
				if matched || strings.Contains(lowerName, strings.ToLower(name)) {
					match = true
				}
			}
			if name == "" && ext == "" {
				// Fallback: content search.
				match = false
			}
			if match {
				entries = append(entries, FileEntry{Path: path, Size: info.Size(), ModTime: info.ModTime()})
			}
			return nil
		})
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].ModTime.After(entries[j].ModTime)
		})
	}

	if len(entries) > maxResults {
		entries = entries[:maxResults]
	}

	if len(entries) == 0 {
		return "No files found.", nil
	}

	return formatFileEntries(entries), nil
}

// FindByContent searches text files for a keyword.
func FindByContent(keyword string, searchPath string, maxResults int) (string, error) {
	if keyword == "" {
		return "", fmt.Errorf("search keyword is required")
	}
	if maxResults <= 0 {
		maxResults = 20
	}
	if searchPath == "" {
		home, _ := os.UserHomeDir()
		searchPath = home
	}

	lowerKey := strings.ToLower(keyword)
	var results []string

	filepath.Walk(searchPath, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return skipHidden(info, err)
		}
		if len(results) >= maxResults {
			return filepath.SkipAll
		}
		// Skip binary/large files.
		if info.Size() > 1<<20 {
			return nil
		}

		f, err := os.Open(path)
		if err != nil {
			return nil
		}
		defer f.Close()

		scanner := bufio.NewScanner(f)
		lineNum := 0
		for scanner.Scan() {
			lineNum++
			if strings.Contains(strings.ToLower(scanner.Text()), lowerKey) {
				results = append(results, fmt.Sprintf("%s:%d: %s", path, lineNum, strings.TrimSpace(scanner.Text())))
				if len(results) >= maxResults {
					return filepath.SkipAll
				}
				break // one match per file
			}
		}
		return nil
	})

	if len(results) == 0 {
		return fmt.Sprintf("No files containing %q found.", keyword), nil
	}

	return fmt.Sprintf("%d match(es):\n%s", len(results), strings.Join(results, "\n")), nil
}

func skipHidden(info os.FileInfo, err error) error {
	if err != nil {
		return nil
	}
	if info != nil && info.IsDir() {
		name := info.Name()
		if strings.HasPrefix(name, ".") || name == "node_modules" || name == "vendor" {
			return filepath.SkipDir
		}
	}
	return nil
}

// ParseFileQuery extracts search parameters from a natural language query.
func ParseFileQuery(input string) (name string, ext string, dir string, sortBy string) {
	lower := strings.ToLower(input)

	// Detect sort mode.
	if strings.Contains(lower, "recent") || strings.Contains(lower, "latest") || strings.Contains(lower, "newest") {
		sortBy = "mtime"
	}
	if strings.Contains(lower, "large") || strings.Contains(lower, "big") || strings.Contains(lower, "biggest") {
		sortBy = "size"
	}

	// Detect directory aliases (use lower for aliases, original input for explicit paths).
	dir = extractDirectory(lower)
	if dir == "" {
		dir = extractExplicitPath(input)
	}

	// Detect extension.
	ext = extractExtension(lower)

	// Extract name pattern: remove known keywords and see what's left.
	cleaned := lower
	for _, kw := range []string{"find", "search", "show", "list", "recent", "latest", "newest", "large", "big", "biggest", "files", "file", "in", "on", "from"} {
		cleaned = strings.ReplaceAll(cleaned, kw, "")
	}
	// Remove directory aliases from cleaned string.
	for _, alias := range []string{"downloads", "download", "desktop", "documents", "home"} {
		cleaned = strings.ReplaceAll(cleaned, alias, "")
	}
	// Remove extension keywords.
	for _, ek := range []string{"pdfs", "pdf", "images", "photos", "videos", "docs", "txt", "go", "py", "js", "json", "csv", "xml", "html", "md", "zip", "tar"} {
		cleaned = strings.ReplaceAll(cleaned, ek, "")
	}
	cleaned = strings.TrimSpace(cleaned)
	if cleaned != "" && ext == "" {
		name = cleaned
	}

	return
}

// extractDirectory maps directory aliases to paths.
func extractDirectory(input string) string {
	home, _ := os.UserHomeDir()
	aliases := map[string]string{
		"downloads": filepath.Join(home, "Downloads"),
		"download":  filepath.Join(home, "Downloads"),
		"desktop":   filepath.Join(home, "Desktop"),
		"documents": filepath.Join(home, "Documents"),
		"home":      home,
	}

	// Check for "in <dir>" or "on <dir>" patterns, or bare alias.
	for alias, path := range aliases {
		if strings.Contains(input, alias) {
			return path
		}
	}

	return ""
}

// extractExplicitPath finds explicit filesystem paths in the input (case-sensitive).
func extractExplicitPath(input string) string {
	home, _ := os.UserHomeDir()
	words := strings.Fields(input)
	for _, w := range words {
		if strings.HasPrefix(w, "~/") || strings.HasPrefix(w, "/") || strings.HasPrefix(w, "./") {
			if strings.HasPrefix(w, "~/") {
				return filepath.Join(home, w[2:])
			}
			return w
		}
	}
	return ""
}

// extractExtension detects file extension from query keywords.
func extractExtension(input string) string {
	extMap := map[string]string{
		"pdfs":   "pdf",
		"pdf":    "pdf",
		"images": "png",
		"photos": "jpg",
		"videos": "mp4",
		"docs":   "docx",
		"go":     "go",
		"python": "py",
		"py":     "py",
		"js":     "js",
		"json":   "json",
		"csv":    "csv",
		"xml":    "xml",
		"html":   "html",
		"md":     "md",
		"zip":    "zip",
		"tar":    "tar",
		"txt":    "txt",
	}

	words := strings.Fields(input)
	for _, w := range words {
		if ext, ok := extMap[w]; ok {
			return ext
		}
		// Handle "*.pdf" or ".pdf" style.
		if strings.HasPrefix(w, "*.") {
			return w[2:]
		}
		if strings.HasPrefix(w, ".") && len(w) > 1 {
			return w[1:]
		}
	}
	return ""
}

func formatFileEntries(entries []FileEntry) string {
	var sb strings.Builder
	for _, e := range entries {
		sizeStr := formatSize(e.Size)
		dateStr := e.ModTime.Format("2006-01-02 15:04")
		fmt.Fprintf(&sb, "%8s  %s  %s\n", sizeStr, dateStr, e.Path)
	}
	return strings.TrimRight(sb.String(), "\n")
}

func formatSize(bytes int64) string {
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

// RegisterFileFinderTools adds the filefinder tool to the registry.
func RegisterFileFinderTools(r *Registry) {
	r.Register(Tool{
		Name:        "filefinder",
		Description: "Search local filesystem for files. Args: query (required, e.g. 'find PDFs in downloads', 'recent files', 'large files'), path (optional), max_results (optional).",
		Execute: func(args map[string]string) (string, error) {
			return toolFileFinder(args)
		},
	})
}

func toolFileFinder(args map[string]string) (string, error) {
	query := args["query"]
	if query == "" {
		return "", fmt.Errorf("filefinder requires 'query' argument")
	}

	searchPath := args["path"]
	maxResults := 20
	if v, ok := args["max_results"]; ok {
		if n, err := fmt.Sscanf(v, "%d", &maxResults); err != nil || n != 1 {
			maxResults = 20
		}
	}

	// Check if it's a content search.
	lower := strings.ToLower(query)
	if strings.Contains(lower, "containing") || strings.Contains(lower, "content") || strings.Contains(lower, "grep") {
		// Extract keyword after "containing".
		keyword := query
		if idx := strings.Index(lower, "containing"); idx >= 0 {
			keyword = strings.TrimSpace(query[idx+len("containing"):])
		}
		return FindByContent(keyword, searchPath, maxResults)
	}

	return FindFiles(query, searchPath, maxResults)
}
