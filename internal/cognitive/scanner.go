package cognitive

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ProjectInfo holds a summary of the project structure for the LLM context.
type ProjectInfo struct {
	RootDir    string
	Name       string
	Language   string
	FileCount  int
	Tree       string
	KeyFiles   []string
}

// ScanProject analyzes the working directory and returns a project summary
// that can be injected into the system prompt for project awareness.
func ScanProject(rootDir string) *ProjectInfo {
	info := &ProjectInfo{
		RootDir: rootDir,
		Name:    filepath.Base(rootDir),
	}

	// Detect language and key files
	langCounts := make(map[string]int)
	var keyFiles []string

	filepath.Walk(rootDir, func(path string, fi os.FileInfo, err error) error {
		if err != nil {
			return nil
		}

		name := fi.Name()

		// Skip hidden dirs and noise
		if fi.IsDir() {
			if strings.HasPrefix(name, ".") || name == "node_modules" || name == "vendor" || name == "__pycache__" {
				return filepath.SkipDir
			}
			return nil
		}

		info.FileCount++

		// Count by extension
		ext := strings.ToLower(filepath.Ext(name))
		if ext != "" {
			langCounts[ext]++
		}

		// Detect key files
		rel, _ := filepath.Rel(rootDir, path)
		switch strings.ToLower(name) {
		case "go.mod", "go.sum":
			keyFiles = append(keyFiles, rel)
		case "package.json", "tsconfig.json":
			keyFiles = append(keyFiles, rel)
		case "cargo.toml":
			keyFiles = append(keyFiles, rel)
		case "pyproject.toml", "setup.py", "requirements.txt":
			keyFiles = append(keyFiles, rel)
		case "makefile", "dockerfile", "docker-compose.yml":
			keyFiles = append(keyFiles, rel)
		case "readme.md", "readme.txt", "readme":
			keyFiles = append(keyFiles, rel)
		case ".env.example", "claude.md":
			keyFiles = append(keyFiles, rel)
		}

		return nil
	})

	info.KeyFiles = keyFiles
	info.Language = detectLanguage(langCounts)
	info.Tree = buildCompactTree(rootDir, 3)

	return info
}

// ContextString returns a formatted string for injecting into the LLM system prompt.
func (p *ProjectInfo) ContextString() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Project: %s\n", p.Name))
	sb.WriteString(fmt.Sprintf("Language: %s\n", p.Language))
	sb.WriteString(fmt.Sprintf("Files: %d\n", p.FileCount))

	if len(p.KeyFiles) > 0 {
		sb.WriteString("Key files: " + strings.Join(p.KeyFiles, ", ") + "\n")
	}

	if p.Tree != "" {
		sb.WriteString("Structure:\n" + p.Tree)
	}

	return sb.String()
}

func detectLanguage(counts map[string]int) string {
	if counts[".go"] > 0 {
		return "Go"
	}
	if counts[".rs"] > 0 {
		return "Rust"
	}
	if counts[".ts"] > 0 || counts[".tsx"] > 0 {
		return "TypeScript"
	}
	if counts[".js"] > 0 || counts[".jsx"] > 0 {
		return "JavaScript"
	}
	if counts[".py"] > 0 {
		return "Python"
	}
	if counts[".java"] > 0 {
		return "Java"
	}
	if counts[".c"] > 0 || counts[".h"] > 0 {
		return "C"
	}
	if counts[".cpp"] > 0 || counts[".hpp"] > 0 {
		return "C++"
	}

	// Find the most common extension
	maxExt := ""
	maxCount := 0
	for ext, count := range counts {
		if count > maxCount {
			maxCount = count
			maxExt = ext
		}
	}

	if maxExt != "" {
		return strings.TrimPrefix(maxExt, ".")
	}

	return "unknown"
}

func buildCompactTree(root string, maxDepth int) string {
	var sb strings.Builder
	buildCompactTreeRecursive(&sb, root, "", 0, maxDepth)
	return sb.String()
}

func buildCompactTreeRecursive(out *strings.Builder, path, prefix string, depth, maxDepth int) {
	if depth >= maxDepth {
		return
	}

	entries, err := os.ReadDir(path)
	if err != nil {
		return
	}

	var visible []os.DirEntry
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, ".") || name == "node_modules" || name == "vendor" || name == "__pycache__" {
			continue
		}
		visible = append(visible, e)
	}

	for i, e := range visible {
		isLast := i == len(visible)-1
		connector := "├── "
		childPrefix := "│   "
		if isLast {
			connector = "└── "
			childPrefix = "    "
		}

		fmt.Fprintf(out, "%s%s%s\n", prefix, connector, e.Name())

		if e.IsDir() {
			buildCompactTreeRecursive(out, filepath.Join(path, e.Name()), prefix+childPrefix, depth+1, maxDepth)
		}
	}
}
