package tools

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// RegisterBuiltins adds all standard tools to the registry.
func RegisterBuiltins(r *Registry, workDir string, allowShell bool) {
	r.Register(Tool{
		Name:        "read",
		Description: "Read a file's contents. Args: path (required), offset (optional line number to start from), limit (optional max lines).",
		Execute: func(args map[string]string) (string, error) {
			return toolRead(workDir, args)
		},
	})

	r.Register(Tool{
		Name:        "write",
		Description: "Create or overwrite a file. Args: path (required), content (required).",
		Execute: func(args map[string]string) (string, error) {
			return toolWrite(workDir, args)
		},
	})

	r.Register(Tool{
		Name:        "edit",
		Description: "Replace a specific string in a file. Args: path (required), old (the exact text to find), new (the replacement text).",
		Execute: func(args map[string]string) (string, error) {
			return toolEdit(workDir, args)
		},
	})

	r.Register(Tool{
		Name:        "glob",
		Description: "Find files matching a glob pattern. Args: pattern (required, e.g. '**/*.go'), path (optional base directory).",
		Execute: func(args map[string]string) (string, error) {
			return toolGlob(workDir, args)
		},
	})

	r.Register(Tool{
		Name:        "grep",
		Description: "Search file contents for a regex pattern. Args: pattern (required), path (optional directory), glob (optional file filter like '*.go').",
		Execute: func(args map[string]string) (string, error) {
			return toolGrep(workDir, args)
		},
	})

	r.Register(Tool{
		Name:        "ls",
		Description: "List directory contents. Args: path (optional, defaults to working directory).",
		Execute: func(args map[string]string) (string, error) {
			return toolLs(workDir, args)
		},
	})

	r.Register(Tool{
		Name:        "shell",
		Description: "Execute a shell command. Args: command (required). Only available with --allow-shell flag.",
		Execute: func(args map[string]string) (string, error) {
			if !allowShell {
				return "", fmt.Errorf("shell execution disabled — start Nous with --allow-shell to enable")
			}
			return toolShell(workDir, args)
		},
	})

	r.Register(Tool{
		Name:        "mkdir",
		Description: "Create a directory (and parents). Args: path (required).",
		Execute: func(args map[string]string) (string, error) {
			return toolMkdir(workDir, args)
		},
	})

	r.Register(Tool{
		Name:        "tree",
		Description: "Show project directory structure. Args: path (optional), depth (optional, default 3).",
		Execute: func(args map[string]string) (string, error) {
			return toolTree(workDir, args)
		},
	})
}

func resolvePath(workDir, path string) string {
	if filepath.IsAbs(path) {
		return path
	}
	return filepath.Join(workDir, path)
}

func toolRead(workDir string, args map[string]string) (string, error) {
	path := resolvePath(workDir, args["path"])
	if path == "" {
		return "", fmt.Errorf("read requires 'path' argument")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", path, err)
	}

	content := string(data)
	lines := strings.Split(content, "\n")

	// Handle offset and limit
	offset := 0
	if v, ok := args["offset"]; ok {
		if n, err := strconv.Atoi(v); err == nil {
			offset = n
		}
	}

	limit := len(lines)
	if v, ok := args["limit"]; ok {
		if n, err := strconv.Atoi(v); err == nil {
			limit = n
		}
	}

	if offset >= len(lines) {
		return "", fmt.Errorf("offset %d exceeds file length %d", offset, len(lines))
	}

	end := offset + limit
	if end > len(lines) {
		end = len(lines)
	}

	// Format with line numbers
	var out strings.Builder
	for i := offset; i < end; i++ {
		fmt.Fprintf(&out, "%4d | %s\n", i+1, lines[i])
	}

	return out.String(), nil
}

func toolWrite(workDir string, args map[string]string) (string, error) {
	path := resolvePath(workDir, args["path"])
	content := args["content"]
	if path == "" {
		return "", fmt.Errorf("write requires 'path' argument")
	}

	// Create parent directories
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return "", fmt.Errorf("mkdir for write: %w", err)
	}

	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return "", fmt.Errorf("write %s: %w", path, err)
	}

	return fmt.Sprintf("wrote %d bytes to %s", len(content), path), nil
}

func toolEdit(workDir string, args map[string]string) (string, error) {
	path := resolvePath(workDir, args["path"])
	oldStr := args["old"]
	newStr := args["new"]

	if path == "" || oldStr == "" {
		return "", fmt.Errorf("edit requires 'path', 'old', and 'new' arguments")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read for edit: %w", err)
	}

	content := string(data)
	count := strings.Count(content, oldStr)

	if count == 0 {
		return "", fmt.Errorf("old string not found in %s", path)
	}
	if count > 1 {
		return "", fmt.Errorf("old string found %d times in %s — must be unique", count, path)
	}

	newContent := strings.Replace(content, oldStr, newStr, 1)
	if err := os.WriteFile(path, []byte(newContent), 0644); err != nil {
		return "", fmt.Errorf("write edited file: %w", err)
	}

	return fmt.Sprintf("edited %s: replaced 1 occurrence", path), nil
}

func toolGlob(workDir string, args map[string]string) (string, error) {
	pattern := args["pattern"]
	if pattern == "" {
		return "", fmt.Errorf("glob requires 'pattern' argument")
	}

	base := workDir
	if v, ok := args["path"]; ok && v != "" {
		base = resolvePath(workDir, v)
	}

	var matches []string

	err := filepath.Walk(base, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // skip errors
		}

		// Skip hidden directories and common noise
		if info.IsDir() {
			name := info.Name()
			if strings.HasPrefix(name, ".") || name == "node_modules" || name == "vendor" {
				return filepath.SkipDir
			}
			return nil
		}

		rel, _ := filepath.Rel(base, path)
		matched, _ := filepath.Match(pattern, info.Name())

		// Also try matching against the relative path for ** patterns
		if !matched {
			matched, _ = filepath.Match(pattern, rel)
		}

		if matched {
			matches = append(matches, rel)
		}

		return nil
	})

	if err != nil {
		return "", fmt.Errorf("glob: %w", err)
	}

	if len(matches) == 0 {
		return "no files matched", nil
	}

	if len(matches) > 100 {
		matches = matches[:100]
		return strings.Join(matches, "\n") + "\n... (truncated to 100 results)", nil
	}

	return strings.Join(matches, "\n"), nil
}

func toolGrep(workDir string, args map[string]string) (string, error) {
	pattern := args["pattern"]
	if pattern == "" {
		return "", fmt.Errorf("grep requires 'pattern' argument")
	}

	searchPath := workDir
	if v, ok := args["path"]; ok && v != "" {
		searchPath = resolvePath(workDir, v)
	}

	grepArgs := []string{"-rn", "--color=never"}

	if glob, ok := args["glob"]; ok && glob != "" {
		grepArgs = append(grepArgs, "--include="+glob)
	}

	// Exclude common noise
	grepArgs = append(grepArgs,
		"--exclude-dir=.git",
		"--exclude-dir=node_modules",
		"--exclude-dir=vendor",
		pattern,
		searchPath,
	)

	cmd := exec.Command("grep", grepArgs...)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout

	_ = cmd.Run() // grep exit 1 = no matches

	result := stdout.String()
	if result == "" {
		return "no matches found", nil
	}

	// Truncate output
	lines := strings.Split(result, "\n")
	if len(lines) > 50 {
		lines = lines[:50]
		return strings.Join(lines, "\n") + "\n... (truncated to 50 results)", nil
	}

	return result, nil
}

func toolLs(workDir string, args map[string]string) (string, error) {
	dir := workDir
	if v, ok := args["path"]; ok && v != "" {
		dir = resolvePath(workDir, v)
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", fmt.Errorf("ls %s: %w", dir, err)
	}

	var out strings.Builder
	for _, e := range entries {
		info, err := e.Info()
		if err != nil {
			continue
		}
		prefix := "  "
		if e.IsDir() {
			prefix = "d "
		}
		fmt.Fprintf(&out, "%s %8d  %s\n", prefix, info.Size(), e.Name())
	}

	return out.String(), nil
}

func toolShell(workDir string, args map[string]string) (string, error) {
	command := args["command"]
	if command == "" {
		return "", fmt.Errorf("shell requires 'command' argument")
	}

	cmd := exec.Command("sh", "-c", command)
	cmd.Dir = workDir

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()

	output := stdout.String()
	if stderr.Len() > 0 {
		output += "\nSTDERR:\n" + stderr.String()
	}

	if err != nil {
		return output, fmt.Errorf("exit %v", err)
	}

	// Truncate
	if len(output) > 8192 {
		output = output[:8192] + "\n... (truncated)"
	}

	return output, nil
}

func toolMkdir(workDir string, args map[string]string) (string, error) {
	path := resolvePath(workDir, args["path"])
	if path == "" {
		return "", fmt.Errorf("mkdir requires 'path' argument")
	}

	if err := os.MkdirAll(path, 0755); err != nil {
		return "", fmt.Errorf("mkdir %s: %w", path, err)
	}

	return fmt.Sprintf("created %s", path), nil
}

func toolTree(workDir string, args map[string]string) (string, error) {
	base := workDir
	if v, ok := args["path"]; ok && v != "" {
		base = resolvePath(workDir, v)
	}

	maxDepth := 3
	if v, ok := args["depth"]; ok {
		if n, err := strconv.Atoi(v); err == nil {
			maxDepth = n
		}
	}

	var out strings.Builder
	buildTree(&out, base, "", 0, maxDepth)
	return out.String(), nil
}

func buildTree(out *strings.Builder, path, prefix string, depth, maxDepth int) {
	if depth >= maxDepth {
		return
	}

	entries, err := os.ReadDir(path)
	if err != nil {
		return
	}

	// Filter hidden and noise
	var visible []os.DirEntry
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, ".") || name == "node_modules" || name == "vendor" {
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
			buildTree(out, filepath.Join(path, e.Name()), prefix+childPrefix, depth+1, maxDepth)
		}
	}
}
