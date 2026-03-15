package tools

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/artaeon/nous/internal/memory"
)

// RegisterBuiltins adds all standard tools to the registry.
// If an UndoStack is provided, write/edit/mkdir/find_replace operations push entries onto it.
func RegisterBuiltins(r *Registry, workDir string, allowShell bool, undo ...*memory.UndoStack) {
	var undoStack *memory.UndoStack
	if len(undo) > 0 {
		undoStack = undo[0]
	}

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
			if undoStack != nil {
				pushUndoForWrite(undoStack, resolvePath(workDir, args["path"]), args["content"])
			}
			return toolWrite(workDir, args)
		},
	})

	r.Register(Tool{
		Name:        "edit",
		Description: "Replace text in a file. Args: path (required), old (text to find), new (replacement). Optional: line (line number for context — reads surrounding lines to find match).",
		Execute: func(args map[string]string) (string, error) {
			if undoStack != nil {
				pushUndoForEdit(undoStack, resolvePath(workDir, args["path"]))
			}
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
			if undoStack != nil {
				path := resolvePath(workDir, args["path"])
				undoStack.Push(memory.UndoEntry{
					Path:   path,
					Action: "mkdir",
					WasNew: true,
				})
			}
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

	r.Register(Tool{
		Name:        "fetch",
		Description: "Fetch content from a URL. Args: url (required). Returns text content (HTML tags stripped).",
		Execute: func(args map[string]string) (string, error) {
			return toolFetch(args)
		},
	})

	r.Register(Tool{
		Name:        "run",
		Description: "Execute a command and capture output. Args: command (required), stdin (optional). Requires --allow-shell.",
		Execute: func(args map[string]string) (string, error) {
			if !allowShell {
				return "", fmt.Errorf("command execution disabled — start Nous with --allow-shell to enable")
			}
			return toolRun(workDir, args)
		},
	})

	r.Register(Tool{
		Name:        "sysinfo",
		Description: "Show system information: OS, architecture, CPU, disk. Args: none.",
		Execute: func(args map[string]string) (string, error) {
			return toolSysinfo(workDir)
		},
	})

	r.Register(Tool{
		Name:        "find_replace",
		Description: "Regex find and replace in a file. Args: path (required), pattern (required regex), replacement (required), all (optional 'true').",
		Execute: func(args map[string]string) (string, error) {
			if undoStack != nil {
				pushUndoForEdit(undoStack, resolvePath(workDir, args["path"]))
			}
			return toolFindReplace(workDir, args)
		},
	})

	r.Register(Tool{
		Name:        "git",
		Description: "Run a git command. Args: command (required, e.g. 'status', 'diff', 'log --oneline -10', 'add .', 'commit -m message').",
		Execute: func(args map[string]string) (string, error) {
			return toolGit(workDir, args)
		},
	})

	r.Register(Tool{
		Name:        "patch",
		Description: "Apply a multi-line edit to a file using a before/after patch. Args: path (required), before (the exact multi-line text to find), after (the replacement text).",
		Execute: func(args map[string]string) (string, error) {
			if undoStack != nil {
				pushUndoForEdit(undoStack, resolvePath(workDir, args["path"]))
			}
			return toolPatch(workDir, args)
		},
	})

	r.Register(Tool{
		Name:        "replace_all",
		Description: "Replace all occurrences of a string across files. Args: old (required), new (required), glob (optional file pattern like '*.go', defaults to all files).",
		Execute: func(args map[string]string) (string, error) {
			return toolReplaceAll(workDir, args)
		},
	})

	r.Register(Tool{
		Name:        "diff",
		Description: "Show the diff between the current state and the last commit. Args: path (optional file path), staged (optional 'true' to show staged changes).",
		Execute: func(args map[string]string) (string, error) {
			return toolDiff(workDir, args)
		},
	})

	r.Register(Tool{
		Name:        "clipboard",
		Description: "Read from or write to the system clipboard. Args: action (required: 'read' or 'write'), content (required for write).",
		Execute: func(args map[string]string) (string, error) {
			return toolClipboard(args)
		},
	})
}

// pushUndoForWrite records the state before a write operation.
func pushUndoForWrite(undo *memory.UndoStack, path, newContent string) {
	before := ""
	wasNew := true
	if data, err := os.ReadFile(path); err == nil {
		before = string(data)
		wasNew = false
	}
	undo.Push(memory.UndoEntry{
		Path:   path,
		Action: "write",
		Before: before,
		After:  newContent,
		WasNew: wasNew,
	})
}

// pushUndoForEdit records the state before an edit/find_replace operation.
func pushUndoForEdit(undo *memory.UndoStack, path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	undo.Push(memory.UndoEntry{
		Path:   path,
		Action: "edit",
		Before: string(data),
		WasNew: false,
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

	// If exact match not found, try line-based context matching
	if count == 0 {
		if lineStr, ok := args["line"]; ok {
			if lineNum, err := strconv.Atoi(lineStr); err == nil {
				// Use the line number to narrow the search context
				result, editErr := lineContextEdit(content, path, lineNum, oldStr, newStr)
				if editErr == nil {
					if writeErr := os.WriteFile(path, []byte(result), 0644); writeErr != nil {
						return "", fmt.Errorf("write edited file: %w", writeErr)
					}
					return fmt.Sprintf("edited %s near line %d", path, lineNum), nil
				}
			}
		}

		// Try trimmed match (ignore leading/trailing whitespace differences)
		trimmedOld := strings.TrimSpace(oldStr)
		if trimmedOld != "" {
			lines := strings.Split(content, "\n")
			for i, line := range lines {
				if strings.Contains(strings.TrimSpace(line), trimmedOld) ||
					strings.Contains(trimmedOld, strings.TrimSpace(line)) {
					return "", fmt.Errorf("old string not found exactly in %s, but line %d looks similar: %q. Read the file around line %d to see exact content", path, i+1, strings.TrimSpace(line), i+1)
				}
			}
		}

		return "", fmt.Errorf("old string not found in %s. Read the file first to see exact content", path)
	}
	if count > 1 {
		return "", fmt.Errorf("old string found %d times in %s — must be unique. Add more surrounding context", count, path)
	}

	newContent := strings.Replace(content, oldStr, newStr, 1)
	if err := os.WriteFile(path, []byte(newContent), 0644); err != nil {
		return "", fmt.Errorf("write edited file: %w", err)
	}

	return fmt.Sprintf("edited %s: replaced 1 occurrence", path), nil
}

// lineContextEdit finds the best match near a line number and applies the edit.
func lineContextEdit(content, path string, lineNum int, oldStr, newStr string) (string, error) {
	lines := strings.Split(content, "\n")
	if lineNum < 1 || lineNum > len(lines) {
		return "", fmt.Errorf("line %d out of range", lineNum)
	}

	// Search a window of ±5 lines around the target
	start := lineNum - 6
	if start < 0 {
		start = 0
	}
	end := lineNum + 5
	if end > len(lines) {
		end = len(lines)
	}

	window := strings.Join(lines[start:end], "\n")
	if strings.Contains(window, oldStr) {
		// Found in the window — replace in full content
		return strings.Replace(content, oldStr, newStr, 1), nil
	}

	// Try matching trimmed old string against lines in the window
	trimmedOld := strings.TrimSpace(oldStr)
	for i := start; i < end; i++ {
		if strings.TrimSpace(lines[i]) == trimmedOld {
			// Replace preserving original indentation
			indent := lines[i][:len(lines[i])-len(strings.TrimLeft(lines[i], " \t"))]
			lines[i] = indent + strings.TrimSpace(newStr)
			return strings.Join(lines, "\n"), nil
		}
	}

	return "", fmt.Errorf("no match near line %d", lineNum)
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

	// Handle ** (doublestar) patterns: Go's filepath.Match doesn't support **
	// so we strip it and match the filename portion against the base pattern.
	isRecursive := strings.Contains(pattern, "**")
	basePattern := pattern
	if isRecursive {
		// "**/*.go" → "*.go", "**/*_test.go" → "*_test.go"
		basePattern = strings.TrimPrefix(pattern, "**/")
		basePattern = strings.TrimPrefix(basePattern, "**")
		if basePattern == "" {
			basePattern = "*"
		}
	}

	var matches []string

	err := filepath.Walk(base, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}

		if info.IsDir() {
			name := info.Name()
			if strings.HasPrefix(name, ".") || name == "node_modules" || name == "vendor" {
				return filepath.SkipDir
			}
			return nil
		}

		rel, _ := filepath.Rel(base, path)

		if isRecursive {
			// For ** patterns, match the filename against the base pattern
			matched, _ := filepath.Match(basePattern, info.Name())
			if matched {
				matches = append(matches, rel)
			}
		} else {
			matched, _ := filepath.Match(pattern, info.Name())
			if !matched {
				matched, _ = filepath.Match(pattern, rel)
			}
			if matched {
				matches = append(matches, rel)
			}
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
		total := len(matches)
		matches = matches[:100]
		return strings.Join(matches, "\n") + fmt.Sprintf("\n... (%d more files, %d total)", total-100, total), nil
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

	_ = cmd.Run()

	result := stdout.String()
	if result == "" {
		return "no matches found", nil
	}

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
	if err := validateSafeCommand(command); err != nil {
		return "", err
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
	if v, ok := args["path"]; ok && v != "" && v != "." {
		resolved := resolvePath(workDir, v)
		if info, err := os.Stat(resolved); err == nil && info.IsDir() {
			base = resolved
		}
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

// --- fetch tool ---

var htmlTagRegex = regexp.MustCompile(`<[^>]*>`)
var whitespaceRegex = regexp.MustCompile(`\s+`)

func toolFetch(args map[string]string) (string, error) {
	url := args["url"]
	if url == "" {
		return "", fmt.Errorf("fetch requires 'url' argument")
	}

	client := &http.Client{Timeout: 30 * time.Second}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("fetch: invalid URL: %w", err)
	}
	req.Header.Set("User-Agent", "Nous/0.6.0")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetch %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("fetch %s: HTTP %d %s", url, resp.StatusCode, resp.Status)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", fmt.Errorf("fetch: reading body: %w", err)
	}

	text := htmlTagRegex.ReplaceAllString(string(body), " ")
	text = whitespaceRegex.ReplaceAllString(text, " ")
	text = strings.TrimSpace(text)

	if len(text) > 8192 {
		text = text[:8192] + "\n... (truncated)"
	}

	return text, nil
}

// --- run tool ---

func toolRun(workDir string, args map[string]string) (string, error) {
	command := args["command"]
	if command == "" {
		return "", fmt.Errorf("run requires 'command' argument")
	}
	if err := validateSafeCommand(command); err != nil {
		return "", err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "sh", "-c", command)
	cmd.Dir = workDir

	var combined bytes.Buffer
	cmd.Stdout = &combined
	cmd.Stderr = &combined

	if stdin, ok := args["stdin"]; ok && stdin != "" {
		cmd.Stdin = strings.NewReader(stdin)
	}

	err := cmd.Run()
	output := combined.String()

	if len(output) > 8192 {
		output = output[:8192] + "\n... (truncated)"
	}

	if ctx.Err() == context.DeadlineExceeded {
		return output, fmt.Errorf("run: command timed out after 60 seconds")
	}

	if err != nil {
		return output, fmt.Errorf("run: %v", err)
	}

	return output, nil
}

// --- sysinfo tool ---

func toolSysinfo(workDir string) (string, error) {
	var out strings.Builder

	hostname, _ := os.Hostname()

	fmt.Fprintf(&out, "OS:           %s\n", runtime.GOOS)
	fmt.Fprintf(&out, "Architecture: %s\n", runtime.GOARCH)
	fmt.Fprintf(&out, "CPU cores:    %d\n", runtime.NumCPU())
	fmt.Fprintf(&out, "Hostname:     %s\n", hostname)
	fmt.Fprintf(&out, "Go version:   %s\n", runtime.Version())

	var stat syscall.Statfs_t
	if err := syscall.Statfs(workDir, &stat); err == nil {
		availableBytes := stat.Bavail * uint64(stat.Bsize)
		totalBytes := stat.Blocks * uint64(stat.Bsize)
		availGB := float64(availableBytes) / (1 << 30)
		totalGB := float64(totalBytes) / (1 << 30)
		fmt.Fprintf(&out, "Disk:         %.1f GB available / %.1f GB total\n", availGB, totalGB)
	}

	return out.String(), nil
}

// --- find_replace tool ---

func toolFindReplace(workDir string, args map[string]string) (string, error) {
	path := resolvePath(workDir, args["path"])
	if path == "" {
		return "", fmt.Errorf("find_replace requires 'path' argument")
	}

	pattern := args["pattern"]
	if pattern == "" {
		return "", fmt.Errorf("find_replace requires 'pattern' argument")
	}

	replacement := args["replacement"]

	re, err := regexp.Compile(pattern)
	if err != nil {
		return "", fmt.Errorf("find_replace: invalid regex %q: %w", pattern, err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("find_replace: read %s: %w", path, err)
	}

	content := string(data)
	replaceAll := args["all"] == "true"

	var newContent string
	var count int

	if replaceAll {
		matches := re.FindAllStringIndex(content, -1)
		count = len(matches)
		newContent = re.ReplaceAllString(content, replacement)
	} else {
		loc := re.FindStringIndex(content)
		if loc != nil {
			count = 1
			newContent = content[:loc[0]] + re.ReplaceAllString(content[loc[0]:loc[1]], replacement) + content[loc[1]:]
		} else {
			newContent = content
		}
	}

	if count == 0 {
		return "", fmt.Errorf("find_replace: pattern %q not found in %s", pattern, path)
	}

	if err := os.WriteFile(path, []byte(newContent), 0644); err != nil {
		return "", fmt.Errorf("find_replace: write %s: %w", path, err)
	}

	return fmt.Sprintf("replaced %d occurrence(s) in %s", count, path), nil
}

func toolGit(workDir string, args map[string]string) (string, error) {
	command := args["command"]
	if command == "" {
		return "", fmt.Errorf("git requires 'command' argument")
	}
	if err := validateSafeGitCommand(command); err != nil {
		return "", err
	}

	// Shell-aware argument splitting that respects quotes
	gitArgs := splitShellArgs(command)

	cmd := exec.Command("git", gitArgs...)
	cmd.Dir = workDir

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()

	output := stdout.String()
	if stderr.Len() > 0 {
		if output != "" {
			output += "\n"
		}
		output += stderr.String()
	}

	if err != nil {
		if output == "" {
			return "", fmt.Errorf("git %s: %v", command, err)
		}
		return output, fmt.Errorf("git %s: %v", command, err)
	}

	// Truncate to 8192 chars
	if len(output) > 8192 {
		output = output[:8192] + "\n... (truncated)"
	}

	return output, nil
}

func toolPatch(workDir string, args map[string]string) (string, error) {
	path := resolvePath(workDir, args["path"])
	before := args["before"]
	after := args["after"]

	if path == "" || before == "" {
		return "", fmt.Errorf("patch requires 'path' and 'before' arguments")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read for patch: %w", err)
	}

	// Preserve original file permissions
	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("stat for patch: %w", err)
	}
	perm := info.Mode().Perm()

	content := string(data)
	count := strings.Count(content, before)

	if count == 0 {
		return "", fmt.Errorf("before text not found in %s", path)
	}
	if count > 1 {
		return "", fmt.Errorf("before text found %d times in %s — must be unique", count, path)
	}

	newContent := strings.Replace(content, before, after, 1)
	if err := os.WriteFile(path, []byte(newContent), perm); err != nil {
		return "", fmt.Errorf("write patched file: %w", err)
	}

	return fmt.Sprintf("patched %s: replaced 1 occurrence (%d lines)", path, strings.Count(before, "\n")+1), nil
}

func toolReplaceAll(workDir string, args map[string]string) (string, error) {
	oldStr := args["old"]
	newStr := args["new"]
	globPattern := args["glob"]

	if oldStr == "" {
		return "", fmt.Errorf("replace_all requires 'old' argument")
	}
	if oldStr == newStr {
		return "", fmt.Errorf("old and new strings are identical")
	}

	totalOccurrences := 0
	filesModified := 0

	err := filepath.Walk(workDir, func(path string, info os.FileInfo, err error) error {
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

		// Apply glob filter if specified
		if globPattern != "" {
			matched, _ := filepath.Match(globPattern, info.Name())
			if !matched {
				return nil
			}
		}

		// Read file
		data, err := os.ReadFile(path)
		if err != nil {
			return nil // skip unreadable files
		}

		// Skip binary files (check for null bytes in first 512 bytes)
		checkLen := len(data)
		if checkLen > 512 {
			checkLen = 512
		}
		if bytes.ContainsRune(data[:checkLen], 0) {
			return nil
		}

		content := string(data)
		count := strings.Count(content, oldStr)
		if count == 0 {
			return nil
		}

		// Replace all occurrences
		newContent := strings.ReplaceAll(content, oldStr, newStr)
		if err := os.WriteFile(path, []byte(newContent), info.Mode().Perm()); err != nil {
			return nil // skip write errors
		}

		totalOccurrences += count
		filesModified++

		return nil
	})

	if err != nil {
		return "", fmt.Errorf("replace_all: %w", err)
	}

	if totalOccurrences == 0 {
		return "no occurrences found", nil
	}

	return fmt.Sprintf("replaced %d occurrences across %d files", totalOccurrences, filesModified), nil
}

func toolDiff(workDir string, args map[string]string) (string, error) {
	gitArgs := []string{"diff"}

	if staged, ok := args["staged"]; ok && staged == "true" {
		gitArgs = append(gitArgs, "--staged")
	}

	if path, ok := args["path"]; ok && path != "" {
		gitArgs = append(gitArgs, "--", resolvePath(workDir, path))
	}

	cmd := exec.Command("git", gitArgs...)
	cmd.Dir = workDir

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()

	output := stdout.String()
	if stderr.Len() > 0 {
		if output != "" {
			output += "\n"
		}
		output += stderr.String()
	}

	if err != nil {
		if output == "" {
			return "", fmt.Errorf("git diff: %v", err)
		}
		return output, fmt.Errorf("git diff: %v", err)
	}

	if output == "" {
		return "no changes", nil
	}

	// Truncate to 8192 chars
	if len(output) > 8192 {
		output = output[:8192] + "\n... (truncated)"
	}

	return output, nil
}

// splitShellArgs splits a command string respecting single and double quotes.
// e.g., `commit -m "fix the bug"` → ["commit", "-m", "fix the bug"]
func splitShellArgs(s string) []string {
	var args []string
	var current strings.Builder
	inSingle := false
	inDouble := false

	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c == '\'' && !inDouble:
			inSingle = !inSingle
		case c == '"' && !inSingle:
			inDouble = !inDouble
		case c == ' ' && !inSingle && !inDouble:
			if current.Len() > 0 {
				args = append(args, current.String())
				current.Reset()
			}
		default:
			current.WriteByte(c)
		}
	}
	if current.Len() > 0 {
		args = append(args, current.String())
	}
	return args
}

func allowUnsafeCommands() bool {
	return strings.TrimSpace(os.Getenv("NOUS_ALLOW_UNSAFE")) == "1"
}

func validateSafeCommand(command string) error {
	if allowUnsafeCommands() {
		return nil
	}
	lower := strings.ToLower(strings.TrimSpace(command))
	blocked := []string{
		"rm -rf /",
		"rm -rf --no-preserve-root /",
		"mkfs",
		"shutdown",
		"reboot",
		"poweroff",
		"halt",
		":(){",
		"dd if=",
		"> /dev/sd",
		"chmod -r 777 /",
		"chmod -r 777",
		"chown -r /",
	}
	for _, fragment := range blocked {
		if strings.Contains(lower, fragment) {
			return fmt.Errorf("refusing risky command fragment %q (set NOUS_ALLOW_UNSAFE=1 to override)", fragment)
		}
	}
	return nil
}

func validateSafeGitCommand(command string) error {
	if allowUnsafeCommands() {
		return nil
	}
	lower := strings.ToLower(strings.TrimSpace(command))
	blocked := []string{
		"reset --hard",
		"clean -fd",
		"clean -xdf",
		"push --force",
		"push -f",
	}
	for _, fragment := range blocked {
		if strings.Contains(lower, fragment) {
			return fmt.Errorf("refusing risky git command fragment %q (set NOUS_ALLOW_UNSAFE=1 to override)", fragment)
		}
	}
	return nil
}

// --- clipboard tool ---

func toolClipboard(args map[string]string) (string, error) {
	action := args["action"]
	if action == "" {
		return "", fmt.Errorf("clipboard requires 'action' argument ('read' or 'write')")
	}

	// Determine which clipboard tool is available
	clipTool := ""
	if path, err := exec.LookPath("xclip"); err == nil && path != "" {
		clipTool = "xclip"
	} else if path, err := exec.LookPath("xsel"); err == nil && path != "" {
		clipTool = "xsel"
	}

	if clipTool == "" {
		return "", fmt.Errorf("clipboard: neither xclip nor xsel is installed — install one with: sudo apt install xclip")
	}

	switch action {
	case "read":
		var cmd *exec.Cmd
		if clipTool == "xclip" {
			cmd = exec.Command("xclip", "-selection", "clipboard", "-o")
		} else {
			cmd = exec.Command("xsel", "--clipboard", "--output")
		}
		out, err := cmd.Output()
		if err != nil {
			return "", fmt.Errorf("clipboard read: %w", err)
		}
		return string(out), nil

	case "write":
		content := args["content"]
		if content == "" {
			return "", fmt.Errorf("clipboard write requires 'content' argument")
		}
		var cmd *exec.Cmd
		if clipTool == "xclip" {
			cmd = exec.Command("xclip", "-selection", "clipboard")
		} else {
			cmd = exec.Command("xsel", "--clipboard", "--input")
		}
		cmd.Stdin = strings.NewReader(content)
		if err := cmd.Run(); err != nil {
			return "", fmt.Errorf("clipboard write: %w", err)
		}
		return fmt.Sprintf("wrote %d bytes to clipboard", len(content)), nil

	default:
		return "", fmt.Errorf("clipboard: unknown action %q (use 'read' or 'write')", action)
	}
}
