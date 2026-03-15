package cognitive

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// IntentCompiler translates natural language into deterministic tool calls.
//
// Instead of forcing a 1.5B model to generate JSON tool calls (which fails ~60%
// of the time), the compiler parses natural language and maps it to structured
// actions. The model does what it's good at (understanding intent) while the
// compiler handles what it's bad at (structured output generation).
//
// This is a "cognitive prosthetic" — a deterministic module that compensates
// for a small model's weakness in structured output, similar to how a calculator
// compensates for human weakness in arithmetic.
//
// Two integration points:
//  1. PRE-LLM: resolve user queries directly into tool calls (skip LLM for tool selection)
//  2. POST-LLM: recover when the model's response contains intent but malformed JSON
type IntentCompiler struct {
	workDir   string
	fileTree  []string // cached project file paths (relative)
	dirTree   []string // cached project directories (relative)
	mu        sync.RWMutex
	lastScan  time.Time
	scanTTL   time.Duration
	maxDepth  int
	maxFiles  int
}

// CompiledAction represents a deterministically resolved tool call.
type CompiledAction struct {
	Tool       string
	Args       map[string]string
	Confidence float64 // 0.0-1.0
	Source     string  // which pattern matched (for debugging/learning)
}

// NewIntentCompiler creates a new compiler grounded to the given working directory.
func NewIntentCompiler(workDir string) *IntentCompiler {
	ic := &IntentCompiler{
		workDir:  workDir,
		scanTTL:  30 * time.Second,
		maxDepth: 6,
		maxFiles: 2000,
	}
	ic.RefreshFS()
	return ic
}

// --- Intent Pattern Definitions ---
// Each pattern category maps natural language to a specific tool.
// Patterns are ordered by specificity — more specific patterns first.

// Read intent: user wants to see file contents
var readPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)(?:read|cat|show me|show|display|open|look at|view|check)\s+(?:the\s+)?(?:file\s+)?["'\x60]?([a-zA-Z0-9_./ -]+\.[a-zA-Z0-9]+)["'\x60]?`),
	regexp.MustCompile(`(?i)what(?:'s| is| are)\s+in\s+(?:the\s+)?(?:file\s+)?["'\x60]?([a-zA-Z0-9_./ -]+\.[a-zA-Z0-9]+)["'\x60]?`),
	regexp.MustCompile(`(?i)(?:read|cat|show|display|open|view)\s+["'\x60]?([a-zA-Z0-9_./ -]+)["'\x60]?\s*$`),
}

// Search intent: user wants to find content in files
var searchPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)(?:search|grep|find|look)\s+(?:for\s+)?(?:the\s+)?(?:word|string|text|pattern|term)?\s*["'\x60]([^"'\x60]+)["'\x60]`),
	regexp.MustCompile(`(?i)(?:search|grep|find|look)\s+(?:for\s+)?["'\x60]([^"'\x60]+)["'\x60]`),
	regexp.MustCompile(`(?i)where\s+is\s+["'\x60]?(\w+)["'\x60]?\s+(?:defined|used|called|declared|implemented)`),
	regexp.MustCompile(`(?i)where\s+(?:is|are)\s+["'\x60]?([A-Z]\w+)["'\x60]?\s`),
	regexp.MustCompile(`(?i)(?:search|grep|find)\s+(?:for\s+)?(\S+)\s+(?:in\s+)?(?:all\s+)?(?:go|\.go|python|\.py|java|\.java|rust|\.rs|js|\.js|ts|\.ts)\s+files`),
	regexp.MustCompile(`(?i)(?:search|grep|find)\s+(?:for\s+)?(?:the\s+)?(?:word|string|text)?\s*(\S+)\s+(?:in\s+)`),
	regexp.MustCompile(`(?i)(?:search|grep|find)\s+(?:for\s+)?(\S+)`),
}

// Search with file filter extraction
var searchFilterPattern = regexp.MustCompile(`(?i)in\s+(?:all\s+)?(?:the\s+)?((?:\*\.)?[a-z]+)\s+files`)
var searchDirPattern = regexp.MustCompile(`(?i)in\s+(?:the\s+)?(?:directory\s+)?["'\x60]?([a-zA-Z0-9_./ -]+)["'\x60]?`)

// List intent: user wants to see directory contents
var listPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)(?:list|ls|show)\s+(?:all\s+)?(?:the\s+)?(?:files|contents?)\s+(?:in\s+|of\s+)?(?:the\s+)?(?:directory\s+)?["'\x60]?([a-zA-Z0-9_./ -]+)["'\x60]?`),
	regexp.MustCompile(`(?i)what\s+(?:files|things)\s+(?:are\s+)?in\s+(?:the\s+)?(?:directory\s+)?(?:["'\x60]?([a-zA-Z0-9_./ -]+)["'\x60]?|(?:current|this)\s+(?:directory|folder|dir))`),
	regexp.MustCompile(`(?i)(?:list|ls|show)\s+(?:the\s+)?(?:current\s+)?(?:directory|folder|dir|files)`),
	regexp.MustCompile(`(?i)what(?:'s| is)\s+(?:in\s+)?(?:the\s+)?(?:current|this)\s+(?:directory|folder|dir)`),
}

// Write intent: user wants to create or modify a file
var writePatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)(?:create|write|save|make)\s+(?:a\s+)?(?:file\s+)?(?:called|named)\s+["'\x60]?([a-zA-Z0-9_./ -]+)["'\x60]?\s+(?:with|containing)\s+(?:the\s+)?(?:text|content|data)?\s*["'\x60](.+?)["'\x60]?\s*$`),
	regexp.MustCompile(`(?i)(?:create|write|save|make)\s+(?:a\s+)?(?:file\s+)?["'\x60]?([a-zA-Z0-9_./ -]+)["'\x60]?\s+(?:with|containing)\s+["'\x60](.+?)["'\x60]?\s*$`),
	regexp.MustCompile(`(?i)(?:write|save|put)\s+["'\x60](.+?)["'\x60]\s+(?:to|into|in)\s+(?:the\s+)?(?:file\s+)?["'\x60]?([a-zA-Z0-9_./ -]+)["'\x60]?`),
}

// Tree intent: user wants directory structure
var treePatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)(?:show|display)\s+(?:the\s+)?(?:directory\s+)?(?:tree|structure)(?:\s+(?:of\s+)?["'\x60]?([a-zA-Z0-9_./ -]*)["'\x60]?)?`),
	regexp.MustCompile(`(?i)(?:project|directory|folder|dir|code)\s+(?:tree|structure|layout|overview)`),
}

// Glob intent: user wants to find files by name pattern
var globPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)(?:find|list|show)\s+(?:all\s+)?(?:files?\s+)?(?:matching|named|called|with\s+(?:extension|ext|name))\s+["'\x60]?([a-zA-Z0-9_.*?/ -]+)["'\x60]?`),
	regexp.MustCompile(`(?i)(?:find|list|show)\s+(?:all\s+)?(?:\*\.)?([a-z]+)\s+files?`),
}

// Git intent: user wants git operations
var gitPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)(?:git|show)\s+(status|log|diff|branch|remote|blame|stash|show)`),
	regexp.MustCompile(`(?i)(?:show|what(?:'s| is| are))\s+(?:the\s+)?(?:git\s+)?(?:status|changes|modifications|modified files?)`),
	regexp.MustCompile(`(?i)(?:show|what(?:'s| is| are))\s+(?:the\s+)?(?:recent\s+)?(?:commit|change)\s*(?:history|log)`),
}

// fileExtToGlob maps common language to glob filters
var fileExtToGlob = map[string]string{
	"go": "*.go", "python": "*.py", "py": "*.py", "java": "*.java",
	"javascript": "*.js", "js": "*.js", "typescript": "*.ts", "ts": "*.ts",
	"rust": "*.rs", "rs": "*.rs", "ruby": "*.rb", "rb": "*.rb",
	"c": "*.c", "cpp": "*.cpp", "h": "*.h", "yaml": "*.yaml",
	"yml": "*.yml", "json": "*.json", "md": "*.md", "toml": "*.toml",
	"xml": "*.xml", "html": "*.html", "css": "*.css", "sql": "*.sql",
	"sh": "*.sh", "bash": "*.sh", "txt": "*.txt",
}

// Compile attempts to deterministically resolve user input into tool calls.
// Returns nil if no pattern matches with sufficient confidence.
func (ic *IntentCompiler) Compile(input string) []CompiledAction {
	ic.ensureFreshFS()
	input = strings.TrimSpace(input)
	if input == "" {
		return nil
	}

	// Try each intent category in order of specificity
	if actions := ic.tryWriteIntent(input); len(actions) > 0 {
		return actions
	}
	if actions := ic.tryGitIntent(input); len(actions) > 0 {
		return actions
	}
	if actions := ic.tryTreeIntent(input); len(actions) > 0 {
		return actions
	}
	if actions := ic.tryGlobIntent(input); len(actions) > 0 {
		return actions
	}
	if actions := ic.tryReadIntent(input); len(actions) > 0 {
		return actions
	}
	if actions := ic.trySearchIntent(input); len(actions) > 0 {
		return actions
	}
	if actions := ic.tryListIntent(input); len(actions) > 0 {
		return actions
	}

	return nil
}

// CompileResponse parses the LLM's natural language response to extract
// intended tool calls that the model failed to express as JSON.
// This is the POST-LLM fallback path.
func (ic *IntentCompiler) CompileResponse(response, originalInput string) []CompiledAction {
	ic.ensureFreshFS()

	// The model said something like "Let me look at the go.mod file" or
	// "I need to search for ReflectionGate" — extract tool intent from that
	combined := response + " " + originalInput
	if actions := ic.Compile(combined); len(actions) > 0 {
		// Reduce confidence since we're inferring from response text
		for i := range actions {
			actions[i].Confidence *= 0.8
			actions[i].Source = "response-recovery:" + actions[i].Source
		}
		return actions
	}
	return nil
}

// --- Intent matchers ---

func (ic *IntentCompiler) tryReadIntent(input string) []CompiledAction {
	for _, pat := range readPatterns {
		if m := pat.FindStringSubmatch(input); len(m) >= 2 {
			path := strings.TrimSpace(m[1])
			resolved := ic.ResolvePath(path)
			if resolved == "" {
				continue
			}
			return []CompiledAction{{
				Tool:       "read",
				Args:       map[string]string{"path": resolved},
				Confidence: 0.9,
				Source:     "read-pattern",
			}}
		}
	}
	return nil
}

func (ic *IntentCompiler) trySearchIntent(input string) []CompiledAction {
	for _, pat := range searchPatterns {
		m := pat.FindStringSubmatch(input)
		if len(m) < 2 {
			continue
		}
		pattern := strings.TrimSpace(m[1])
		if pattern == "" || len(pattern) < 2 {
			continue
		}

		args := map[string]string{"pattern": pattern}

		// Extract file type filter
		if fm := searchFilterPattern.FindStringSubmatch(input); len(fm) >= 2 {
			ext := strings.ToLower(fm[1])
			if glob, ok := fileExtToGlob[ext]; ok {
				args["glob"] = glob
			} else if strings.HasPrefix(ext, "*.") {
				args["glob"] = ext
			} else {
				args["glob"] = "*." + ext
			}
		}

		// Extract directory scope
		if dm := searchDirPattern.FindStringSubmatch(input); len(dm) >= 2 {
			dir := strings.TrimSpace(dm[1])
			if resolved := ic.ResolveDir(dir); resolved != "" {
				args["path"] = resolved
			}
		}

		return []CompiledAction{{
			Tool:       "grep",
			Args:       args,
			Confidence: 0.85,
			Source:     "search-pattern",
		}}
	}
	return nil
}

func (ic *IntentCompiler) tryListIntent(input string) []CompiledAction {
	for _, pat := range listPatterns {
		m := pat.FindStringSubmatch(input)
		if m == nil {
			continue
		}
		args := map[string]string{}

		if len(m) >= 2 && strings.TrimSpace(m[1]) != "" {
			dir := strings.TrimSpace(m[1])
			if resolved := ic.ResolveDir(dir); resolved != "" {
				args["path"] = resolved
			} else if resolved := ic.ResolvePath(dir); resolved != "" {
				args["path"] = resolved
			}
		}

		return []CompiledAction{{
			Tool:       "ls",
			Args:       args,
			Confidence: 0.85,
			Source:     "list-pattern",
		}}
	}
	return nil
}

func (ic *IntentCompiler) tryWriteIntent(input string) []CompiledAction {
	for _, pat := range writePatterns {
		m := pat.FindStringSubmatch(input)
		if len(m) < 3 {
			continue
		}

		// Patterns 1&2: write(path, content), Pattern 3: write(content, path) - reversed
		path := strings.TrimSpace(m[1])
		content := strings.TrimSpace(m[2])

		// Pattern 3 reverses path and content
		if strings.Contains(pat.String(), `(?:to|into|in)`) {
			path, content = content, path
		}

		return []CompiledAction{{
			Tool:       "write",
			Args:       map[string]string{"path": path, "content": content},
			Confidence: 0.8,
			Source:     "write-pattern",
		}}
	}
	return nil
}

func (ic *IntentCompiler) tryTreeIntent(input string) []CompiledAction {
	for _, pat := range treePatterns {
		m := pat.FindStringSubmatch(input)
		if m == nil {
			continue
		}
		args := map[string]string{}
		if len(m) >= 2 && strings.TrimSpace(m[1]) != "" {
			dir := strings.TrimSpace(m[1])
			if resolved := ic.ResolveDir(dir); resolved != "" {
				args["path"] = resolved
			}
		}
		return []CompiledAction{{
			Tool:       "tree",
			Args:       args,
			Confidence: 0.85,
			Source:     "tree-pattern",
		}}
	}
	return nil
}

func (ic *IntentCompiler) tryGlobIntent(input string) []CompiledAction {
	for _, pat := range globPatterns {
		m := pat.FindStringSubmatch(input)
		if m == nil || len(m) < 2 {
			continue
		}
		raw := strings.TrimSpace(m[1])
		if raw == "" {
			continue
		}

		// If it's a bare extension name like "go", convert to glob
		pattern := raw
		if glob, ok := fileExtToGlob[strings.ToLower(raw)]; ok {
			pattern = "**/" + glob
		} else if !strings.Contains(raw, "*") && !strings.Contains(raw, "/") {
			pattern = "**/*." + raw
		}

		return []CompiledAction{{
			Tool:       "glob",
			Args:       map[string]string{"pattern": pattern},
			Confidence: 0.85,
			Source:     "glob-pattern",
		}}
	}
	return nil
}

func (ic *IntentCompiler) tryGitIntent(input string) []CompiledAction {
	for _, pat := range gitPatterns {
		m := pat.FindStringSubmatch(input)
		if m == nil {
			continue
		}
		cmd := "status"
		if len(m) >= 2 && strings.TrimSpace(m[1]) != "" {
			cmd = strings.TrimSpace(m[1])
		}

		// Map natural language to git subcommands
		lower := strings.ToLower(input)
		switch {
		case strings.Contains(lower, "status") || strings.Contains(lower, "changes") || strings.Contains(lower, "modified"):
			cmd = "status"
		case strings.Contains(lower, "log") || strings.Contains(lower, "history") || strings.Contains(lower, "commit"):
			cmd = "log --oneline -15"
		case strings.Contains(lower, "diff"):
			cmd = "diff"
		case strings.Contains(lower, "branch"):
			cmd = "branch -a"
		}

		return []CompiledAction{{
			Tool:       "git",
			Args:       map[string]string{"command": cmd},
			Confidence: 0.9,
			Source:     "git-pattern",
		}}
	}
	return nil
}

// --- Filesystem Grounding ---

// ResolvePath maps a potentially vague or partial path to a real file.
// Uses multiple strategies in order of confidence:
//  1. Exact match (relative to workDir)
//  2. Suffix match ("main.go" → "cmd/nous/main.go")
//  3. Basename match ("reasoner" → "internal/cognitive/reasoner.go")
//  4. Fuzzy match (Levenshtein-like partial matching)
func (ic *IntentCompiler) ResolvePath(fragment string) string {
	if fragment == "" {
		return ""
	}

	ic.mu.RLock()
	defer ic.mu.RUnlock()

	fragment = strings.TrimSpace(fragment)
	fragment = strings.Trim(fragment, `"'` + "`")

	// Strategy 1: Exact match
	if fileExistsAt(ic.workDir, fragment) {
		return fragment
	}

	// Strategy 2: Try as absolute path
	if strings.HasPrefix(fragment, "/") {
		if _, err := os.Stat(fragment); err == nil {
			return fragment
		}
	}

	// Strategy 3: Suffix match — "main.go" matches "cmd/nous/main.go"
	normalFrag := filepath.Clean(fragment)
	var suffixMatches []string
	for _, f := range ic.fileTree {
		if strings.HasSuffix(f, "/"+normalFrag) || f == normalFrag {
			suffixMatches = append(suffixMatches, f)
		}
	}
	if len(suffixMatches) == 1 {
		return suffixMatches[0]
	}

	// Strategy 4: Basename match — "reasoner" matches "internal/cognitive/reasoner.go"
	base := filepath.Base(normalFrag)
	baseNoExt := strings.TrimSuffix(base, filepath.Ext(base))
	var baseMatches []string
	for _, f := range ic.fileTree {
		fb := filepath.Base(f)
		fbNoExt := strings.TrimSuffix(fb, filepath.Ext(fb))
		if fb == base || fbNoExt == baseNoExt {
			baseMatches = append(baseMatches, f)
		}
	}
	if len(baseMatches) == 1 {
		return baseMatches[0]
	}

	// Strategy 5: Contains match — "cognitive/reasoner" matches the path
	if strings.Contains(normalFrag, "/") {
		for _, f := range ic.fileTree {
			if strings.Contains(f, normalFrag) {
				return f
			}
		}
	}

	// Strategy 6: Shortest suffix match if multiple found
	if len(suffixMatches) > 1 {
		sort.Slice(suffixMatches, func(i, j int) bool {
			return len(suffixMatches[i]) < len(suffixMatches[j])
		})
		return suffixMatches[0]
	}
	if len(baseMatches) > 1 {
		sort.Slice(baseMatches, func(i, j int) bool {
			return len(baseMatches[i]) < len(baseMatches[j])
		})
		return baseMatches[0]
	}

	return ""
}

// ResolveDir maps a path fragment to a real directory.
func (ic *IntentCompiler) ResolveDir(fragment string) string {
	if fragment == "" {
		return ""
	}

	ic.mu.RLock()
	defer ic.mu.RUnlock()

	fragment = strings.TrimSpace(fragment)
	fragment = strings.Trim(fragment, `"'` + "`")
	normalFrag := filepath.Clean(fragment)

	// Exact match
	for _, d := range ic.dirTree {
		if d == normalFrag {
			return d
		}
	}

	// Suffix match
	for _, d := range ic.dirTree {
		if strings.HasSuffix(d, "/"+normalFrag) || strings.HasSuffix(d, normalFrag) {
			return d
		}
	}

	// Contains match
	for _, d := range ic.dirTree {
		if strings.Contains(d, normalFrag) {
			return d
		}
	}

	// Also check if it resolves as a file's directory
	if resolved := ic.ResolvePath(fragment); resolved != "" {
		return filepath.Dir(resolved)
	}

	return ""
}

// RefreshFS scans the working directory to build the file/dir tree.
func (ic *IntentCompiler) RefreshFS() {
	ic.mu.Lock()
	defer ic.mu.Unlock()

	ic.fileTree = nil
	ic.dirTree = nil
	ic.lastScan = time.Now()

	if ic.workDir == "" {
		return
	}

	seen := make(map[string]bool)
	count := 0
	_ = filepath.Walk(ic.workDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // skip errors
		}
		if count >= ic.maxFiles {
			return filepath.SkipDir
		}

		// Skip hidden directories and common noise
		name := info.Name()
		if name != "." && strings.HasPrefix(name, ".") {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if isNoiseDir(name) && info.IsDir() {
			return filepath.SkipDir
		}

		rel, err := filepath.Rel(ic.workDir, path)
		if err != nil || rel == "." {
			return nil
		}

		// Check depth
		depth := strings.Count(rel, string(filepath.Separator))
		if depth > ic.maxDepth {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		if info.IsDir() {
			if !seen[rel] {
				ic.dirTree = append(ic.dirTree, rel)
				seen[rel] = true
			}
		} else {
			ic.fileTree = append(ic.fileTree, rel)
			count++
		}
		return nil
	})

	sort.Strings(ic.fileTree)
	sort.Strings(ic.dirTree)
}

// ensureFreshFS refreshes the filesystem cache if stale.
func (ic *IntentCompiler) ensureFreshFS() {
	ic.mu.RLock()
	stale := time.Since(ic.lastScan) > ic.scanTTL
	ic.mu.RUnlock()
	if stale {
		ic.RefreshFS()
	}
}

// FileTree returns the cached file list (for testing/debugging).
func (ic *IntentCompiler) FileTree() []string {
	ic.mu.RLock()
	defer ic.mu.RUnlock()
	out := make([]string, len(ic.fileTree))
	copy(out, ic.fileTree)
	return out
}

// DirTree returns the cached directory list (for testing/debugging).
func (ic *IntentCompiler) DirTree() []string {
	ic.mu.RLock()
	defer ic.mu.RUnlock()
	out := make([]string, len(ic.dirTree))
	copy(out, ic.dirTree)
	return out
}

func fileExistsAt(base, rel string) bool {
	path := filepath.Join(base, rel)
	_, err := os.Stat(path)
	return err == nil
}

func isNoiseDir(name string) bool {
	switch name {
	case "node_modules", "vendor", "__pycache__", ".git", "dist", "build", "target", ".cache":
		return true
	}
	return false
}
