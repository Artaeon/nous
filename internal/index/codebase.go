package index

import (
	"bufio"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/artaeon/nous/internal/safefile"
)

// Symbol represents a code symbol (function, type, method, etc).
type Symbol struct {
	Name      string `json:"name"`
	Kind      string `json:"kind"`      // "func", "type", "struct", "interface", "method", "const", "var"
	Package   string `json:"package"`
	File      string `json:"file"`      // relative path
	Line      int    `json:"line"`
	Signature string `json:"signature"` // e.g., "func NewReasoner(board *Blackboard, llm *Client, toolReg *Registry) *Reasoner"
	Doc       string `json:"doc"`       // first line of doc comment, if any
}

// FileInfo holds metadata about an indexed file.
type FileInfo struct {
	Path    string   `json:"path"`
	Hash    string   `json:"hash"` // SHA256 for change detection
	Package string   `json:"package"`
	Imports []string `json:"imports"`
	Lines   int      `json:"lines"`
}

// CodebaseIndex is a persistent structural index of the project.
type CodebaseIndex struct {
	mu      sync.RWMutex
	Symbols []Symbol   `json:"symbols"`
	Files   []FileInfo `json:"files"`
	path    string     // path to index.json
}

// NewCodebaseIndex creates or loads a codebase index.
// storePath is the directory where index.json will be stored.
func NewCodebaseIndex(storePath string) *CodebaseIndex {
	idx := &CodebaseIndex{
		path: filepath.Join(storePath, "index.json"),
	}
	// Attempt to load existing index; ignore errors (fresh start).
	_ = idx.Load()
	return idx
}

// Build walks source files under rootDir, parses Go files for symbols,
// and records basic file metadata for all recognized source files.
func (idx *CodebaseIndex) Build(rootDir string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.Symbols = nil
	idx.Files = nil

	return filepath.Walk(rootDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}

		name := info.Name()

		// Skip hidden dirs, vendor, node_modules, etc.
		if info.IsDir() {
			if strings.HasPrefix(name, ".") || name == "vendor" || name == "node_modules" || name == "testdata" {
				return filepath.SkipDir
			}
			return nil
		}

		rel, _ := filepath.Rel(rootDir, path)

		// Go files: full AST parsing for symbols (skip test files)
		if strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go") {
			if err := idx.indexFile(path, rel); err != nil {
				// Non-fatal: skip files that fail to parse
				return nil
			}
			return nil
		}

		// Other source files: record file metadata (for repo awareness)
		if isSourceFile(name) {
			hash, _ := fileHash(path)
			lines := 0
			if src, err := os.ReadFile(path); err == nil {
				lines = countLines(src)
			}
			idx.Files = append(idx.Files, FileInfo{
				Path:    rel,
				Hash:    hash,
				Package: langFromExt(name),
				Lines:   lines,
			})
		}

		return nil
	})
}

// isSourceFile checks if a filename is a recognized source/config file.
func isSourceFile(name string) bool {
	ext := strings.ToLower(filepath.Ext(name))
	switch ext {
	case ".py", ".js", ".ts", ".tsx", ".jsx", ".rb", ".rs", ".c", ".cpp", ".h",
		".java", ".kt", ".swift", ".cs", ".php", ".lua", ".sh", ".bash",
		".yaml", ".yml", ".toml", ".json", ".xml", ".html", ".css", ".scss",
		".md", ".txt", ".sql", ".proto", ".graphql", ".zig", ".nim", ".ex", ".exs":
		return true
	}
	// Dotfiles like Makefile, Dockerfile, etc.
	switch name {
	case "Makefile", "Dockerfile", "Containerfile", "Vagrantfile",
		"Rakefile", "Gemfile", "Procfile", ".gitignore", ".dockerignore":
		return true
	}
	return false
}

// langFromExt returns a language tag from a file extension.
func langFromExt(name string) string {
	ext := strings.ToLower(filepath.Ext(name))
	switch ext {
	case ".py":
		return "python"
	case ".js", ".jsx":
		return "javascript"
	case ".ts", ".tsx":
		return "typescript"
	case ".rb":
		return "ruby"
	case ".rs":
		return "rust"
	case ".c", ".h":
		return "c"
	case ".cpp":
		return "cpp"
	case ".java":
		return "java"
	case ".go":
		return "go"
	case ".sh", ".bash":
		return "shell"
	case ".yaml", ".yml":
		return "yaml"
	case ".json":
		return "json"
	case ".md":
		return "markdown"
	case ".sql":
		return "sql"
	default:
		return strings.TrimPrefix(ext, ".")
	}
}

// IncrementalUpdate re-indexes only files whose content hash has changed.
func (idx *CodebaseIndex) IncrementalUpdate(rootDir string, changedPaths []string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, relPath := range changedPaths {
		absPath := filepath.Join(rootDir, relPath)

		// Check if file still exists
		fi, err := os.Stat(absPath)
		if err != nil {
			// File deleted — remove from index
			idx.removeFile(relPath)
			continue
		}
		if fi.IsDir() {
			continue
		}

		// Check hash
		newHash, err := fileHash(absPath)
		if err != nil {
			continue
		}

		oldHash := ""
		for _, f := range idx.Files {
			if f.Path == relPath {
				oldHash = f.Hash
				break
			}
		}

		if newHash == oldHash {
			continue // unchanged
		}

		// Remove old entries for this file
		idx.removeFile(relPath)

		// Re-index
		_ = idx.indexFile(absPath, relPath)
	}

	return nil
}

// Lookup performs a case-insensitive substring match on symbol names, returns top 10.
func (idx *CodebaseIndex) Lookup(query string) []Symbol {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	query = strings.ToLower(query)
	var results []Symbol

	for _, s := range idx.Symbols {
		if strings.Contains(strings.ToLower(s.Name), query) {
			results = append(results, s)
			if len(results) >= 10 {
				break
			}
		}
	}

	return results
}

// FileContext returns a compact one-line summary of a file's symbols.
// Format: "package cognitive: NewReasoner, reason, callLLM, ..."
func (idx *CodebaseIndex) FileContext(relPath string) string {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	var pkg string
	var names []string

	for _, s := range idx.Symbols {
		if s.File == relPath {
			pkg = s.Package
			names = append(names, s.Name)
		}
	}

	if len(names) == 0 {
		return ""
	}

	return fmt.Sprintf("package %s: %s", pkg, strings.Join(names, ", "))
}

// RelevantContext finds symbols matching keywords in the query and returns
// a formatted block suitable for injection into an LLM system prompt.
func (idx *CodebaseIndex) RelevantContext(query string, maxSymbols int) string {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	words := strings.Fields(strings.ToLower(query))
	if len(words) == 0 {
		return ""
	}

	type scored struct {
		sym   Symbol
		score int
	}

	var matches []scored

	for _, s := range idx.Symbols {
		nameLower := strings.ToLower(s.Name)
		sigLower := strings.ToLower(s.Signature)
		score := 0
		for _, w := range words {
			if len(w) < 2 {
				continue
			}
			if strings.Contains(nameLower, w) {
				score += 3
			}
			if strings.Contains(sigLower, w) {
				score++
			}
		}
		if score > 0 {
			matches = append(matches, scored{sym: s, score: score})
		}
	}

	// Simple sort by score descending
	for i := 0; i < len(matches); i++ {
		for j := i + 1; j < len(matches); j++ {
			if matches[j].score > matches[i].score {
				matches[i], matches[j] = matches[j], matches[i]
			}
		}
	}

	if len(matches) > maxSymbols {
		matches = matches[:maxSymbols]
	}

	if len(matches) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("[Codebase]\n")
	for _, m := range matches {
		s := m.sym
		sig := s.Signature
		if sig == "" {
			sig = s.Kind + " " + s.Name
		}
		sb.WriteString(fmt.Sprintf("- %s [%s:%d]\n", sig, s.File, s.Line))
	}

	return sb.String()
}

// BestFileForQuery returns the file path most relevant to the query,
// based on symbol name and signature matching. Returns "" if no match.
func (idx *CodebaseIndex) BestFileForQuery(query string) string {
	sym := idx.BestSymbolForQuery(query)
	if sym == nil {
		return ""
	}
	return sym.File
}

// BestSymbolForQuery returns the single best-matching symbol for the query,
// including its file path and line number. Returns nil if no match.
func (idx *CodebaseIndex) BestSymbolForQuery(query string) *Symbol {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	words := strings.Fields(strings.ToLower(query))
	if len(words) == 0 {
		return nil
	}

	var bestSym *Symbol
	bestScore := 0

	for i := range idx.Symbols {
		s := &idx.Symbols[i]
		nameLower := strings.ToLower(s.Name)
		sigLower := strings.ToLower(s.Signature)
		score := 0
		for _, w := range words {
			if len(w) < 2 {
				continue
			}
			// Exact name match is strongest signal
			if nameLower == w {
				score += 10
			} else if strings.Contains(nameLower, w) {
				score += 3
			}
			if strings.Contains(sigLower, w) {
				score++
			}
		}
		if score > bestScore {
			bestScore = score
			bestSym = s
		}
	}

	return bestSym
}

// Save marshals the index to JSON and writes it to disk.
func (idx *CodebaseIndex) Save() error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	dir := filepath.Dir(idx.path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	data, err := json.MarshalIndent(struct {
		Symbols []Symbol   `json:"symbols"`
		Files   []FileInfo `json:"files"`
	}{
		Symbols: idx.Symbols,
		Files:   idx.Files,
	}, "", "  ")
	if err != nil {
		return err
	}

	return safefile.WriteAtomic(idx.path, data, 0644)
}

// Load reads the index from disk.
func (idx *CodebaseIndex) Load() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	data, err := os.ReadFile(idx.path)
	if err != nil {
		return err
	}

	var stored struct {
		Symbols []Symbol   `json:"symbols"`
		Files   []FileInfo `json:"files"`
	}
	if err := json.Unmarshal(data, &stored); err != nil {
		return err
	}

	idx.Symbols = stored.Symbols
	idx.Files = stored.Files
	return nil
}

// Size returns the number of symbols in the index.
func (idx *CodebaseIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.Symbols)
}

// FileSummary returns a compact overview of the indexed files grouped by language.
// Format: "go: 12 files (3,400 lines), python: 3 files (450 lines)"
func (idx *CodebaseIndex) FileSummary() string {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	type langStats struct {
		files int
		lines int
	}
	stats := make(map[string]*langStats)

	for _, f := range idx.Files {
		lang := f.Package
		if lang == "" {
			lang = "other"
		}
		s, ok := stats[lang]
		if !ok {
			s = &langStats{}
			stats[lang] = s
		}
		s.files++
		s.lines += f.Lines
	}

	if len(stats) == 0 {
		return ""
	}

	var parts []string
	for lang, s := range stats {
		parts = append(parts, fmt.Sprintf("%s: %d files (%d lines)", lang, s.files, s.lines))
	}
	return strings.Join(parts, ", ")
}

// indexFile parses a single Go file and adds its symbols and metadata to the index.
func (idx *CodebaseIndex) indexFile(absPath, relPath string) error {
	src, err := os.ReadFile(absPath)
	if err != nil {
		return err
	}

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, absPath, src, parser.ParseComments)
	if err != nil {
		return err
	}

	pkgName := file.Name.Name

	// File hash
	hash := fmt.Sprintf("%x", sha256.Sum256(src))

	// Count lines
	lines := countLines(src)

	// Extract imports
	var imports []string
	for _, imp := range file.Imports {
		path := imp.Path.Value
		path = strings.Trim(path, `"`)
		imports = append(imports, path)
	}

	idx.Files = append(idx.Files, FileInfo{
		Path:    relPath,
		Hash:    hash,
		Package: pkgName,
		Imports: imports,
		Lines:   lines,
	})

	// Source lines for signature extraction
	srcLines := strings.Split(string(src), "\n")

	// Walk declarations
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			sym := Symbol{
				Name:    d.Name.Name,
				Kind:    "func",
				Package: pkgName,
				File:    relPath,
				Line:    fset.Position(d.Pos()).Line,
			}

			// Method with receiver
			if d.Recv != nil && len(d.Recv.List) > 0 {
				sym.Kind = "method"
			}

			// Extract signature from source line
			sym.Signature = extractSignature(srcLines, sym.Line)

			// Doc comment
			if d.Doc != nil {
				sym.Doc = firstLine(d.Doc.Text())
			}

			idx.Symbols = append(idx.Symbols, sym)

		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					sym := Symbol{
						Name:    s.Name.Name,
						Kind:    "type",
						Package: pkgName,
						File:    relPath,
						Line:    fset.Position(s.Pos()).Line,
					}

					// Determine concrete kind
					switch s.Type.(type) {
					case *ast.StructType:
						sym.Kind = "struct"
						sym.Signature = fmt.Sprintf("type %s struct", s.Name.Name)
					case *ast.InterfaceType:
						sym.Kind = "interface"
						sym.Signature = fmt.Sprintf("type %s interface", s.Name.Name)
					default:
						sym.Signature = fmt.Sprintf("type %s", s.Name.Name)
					}

					// Doc comment (from GenDecl or TypeSpec)
					if s.Doc != nil {
						sym.Doc = firstLine(s.Doc.Text())
					} else if d.Doc != nil {
						sym.Doc = firstLine(d.Doc.Text())
					}

					idx.Symbols = append(idx.Symbols, sym)

				case *ast.ValueSpec:
					kind := "var"
					if d.Tok == token.CONST {
						kind = "const"
					}
					for _, name := range s.Names {
						if name.Name == "_" {
							continue
						}
						sym := Symbol{
							Name:    name.Name,
							Kind:    kind,
							Package: pkgName,
							File:    relPath,
							Line:    fset.Position(name.Pos()).Line,
						}
						sym.Signature = fmt.Sprintf("%s %s", kind, name.Name)

						if s.Doc != nil {
							sym.Doc = firstLine(s.Doc.Text())
						} else if d.Doc != nil {
							sym.Doc = firstLine(d.Doc.Text())
						}

						idx.Symbols = append(idx.Symbols, sym)
					}
				}
			}
		}
	}

	return nil
}

// removeFile removes all symbols and file entries for a given relative path.
func (idx *CodebaseIndex) removeFile(relPath string) {
	// Remove symbols
	filtered := idx.Symbols[:0]
	for _, s := range idx.Symbols {
		if s.File != relPath {
			filtered = append(filtered, s)
		}
	}
	idx.Symbols = filtered

	// Remove file info
	filteredFiles := idx.Files[:0]
	for _, f := range idx.Files {
		if f.Path != relPath {
			filteredFiles = append(filteredFiles, f)
		}
	}
	idx.Files = filteredFiles
}

// extractSignature reads the function signature from the source line.
func extractSignature(lines []string, lineNum int) string {
	if lineNum < 1 || lineNum > len(lines) {
		return ""
	}

	// Get the line (1-indexed)
	line := lines[lineNum-1]
	sig := strings.TrimSpace(line)

	// If the signature spans multiple lines (opening paren but no closing),
	// concatenate subsequent lines until we find the closing paren or opening brace.
	if strings.Contains(sig, "(") && !strings.Contains(sig, ")") {
		for i := lineNum; i < len(lines) && i < lineNum+10; i++ {
			next := strings.TrimSpace(lines[i])
			sig += " " + next
			if strings.Contains(next, ")") || strings.Contains(next, "{") {
				break
			}
		}
	}

	// Trim the function body opening brace
	if idx := strings.Index(sig, "{"); idx >= 0 {
		sig = strings.TrimSpace(sig[:idx])
	}

	return sig
}

// fileHash computes SHA256 of a file's contents.
func fileHash(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", sha256.Sum256(data)), nil
}

// countLines counts the number of newlines in the source.
func countLines(src []byte) int {
	n := 0
	scanner := bufio.NewScanner(strings.NewReader(string(src)))
	for scanner.Scan() {
		n++
	}
	return n
}

// firstLine returns the first line of text, trimmed.
func firstLine(text string) string {
	text = strings.TrimSpace(text)
	if idx := strings.IndexByte(text, '\n'); idx >= 0 {
		return strings.TrimSpace(text[:idx])
	}
	return text
}
