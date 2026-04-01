package cognitive

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

// AutoDocGenerator produces Go doc comments from code analysis.
// It parses Go source files, inspects function signatures and body
// patterns, and generates idiomatic doc comments for exported
// functions that lack them.
type AutoDocGenerator struct{}

// NewAutoDocGenerator creates a new AutoDocGenerator.
func NewAutoDocGenerator() *AutoDocGenerator {
	return &AutoDocGenerator{}
}

// GenerateDoc parses the file at path, finds the function named
// funcName, and returns a Go doc comment string. Returns an empty
// string if the function is not found or already has a doc comment.
func (g *AutoDocGenerator) GenerateDoc(path string, funcName string) string {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
	if err != nil {
		return ""
	}

	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		if fn.Name.Name != funcName {
			continue
		}
		// Skip if already documented.
		if fn.Doc != nil && fn.Doc.Text() != "" {
			return ""
		}
		return g.docForFunc(fn)
	}
	return ""
}

// GenerateDocForFile generates doc comments for all undocumented
// exported functions in the given file and returns a unified diff
// that can be applied with patch(1).
func (g *AutoDocGenerator) GenerateDocForFile(path string) string {
	fset := token.NewFileSet()
	src, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	f, err := parser.ParseFile(fset, path, src, parser.ParseComments)
	if err != nil {
		return ""
	}

	type insertion struct {
		line int    // 1-based line where the comment should be inserted before
		text string // the doc comment (with leading //)
	}
	var inserts []insertion

	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		if !fn.Name.IsExported() {
			continue
		}
		if fn.Doc != nil && fn.Doc.Text() != "" {
			continue
		}
		doc := g.docForFunc(fn)
		if doc == "" {
			continue
		}
		line := fset.Position(fn.Pos()).Line
		inserts = append(inserts, insertion{line: line, text: doc})
	}

	if len(inserts) == 0 {
		return ""
	}

	lines := strings.Split(string(src), "\n")
	var diff strings.Builder
	diff.WriteString(fmt.Sprintf("--- %s\n", path))
	diff.WriteString(fmt.Sprintf("+++ %s\n", path))

	for _, ins := range inserts {
		lineIdx := ins.line - 1 // 0-based
		ctx := ""
		if lineIdx < len(lines) {
			ctx = lines[lineIdx]
		}
		diff.WriteString(fmt.Sprintf("@@ -%d,1 +%d,%d @@\n", ins.line, ins.line, 1+strings.Count(ins.text, "\n")+1))
		for _, cl := range strings.Split(ins.text, "\n") {
			diff.WriteString("+" + cl + "\n")
		}
		diff.WriteString(" " + ctx + "\n")
	}

	return diff.String()
}

// DocumentPackage generates and writes doc comments for all
// undocumented exported functions in every Go file under dir.
// Returns the number of functions documented.
func (g *AutoDocGenerator) DocumentPackage(dir string) (int, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, fmt.Errorf("read dir: %w", err)
	}

	total := 0
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if !strings.HasSuffix(e.Name(), ".go") {
			continue
		}
		if strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		path := filepath.Join(dir, e.Name())
		n, err := g.documentFile(path)
		if err != nil {
			return total, fmt.Errorf("document %s: %w", e.Name(), err)
		}
		total += n
	}
	return total, nil
}

// documentFile inserts doc comments into a single file and writes
// the result back. Returns the number of functions documented.
func (g *AutoDocGenerator) documentFile(path string) (int, error) {
	fset := token.NewFileSet()
	src, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	f, err := parser.ParseFile(fset, path, src, parser.ParseComments)
	if err != nil {
		return 0, err
	}

	type insertion struct {
		line int
		text string
	}
	var inserts []insertion

	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		if !fn.Name.IsExported() {
			continue
		}
		if fn.Doc != nil && fn.Doc.Text() != "" {
			continue
		}
		doc := g.docForFunc(fn)
		if doc == "" {
			continue
		}
		line := fset.Position(fn.Pos()).Line
		inserts = append(inserts, insertion{line: line, text: doc})
	}

	if len(inserts) == 0 {
		return 0, nil
	}

	lines := strings.Split(string(src), "\n")
	// Insert in reverse order so line numbers stay valid.
	for i := len(inserts) - 1; i >= 0; i-- {
		ins := inserts[i]
		idx := ins.line - 1 // 0-based
		docLines := strings.Split(ins.text, "\n")
		// Build new slice: lines[:idx] + docLines + lines[idx:]
		after := make([]string, len(lines[idx:]))
		copy(after, lines[idx:])
		lines = append(lines[:idx], append(docLines, after...)...)
	}

	result := strings.Join(lines, "\n")
	if err := os.WriteFile(path, []byte(result), 0644); err != nil {
		return 0, err
	}
	return len(inserts), nil
}

// docForFunc builds a doc comment string for the given function declaration.
func (g *AutoDocGenerator) docForFunc(fn *ast.FuncDecl) string {
	name := fn.Name.Name
	if name == "" {
		return ""
	}

	var parts []string

	// Determine the opening phrase from the function name prefix.
	phrase := g.openingPhrase(fn)
	parts = append(parts, fmt.Sprintf("// %s %s", name, phrase))

	// Document return characteristics.
	if fn.Type.Results != nil {
		retDoc := g.returnDoc(fn)
		if retDoc != "" {
			parts = append(parts, "// "+retDoc)
		}
	}

	return strings.Join(parts, "\n")
}

// verbPrefixes maps Go function name prefixes to doc verb phrases.
var verbPrefixes = []struct {
	prefix string
	verb   string
}{
	{"New", "creates a new"},
	{"Is", "reports whether"},
	{"Has", "reports whether"},
	{"Get", "returns the"},
	{"Set", "sets the"},
	{"Load", "loads"},
	{"Parse", "parses"},
	{"Handle", "handles"},
	{"Write", "writes"},
	{"Read", "reads"},
	{"Delete", "removes"},
	{"Remove", "removes"},
	{"Update", "updates"},
	{"Find", "searches for"},
	{"Search", "searches for"},
	{"Validate", "validates"},
	{"Convert", "converts"},
	{"Init", "initializes"},
}

// openingPhrase generates the verb phrase that follows the function
// name in the doc comment.
func (g *AutoDocGenerator) openingPhrase(fn *ast.FuncDecl) string {
	name := fn.Name.Name

	// Constructor: NewFoo → "creates a new Foo with the given [params]."
	if strings.HasPrefix(name, "New") && len(name) > 3 {
		typeName := name[3:]
		params := g.paramNames(fn)
		if len(params) > 0 {
			return fmt.Sprintf("creates a new %s with the given %s.", typeName, strings.Join(params, ", "))
		}
		return fmt.Sprintf("creates a new %s.", typeName)
	}

	// Match known prefixes.
	for _, vp := range verbPrefixes {
		if strings.HasPrefix(name, vp.prefix) {
			rest := name[len(vp.prefix):]
			if rest == "" {
				params := g.paramNames(fn)
				if len(params) > 0 {
					return fmt.Sprintf("%s the given %s.", vp.verb, strings.Join(params, ", "))
				}
				return vp.verb + "."
			}
			subject := splitCamelCase(rest)
			params := g.paramNames(fn)
			if len(params) > 0 {
				return fmt.Sprintf("%s %s from the given %s.", vp.verb, subject, strings.Join(params, ", "))
			}
			return fmt.Sprintf("%s %s.", vp.verb, subject)
		}
	}

	// Receiver method: try to describe via "verb + rest".
	subject := splitCamelCase(name)
	params := g.paramNames(fn)
	if len(params) > 0 {
		return fmt.Sprintf("%s using the given %s.", lowercaseVerb(subject), strings.Join(params, ", "))
	}
	return lowercaseVerb(subject) + "."
}

// returnDoc generates a documentation line about the return values.
func (g *AutoDocGenerator) returnDoc(fn *ast.FuncDecl) string {
	if fn.Type.Results == nil {
		return ""
	}
	results := fn.Type.Results.List
	if len(results) == 0 {
		return ""
	}

	hasError := false
	hasBool := false
	hasPointer := false

	for _, r := range results {
		typeName := exprString(r.Type)
		if typeName == "error" {
			hasError = true
		}
		if typeName == "bool" {
			hasBool = true
		}
		if strings.HasPrefix(typeName, "*") {
			hasPointer = true
		}
	}

	var docs []string
	if hasError {
		op := lowercaseVerb(splitCamelCase(fn.Name.Name))
		docs = append(docs, fmt.Sprintf("Returns an error if %s fails.", op))
	}
	if hasBool && !hasError {
		// Already handled in opening phrase for Is/Has.
		if !strings.HasPrefix(fn.Name.Name, "Is") && !strings.HasPrefix(fn.Name.Name, "Has") {
			docs = append(docs, "Returns true on success.")
		}
	}
	if hasPointer {
		docs = append(docs, "Returns nil if not found.")
	}

	return strings.Join(docs, " ")
}

// paramNames returns human-readable names for the function's parameters.
func (g *AutoDocGenerator) paramNames(fn *ast.FuncDecl) []string {
	if fn.Type.Params == nil {
		return nil
	}
	var names []string
	for _, field := range fn.Type.Params.List {
		for _, ident := range field.Names {
			n := ident.Name
			if n == "_" || n == "" {
				continue
			}
			names = append(names, n)
		}
	}
	return names
}

// exprString returns a simple string representation of an AST type expression.
func exprString(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return "*" + exprString(t.X)
	case *ast.SelectorExpr:
		return exprString(t.X) + "." + t.Sel.Name
	case *ast.ArrayType:
		return "[]" + exprString(t.Elt)
	case *ast.MapType:
		return "map[" + exprString(t.Key) + "]" + exprString(t.Value)
	case *ast.InterfaceType:
		return "interface{}"
	case *ast.FuncType:
		return "func"
	case *ast.Ellipsis:
		return "..." + exprString(t.Elt)
	default:
		return "unknown"
	}
}

// splitCamelCase converts "ValidateToken" to "validate token".
func splitCamelCase(s string) string {
	if s == "" {
		return ""
	}
	var words []string
	current := strings.Builder{}
	for i, r := range s {
		if unicode.IsUpper(r) && i > 0 {
			if current.Len() > 0 {
				words = append(words, strings.ToLower(current.String()))
				current.Reset()
			}
		}
		current.WriteRune(r)
	}
	if current.Len() > 0 {
		words = append(words, strings.ToLower(current.String()))
	}
	return strings.Join(words, " ")
}

// lowercaseVerb lowercases the first letter of a phrase without
// affecting the rest. This turns "Validate token" into "validate token".
func lowercaseVerb(s string) string {
	if s == "" {
		return ""
	}
	runes := []rune(s)
	runes[0] = unicode.ToLower(runes[0])
	return string(runes)
}
