package cognitive

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

// DepGraph maps function call relationships across a Go codebase.
// It maintains forward (callees) and reverse (callers) maps so
// that both "who uses this?" and "what does this depend on?" can
// be answered in O(1).
type DepGraph struct {
	Callers  map[string][]string // function → functions that call it
	Callees  map[string][]string // function → functions it calls
	Packages map[string][]string // package → its functions
}

// NewDepGraph creates a new empty DepGraph.
func NewDepGraph() *DepGraph {
	return &DepGraph{
		Callers:  make(map[string][]string),
		Callees:  make(map[string][]string),
		Packages: make(map[string][]string),
	}
}

// Build walks all Go files under dir, parses their AST, and
// populates the caller/callee/package maps.
func (dg *DepGraph) Build(dir string) error {
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() {
			name := info.Name()
			if strings.HasPrefix(name, ".") || name == "vendor" || name == "testdata" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(info.Name(), ".go") {
			return nil
		}
		if strings.HasSuffix(info.Name(), "_test.go") {
			return nil
		}
		return dg.parseFile(path)
	})
	if err != nil {
		return err
	}
	// Normalize: replace unqualified callee names with their
	// qualified form when an exact match exists in the package list.
	dg.normalize()
	return nil
}

// parseFile parses a single Go file and records all call relationships.
func (dg *DepGraph) parseFile(path string) error {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil {
		return nil // skip unparseable files
	}

	pkgName := f.Name.Name

	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}

		callerName := qualifiedFuncName(pkgName, fn)

		// Register function in its package.
		dg.Packages[pkgName] = appendUniqueStr(dg.Packages[pkgName], callerName)

		// Walk the function body for call expressions.
		if fn.Body == nil {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			calleeName := callExprName(call)
			if calleeName == "" {
				return true
			}

			dg.Callees[callerName] = appendUniqueStr(dg.Callees[callerName], calleeName)
			dg.Callers[calleeName] = appendUniqueStr(dg.Callers[calleeName], callerName)
			return true
		})
	}
	return nil
}

// normalize replaces unqualified keys in Callers and callee
// entries in Callees with their qualified form from Packages.
func (dg *DepGraph) normalize() {
	// Build lookup: unqualified name → qualified name.
	qualified := make(map[string]string)
	for _, fns := range dg.Packages {
		for _, fn := range fns {
			short := unqualified(fn)
			qualified[short] = fn
		}
	}

	// Normalize Callees: replace unqualified callee values.
	for caller, callees := range dg.Callees {
		for i, c := range callees {
			if q, ok := qualified[c]; ok {
				callees[i] = q
			}
		}
		dg.Callees[caller] = callees
	}

	// Rebuild Callers from the normalized Callees.
	dg.Callers = make(map[string][]string)
	for caller, callees := range dg.Callees {
		for _, callee := range callees {
			dg.Callers[callee] = appendUniqueStr(dg.Callers[callee], caller)
		}
	}
}

// WhoCalls returns all functions that call funcName.
// This answers "who uses this?"
func (dg *DepGraph) WhoCalls(funcName string) []string {
	// Try exact match first.
	if callers, ok := dg.Callers[funcName]; ok {
		return callers
	}
	// Try unqualified match.
	for key, callers := range dg.Callers {
		if unqualified(key) == funcName {
			return callers
		}
	}
	return nil
}

// WhatCalls returns all functions called by funcName.
// This answers "what does this depend on?"
func (dg *DepGraph) WhatCalls(funcName string) []string {
	// Try exact match first.
	if callees, ok := dg.Callees[funcName]; ok {
		return callees
	}
	// Try unqualified match.
	for key, callees := range dg.Callees {
		if unqualified(key) == funcName {
			return callees
		}
	}
	return nil
}

// Impact returns the transitive callers of funcName — every
// function that would be affected if funcName's behavior changed.
// Uses BFS up the caller graph.
func (dg *DepGraph) Impact(funcName string) []string {
	// Resolve the canonical key.
	canonical := dg.resolve(funcName)
	if canonical == "" {
		return nil
	}

	visited := map[string]bool{canonical: true}
	queue := []string{canonical}
	var result []string

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		callers := dg.Callers[current]
		for _, c := range callers {
			if visited[c] {
				continue
			}
			visited[c] = true
			result = append(result, c)
			queue = append(queue, c)
		}
	}
	return result
}

// Render produces an ASCII visualization of a function's call
// relationships up to the given depth.
func (dg *DepGraph) Render(funcName string, depth int) string {
	canonical := dg.resolve(funcName)
	if canonical == "" {
		return funcName + "\n  (not found in dependency graph)\n"
	}

	var b strings.Builder
	b.WriteString(canonical + "\n")

	// Callees (what this function calls).
	callees := dg.Callees[canonical]
	if len(callees) > 0 {
		b.WriteString("  ├─ calls: " + strings.Join(shortNames(callees), ", ") + "\n")
	}

	// Callers (who calls this function).
	callers := dg.Callers[canonical]
	if len(callers) > 0 {
		b.WriteString("  └─ called by: " + strings.Join(shortNames(callers), ", ") + "\n")

		// Show impact chain if depth > 1.
		if depth > 1 {
			impact := dg.Impact(canonical)
			if len(impact) > 0 {
				b.WriteString("      └─ impact: " + strings.Join(shortNames(impact), " → ") + "\n")
			}
		}
	}

	if len(callees) == 0 && len(callers) == 0 {
		b.WriteString("  (no recorded call relationships)\n")
	}

	return b.String()
}

// resolve finds the canonical (qualified) key for a function name.
func (dg *DepGraph) resolve(funcName string) string {
	if _, ok := dg.Callees[funcName]; ok {
		return funcName
	}
	if _, ok := dg.Callers[funcName]; ok {
		return funcName
	}
	// Scan all keys for an unqualified match.
	for key := range dg.Callees {
		if unqualified(key) == funcName {
			return key
		}
	}
	for key := range dg.Callers {
		if unqualified(key) == funcName {
			return key
		}
	}
	// Check package lists.
	for _, fns := range dg.Packages {
		for _, fn := range fns {
			if unqualified(fn) == funcName {
				return fn
			}
		}
	}
	return ""
}

// qualifiedFuncName builds "pkg.FuncName" or "pkg.Type.Method".
func qualifiedFuncName(pkg string, fn *ast.FuncDecl) string {
	name := fn.Name.Name
	if fn.Recv != nil && len(fn.Recv.List) > 0 {
		recvType := exprString(fn.Recv.List[0].Type)
		recvType = strings.TrimPrefix(recvType, "*")
		return fmt.Sprintf("%s.%s.%s", pkg, recvType, name)
	}
	return fmt.Sprintf("%s.%s", pkg, name)
}

// callExprName extracts a human-readable name from a call expression.
func callExprName(call *ast.CallExpr) string {
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		return fun.Name
	case *ast.SelectorExpr:
		x := ""
		if ident, ok := fun.X.(*ast.Ident); ok {
			x = ident.Name
		}
		if x != "" {
			return x + "." + fun.Sel.Name
		}
		return fun.Sel.Name
	default:
		return ""
	}
}

// unqualified returns the last component: "pkg.Foo" → "Foo",
// "pkg.Type.Method" → "Method".
func unqualified(name string) string {
	if idx := strings.LastIndex(name, "."); idx >= 0 {
		return name[idx+1:]
	}
	return name
}

// shortNames strips package prefixes for display.
func shortNames(names []string) []string {
	out := make([]string, len(names))
	for i, n := range names {
		out[i] = n
	}
	return out
}

// appendUniqueStr appends s to slice only if not already present.
func appendUniqueStr(slice []string, s string) []string {
	for _, existing := range slice {
		if existing == s {
			return slice
		}
	}
	return append(slice, s)
}
