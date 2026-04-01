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

// CodeReviewer performs static analysis to find common code issues.
type CodeReviewer struct{}

// NewCodeReviewer creates a code reviewer.
func NewCodeReviewer() *CodeReviewer {
	return &CodeReviewer{}
}

// ReviewResult represents a single finding.
type ReviewResult struct {
	File     string `json:"file"`
	Line     int    `json:"line"`
	Severity string `json:"severity"` // "error", "warning", "info"
	Rule     string `json:"rule"`
	Message  string `json:"message"`
}

// ReviewFile analyzes a single Go file and returns findings.
func (cr *CodeReviewer) ReviewFile(path string) ([]ReviewResult, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	var results []ReviewResult

	results = append(results, cr.checkUncheckedErrors(fset, file, path)...)
	results = append(results, cr.checkUnusedParameters(fset, file, path)...)
	results = append(results, cr.checkEmptyErrorHandlers(fset, file, path)...)
	results = append(results, cr.checkHardcodedSecrets(fset, file, path)...)
	results = append(results, cr.checkTodoComments(fset, file, path)...)
	results = append(results, cr.checkLongFunctions(fset, file, path)...)
	results = append(results, cr.checkDeepNesting(fset, file, path)...)
	results = append(results, cr.checkMissingDoc(fset, file, path)...)

	return results, nil
}

// ReviewDir analyzes all Go files in a directory.
func (cr *CodeReviewer) ReviewDir(dir string) ([]ReviewResult, error) {
	var allResults []ReviewResult

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read dir: %w", err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		path := filepath.Join(dir, name)
		results, err := cr.ReviewFile(path)
		if err != nil {
			continue // skip files that fail to parse
		}
		allResults = append(allResults, results...)
	}

	return allResults, nil
}

// checkUncheckedErrors finds function calls that return error but the error
// value is assigned to _ or ignored entirely.
func (cr *CodeReviewer) checkUncheckedErrors(fset *token.FileSet, file *ast.File, path string) []ReviewResult {
	var results []ReviewResult

	ast.Inspect(file, func(n ast.Node) bool {
		assign, ok := n.(*ast.AssignStmt)
		if !ok {
			return true
		}

		// Check if any RHS is a call expression
		for _, rhs := range assign.Rhs {
			call, ok := rhs.(*ast.CallExpr)
			if !ok {
				continue
			}

			// Check if last LHS is blank identifier
			if len(assign.Lhs) > 0 {
				last := assign.Lhs[len(assign.Lhs)-1]
				ident, ok := last.(*ast.Ident)
				if ok && ident.Name == "_" {
					// Heuristic: if the function name suggests it returns error
					funcName := cr.callName(call)
					if cr.likelyReturnsError(funcName) {
						results = append(results, ReviewResult{
							File:     path,
							Line:     fset.Position(assign.Pos()).Line,
							Severity: "warning",
							Rule:     "unchecked-error",
							Message:  fmt.Sprintf("Error return from %s is discarded", funcName),
						})
					}
				}
			}
		}

		return true
	})

	return results
}

// likelyReturnsError returns true if the function name suggests it returns an error.
func (cr *CodeReviewer) likelyReturnsError(name string) bool {
	lower := strings.ToLower(name)
	// Common patterns that return errors
	errorFuncs := []string{
		"open", "read", "write", "close", "create", "remove", "stat",
		"parse", "marshal", "unmarshal", "decode", "encode",
		"dial", "listen", "accept", "connect",
		"exec", "run", "start",
		"get", "post", "put", "delete", "do",
		"scan", "query", "prepare",
		"save", "load", "flush",
	}
	for _, f := range errorFuncs {
		if strings.Contains(lower, f) {
			return true
		}
	}
	return false
}

// callName extracts a readable name from a call expression.
func (cr *CodeReviewer) callName(call *ast.CallExpr) string {
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		return fun.Name
	case *ast.SelectorExpr:
		if x, ok := fun.X.(*ast.Ident); ok {
			return x.Name + "." + fun.Sel.Name
		}
		return fun.Sel.Name
	}
	return "unknown"
}

// checkUnusedParameters finds function parameters that are never referenced
// in the function body.
func (cr *CodeReviewer) checkUnusedParameters(fset *token.FileSet, file *ast.File, path string) []ReviewResult {
	var results []ReviewResult

	for _, decl := range file.Decls {
		fd, ok := decl.(*ast.FuncDecl)
		if !ok || fd.Body == nil || fd.Type.Params == nil {
			continue
		}

		// Collect parameter names
		for _, field := range fd.Type.Params.List {
			for _, name := range field.Names {
				if name.Name == "_" {
					continue
				}
				if !cr.identUsedInBlock(name.Name, fd.Body) {
					results = append(results, ReviewResult{
						File:     path,
						Line:     fset.Position(name.Pos()).Line,
						Severity: "warning",
						Rule:     "unused-parameter",
						Message:  fmt.Sprintf("Parameter %q in function %s is never used", name.Name, fd.Name.Name),
					})
				}
			}
		}
	}

	return results
}

// identUsedInBlock checks if an identifier name appears in a block statement,
// excluding the parameter list itself.
func (cr *CodeReviewer) identUsedInBlock(name string, block *ast.BlockStmt) bool {
	found := false
	ast.Inspect(block, func(n ast.Node) bool {
		if found {
			return false
		}
		ident, ok := n.(*ast.Ident)
		if ok && ident.Name == name {
			found = true
			return false
		}
		return true
	})
	return found
}

// checkEmptyErrorHandlers finds `if err != nil { }` blocks with empty or
// trivially empty bodies (only a bare return).
func (cr *CodeReviewer) checkEmptyErrorHandlers(fset *token.FileSet, file *ast.File, path string) []ReviewResult {
	var results []ReviewResult

	ast.Inspect(file, func(n ast.Node) bool {
		ifStmt, ok := n.(*ast.IfStmt)
		if !ok {
			return true
		}

		// Check if condition is `err != nil`
		bin, ok := ifStmt.Cond.(*ast.BinaryExpr)
		if !ok || bin.Op != token.NEQ {
			return true
		}
		xIdent, ok := bin.X.(*ast.Ident)
		if !ok || xIdent.Name != "err" {
			return true
		}
		yIdent, ok := bin.Y.(*ast.Ident)
		if !ok || yIdent.Name != "nil" {
			return true
		}

		// Check if body is empty or just a bare return
		body := ifStmt.Body
		if body == nil || len(body.List) == 0 {
			results = append(results, ReviewResult{
				File:     path,
				Line:     fset.Position(ifStmt.Pos()).Line,
				Severity: "warning",
				Rule:     "empty-error-handler",
				Message:  "Error is checked but not handled (empty block)",
			})
		} else if len(body.List) == 1 {
			if ret, ok := body.List[0].(*ast.ReturnStmt); ok && len(ret.Results) == 0 {
				results = append(results, ReviewResult{
					File:     path,
					Line:     fset.Position(ifStmt.Pos()).Line,
					Severity: "warning",
					Rule:     "empty-error-handler",
					Message:  "Error is checked but handler only does a bare return",
				})
			}
		}

		return true
	})

	return results
}

// checkHardcodedSecrets finds string literal assignments to variables whose
// names suggest they hold secrets (password, secret, apiKey, token, etc).
func (cr *CodeReviewer) checkHardcodedSecrets(fset *token.FileSet, file *ast.File, path string) []ReviewResult {
	var results []ReviewResult

	secretNames := []string{"password", "secret", "apikey", "api_key", "token", "private_key", "privatekey"}

	ast.Inspect(file, func(n ast.Node) bool {
		assign, ok := n.(*ast.AssignStmt)
		if !ok {
			return true
		}

		for i, lhs := range assign.Lhs {
			ident, ok := lhs.(*ast.Ident)
			if !ok {
				continue
			}
			nameLower := strings.ToLower(ident.Name)

			isSecret := false
			for _, s := range secretNames {
				if strings.Contains(nameLower, s) {
					isSecret = true
					break
				}
			}
			if !isSecret {
				continue
			}

			// Check if RHS at this index is a string literal
			if i < len(assign.Rhs) {
				if lit, ok := assign.Rhs[i].(*ast.BasicLit); ok && lit.Kind == token.STRING {
					val := strings.Trim(lit.Value, `"` + "`")
					if len(val) > 0 {
						results = append(results, ReviewResult{
							File:     path,
							Line:     fset.Position(assign.Pos()).Line,
							Severity: "error",
							Rule:     "hardcoded-secret",
							Message:  fmt.Sprintf("Hardcoded secret in variable %q", ident.Name),
						})
					}
				}
			}
		}

		return true
	})

	// Also check ValueSpec (var/const declarations)
	ast.Inspect(file, func(n ast.Node) bool {
		vs, ok := n.(*ast.ValueSpec)
		if !ok {
			return true
		}

		for i, name := range vs.Names {
			nameLower := strings.ToLower(name.Name)
			isSecret := false
			for _, s := range secretNames {
				if strings.Contains(nameLower, s) {
					isSecret = true
					break
				}
			}
			if !isSecret {
				continue
			}
			if i < len(vs.Values) {
				if lit, ok := vs.Values[i].(*ast.BasicLit); ok && lit.Kind == token.STRING {
					val := strings.Trim(lit.Value, `"` + "`")
					if len(val) > 0 {
						results = append(results, ReviewResult{
							File:     path,
							Line:     fset.Position(name.Pos()).Line,
							Severity: "error",
							Rule:     "hardcoded-secret",
							Message:  fmt.Sprintf("Hardcoded secret in variable %q", name.Name),
						})
					}
				}
			}
		}

		return true
	})

	return results
}

// checkTodoComments finds TODO, FIXME, HACK, and XXX comments.
func (cr *CodeReviewer) checkTodoComments(fset *token.FileSet, file *ast.File, path string) []ReviewResult {
	var results []ReviewResult

	markers := []string{"TODO", "FIXME", "HACK", "XXX"}

	for _, cg := range file.Comments {
		for _, c := range cg.List {
			text := c.Text
			upper := strings.ToUpper(text)
			for _, marker := range markers {
				if strings.Contains(upper, marker) {
					results = append(results, ReviewResult{
						File:     path,
						Line:     fset.Position(c.Pos()).Line,
						Severity: "info",
						Rule:     "todo-comment",
						Message:  fmt.Sprintf("Found %s comment: %s", marker, strings.TrimSpace(text)),
					})
					break // only report each comment once
				}
			}
		}
	}

	return results
}

// checkLongFunctions flags functions longer than 50 lines.
func (cr *CodeReviewer) checkLongFunctions(fset *token.FileSet, file *ast.File, path string) []ReviewResult {
	var results []ReviewResult

	for _, decl := range file.Decls {
		fd, ok := decl.(*ast.FuncDecl)
		if !ok || fd.Body == nil {
			continue
		}

		startLine := fset.Position(fd.Body.Lbrace).Line
		endLine := fset.Position(fd.Body.Rbrace).Line
		lineCount := endLine - startLine + 1

		if lineCount > 50 {
			results = append(results, ReviewResult{
				File:     path,
				Line:     fset.Position(fd.Pos()).Line,
				Severity: "info",
				Rule:     "long-function",
				Message:  fmt.Sprintf("Function %s is %d lines long (threshold: 50)", fd.Name.Name, lineCount),
			})
		}
	}

	return results
}

// checkDeepNesting flags if/for nesting deeper than 4 levels.
func (cr *CodeReviewer) checkDeepNesting(fset *token.FileSet, file *ast.File, path string) []ReviewResult {
	var results []ReviewResult

	for _, decl := range file.Decls {
		fd, ok := decl.(*ast.FuncDecl)
		if !ok || fd.Body == nil {
			continue
		}

		cr.walkNesting(fset, fd.Body, 0, fd.Name.Name, path, &results)
	}

	return results
}

// walkNesting recursively walks the AST tracking nesting depth of if/for/select.
func (cr *CodeReviewer) walkNesting(fset *token.FileSet, node ast.Node, depth int, funcName string, path string, results *[]ReviewResult) {
	if node == nil {
		return
	}

	ast.Inspect(node, func(n ast.Node) bool {
		if n == nil {
			return false
		}

		switch n.(type) {
		case *ast.IfStmt, *ast.ForStmt, *ast.RangeStmt, *ast.SelectStmt, *ast.SwitchStmt, *ast.TypeSwitchStmt:
			newDepth := depth + 1
			if newDepth > 4 {
				*results = append(*results, ReviewResult{
					File:     path,
					Line:     fset.Position(n.Pos()).Line,
					Severity: "warning",
					Rule:     "deep-nesting",
					Message:  fmt.Sprintf("Nesting depth %d in function %s (threshold: 4)", newDepth, funcName),
				})
			}

			// Manually recurse into the body at incremented depth
			switch stmt := n.(type) {
			case *ast.IfStmt:
				cr.walkNestingBlock(fset, stmt.Body, newDepth, funcName, path, results)
				if stmt.Else != nil {
					cr.walkNesting(fset, stmt.Else, newDepth, funcName, path, results)
				}
			case *ast.ForStmt:
				cr.walkNestingBlock(fset, stmt.Body, newDepth, funcName, path, results)
			case *ast.RangeStmt:
				cr.walkNestingBlock(fset, stmt.Body, newDepth, funcName, path, results)
			case *ast.SelectStmt:
				cr.walkNestingBlock(fset, stmt.Body, newDepth, funcName, path, results)
			case *ast.SwitchStmt:
				cr.walkNestingBlock(fset, stmt.Body, newDepth, funcName, path, results)
			case *ast.TypeSwitchStmt:
				cr.walkNestingBlock(fset, stmt.Body, newDepth, funcName, path, results)
			}
			return false // don't recurse via Inspect, we did it manually
		}

		return true
	})
}

// walkNestingBlock walks a block statement at a given nesting depth.
func (cr *CodeReviewer) walkNestingBlock(fset *token.FileSet, block *ast.BlockStmt, depth int, funcName string, path string, results *[]ReviewResult) {
	if block == nil {
		return
	}
	for _, stmt := range block.List {
		cr.walkNesting(fset, stmt, depth, funcName, path, results)
	}
}

// checkMissingDoc flags exported functions and types without doc comments.
func (cr *CodeReviewer) checkMissingDoc(fset *token.FileSet, file *ast.File, path string) []ReviewResult {
	var results []ReviewResult

	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if !isExportedName(d.Name.Name) {
				continue
			}
			if d.Doc == nil || strings.TrimSpace(d.Doc.Text()) == "" {
				results = append(results, ReviewResult{
					File:     path,
					Line:     fset.Position(d.Pos()).Line,
					Severity: "info",
					Rule:     "missing-doc",
					Message:  fmt.Sprintf("Exported function %s has no doc comment", d.Name.Name),
				})
			}

		case *ast.GenDecl:
			for _, spec := range d.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if !ok {
					continue
				}
				if !isExportedName(ts.Name.Name) {
					continue
				}
				hasDoc := false
				if ts.Doc != nil && strings.TrimSpace(ts.Doc.Text()) != "" {
					hasDoc = true
				}
				if d.Doc != nil && strings.TrimSpace(d.Doc.Text()) != "" {
					hasDoc = true
				}
				if !hasDoc {
					results = append(results, ReviewResult{
						File:     path,
						Line:     fset.Position(ts.Pos()).Line,
						Severity: "info",
						Rule:     "missing-doc",
						Message:  fmt.Sprintf("Exported type %s has no doc comment", ts.Name.Name),
					})
				}
			}
		}
	}

	return results
}

// isExportedName returns true if the name starts with an uppercase letter.
func isExportedName(name string) bool {
	if name == "" {
		return false
	}
	return unicode.IsUpper(rune(name[0]))
}
