package cognitive

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
)

// CodeExplainer generates natural language explanations of Go source code.
// Uses AST parsing for structure and pattern matching for semantics.
type CodeExplainer struct{}

// NewCodeExplainer creates a code explainer.
func NewCodeExplainer() *CodeExplainer {
	return &CodeExplainer{}
}

// ExplainFile generates an overview explanation of a Go source file.
func (ce *CodeExplainer) ExplainFile(path string) (string, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
	if err != nil {
		return "", fmt.Errorf("parse error: %w", err)
	}

	var sb strings.Builder

	// Header
	sb.WriteString(fmt.Sprintf("File: %s\n", path))
	sb.WriteString(fmt.Sprintf("Package: %s\n", file.Name.Name))
	sb.WriteString("\n")

	// File-level doc comment
	if file.Doc != nil {
		docText := strings.TrimSpace(file.Doc.Text())
		if docText != "" {
			sb.WriteString(docText + "\n\n")
		}
	}

	// Imports
	var imports []string
	for _, imp := range file.Imports {
		impPath := strings.Trim(imp.Path.Value, `"`)
		imports = append(imports, impPath)
	}
	if len(imports) > 0 {
		sb.WriteString(fmt.Sprintf("Imports: %s\n\n", strings.Join(imports, ", ")))
	}

	// Collect types and functions
	var typeDescs []string
	var funcDescs []string

	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if !ok {
					continue
				}
				kind := "type"
				switch ts.Type.(type) {
				case *ast.StructType:
					kind = "struct"
				case *ast.InterfaceType:
					kind = "interface"
				}
				doc := ""
				if ts.Doc != nil {
					doc = strings.TrimSpace(ts.Doc.Text())
				} else if d.Doc != nil {
					doc = strings.TrimSpace(d.Doc.Text())
				}
				desc := fmt.Sprintf("- %s %s", ts.Name.Name, kind)
				if doc != "" {
					firstLine := strings.SplitN(doc, "\n", 2)[0]
					desc += " — " + firstLine
				}
				typeDescs = append(typeDescs, desc)
			}

		case *ast.FuncDecl:
			sig := ce.formatFuncSignature(d)
			doc := ""
			if d.Doc != nil {
				doc = strings.TrimSpace(d.Doc.Text())
			}
			desc := fmt.Sprintf("- %s", sig)
			if doc != "" {
				firstLine := strings.SplitN(doc, "\n", 2)[0]
				desc += " — " + firstLine
			}
			funcDescs = append(funcDescs, desc)
		}
	}

	if len(typeDescs) > 0 {
		sb.WriteString("Types:\n")
		for _, d := range typeDescs {
			sb.WriteString(d + "\n")
		}
		sb.WriteString("\n")
	}

	if len(funcDescs) > 0 {
		sb.WriteString("Functions:\n")
		for _, d := range funcDescs {
			sb.WriteString(d + "\n")
		}
		sb.WriteString("\n")
	}

	return strings.TrimRight(sb.String(), "\n") + "\n", nil
}

// ExplainFunction explains a specific function by name within a file.
func (ce *CodeExplainer) ExplainFunction(path string, funcName string) (string, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
	if err != nil {
		return "", fmt.Errorf("parse error: %w", err)
	}

	for _, decl := range file.Decls {
		fd, ok := decl.(*ast.FuncDecl)
		if !ok || fd.Name.Name != funcName {
			continue
		}
		return ce.explainFuncDecl(fset, fd), nil
	}

	return "", fmt.Errorf("function %q not found in %s", funcName, path)
}

// ExplainLine explains what's happening at a specific line number.
func (ce *CodeExplainer) ExplainLine(path string, lineNum int) (string, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read error: %w", err)
	}

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, src, parser.ParseComments)
	if err != nil {
		return "", fmt.Errorf("parse error: %w", err)
	}

	// Find the innermost node containing this line
	var result string
	ast.Inspect(file, func(n ast.Node) bool {
		if n == nil {
			return false
		}
		startLine := fset.Position(n.Pos()).Line
		endLine := fset.Position(n.End()).Line

		if lineNum < startLine || lineNum > endLine {
			return false
		}

		// Only describe nodes that start on the target line for precision
		if startLine == lineNum {
			desc := ce.describeNode(n)
			if desc != "" {
				result = desc
			}
		}

		return true
	})

	if result == "" {
		// Fall back to reading the raw line
		lines := strings.Split(string(src), "\n")
		if lineNum >= 1 && lineNum <= len(lines) {
			line := strings.TrimSpace(lines[lineNum-1])
			if line == "" {
				return "Empty line", nil
			}
			if strings.HasPrefix(line, "//") {
				return fmt.Sprintf("Comment: %s", line), nil
			}
			return fmt.Sprintf("Line %d: %s", lineNum, line), nil
		}
		return "", fmt.Errorf("line %d not found in file", lineNum)
	}

	return result, nil
}

// explainFuncDecl generates a detailed explanation of a function declaration.
func (ce *CodeExplainer) explainFuncDecl(fset *token.FileSet, fd *ast.FuncDecl) string {
	var sb strings.Builder

	// Signature line
	sig := ce.formatFuncSignature(fd)
	sb.WriteString(fmt.Sprintf("func %s\n\n", sig))

	// Doc comment
	if fd.Doc != nil {
		sb.WriteString(strings.TrimSpace(fd.Doc.Text()) + "\n\n")
	}

	// Analyze body patterns
	patterns := ce.analyzeBody(fd.Body)
	if len(patterns) > 0 {
		sb.WriteString(strings.Join(patterns, ". ") + ".\n\n")
	}

	// Parameters
	if fd.Type.Params != nil && len(fd.Type.Params.List) > 0 {
		sb.WriteString("Parameters:\n")
		for _, field := range fd.Type.Params.List {
			typeStr := ce.typeString(field.Type)
			for _, name := range field.Names {
				sb.WriteString(fmt.Sprintf("  %s (%s)\n", name.Name, typeStr))
			}
		}
		sb.WriteString("\n")
	}

	// Returns
	if fd.Type.Results != nil && len(fd.Type.Results.List) > 0 {
		sb.WriteString("Returns: ")
		var retTypes []string
		for _, field := range fd.Type.Results.List {
			typeStr := ce.typeString(field.Type)
			if len(field.Names) > 0 {
				for _, name := range field.Names {
					retTypes = append(retTypes, fmt.Sprintf("%s %s", name.Name, typeStr))
				}
			} else {
				retTypes = append(retTypes, typeStr)
			}
		}
		sb.WriteString(strings.Join(retTypes, ", "))
		sb.WriteString("\n")
	}

	// Line range
	if fd.Body != nil {
		startLine := fset.Position(fd.Pos()).Line
		endLine := fset.Position(fd.End()).Line
		lineCount := endLine - startLine + 1
		sb.WriteString(fmt.Sprintf("Lines: %d-%d (%d lines)\n", startLine, endLine, lineCount))
	}

	return sb.String()
}

// analyzeBody inspects a function body for common patterns and returns
// natural language descriptions of what it finds.
func (ce *CodeExplainer) analyzeBody(body *ast.BlockStmt) []string {
	if body == nil {
		return nil
	}

	var patterns []string
	seen := make(map[string]bool)

	add := func(s string) {
		if !seen[s] {
			seen[s] = true
			patterns = append(patterns, s)
		}
	}

	ast.Inspect(body, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.IfStmt:
			if ce.isErrCheck(node.Cond) {
				add("Includes error handling")
			}
		case *ast.RangeStmt:
			add("Iterates over a collection")
		case *ast.ForStmt:
			add("Uses a loop")
		case *ast.GoStmt:
			add("Spawns a goroutine")
		case *ast.SelectStmt:
			add("Uses channel selection")
		case *ast.DeferStmt:
			add("Uses deferred cleanup")
		case *ast.CallExpr:
			ce.analyzeCall(node, add)
		}
		return true
	})

	return patterns
}

// analyzeCall checks a function call for known package patterns.
func (ce *CodeExplainer) analyzeCall(call *ast.CallExpr, add func(string)) {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return
	}
	ident, ok := sel.X.(*ast.Ident)
	if !ok {
		return
	}

	switch ident.Name {
	case "http":
		add("Makes HTTP operations")
	case "json":
		add("Handles JSON encoding/decoding")
	case "sql", "db":
		add("Interacts with a database")
	case "os", "filepath":
		add("Performs file operations")
	case "fmt":
		add("Formats output")
	case "log":
		add("Logs information")
	case "sync":
		add("Uses synchronization primitives")
	}
}

// isErrCheck returns true if the expression is checking `err != nil` or `err == nil`.
func (ce *CodeExplainer) isErrCheck(expr ast.Expr) bool {
	bin, ok := expr.(*ast.BinaryExpr)
	if !ok {
		return false
	}
	ident, ok := bin.X.(*ast.Ident)
	if !ok {
		return false
	}
	return ident.Name == "err" && (bin.Op == token.NEQ || bin.Op == token.EQL)
}

// describeNode returns a natural language description for an AST node.
func (ce *CodeExplainer) describeNode(n ast.Node) string {
	switch node := n.(type) {
	case *ast.AssignStmt:
		return ce.describeAssign(node)
	case *ast.ExprStmt:
		if call, ok := node.X.(*ast.CallExpr); ok {
			return fmt.Sprintf("Calls %s", ce.exprString(call.Fun))
		}
	case *ast.ReturnStmt:
		if len(node.Results) == 0 {
			return "Returns (no value)"
		}
		var vals []string
		for _, r := range node.Results {
			vals = append(vals, ce.exprString(r))
		}
		return fmt.Sprintf("Returns %s", strings.Join(vals, ", "))
	case *ast.IfStmt:
		if ce.isErrCheck(node.Cond) {
			return "Checks if err != nil"
		}
		return fmt.Sprintf("Checks if %s", ce.exprString(node.Cond))
	case *ast.RangeStmt:
		return fmt.Sprintf("Iterates over %s", ce.exprString(node.X))
	case *ast.ForStmt:
		return "Begins a for loop"
	case *ast.DeferStmt:
		return fmt.Sprintf("Defers call to %s", ce.exprString(node.Call.Fun))
	case *ast.GoStmt:
		return "Spawns a goroutine"
	case *ast.FuncDecl:
		return fmt.Sprintf("Declares function %s", node.Name.Name)
	case *ast.GenDecl:
		if len(node.Specs) > 0 {
			if ts, ok := node.Specs[0].(*ast.TypeSpec); ok {
				return fmt.Sprintf("Declares type %s", ts.Name.Name)
			}
		}
	case *ast.SelectStmt:
		return "Begins a select statement for channel operations"
	case *ast.SendStmt:
		return fmt.Sprintf("Sends %s to channel %s", ce.exprString(node.Value), ce.exprString(node.Chan))
	}
	return ""
}

// describeAssign produces a description for an assignment statement.
func (ce *CodeExplainer) describeAssign(a *ast.AssignStmt) string {
	if len(a.Lhs) == 0 {
		return ""
	}

	var lhs []string
	for _, l := range a.Lhs {
		lhs = append(lhs, ce.exprString(l))
	}
	names := strings.Join(lhs, ", ")

	switch a.Tok {
	case token.DEFINE:
		if len(a.Rhs) == 1 {
			return fmt.Sprintf("Defines %s as %s", names, ce.exprString(a.Rhs[0]))
		}
		return fmt.Sprintf("Defines %s", names)
	case token.ASSIGN:
		if len(a.Rhs) == 1 {
			return fmt.Sprintf("Assigns %s to %s", ce.exprString(a.Rhs[0]), names)
		}
		return fmt.Sprintf("Assigns to %s", names)
	default:
		return fmt.Sprintf("Updates %s", names)
	}
}

// formatFuncSignature builds a readable signature from a function declaration.
func (ce *CodeExplainer) formatFuncSignature(fd *ast.FuncDecl) string {
	var sb strings.Builder

	// Receiver
	if fd.Recv != nil && len(fd.Recv.List) > 0 {
		recv := fd.Recv.List[0]
		sb.WriteString("(")
		if len(recv.Names) > 0 {
			sb.WriteString(recv.Names[0].Name + " ")
		}
		sb.WriteString(ce.typeString(recv.Type))
		sb.WriteString(") ")
	}

	sb.WriteString(fd.Name.Name)
	sb.WriteString("(")

	// Parameters
	var params []string
	if fd.Type.Params != nil {
		for _, field := range fd.Type.Params.List {
			typeStr := ce.typeString(field.Type)
			if len(field.Names) > 0 {
				for _, name := range field.Names {
					params = append(params, name.Name+" "+typeStr)
				}
			} else {
				params = append(params, typeStr)
			}
		}
	}
	sb.WriteString(strings.Join(params, ", "))
	sb.WriteString(")")

	// Returns
	if fd.Type.Results != nil && len(fd.Type.Results.List) > 0 {
		sb.WriteString(" ")
		var rets []string
		for _, field := range fd.Type.Results.List {
			typeStr := ce.typeString(field.Type)
			if len(field.Names) > 0 {
				for _, name := range field.Names {
					rets = append(rets, name.Name+" "+typeStr)
				}
			} else {
				rets = append(rets, typeStr)
			}
		}
		if len(rets) == 1 {
			sb.WriteString(rets[0])
		} else {
			sb.WriteString("(" + strings.Join(rets, ", ") + ")")
		}
	}

	return sb.String()
}

// typeString returns a simplified string representation of a type expression.
func (ce *CodeExplainer) typeString(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return "*" + ce.typeString(t.X)
	case *ast.SelectorExpr:
		return ce.typeString(t.X) + "." + t.Sel.Name
	case *ast.ArrayType:
		return "[]" + ce.typeString(t.Elt)
	case *ast.MapType:
		return "map[" + ce.typeString(t.Key) + "]" + ce.typeString(t.Value)
	case *ast.InterfaceType:
		return "interface{}"
	case *ast.FuncType:
		return "func(...)"
	case *ast.ChanType:
		return "chan " + ce.typeString(t.Value)
	case *ast.Ellipsis:
		return "..." + ce.typeString(t.Elt)
	default:
		return "unknown"
	}
}

// exprString returns a simplified string for any expression (best-effort).
func (ce *CodeExplainer) exprString(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.BasicLit:
		return e.Value
	case *ast.SelectorExpr:
		return ce.exprString(e.X) + "." + e.Sel.Name
	case *ast.CallExpr:
		return ce.exprString(e.Fun) + "(...)"
	case *ast.StarExpr:
		return "*" + ce.exprString(e.X)
	case *ast.UnaryExpr:
		return e.Op.String() + ce.exprString(e.X)
	case *ast.BinaryExpr:
		return ce.exprString(e.X) + " " + e.Op.String() + " " + ce.exprString(e.Y)
	case *ast.IndexExpr:
		return ce.exprString(e.X) + "[" + ce.exprString(e.Index) + "]"
	case *ast.CompositeLit:
		if e.Type != nil {
			return ce.exprString(e.Type) + "{...}"
		}
		return "{...}"
	case *ast.FuncLit:
		return "func literal"
	case *ast.ParenExpr:
		return "(" + ce.exprString(e.X) + ")"
	case *ast.TypeAssertExpr:
		return ce.exprString(e.X) + ".(" + ce.typeString(e.Type) + ")"
	case *ast.SliceExpr:
		return ce.exprString(e.X) + "[:]"
	default:
		return "expr"
	}
}
