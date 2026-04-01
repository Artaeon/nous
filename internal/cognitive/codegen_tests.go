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

// SmartTestGenerator analyzes Go source files and generates meaningful,
// compilable test files with table-driven tests and realistic test cases.
type SmartTestGenerator struct{}

// NewSmartTestGenerator creates a SmartTestGenerator.
func NewSmartTestGenerator() *SmartTestGenerator {
	return &SmartTestGenerator{}
}

// parsedFunc holds the signature information extracted from an exported function.
type parsedFunc struct {
	Name    string
	Params  []parsedParam
	Returns []parsedReturn
}

type parsedParam struct {
	Name string
	Type string
}

type parsedReturn struct {
	Name string
	Type string
}

// GenerateTests parses a Go source file and generates a complete test file
// with table-driven tests and realistic test cases for each exported function.
func (stg *SmartTestGenerator) GenerateTests(path string) (string, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", path, err)
	}

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filepath.Base(path), src, parser.ParseComments)
	if err != nil {
		return "", fmt.Errorf("parse %s: %w", path, err)
	}

	pkgName := file.Name.Name

	var funcs []parsedFunc
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		// Skip unexported functions and methods.
		if !fn.Name.IsExported() || fn.Recv != nil {
			continue
		}
		pf := parsedFunc{Name: fn.Name.Name}

		// Parameters.
		if fn.Type.Params != nil {
			for _, field := range fn.Type.Params.List {
				typStr := exprToString(field.Type)
				if len(field.Names) == 0 {
					pf.Params = append(pf.Params, parsedParam{Type: typStr})
				}
				for _, name := range field.Names {
					pf.Params = append(pf.Params, parsedParam{Name: name.Name, Type: typStr})
				}
			}
		}

		// Return types.
		if fn.Type.Results != nil {
			for _, field := range fn.Type.Results.List {
				typStr := exprToString(field.Type)
				name := ""
				if len(field.Names) > 0 {
					name = field.Names[0].Name
				}
				pf.Returns = append(pf.Returns, parsedReturn{Name: name, Type: typStr})
			}
		}

		funcs = append(funcs, pf)
	}

	if len(funcs) == 0 {
		return "", fmt.Errorf("no exported functions found in %s", path)
	}

	// Build the test file.
	var b strings.Builder
	b.WriteString("package " + pkgName + "\n\n")
	b.WriteString("import (\n\t\"testing\"\n)\n")

	for _, fn := range funcs {
		b.WriteString("\n")
		b.WriteString(stg.generateTestFunc(fn))
	}

	return b.String(), nil
}

// generateTestFunc builds a complete table-driven test function for one func.
func (stg *SmartTestGenerator) generateTestFunc(fn parsedFunc) string {
	var b strings.Builder

	cases := stg.generateCases(fn)

	b.WriteString(fmt.Sprintf("func Test%s(t *testing.T) {\n", fn.Name))

	// Build the struct fields.
	b.WriteString("\ttests := []struct {\n")
	b.WriteString("\t\tname string\n")

	for _, p := range fn.Params {
		fieldName := p.Name
		if fieldName == "" {
			fieldName = inferFieldName(p.Type)
		}
		b.WriteString(fmt.Sprintf("\t\t%s %s\n", fieldName, p.Type))
	}

	// Add expectation fields.
	hasError := stg.hasErrorReturn(fn)
	hasBool := stg.hasBoolReturn(fn)
	hasPointer := stg.hasPointerReturn(fn)
	hasSlice := stg.hasSliceReturn(fn)
	hasValueReturn := stg.hasValueReturn(fn)

	if hasError {
		b.WriteString("\t\twantErr bool\n")
	}
	if hasBool && !hasError {
		b.WriteString("\t\twant bool\n")
	}
	if hasValueReturn && !hasBool {
		ret := stg.firstValueReturn(fn)
		b.WriteString(fmt.Sprintf("\t\twant %s\n", ret.Type))
	}

	b.WriteString("\t}{\n")

	// Write test cases.
	for _, tc := range cases {
		b.WriteString("\t\t{")
		parts := []string{fmt.Sprintf("name: %q", tc.name)}
		for _, arg := range tc.args {
			parts = append(parts, fmt.Sprintf("%s: %s", arg.field, arg.value))
		}
		if hasError {
			parts = append(parts, fmt.Sprintf("wantErr: %v", tc.wantErr))
		}
		if hasBool && !hasError {
			parts = append(parts, fmt.Sprintf("want: %v", tc.wantBool))
		}
		if hasValueReturn && !hasBool {
			parts = append(parts, fmt.Sprintf("want: %s", tc.wantValue))
		}
		b.WriteString(strings.Join(parts, ", "))
		b.WriteString("},\n")
	}

	b.WriteString("\t}\n\n")

	// Write the test loop.
	b.WriteString("\tfor _, tt := range tests {\n")
	b.WriteString("\t\tt.Run(tt.name, func(t *testing.T) {\n")

	// Build the call expression.
	callArgs := make([]string, len(fn.Params))
	for i, p := range fn.Params {
		fieldName := p.Name
		if fieldName == "" {
			fieldName = inferFieldName(p.Type)
		}
		callArgs[i] = "tt." + fieldName
	}
	callExpr := fmt.Sprintf("%s(%s)", fn.Name, strings.Join(callArgs, ", "))

	// Assign results based on return types.
	switch {
	case hasError && (hasPointer || hasSlice || hasBool || hasValueReturn):
		b.WriteString(fmt.Sprintf("\t\t\tgot, err := %s\n", callExpr))
		b.WriteString("\t\t\tif (err != nil) != tt.wantErr {\n")
		b.WriteString(fmt.Sprintf("\t\t\t\tt.Errorf(\"%s() error = %%v, wantErr %%v\", err, tt.wantErr)\n", fn.Name))
		b.WriteString("\t\t\t\treturn\n")
		b.WriteString("\t\t\t}\n")
		if hasPointer {
			b.WriteString("\t\t\tif !tt.wantErr && got == nil {\n")
			b.WriteString(fmt.Sprintf("\t\t\t\tt.Errorf(\"%s() returned nil, want non-nil\")\n", fn.Name))
			b.WriteString("\t\t\t}\n")
		} else if hasSlice {
			// no extra check for slices in error case — just the error check suffices
		} else if hasBool {
			b.WriteString("\t\t\tif !tt.wantErr && got != tt.want {\n")
			b.WriteString(fmt.Sprintf("\t\t\t\tt.Errorf(\"%s() = %%v, want %%v\", got, tt.want)\n", fn.Name))
			b.WriteString("\t\t\t}\n")
		} else if hasValueReturn {
			b.WriteString("\t\t\tif !tt.wantErr && got != tt.want {\n")
			b.WriteString(fmt.Sprintf("\t\t\t\tt.Errorf(\"%s() = %%v, want %%v\", got, tt.want)\n", fn.Name))
			b.WriteString("\t\t\t}\n")
		}

	case hasError:
		b.WriteString(fmt.Sprintf("\t\t\t_, err := %s\n", callExpr))
		b.WriteString("\t\t\tif (err != nil) != tt.wantErr {\n")
		b.WriteString(fmt.Sprintf("\t\t\t\tt.Errorf(\"%s() error = %%v, wantErr %%v\", err, tt.wantErr)\n", fn.Name))
		b.WriteString("\t\t\t}\n")

	case hasPointer:
		b.WriteString(fmt.Sprintf("\t\t\tgot := %s\n", callExpr))
		b.WriteString("\t\t\tif got == nil {\n")
		b.WriteString(fmt.Sprintf("\t\t\t\tt.Fatal(\"%s() returned nil\")\n", fn.Name))
		b.WriteString("\t\t\t}\n")

	case hasBool:
		b.WriteString(fmt.Sprintf("\t\t\tgot := %s\n", callExpr))
		b.WriteString("\t\t\tif got != tt.want {\n")
		b.WriteString(fmt.Sprintf("\t\t\t\tt.Errorf(\"%s() = %%v, want %%v\", got, tt.want)\n", fn.Name))
		b.WriteString("\t\t\t}\n")

	case hasSlice:
		b.WriteString(fmt.Sprintf("\t\t\tgot := %s\n", callExpr))
		b.WriteString("\t\t\t_ = got\n")

	case hasValueReturn:
		b.WriteString(fmt.Sprintf("\t\t\tgot := %s\n", callExpr))
		b.WriteString("\t\t\tif got != tt.want {\n")
		b.WriteString(fmt.Sprintf("\t\t\t\tt.Errorf(\"%s() = %%v, want %%v\", got, tt.want)\n", fn.Name))
		b.WriteString("\t\t\t}\n")

	default:
		// Void function — just call it.
		b.WriteString(fmt.Sprintf("\t\t\t%s\n", callExpr))
	}

	b.WriteString("\t\t})\n")
	b.WriteString("\t}\n")
	b.WriteString("}\n")

	return b.String()
}

// testCase holds one row of the table-driven test.
type testCase struct {
	name      string
	args      []testArg
	wantErr   bool
	wantBool  bool
	wantValue string
}

type testArg struct {
	field string
	value string
}

// generateCases builds test cases by analyzing parameter types and names.
func (stg *SmartTestGenerator) generateCases(fn parsedFunc) []testCase {
	hasErr := stg.hasErrorReturn(fn)
	hasBool := stg.hasBoolReturn(fn) && !hasErr

	// Generate per-parameter value sets.
	paramSets := make([][]testArg, len(fn.Params))
	for i, p := range fn.Params {
		fieldName := p.Name
		if fieldName == "" {
			fieldName = inferFieldName(p.Type)
		}
		paramSets[i] = stg.valuesForParam(p, fieldName)
	}

	// If there are no params, generate one or two cases based on return type.
	if len(fn.Params) == 0 {
		tc := testCase{name: "basic"}
		if hasErr {
			tc.wantErr = false
		}
		return []testCase{tc}
	}

	// Build cases from the first parameter's values, filling others with defaults.
	var cases []testCase
	if len(paramSets) > 0 && len(paramSets[0]) > 0 {
		for _, primary := range paramSets[0] {
			tc := testCase{
				name: primary.value,
				args: []testArg{primary},
			}

			// Use first value of each subsequent param as default.
			for j := 1; j < len(paramSets); j++ {
				if len(paramSets[j]) > 0 {
					tc.args = append(tc.args, paramSets[j][0])
				}
			}

			// Determine expected results.
			if hasErr {
				tc.wantErr = stg.isFailureCase(primary, fn.Params[0])
			}
			if hasBool {
				tc.wantBool = !stg.isFailureCase(primary, fn.Params[0])
			}

			// Clean up the name.
			tc.name = stg.caseName(primary, fn.Params[0])

			cases = append(cases, tc)
		}

		// Also generate cases from subsequent parameters' edge values.
		for j := 1; j < len(paramSets); j++ {
			for _, arg := range paramSets[j] {
				if stg.isEdgeCase(arg, fn.Params[j]) {
					tc := testCase{
						name: stg.caseName(arg, fn.Params[j]),
					}
					// Use first value for all params, override param j.
					for k := 0; k < len(paramSets); k++ {
						if k == j {
							tc.args = append(tc.args, arg)
						} else if len(paramSets[k]) > 0 {
							tc.args = append(tc.args, paramSets[k][0])
						}
					}
					if hasErr {
						tc.wantErr = stg.isFailureCase(arg, fn.Params[j])
					}
					if hasBool {
						tc.wantBool = !stg.isFailureCase(arg, fn.Params[j])
					}
					cases = append(cases, tc)
				}
			}
		}
	}

	// Deduplicate by name.
	seen := map[string]bool{}
	var deduped []testCase
	for _, tc := range cases {
		if !seen[tc.name] {
			seen[tc.name] = true
			deduped = append(deduped, tc)
		}
	}

	return deduped
}

// valuesForParam returns test argument values for a parameter based on its
// type and name.
func (stg *SmartTestGenerator) valuesForParam(p parsedParam, fieldName string) []testArg {
	low := strings.ToLower(p.Name)
	typ := p.Type

	switch {
	case typ == "string":
		return stg.stringValues(low, fieldName)
	case typ == "int" || typ == "int64" || typ == "int32":
		return stg.intValues(low, fieldName)
	case typ == "uint" || typ == "uint64" || typ == "uint32":
		return stg.uintValues(low, fieldName)
	case typ == "float64" || typ == "float32":
		return []testArg{
			{field: fieldName, value: "0.0"},
			{field: fieldName, value: "1.5"},
			{field: fieldName, value: "-1.0"},
		}
	case typ == "bool":
		return []testArg{
			{field: fieldName, value: "true"},
			{field: fieldName, value: "false"},
		}
	case strings.HasPrefix(typ, "[]"):
		return []testArg{
			{field: fieldName, value: "nil"},
			{field: fieldName, value: typ + "{}"},
		}
	default:
		// For unknown types, just use a zero value.
		return []testArg{
			{field: fieldName, value: zeroValue(typ)},
		}
	}
}

func (stg *SmartTestGenerator) stringValues(paramNameLower, fieldName string) []testArg {
	switch {
	case strings.Contains(paramNameLower, "url"):
		return []testArg{
			{field: fieldName, value: `"https://example.com"`},
			{field: fieldName, value: `""`},
			{field: fieldName, value: `"://bad"`},
		}
	case strings.Contains(paramNameLower, "path") || strings.Contains(paramNameLower, "file"):
		return []testArg{
			{field: fieldName, value: `"/tmp/test.txt"`},
			{field: fieldName, value: `""`},
			{field: fieldName, value: `"/nonexistent/path"`},
		}
	case strings.Contains(paramNameLower, "email"):
		return []testArg{
			{field: fieldName, value: `"user@example.com"`},
			{field: fieldName, value: `""`},
			{field: fieldName, value: `"no-at-sign"`},
		}
	case strings.Contains(paramNameLower, "name") || strings.Contains(paramNameLower, "key"):
		return []testArg{
			{field: fieldName, value: `"test"`},
			{field: fieldName, value: `""`},
			{field: fieldName, value: `"with spaces"`},
			{field: fieldName, value: `"special!@#"`},
		}
	default:
		return []testArg{
			{field: fieldName, value: `"hello"`},
			{field: fieldName, value: `""`},
		}
	}
}

func (stg *SmartTestGenerator) intValues(paramNameLower, fieldName string) []testArg {
	switch {
	case strings.Contains(paramNameLower, "port"):
		return []testArg{
			{field: fieldName, value: "8080"},
			{field: fieldName, value: "0"},
			{field: fieldName, value: "65535"},
			{field: fieldName, value: "-1"},
		}
	case strings.Contains(paramNameLower, "count") || strings.Contains(paramNameLower, "size") || strings.Contains(paramNameLower, "len"):
		return []testArg{
			{field: fieldName, value: "0"},
			{field: fieldName, value: "1"},
			{field: fieldName, value: "100"},
		}
	default:
		return []testArg{
			{field: fieldName, value: "0"},
			{field: fieldName, value: "1"},
			{field: fieldName, value: "-1"},
		}
	}
}

func (stg *SmartTestGenerator) uintValues(paramNameLower, fieldName string) []testArg {
	switch {
	case strings.Contains(paramNameLower, "port"):
		return []testArg{
			{field: fieldName, value: "8080"},
			{field: fieldName, value: "0"},
			{field: fieldName, value: "65535"},
		}
	case strings.Contains(paramNameLower, "count") || strings.Contains(paramNameLower, "size") || strings.Contains(paramNameLower, "len"):
		return []testArg{
			{field: fieldName, value: "0"},
			{field: fieldName, value: "1"},
			{field: fieldName, value: "100"},
		}
	default:
		return []testArg{
			{field: fieldName, value: "0"},
			{field: fieldName, value: "1"},
			{field: fieldName, value: "42"},
		}
	}
}

// caseName produces a human-readable test case name.
func (stg *SmartTestGenerator) caseName(arg testArg, p parsedParam) string {
	val := arg.value
	switch {
	case val == `""`:
		return "empty " + arg.field
	case val == "0" || val == "0.0":
		return "zero " + arg.field
	case val == "-1" || val == "-1.0":
		return "negative " + arg.field
	case val == "nil":
		return "nil " + arg.field
	case val == "true":
		return arg.field + " true"
	case val == "false":
		return arg.field + " false"
	case strings.HasSuffix(val, "{}"):
		return "empty slice"
	default:
		// Strip quotes for string values.
		clean := strings.Trim(val, `"`)
		if len(clean) > 20 {
			clean = clean[:20]
		}
		return clean
	}
}

// isFailureCase guesses whether a test value represents a failure input.
func (stg *SmartTestGenerator) isFailureCase(arg testArg, p parsedParam) bool {
	val := arg.value
	switch {
	case val == `""`:
		return true
	case val == "nil":
		return true
	case val == "-1" && (p.Type == "int" || p.Type == "int64" || p.Type == "int32"):
		low := strings.ToLower(p.Name)
		if strings.Contains(low, "port") || strings.Contains(low, "count") || strings.Contains(low, "size") {
			return true
		}
	case val == `"://bad"` || val == `"no-at-sign"` || val == `"/nonexistent/path"`:
		return true
	case strings.HasSuffix(val, "{}"):
		return false
	}
	return false
}

// isEdgeCase identifies values worth adding as extra dedicated test rows.
func (stg *SmartTestGenerator) isEdgeCase(arg testArg, p parsedParam) bool {
	val := arg.value
	return val == `""` || val == "0" || val == "-1" || val == "nil"
}

// Return type checks.

func (stg *SmartTestGenerator) hasErrorReturn(fn parsedFunc) bool {
	for _, r := range fn.Returns {
		if r.Type == "error" {
			return true
		}
	}
	return false
}

func (stg *SmartTestGenerator) hasBoolReturn(fn parsedFunc) bool {
	for _, r := range fn.Returns {
		if r.Type == "bool" {
			return true
		}
	}
	return false
}

func (stg *SmartTestGenerator) hasPointerReturn(fn parsedFunc) bool {
	for _, r := range fn.Returns {
		if strings.HasPrefix(r.Type, "*") {
			return true
		}
	}
	return false
}

func (stg *SmartTestGenerator) hasSliceReturn(fn parsedFunc) bool {
	for _, r := range fn.Returns {
		if strings.HasPrefix(r.Type, "[]") {
			return true
		}
	}
	return false
}

func (stg *SmartTestGenerator) hasValueReturn(fn parsedFunc) bool {
	for _, r := range fn.Returns {
		t := r.Type
		if t != "error" && !strings.HasPrefix(t, "*") && !strings.HasPrefix(t, "[]") {
			return true
		}
	}
	return false
}

func (stg *SmartTestGenerator) firstValueReturn(fn parsedFunc) parsedReturn {
	for _, r := range fn.Returns {
		t := r.Type
		if t != "error" && !strings.HasPrefix(t, "*") && !strings.HasPrefix(t, "[]") {
			return r
		}
	}
	return parsedReturn{}
}

// exprToString converts an AST type expression into a readable string.
func exprToString(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return "*" + exprToString(t.X)
	case *ast.ArrayType:
		if t.Len == nil {
			return "[]" + exprToString(t.Elt)
		}
		return "[...]" + exprToString(t.Elt)
	case *ast.SelectorExpr:
		return exprToString(t.X) + "." + t.Sel.Name
	case *ast.MapType:
		return "map[" + exprToString(t.Key) + "]" + exprToString(t.Value)
	case *ast.InterfaceType:
		return "interface{}"
	case *ast.Ellipsis:
		return "..." + exprToString(t.Elt)
	case *ast.FuncType:
		return "func()"
	case *ast.ChanType:
		return "chan " + exprToString(t.Value)
	default:
		return "interface{}"
	}
}

// inferFieldName produces a struct field name from a type when the parameter
// is unnamed.
func inferFieldName(typ string) string {
	typ = strings.TrimPrefix(typ, "*")
	typ = strings.TrimPrefix(typ, "[]")
	if idx := strings.LastIndex(typ, "."); idx >= 0 {
		typ = typ[idx+1:]
	}
	if typ == "" {
		return "arg"
	}
	// Lowercase first letter.
	return strings.ToLower(typ[:1]) + typ[1:]
}

// zeroValue returns a Go zero-value literal for a type.
func zeroValue(typ string) string {
	switch {
	case typ == "string":
		return `""`
	case typ == "int" || typ == "int64" || typ == "int32" || typ == "int16" || typ == "int8":
		return "0"
	case typ == "uint" || typ == "uint64" || typ == "uint32" || typ == "uint16" || typ == "uint8":
		return "0"
	case typ == "float64" || typ == "float32":
		return "0.0"
	case typ == "bool":
		return "false"
	case typ == "error":
		return "nil"
	case strings.HasPrefix(typ, "*"):
		return "nil"
	case strings.HasPrefix(typ, "[]"):
		return "nil"
	case strings.HasPrefix(typ, "map["):
		return "nil"
	default:
		return typ + "{}"
	}
}
