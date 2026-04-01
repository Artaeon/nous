package agent

import (
	"fmt"
	"strings"
)

// -----------------------------------------------------------------------
// Smart Code Generation — generates domain-specific CLI tools from
// the domain knowledge base. Each generated project includes:
// - Data model with JSON serialization
// - JSON file-based persistent storage
// - CLI with subcommands for each operation
// - Help text and usage information
// - Tests for core functionality
// -----------------------------------------------------------------------

// GenerateSmartCLI generates a complete CLI tool from a domain definition.
func (ca *CodeAgent) GenerateSmartCLI(plan *CodePlan, domain *DomainDef) ([]string, error) {
	var files []string

	entity := domain.Entity
	lower := strings.ToLower(entity.Name)
	projectName := plan.ProjectName

	// 1. model.go — data structures
	if err := ca.writeFile("model.go", genSmartModel(entity)); err != nil {
		return nil, err
	}
	files = append(files, "model.go")

	// 2. store.go — JSON file-based persistent storage
	if err := ca.writeFile("store.go", genSmartStore(entity, lower)); err != nil {
		return nil, err
	}
	files = append(files, "store.go")

	// 3. commands.go — implementation of each operation
	if err := ca.writeFile("commands.go", genSmartCommands(entity, domain.Operations, lower)); err != nil {
		return nil, err
	}
	files = append(files, "commands.go")

	// 4. main.go — CLI entry point with subcommand routing
	if err := ca.writeFile("main.go", genSmartMain(projectName, domain.Operations, lower)); err != nil {
		return nil, err
	}
	files = append(files, "main.go")

	// 5. store_test.go — tests for storage
	if err := ca.writeFile("store_test.go", genSmartTests(entity, lower)); err != nil {
		return nil, err
	}
	files = append(files, "store_test.go")

	return files, nil
}

func genSmartModel(e Entity) string {
	var sb strings.Builder
	sb.WriteString("package main\n\nimport \"time\"\n\n")
	sb.WriteString(fmt.Sprintf("// %s is the core data type.\n", e.Name))
	sb.WriteString(fmt.Sprintf("type %s struct {\n", e.Name))
	sb.WriteString("\tID        int       `json:\"id\"`\n")
	for _, f := range e.Fields {
		sb.WriteString(fmt.Sprintf("\t%-10s %-10s `json:\"%s\"`\n", f.Name, f.Type, f.JSON))
	}
	sb.WriteString("\tCreatedAt time.Time `json:\"created_at\"`\n")
	sb.WriteString("\tUpdatedAt time.Time `json:\"updated_at\"`\n")
	sb.WriteString("}\n")
	return sb.String()
}

func genSmartStore(e Entity, lower string) string {
	name := e.Name
	return fmt.Sprintf(`package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Store provides persistent JSON file storage for %ss.
type Store struct {
	mu       sync.RWMutex
	items    []*%s
	nextID   int
	filepath string
}

// NewStore creates or loads a store from the given directory.
func NewStore(dir string) *Store {
	if err := os.MkdirAll(dir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "warning: cannot create data dir: %%v\n", err)
	}
	s := &Store{
		filepath: filepath.Join(dir, "%ss.json"),
		nextID:   1,
	}
	s.load()
	return s
}

func (s *Store) load() {
	data, err := os.ReadFile(s.filepath)
	if err != nil {
		return
	}
	var items []*%s
	if err := json.Unmarshal(data, &items); err != nil {
		return
	}
	s.items = items
	for _, item := range items {
		if item.ID >= s.nextID {
			s.nextID = item.ID + 1
		}
	}
}

func (s *Store) save() error {
	data, err := json.MarshalIndent(s.items, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.filepath, data, 0644)
}

// Create adds a new %s and persists it.
func (s *Store) Create(item *%s) *%s {
	s.mu.Lock()
	defer s.mu.Unlock()
	item.ID = s.nextID
	s.nextID++
	item.CreatedAt = time.Now()
	item.UpdatedAt = item.CreatedAt
	s.items = append(s.items, item)
	s.save()
	return item
}

// Get returns a %s by ID.
func (s *Store) Get(id int) (*%s, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, item := range s.items {
		if item.ID == id {
			return item, nil
		}
	}
	return nil, fmt.Errorf("%s %%d not found", id)
}

// List returns all %ss, optionally filtered.
func (s *Store) List() []*%s {
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := make([]*%s, len(s.items))
	copy(result, s.items)
	return result
}

// Update modifies a %s by ID.
func (s *Store) Update(id int, fn func(*%s)) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, item := range s.items {
		if item.ID == id {
			fn(item)
			item.UpdatedAt = time.Now()
			return s.save()
		}
	}
	return fmt.Errorf("%s %%d not found", id)
}

// Delete removes a %s by ID.
func (s *Store) Delete(id int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, item := range s.items {
		if item.ID == id {
			s.items = append(s.items[:i], s.items[i+1:]...)
			return s.save()
		}
	}
	return fmt.Errorf("%s %%d not found", id)
}

// Search finds items matching a query string.
func (s *Store) Search(query string) []*%s {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var results []*%s
	q := query
	for _, item := range s.items {
		data, _ := json.Marshal(item)
		if containsIgnoreCase(string(data), q) {
			results = append(results, item)
		}
	}
	return results
}

func containsIgnoreCase(haystack, needle string) bool {
	return len(needle) > 0 && len(haystack) >= len(needle) &&
		(haystack == needle || len(needle) <= len(haystack) &&
			findIgnoreCase(haystack, needle))
}

func findIgnoreCase(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		match := true
		for j := 0; j < len(sub); j++ {
			a, b := s[i+j], sub[j]
			if a >= 'A' && a <= 'Z' { a += 32 }
			if b >= 'A' && b <= 'Z' { b += 32 }
			if a != b { match = false; break }
		}
		if match { return true }
	}
	return false
}
`,
		lower, name, lower, name,
		lower, name, name,
		lower, name, lower,
		lower, name, name,
		lower, name, lower,
		lower, name,
		name, name,
	)
}

func genSmartCommands(e Entity, ops []Operation, lower string) string {
	name := e.Name
	var sb strings.Builder

	// Determine which imports are needed
	needsStrings := false
	needsTabwriter := false
	for _, op := range ops {
		if op.Name == "inbox" || op.Name == "list" {
			needsTabwriter = true
		}
		if op.Name == "search" || op.Name == "find" {
			needsStrings = true
		}
	}
	// Check if any entity has []string fields
	for _, f := range e.Fields {
		if f.Type == "[]string" {
			needsStrings = true
		}
	}
	sb.WriteString("package main\n\nimport (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n")
	if needsStrings {
		sb.WriteString("\t\"strings\"\n")
	}
	if needsTabwriter {
		sb.WriteString("\t\"text/tabwriter\"\n")
	}
	sb.WriteString(")\n\n")

	// Generate a command function for each operation
	for _, op := range ops {
		switch op.Name {
		case "inbox", "list":
			sb.WriteString(fmt.Sprintf(`// cmd%s shows all %ss.
func cmd%s(store *Store, args []string) {
	items := store.List()
	if len(items) == 0 {
		fmt.Println("No %ss found.")
		return
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "ID\t%s")
`, op.Verb, lower, op.Verb, lower, fieldsHeader(e)))

			sb.WriteString(fmt.Sprintf(`	for _, item := range items {
		fmt.Fprintf(w, "%%d\t%s\n", item.ID, %s)
	}
	w.Flush()
	fmt.Printf("\n%%d %s(s) total\n", len(items))
}

`, fieldsFormat(e), fieldsAccess(e, "item"), lower))

		case "read", "view", "get", "info":
			sb.WriteString(fmt.Sprintf(`// cmd%s shows details of a single %s.
func cmd%s(store *Store, args []string) {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: %s %s <id>")
		os.Exit(1)
	}
	id, err := strconv.Atoi(args[0])
	if err != nil {
		fmt.Fprintln(os.Stderr, "invalid id:", args[0])
		os.Exit(1)
	}
	item, err := store.Get(id)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("ID:       %%d\n", item.ID)
`, op.Verb, lower, op.Verb, lower, op.Name))

			for _, f := range e.Fields {
				if f.Type == "bool" {
					sb.WriteString(fmt.Sprintf("\tfmt.Printf(\"%s:  %%v\\n\", item.%s)\n", f.Name, f.Name))
				} else {
					sb.WriteString(fmt.Sprintf("\tfmt.Printf(\"%s:  %%s\\n\", item.%s)\n", f.Name, f.Name))
				}
			}
			sb.WriteString("\tfmt.Printf(\"Created:  %s\\n\", item.CreatedAt.Format(\"2006-01-02 15:04\"))\n")
			sb.WriteString("}\n\n")

		case "compose", "add", "new", "create", "save":
			sb.WriteString(fmt.Sprintf(`// cmd%s creates a new %s.
func cmd%s(store *Store, args []string) {
	item := &%s{}
	// Parse flags
`, op.Verb, lower, op.Verb, name))
			// Parse each field from flags
			for _, f := range e.Fields {
				flagName := strings.ToLower(f.Name)
				sb.WriteString(fmt.Sprintf("\t%sVal := \"\"\n", flagName))
			}
			sb.WriteString("\tfor i := 0; i < len(args); i++ {\n\t\tswitch args[i] {\n")
			for _, f := range e.Fields {
				flagName := strings.ToLower(f.Name)
				sb.WriteString(fmt.Sprintf("\t\tcase \"--%s\":\n\t\t\tif i+1 < len(args) { i++; %sVal = args[i] }\n", flagName, flagName))
			}
			sb.WriteString("\t\tdefault:\n")
			// First positional arg goes to first field
			if len(e.Fields) > 0 {
				first := strings.ToLower(e.Fields[0].Name)
				sb.WriteString(fmt.Sprintf("\t\t\tif %sVal == \"\" { %sVal = args[i] }\n", first, first))
			}
			sb.WriteString("\t\t}\n\t}\n\n")

			// Assign fields
			for _, f := range e.Fields {
				flagName := strings.ToLower(f.Name)
				switch f.Type {
				case "float64":
					sb.WriteString(fmt.Sprintf("\tif v, err := strconv.ParseFloat(%sVal, 64); err == nil { item.%s = v }\n", flagName, f.Name))
				case "int":
					sb.WriteString(fmt.Sprintf("\tif v, err := strconv.Atoi(%sVal); err == nil { item.%s = v }\n", flagName, f.Name))
				case "bool":
					sb.WriteString(fmt.Sprintf("\titem.%s = %sVal == \"true\" || %sVal == \"yes\" || %sVal == \"1\"\n", f.Name, flagName, flagName, flagName))
				case "[]string":
					sb.WriteString(fmt.Sprintf("\tif %sVal != \"\" { item.%s = strings.Split(%sVal, \",\") }\n", flagName, f.Name, flagName))
				default:
					sb.WriteString(fmt.Sprintf("\titem.%s = %sVal\n", f.Name, flagName))
				}
			}

			// Validate first field is non-empty
			if len(e.Fields) > 0 {
				f := e.Fields[0]
				if f.Type == "string" {
					sb.WriteString(fmt.Sprintf("\n\tif item.%s == \"\" {\n\t\tfmt.Fprintln(os.Stderr, \"%s is required\")\n\t\tos.Exit(1)\n\t}\n", f.Name, strings.ToLower(f.Name)))
				}
			}

			sb.WriteString(fmt.Sprintf(`
	created := store.Create(item)
	fmt.Printf("%s #%%d created\n", created.ID)
}

`, name))

		case "delete", "remove":
			sb.WriteString(fmt.Sprintf(`// cmd%s removes a %s.
func cmd%s(store *Store, args []string) {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: %s delete <id>")
		os.Exit(1)
	}
	id, err := strconv.Atoi(args[0])
	if err != nil {
		fmt.Fprintln(os.Stderr, "invalid id:", args[0])
		os.Exit(1)
	}
	if err := store.Delete(id); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("%s #%%d deleted\n", id)
}

`, op.Verb, lower, op.Verb, lower, name))

		case "search", "find":
			sb.WriteString(fmt.Sprintf(`// cmd%s searches %ss.
func cmd%s(store *Store, args []string) {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: %s search <query>")
		os.Exit(1)
	}
	query := strings.Join(args, " ")
	results := store.Search(query)
	if len(results) == 0 {
		fmt.Printf("No %ss matching \"%%s\"\n", query)
		return
	}
	fmt.Printf("Found %%d %s(s) matching \"%%s\":\n", len(results), query)
	for _, item := range results {
		fmt.Printf("  #%%d  %s\n", item.ID, %s)
	}
}

`, op.Verb, lower, op.Verb, lower, lower, lower, firstFieldFmt(e), firstFieldAccess(e, "item")))

		case "send", "reply":
			sb.WriteString(fmt.Sprintf(`// cmd%s handles the %s operation.
func cmd%s(store *Store, args []string) {
	fmt.Println("%s operation — implement with SMTP/IMAP integration")
	// TODO: integrate with actual email sending
}

`, op.Verb, op.Name, op.Verb, op.Name))

		default:
			sb.WriteString(fmt.Sprintf(`// cmd%s handles the %s operation.
func cmd%s(store *Store, args []string) {
	fmt.Println("%s — not yet implemented")
}

`, op.Verb, op.Name, op.Verb, op.Name))
		}
	}

	return sb.String()
}

func genSmartMain(projectName string, ops []Operation, lower string) string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf(`package main

import (
	"fmt"
	"os"
	"path/filepath"
)

const appName = "%s"

func main() {
	// Store data in ~/.%s/
	home, _ := os.UserHomeDir()
	dataDir := filepath.Join(home, ".%s")
	store := NewStore(dataDir)

	if len(os.Args) < 2 {
		printUsage()
		os.Exit(0)
	}

	cmd := os.Args[1]
	args := os.Args[2:]

	switch cmd {
`, projectName, lower, lower))

	for _, op := range ops {
		sb.WriteString(fmt.Sprintf("\tcase \"%s\":\n\t\tcmd%s(store, args)\n", op.Name, op.Verb))
	}

	sb.WriteString(`	case "help", "-h", "--help":
		printUsage()
	case "version", "-v", "--version":
		fmt.Printf("%s v1.0.0\n", appName)
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\nRun '%s help' for usage.\n", cmd, appName)
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Printf("%s — a fast, local command-line tool\n\n", appName)
	fmt.Println("Usage:")
`)

	for _, op := range ops {
		usage := op.Name
		if op.CLIUsage != "" {
			usage = op.CLIUsage
		}
		sb.WriteString(fmt.Sprintf("\tfmt.Printf(\"  %%s %-30s %%s\\n\", appName, \"%s\")\n", usage, op.Description))
	}

	sb.WriteString(`	fmt.Printf("  %s %-30s %s\n", appName, "help", "show this help")
	fmt.Printf("  %s %-30s %s\n", appName, "version", "show version")
}
`)
	return sb.String()
}

func testValue(f Field) string {
	switch f.Type {
	case "string":
		return "\"Test\""
	case "int":
		return "42"
	case "float64":
		return "9.99"
	case "bool":
		return "true"
	case "[]string":
		return "[]string{\"a\", \"b\"}"
	default:
		return "\"Test\""
	}
}

func testComparison(f Field) string {
	switch f.Type {
	case "string":
		return fmt.Sprintf("if got.%s != \"Test\" {\n\t\tt.Errorf(\"%s: got %%v, want Test\", got.%s)\n\t}", f.Name, f.Name, f.Name)
	case "int":
		return fmt.Sprintf("if got.%s != 42 {\n\t\tt.Errorf(\"%s: got %%v, want 42\", got.%s)\n\t}", f.Name, f.Name, f.Name)
	case "float64":
		return fmt.Sprintf("if got.%s != 9.99 {\n\t\tt.Errorf(\"%s: got %%v, want 9.99\", got.%s)\n\t}", f.Name, f.Name, f.Name)
	case "bool":
		return fmt.Sprintf("if got.%s != true {\n\t\tt.Errorf(\"%s: got %%v, want true\", got.%s)\n\t}", f.Name, f.Name, f.Name)
	default:
		return fmt.Sprintf("if got.%s != \"Test\" {\n\t\tt.Errorf(\"%s: got %%v, want Test\", got.%s)\n\t}", f.Name, f.Name, f.Name)
	}
}

func genSmartTests(e Entity, lower string) string {
	name := e.Name
	firstField := Field{Name: "Name", Type: "string"}
	if len(e.Fields) > 0 {
		firstField = e.Fields[0]
	}

	ff := firstField
	val := testValue(ff)
	cmp := testComparison(ff)

	var sb strings.Builder
	sb.WriteString("package main\n\nimport (\n\t\"os\"\n\t\"testing\"\n)\n\n")

	// TestStoreCreateAndGet
	sb.WriteString(fmt.Sprintf("func TestStoreCreateAndGet(t *testing.T) {\n\tdir := t.TempDir()\n\tstore := NewStore(dir)\n\n\titem := &%s{%s: %s}\n\tcreated := store.Create(item)\n\tif created.ID != 1 {\n\t\tt.Errorf(\"ID: got %%d, want 1\", created.ID)\n\t}\n\n\tgot, err := store.Get(1)\n\tif err != nil {\n\t\tt.Fatalf(\"Get: %%v\", err)\n\t}\n\t%s\n}\n\n",
		name, ff.Name, val, cmp))

	// TestStoreList
	sb.WriteString(fmt.Sprintf("func TestStoreList(t *testing.T) {\n\tdir := t.TempDir()\n\tstore := NewStore(dir)\n\n\tstore.Create(&%s{%s: %s})\n\tstore.Create(&%s{%s: %s})\n\n\titems := store.List()\n\tif len(items) != 2 {\n\t\tt.Errorf(\"List: got %%d items, want 2\", len(items))\n\t}\n}\n\n",
		name, ff.Name, val, name, ff.Name, val))

	// TestStoreDelete
	sb.WriteString(fmt.Sprintf("func TestStoreDelete(t *testing.T) {\n\tdir := t.TempDir()\n\tstore := NewStore(dir)\n\n\tstore.Create(&%s{%s: %s})\n\tif err := store.Delete(1); err != nil {\n\t\tt.Fatalf(\"Delete: %%v\", err)\n\t}\n\t_, err := store.Get(1)\n\tif err == nil {\n\t\tt.Error(\"expected error after delete\")\n\t}\n}\n\n",
		name, ff.Name, val))

	// TestStoreSearch
	sb.WriteString(fmt.Sprintf("func TestStoreSearch(t *testing.T) {\n\tdir := t.TempDir()\n\tstore := NewStore(dir)\n\n\tstore.Create(&%s{%s: %s})\n\n\tresults := store.Search(\"%s\")\n\tif len(results) != 1 {\n\t\tt.Errorf(\"Search: got %%d results, want 1\", len(results))\n\t}\n}\n\n",
		name, ff.Name, val, strings.Trim(val, "\"")))

	// TestStorePersistence
	sb.WriteString(fmt.Sprintf("func TestStorePersistence(t *testing.T) {\n\tdir := t.TempDir()\n\tstore := NewStore(dir)\n\tstore.Create(&%s{%s: %s})\n\n\tstore2 := NewStore(dir)\n\titems := store2.List()\n\tif len(items) != 1 {\n\t\tt.Errorf(\"Persistence: got %%d items, want 1\", len(items))\n\t}\n}\n\n",
		name, ff.Name, val))

	// TestStoreGetNotFound
	sb.WriteString("func TestStoreGetNotFound(t *testing.T) {\n\tdir := t.TempDir()\n\tstore := NewStore(dir)\n\t_, err := store.Get(999)\n\tif err == nil {\n\t\tt.Error(\"expected error for missing ID\")\n\t}\n}\n\n")

	sb.WriteString("func TestMain(m *testing.M) { os.Exit(m.Run()) }\n")

	return sb.String()
}

// -----------------------------------------------------------------------
// Helpers for field formatting
// -----------------------------------------------------------------------

func fieldsHeader(e Entity) string {
	var parts []string
	for _, f := range e.Fields {
		parts = append(parts, strings.ToUpper(f.Name))
	}
	return strings.Join(parts, "\\t")
}

func fieldsHeaderValues(e Entity) string {
	if len(e.Fields) == 0 {
		return ""
	}
	return strings.ToLower(e.Fields[0].Name)
}

func fieldsFormat(e Entity) string {
	var parts []string
	for _, f := range e.Fields {
		switch f.Type {
		case "int", "int64":
			parts = append(parts, "%d")
		case "float64":
			parts = append(parts, "%.2f")
		case "bool":
			parts = append(parts, "%v")
		case "[]string":
			parts = append(parts, "%s")
		default:
			parts = append(parts, "%s")
		}
	}
	return strings.Join(parts, "\\t")
}

func fieldsAccess(e Entity, varName string) string {
	var parts []string
	for _, f := range e.Fields {
		if f.Type == "[]string" {
			parts = append(parts, fmt.Sprintf("strings.Join(%s.%s, \",\")", varName, f.Name))
		} else {
			parts = append(parts, fmt.Sprintf("%s.%s", varName, f.Name))
		}
	}
	return strings.Join(parts, ", ")
}

func firstFieldFmt(e Entity) string {
	if len(e.Fields) == 0 {
		return "%v"
	}
	return "%s"
}

func firstFieldAccess(e Entity, varName string) string {
	if len(e.Fields) == 0 {
		return varName
	}
	return varName + "." + e.Fields[0].Name
}
