package agent

import "strings"

// -----------------------------------------------------------------------
// Domain Knowledge Base
//
// Maps real-world domains to their standard operations, entities, and
// data structures. This is what allows the coding agent to understand
// "build me an email manager" without an LLM — it knows that emails
// have subjects, bodies, senders, and that managing them means
// list/read/compose/send/delete/search.
// -----------------------------------------------------------------------

// DomainDef defines a real-world domain with its entities and operations.
type DomainDef struct {
	Name       string       // "email", "file", "task", etc.
	Keywords   []string     // words that indicate this domain
	Entity     Entity       // primary entity
	Operations []Operation  // standard operations for this domain
	Imports    []string     // Go imports needed
	ExtraCode  string       // domain-specific helper code
}

// Operation is something you can do in a domain.
type Operation struct {
	Name        string   // "list", "read", "send", etc.
	Verb        string   // Go method name: "List", "Read", "Send"
	Description string   // what it does
	HTTPMethod  string   // for APIs: GET, POST, PUT, DELETE
	HTTPPath    string   // for APIs: /api/emails, /api/emails/{id}
	CLIUsage    string   // for CLIs: "list [--unread]"
	Flags       []CLIFlag // CLI flags for this operation
}

// CLIFlag is a command-line flag.
type CLIFlag struct {
	Name    string
	Type    string // "string", "int", "bool"
	Default string
	Usage   string
}

// -----------------------------------------------------------------------
// Domain Registry — 20+ real-world domains
// -----------------------------------------------------------------------

var domainRegistry = []DomainDef{
	{
		Name:     "email",
		Keywords: []string{"email", "mail", "inbox", "gmail", "outlook", "smtp", "imap", "message"},
		Entity: Entity{
			Name: "Email",
			Fields: []Field{
				{Name: "From", Type: "string", JSON: "from"},
				{Name: "To", Type: "string", JSON: "to"},
				{Name: "Subject", Type: "string", JSON: "subject"},
				{Name: "Body", Type: "string", JSON: "body"},
				{Name: "Read", Type: "bool", JSON: "read"},
				{Name: "Date", Type: "string", JSON: "date"},
			},
		},
		Operations: []Operation{
			{Name: "inbox", Verb: "Inbox", Description: "show inbox messages", CLIUsage: "inbox [--unread] [--limit N]",
				Flags: []CLIFlag{{Name: "unread", Type: "bool", Default: "false", Usage: "show only unread"}, {Name: "limit", Type: "int", Default: "20", Usage: "max messages"}}},
			{Name: "read", Verb: "Read", Description: "read a specific email", CLIUsage: "read <id>"},
			{Name: "compose", Verb: "Compose", Description: "compose a new email", CLIUsage: "compose --to <addr> --subject <subj> [--body <text>]",
				Flags: []CLIFlag{{Name: "to", Type: "string", Usage: "recipient address"}, {Name: "subject", Type: "string", Usage: "email subject"}, {Name: "body", Type: "string", Usage: "email body"}}},
			{Name: "send", Verb: "Send", Description: "send a composed email", CLIUsage: "send --to <addr> --subject <subj> --body <text>"},
			{Name: "reply", Verb: "Reply", Description: "reply to an email", CLIUsage: "reply <id> --body <text>"},
			{Name: "delete", Verb: "Delete", Description: "delete an email", CLIUsage: "delete <id>"},
			{Name: "search", Verb: "Search", Description: "search emails by keyword", CLIUsage: "search <query>",
				Flags: []CLIFlag{{Name: "from", Type: "string", Usage: "filter by sender"}}},
		},
	},
	{
		Name:     "file",
		Keywords: []string{"file", "files", "directory", "folder", "filesystem", "disk"},
		Entity: Entity{
			Name: "FileEntry",
			Fields: []Field{
				{Name: "Path", Type: "string", JSON: "path"},
				{Name: "Name", Type: "string", JSON: "name"},
				{Name: "Size", Type: "int64", JSON: "size"},
				{Name: "IsDir", Type: "bool", JSON: "is_dir"},
				{Name: "Modified", Type: "string", JSON: "modified"},
			},
		},
		Operations: []Operation{
			{Name: "list", Verb: "List", Description: "list files in a directory", CLIUsage: "list [path] [--all] [--long]"},
			{Name: "search", Verb: "Search", Description: "search for files by name", CLIUsage: "search <pattern> [--dir path]"},
			{Name: "rename", Verb: "Rename", Description: "rename a file", CLIUsage: "rename <old> <new>"},
			{Name: "move", Verb: "Move", Description: "move a file", CLIUsage: "move <src> <dst>"},
			{Name: "copy", Verb: "Copy", Description: "copy a file", CLIUsage: "copy <src> <dst>"},
			{Name: "delete", Verb: "Delete", Description: "delete a file", CLIUsage: "delete <path> [--force]"},
			{Name: "info", Verb: "Info", Description: "show file details", CLIUsage: "info <path>"},
			{Name: "tree", Verb: "Tree", Description: "show directory tree", CLIUsage: "tree [path] [--depth N]"},
		},
		Imports: []string{"os", "path/filepath", "io", "io/fs"},
	},
	{
		Name:     "note",
		Keywords: []string{"note", "notes", "notebook", "memo", "jot"},
		Entity: Entity{
			Name: "Note",
			Fields: []Field{
				{Name: "Title", Type: "string", JSON: "title"},
				{Name: "Body", Type: "string", JSON: "body"},
				{Name: "Tags", Type: "[]string", JSON: "tags"},
			},
		},
		Operations: []Operation{
			{Name: "new", Verb: "Create", Description: "create a new note", CLIUsage: "new <title> [--tags tag1,tag2]"},
			{Name: "list", Verb: "List", Description: "list all notes", CLIUsage: "list [--tag tag]"},
			{Name: "view", Verb: "View", Description: "view a note", CLIUsage: "view <id>"},
			{Name: "edit", Verb: "Edit", Description: "edit a note", CLIUsage: "edit <id>"},
			{Name: "delete", Verb: "Delete", Description: "delete a note", CLIUsage: "delete <id>"},
			{Name: "search", Verb: "Search", Description: "search notes", CLIUsage: "search <query>"},
		},
	},
	{
		Name:     "password",
		Keywords: []string{"password", "passwords", "vault", "secret", "credential", "keychain"},
		Entity: Entity{
			Name: "Credential",
			Fields: []Field{
				{Name: "Service", Type: "string", JSON: "service"},
				{Name: "Username", Type: "string", JSON: "username"},
				{Name: "Password", Type: "string", JSON: "password"},
				{Name: "URL", Type: "string", JSON: "url"},
				{Name: "Notes", Type: "string", JSON: "notes"},
			},
		},
		Operations: []Operation{
			{Name: "add", Verb: "Add", Description: "store a credential", CLIUsage: "add <service> --user <u> --pass <p>"},
			{Name: "get", Verb: "Get", Description: "retrieve a credential", CLIUsage: "get <service>"},
			{Name: "list", Verb: "List", Description: "list all services", CLIUsage: "list"},
			{Name: "generate", Verb: "Generate", Description: "generate a strong password", CLIUsage: "generate [--length N]"},
			{Name: "delete", Verb: "Delete", Description: "remove a credential", CLIUsage: "delete <service>"},
		},
	},
	{
		Name:     "expense",
		Keywords: []string{"expense", "expenses", "budget", "finance", "money", "spending", "receipt"},
		Entity: Entity{
			Name: "Expense",
			Fields: []Field{
				{Name: "Amount", Type: "float64", JSON: "amount"},
				{Name: "Category", Type: "string", JSON: "category"},
				{Name: "Description", Type: "string", JSON: "description"},
				{Name: "Date", Type: "string", JSON: "date"},
			},
		},
		Operations: []Operation{
			{Name: "add", Verb: "Add", Description: "log an expense", CLIUsage: "add <amount> <category> [description]"},
			{Name: "list", Verb: "List", Description: "list expenses", CLIUsage: "list [--month M] [--category C]"},
			{Name: "summary", Verb: "Summary", Description: "show spending summary", CLIUsage: "summary [--month M]"},
			{Name: "export", Verb: "Export", Description: "export to CSV", CLIUsage: "export [--output file.csv]"},
			{Name: "delete", Verb: "Delete", Description: "remove an expense", CLIUsage: "delete <id>"},
		},
	},
	{
		Name:     "bookmark",
		Keywords: []string{"bookmark", "bookmarks", "links", "url", "read later", "pinboard"},
		Entity: Entity{
			Name: "Bookmark",
			Fields: []Field{
				{Name: "URL", Type: "string", JSON: "url"},
				{Name: "Title", Type: "string", JSON: "title"},
				{Name: "Tags", Type: "[]string", JSON: "tags"},
				{Name: "Description", Type: "string", JSON: "description"},
			},
		},
		Operations: []Operation{
			{Name: "add", Verb: "Add", Description: "add a bookmark", CLIUsage: "add <url> [--title T] [--tags t1,t2]"},
			{Name: "list", Verb: "List", Description: "list bookmarks", CLIUsage: "list [--tag tag]"},
			{Name: "open", Verb: "Open", Description: "open bookmark in browser", CLIUsage: "open <id>"},
			{Name: "search", Verb: "Search", Description: "search bookmarks", CLIUsage: "search <query>"},
			{Name: "delete", Verb: "Delete", Description: "remove a bookmark", CLIUsage: "delete <id>"},
			{Name: "export", Verb: "Export", Description: "export bookmarks", CLIUsage: "export [--format html|json]"},
		},
	},
	{
		Name:     "contact",
		Keywords: []string{"contact", "contacts", "address book", "people", "phonebook"},
		Entity: Entity{
			Name: "Contact",
			Fields: []Field{
				{Name: "Name", Type: "string", JSON: "name"},
				{Name: "Email", Type: "string", JSON: "email"},
				{Name: "Phone", Type: "string", JSON: "phone"},
				{Name: "Company", Type: "string", JSON: "company"},
				{Name: "Notes", Type: "string", JSON: "notes"},
			},
		},
		Operations: []Operation{
			{Name: "add", Verb: "Add", Description: "add a contact", CLIUsage: "add <name> --email <e> [--phone p]"},
			{Name: "list", Verb: "List", Description: "list contacts", CLIUsage: "list"},
			{Name: "find", Verb: "Find", Description: "find a contact", CLIUsage: "find <query>"},
			{Name: "edit", Verb: "Edit", Description: "edit a contact", CLIUsage: "edit <id>"},
			{Name: "delete", Verb: "Delete", Description: "remove a contact", CLIUsage: "delete <id>"},
			{Name: "export", Verb: "Export", Description: "export contacts", CLIUsage: "export [--format csv|json]"},
		},
	},
	{
		Name:     "habit",
		Keywords: []string{"habit", "habits", "tracker", "streak", "daily", "routine"},
		Entity: Entity{
			Name: "Habit",
			Fields: []Field{
				{Name: "Name", Type: "string", JSON: "name"},
				{Name: "Frequency", Type: "string", JSON: "frequency"},
				{Name: "Streak", Type: "int", JSON: "streak"},
				{Name: "LastDone", Type: "string", JSON: "last_done"},
			},
		},
		Operations: []Operation{
			{Name: "add", Verb: "Add", Description: "add a habit to track", CLIUsage: "add <name> [--frequency daily|weekly]"},
			{Name: "done", Verb: "Done", Description: "mark habit as done today", CLIUsage: "done <name>"},
			{Name: "list", Verb: "List", Description: "list habits with streaks", CLIUsage: "list"},
			{Name: "stats", Verb: "Stats", Description: "show habit statistics", CLIUsage: "stats [name]"},
			{Name: "delete", Verb: "Delete", Description: "remove a habit", CLIUsage: "delete <name>"},
		},
	},
	{
		Name:     "snippet",
		Keywords: []string{"snippet", "snippets", "gist", "code snippet", "clipboard", "paste"},
		Entity: Entity{
			Name: "Snippet",
			Fields: []Field{
				{Name: "Title", Type: "string", JSON: "title"},
				{Name: "Language", Type: "string", JSON: "language"},
				{Name: "Code", Type: "string", JSON: "code"},
				{Name: "Tags", Type: "[]string", JSON: "tags"},
			},
		},
		Operations: []Operation{
			{Name: "save", Verb: "Save", Description: "save a code snippet", CLIUsage: "save <title> --lang <L> [--tags t1,t2]"},
			{Name: "list", Verb: "List", Description: "list snippets", CLIUsage: "list [--lang L] [--tag T]"},
			{Name: "view", Verb: "View", Description: "view a snippet", CLIUsage: "view <id>"},
			{Name: "copy", Verb: "Copy", Description: "copy snippet to clipboard", CLIUsage: "copy <id>"},
			{Name: "search", Verb: "Search", Description: "search snippets", CLIUsage: "search <query>"},
			{Name: "delete", Verb: "Delete", Description: "delete a snippet", CLIUsage: "delete <id>"},
		},
	},
	{
		Name:     "project",
		Keywords: []string{"project", "projects", "kanban", "board", "sprint", "backlog"},
		Entity: Entity{
			Name: "Project",
			Fields: []Field{
				{Name: "Name", Type: "string", JSON: "name"},
				{Name: "Description", Type: "string", JSON: "description"},
				{Name: "Status", Type: "string", JSON: "status"},
				{Name: "Deadline", Type: "string", JSON: "deadline"},
			},
		},
		Operations: []Operation{
			{Name: "create", Verb: "Create", Description: "create a project", CLIUsage: "create <name> [--deadline D]"},
			{Name: "list", Verb: "List", Description: "list projects", CLIUsage: "list [--status S]"},
			{Name: "status", Verb: "Status", Description: "show project status", CLIUsage: "status <name>"},
			{Name: "update", Verb: "Update", Description: "update project status", CLIUsage: "update <name> --status <S>"},
			{Name: "delete", Verb: "Delete", Description: "archive a project", CLIUsage: "delete <name>"},
		},
	},
}

// -----------------------------------------------------------------------
// Domain Matching
// -----------------------------------------------------------------------

// FindDomain matches a description to the best domain, or returns nil.
// Only returns a match if the domain's primary keywords (name-related)
// are present — not just generic keywords like "tracker" or "manager".
func FindDomain(desc string) *DomainDef {
	lower := strings.ToLower(desc)
	var best *DomainDef
	bestScore := 0

	// Generic words that shouldn't count as domain matches on their own
	genericWords := map[string]bool{
		"tracker": true, "manager": true, "organizer": true, "system": true,
		"tool": true, "app": true, "application": true, "service": true,
	}

	for i := range domainRegistry {
		d := &domainRegistry[i]
		score := 0
		hasSpecificMatch := false
		for _, kw := range d.Keywords {
			if strings.Contains(lower, kw) {
				score += len(kw)
				if !genericWords[kw] {
					hasSpecificMatch = true
				}
			}
		}
		// Only count this domain if a specific (non-generic) keyword matched
		if hasSpecificMatch && score > bestScore {
			bestScore = score
			best = d
		}
	}

	return best
}

// -----------------------------------------------------------------------
// Operation Extraction
// -----------------------------------------------------------------------

// operationVerbs maps natural language to standard operations.
var operationVerbs = map[string]string{
	"manage":    "crud",
	"managing":  "crud",
	"track":     "crud",
	"tracking":  "crud",
	"organize":  "crud",
	"store":     "crud",
	"list":      "list",
	"view":      "read",
	"read":      "read",
	"show":      "read",
	"display":   "read",
	"create":    "create",
	"add":       "create",
	"new":       "create",
	"write":     "create",
	"compose":   "create",
	"edit":      "update",
	"update":    "update",
	"modify":    "update",
	"change":    "update",
	"delete":    "delete",
	"remove":    "delete",
	"search":    "search",
	"find":      "search",
	"filter":    "search",
	"query":     "search",
	"send":      "send",
	"export":    "export",
	"import":    "import",
	"sync":      "sync",
	"backup":    "backup",
	"monitor":   "monitor",
	"analyze":   "analyze",
	"report":    "report",
	"summarize": "summary",
}

// ExtractOperations determines what operations the user wants from their description.
func ExtractOperations(desc string) []string {
	lower := strings.ToLower(desc)
	var ops []string
	seen := make(map[string]bool)

	for word, op := range operationVerbs {
		if strings.Contains(lower, word) {
			if op == "crud" {
				// CRUD expands to all basic operations
				for _, o := range []string{"create", "list", "read", "update", "delete"} {
					if !seen[o] {
						seen[o] = true
						ops = append(ops, o)
					}
				}
			} else if !seen[op] {
				seen[op] = true
				ops = append(ops, op)
			}
		}
	}

	// Default: CRUD if nothing specific detected
	if len(ops) == 0 {
		return []string{"create", "list", "read", "update", "delete"}
	}
	return ops
}

// -----------------------------------------------------------------------
// Context Extraction
// -----------------------------------------------------------------------

// ExtractContext pulls qualitative hints from the description.
func ExtractContext(desc string) map[string]string {
	lower := strings.ToLower(desc)
	ctx := make(map[string]string)

	// Detect "instead of X" → lightweight alternative
	if idx := strings.Index(lower, "instead of"); idx >= 0 {
		after := strings.TrimSpace(lower[idx+10:])
		word := strings.Fields(after)
		if len(word) > 0 {
			ctx["replaces"] = word[0]
			ctx["style"] = "lightweight"
		}
	}

	// Detect style hints
	for _, word := range []string{"simple", "minimal", "lightweight", "fast", "quick"} {
		if strings.Contains(lower, word) {
			ctx["style"] = "minimal"
		}
	}
	for _, word := range []string{"full", "complete", "comprehensive", "advanced", "powerful"} {
		if strings.Contains(lower, word) {
			ctx["style"] = "full"
		}
	}

	// Detect storage
	if containsAnyWord(lower, "sqlite", "database", "postgres", "mysql", "db") {
		ctx["storage"] = "database"
	} else if containsAnyWord(lower, "json file", "file storage", "local file", "disk") {
		ctx["storage"] = "json_file"
	} else {
		ctx["storage"] = "json_file" // sensible default for CLI tools
	}

	return ctx
}
