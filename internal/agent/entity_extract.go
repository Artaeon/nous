package agent

import (
	"strings"
	"unicode"
)

// -----------------------------------------------------------------------
// Smart Entity Extraction
//
// Extracts entities and their fields from ANY natural language description,
// even for domains not in the registry. This makes the coding agent work
// for unlimited domains — not just the 10 pre-defined ones.
//
// Examples:
//   "recipe manager with ingredients, cook time, and difficulty"
//   → Entity{Name: "Recipe", Fields: [{Ingredients, []string}, {CookTime, string}, {Difficulty, string}]}
//
//   "inventory system for tracking products with price, quantity, and SKU"
//   → Entity{Name: "Product", Fields: [{Price, float64}, {Quantity, int}, {SKU, string}]}
//
//   "CLI for managing blog posts with title, content, author, and tags"
//   → Entity{Name: "Post", Fields: [{Title, string}, {Content, string}, {Author, string}, {Tags, []string}]}
// -----------------------------------------------------------------------

// ExtractSmartEntity parses a natural language description and extracts
// the primary entity with its fields. Works for any domain.
func ExtractSmartEntity(desc string) *Entity {
	lower := strings.ToLower(desc)

	// Step 1: Find the entity name
	name := extractEntityName(lower)
	if name == "" {
		return nil
	}

	// Step 2: Extract explicitly mentioned fields
	fields := extractFieldsFromDesc(lower)

	// Step 3: If no explicit fields found, infer defaults from the entity name
	if len(fields) == 0 {
		fields = inferDefaultFields(name)
	}

	// Capitalize entity name
	capName := capitalize(name)
	// Singularize
	capName = singularize(capName)

	return &Entity{Name: capName, Fields: fields}
}

// extractEntityName finds the main entity noun from the description.
func extractEntityName(desc string) string {
	// Pattern: "for managing X", "for X management", "X manager", "for tracking X"
	patterns := []struct {
		prefix string
		suffix string
	}{
		{"for managing ", ""},
		{"for tracking ", ""},
		{"to manage ", ""},
		{"to track ", ""},
		{"for organizing ", ""},
		{"for storing ", ""},
		{"that manages ", ""},
		{"that tracks ", ""},
		{"to store ", ""},
		{"for ", " management"},
		{"", " manager"},
		{"", " tracker"},
		{"", " organizer"},
		{"", " system"},
		{"for ", ""},
	}

	for _, p := range patterns {
		var start int
		if p.prefix != "" {
			idx := strings.Index(desc, p.prefix)
			if idx < 0 {
				continue
			}
			start = idx + len(p.prefix)
		}

		remaining := desc[start:]

		if p.suffix != "" {
			idx := strings.Index(remaining, p.suffix)
			if idx < 0 {
				continue
			}
			word := strings.TrimSpace(remaining[:idx])
			words := strings.Fields(word)
			if len(words) > 0 {
				return words[len(words)-1]
			}
			continue
		}

		// Take the first noun-like word after the prefix
		words := strings.Fields(remaining)
		for _, w := range words {
			w = strings.Trim(w, ".,;:!?\"'()")
			if len(w) < 2 {
				continue
			}
			// Skip common non-entity words
			if isStopWord(w) {
				continue
			}
			return w
		}
	}

	return ""
}

// extractFieldsFromDesc finds explicitly mentioned fields.
// Looks for "with X, Y, and Z" or "including X, Y, Z" patterns.
func extractFieldsFromDesc(desc string) []Field {
	// Find "with ...", "including ...", "that has ...", "containing ..."
	triggers := []string{" with ", " including ", " that has ", " containing ", " having ", " fields: "}
	var fieldSection string

	for _, t := range triggers {
		idx := strings.Index(desc, t)
		if idx >= 0 {
			fieldSection = desc[idx+len(t):]
			break
		}
	}

	if fieldSection == "" {
		return nil
	}

	// Clean up: take until end of sentence or next clause
	for _, stop := range []string{". ", " that ", " which ", " and also ", " instead ", " rather "} {
		if idx := strings.Index(fieldSection, stop); idx > 0 {
			fieldSection = fieldSection[:idx]
		}
	}

	// Split on commas and "and"
	fieldSection = strings.ReplaceAll(fieldSection, ", and ", ", ")
	fieldSection = strings.ReplaceAll(fieldSection, " and ", ", ")
	parts := strings.Split(fieldSection, ",")

	var fields []Field
	for _, part := range parts {
		part = strings.TrimSpace(part)
		part = strings.Trim(part, ".,;:!?\"'()")
		if part == "" || len(part) > 40 {
			continue
		}

		fieldName, fieldType := parseFieldSpec(part)
		if fieldName == "" {
			continue
		}

		fields = append(fields, Field{
			Name: capitalize(fieldName),
			Type: fieldType,
			JSON: toSnakeCase(fieldName),
		})
	}

	return fields
}

// parseFieldSpec parses a field mention like "cook time", "price", "tags"
// and infers name and Go type.
func parseFieldSpec(spec string) (string, string) {
	spec = strings.TrimSpace(strings.ToLower(spec))
	if spec == "" {
		return "", ""
	}

	// Remove articles
	for _, prefix := range []string{"a ", "an ", "the ", "its ", "their "} {
		spec = strings.TrimPrefix(spec, prefix)
	}

	// Check if it ends with a type hint
	words := strings.Fields(spec)
	if len(words) == 0 {
		return "", ""
	}

	// Infer type from the field name
	fieldName := toCamelCase(spec)
	fieldType := inferType(spec)

	return fieldName, fieldType
}

// inferType guesses the Go type from a field name/description.
func inferType(name string) string {
	lower := strings.ToLower(name)

	// Plural/list indicators → slice
	if strings.HasSuffix(lower, "s") && containsAnyWord(lower, "tag", "item", "ingredient", "categor", "label", "keyword", "skill", "member") {
		return "[]string"
	}
	if containsAnyWord(lower, "list of", "array of", "multiple") {
		return "[]string"
	}

	// Numeric indicators
	if containsAnyWord(lower, "price", "cost", "amount", "salary", "rate", "score", "rating", "percentage", "weight", "height") {
		return "float64"
	}
	if containsAnyWord(lower, "count", "quantity", "number", "age", "year", "port", "size", "length", "width", "priority", "level") {
		return "int"
	}

	// Boolean indicators
	if containsAnyWord(lower, "completed", "done", "active", "enabled", "verified", "read", "public", "archived", "deleted", "favorite") {
		return "bool"
	}
	if strings.HasPrefix(lower, "is ") || strings.HasPrefix(lower, "has ") {
		return "bool"
	}

	// Default: string
	return "string"
}

// inferDefaultFields returns sensible default fields for common entity names.
func inferDefaultFields(entityName string) []Field {
	lower := strings.ToLower(entityName)

	// Check known entities first
	if fields, ok := knownEntities[lower]; ok {
		return fields
	}

	// Generic defaults based on the entity name
	defaults := []Field{
		{Name: "Name", Type: "string", JSON: "name"},
		{Name: "Description", Type: "string", JSON: "description"},
	}

	// Add domain-specific defaults
	switch {
	case containsAnyWord(lower, "recipe", "dish", "meal", "food"):
		return []Field{
			{Name: "Name", Type: "string", JSON: "name"},
			{Name: "Ingredients", Type: "[]string", JSON: "ingredients"},
			{Name: "Instructions", Type: "string", JSON: "instructions"},
			{Name: "PrepTime", Type: "string", JSON: "prep_time"},
			{Name: "Servings", Type: "int", JSON: "servings"},
		}
	case containsAnyWord(lower, "movie", "film", "show", "series"):
		return []Field{
			{Name: "Title", Type: "string", JSON: "title"},
			{Name: "Director", Type: "string", JSON: "director"},
			{Name: "Year", Type: "int", JSON: "year"},
			{Name: "Rating", Type: "float64", JSON: "rating"},
			{Name: "Genre", Type: "string", JSON: "genre"},
		}
	case containsAnyWord(lower, "song", "music", "track", "album"):
		return []Field{
			{Name: "Title", Type: "string", JSON: "title"},
			{Name: "Artist", Type: "string", JSON: "artist"},
			{Name: "Album", Type: "string", JSON: "album"},
			{Name: "Duration", Type: "string", JSON: "duration"},
			{Name: "Genre", Type: "string", JSON: "genre"},
		}
	case containsAnyWord(lower, "log", "entry", "record"):
		return []Field{
			{Name: "Message", Type: "string", JSON: "message"},
			{Name: "Level", Type: "string", JSON: "level"},
			{Name: "Source", Type: "string", JSON: "source"},
		}
	case containsAnyWord(lower, "customer", "client", "person", "member"):
		return []Field{
			{Name: "Name", Type: "string", JSON: "name"},
			{Name: "Email", Type: "string", JSON: "email"},
			{Name: "Phone", Type: "string", JSON: "phone"},
			{Name: "Address", Type: "string", JSON: "address"},
		}
	case containsAnyWord(lower, "server", "service", "endpoint", "host"):
		return []Field{
			{Name: "Name", Type: "string", JSON: "name"},
			{Name: "Host", Type: "string", JSON: "host"},
			{Name: "Port", Type: "int", JSON: "port"},
			{Name: "Status", Type: "string", JSON: "status"},
		}
	case containsAnyWord(lower, "order", "purchase", "transaction"):
		return []Field{
			{Name: "Customer", Type: "string", JSON: "customer"},
			{Name: "Total", Type: "float64", JSON: "total"},
			{Name: "Status", Type: "string", JSON: "status"},
			{Name: "Items", Type: "[]string", JSON: "items"},
		}
	case containsAnyWord(lower, "inventory", "stock", "warehouse"):
		return []Field{
			{Name: "Name", Type: "string", JSON: "name"},
			{Name: "SKU", Type: "string", JSON: "sku"},
			{Name: "Quantity", Type: "int", JSON: "quantity"},
			{Name: "Price", Type: "float64", JSON: "price"},
		}
	case containsAnyWord(lower, "goal", "objective", "milestone"):
		return []Field{
			{Name: "Title", Type: "string", JSON: "title"},
			{Name: "Description", Type: "string", JSON: "description"},
			{Name: "Deadline", Type: "string", JSON: "deadline"},
			{Name: "Progress", Type: "int", JSON: "progress"},
			{Name: "Completed", Type: "bool", JSON: "completed"},
		}
	}

	return defaults
}

// -----------------------------------------------------------------------
// String Helpers
// -----------------------------------------------------------------------

func capitalize(s string) string {
	if s == "" {
		return ""
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

func singularize(s string) string {
	if strings.HasSuffix(s, "ies") {
		return s[:len(s)-3] + "y"
	}
	if strings.HasSuffix(s, "ses") || strings.HasSuffix(s, "xes") {
		return s[:len(s)-2]
	}
	if strings.HasSuffix(s, "s") && !strings.HasSuffix(s, "ss") {
		return s[:len(s)-1]
	}
	return s
}

func toCamelCase(s string) string {
	words := strings.Fields(s)
	for i := range words {
		words[i] = capitalize(strings.ToLower(words[i]))
	}
	return strings.Join(words, "")
}

func toSnakeCase(s string) string {
	var result strings.Builder
	for i, r := range s {
		if unicode.IsUpper(r) && i > 0 {
			result.WriteRune('_')
		}
		result.WriteRune(unicode.ToLower(r))
	}
	return result.String()
}

var stopWords = map[string]bool{
	"a": true, "an": true, "the": true, "in": true, "on": true, "at": true,
	"to": true, "for": true, "of": true, "with": true, "and": true, "or": true,
	"that": true, "this": true, "it": true, "is": true, "are": true, "was": true,
	"be": true, "been": true, "being": true, "have": true, "has": true, "had": true,
	"do": true, "does": true, "did": true, "will": true, "would": true, "could": true,
	"should": true, "can": true, "may": true, "might": true, "must": true,
	"my": true, "your": true, "our": true, "their": true, "its": true,
	"me": true, "you": true, "him": true, "her": true, "us": true, "them": true,
	"i": true, "we": true, "he": true, "she": true, "they": true,
	"not": true, "no": true, "nor": true, "but": true, "if": true, "then": true,
	"than": true, "when": true, "where": true, "how": true, "what": true, "which": true,
	"who": true, "whom": true, "why": true, "all": true, "each": true, "every": true,
	"both": true, "few": true, "more": true, "most": true, "some": true, "any": true,
	"such": true, "very": true, "just": true, "also": true, "so": true, "too": true,
	"using": true, "instead": true, "bloated": true, "simple": true, "fast": true,
	"helps": true, "help": true, "build": true, "create": true, "make": true,
	"go": true, "golang": true, "cli": true, "tool": true, "app": true,
	"application": true, "program": true, "software": true,
}

func isStopWord(w string) bool {
	return stopWords[strings.ToLower(w)]
}
