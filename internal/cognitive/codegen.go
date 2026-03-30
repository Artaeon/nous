package cognitive

import (
	"fmt"
	"strings"
)

// -----------------------------------------------------------------------
// Code Template Generator
//
// Generates common code patterns in Python, JavaScript, and Go by
// filling templates with user-specified parameters. Not a language model —
// this produces working, idiomatic code for common patterns.
// -----------------------------------------------------------------------

// CodeGenerator creates code from templates with slot filling.
type CodeGenerator struct{}

// NewCodeGenerator creates a code generator.
func NewCodeGenerator() *CodeGenerator {
	return &CodeGenerator{}
}

// CodeRequest describes what code to generate.
type CodeRequest struct {
	Language    string // "python", "javascript", "go"
	Pattern     string // "csv_reader", "http_handler", "test", etc.
	Description string // original user request (for slot extraction)
	Params      map[string]string // extracted parameters
}

// GeneratedCode is the output of code generation.
type GeneratedCode struct {
	Code     string // the generated code
	Language string // language name
	Pattern  string // which pattern was used
	Explanation string // brief explanation of the code
}

// Generate produces code for a request. Returns nil if the pattern
// isn't recognized.
func (cg *CodeGenerator) Generate(req CodeRequest) *GeneratedCode {
	lang := strings.ToLower(req.Language)
	pattern := req.Pattern

	// Auto-detect pattern from description if not set
	if pattern == "" {
		pattern = detectCodePattern(req.Description, lang)
	}

	switch lang {
	case "python", "py":
		return cg.generatePython(pattern, req)
	case "javascript", "js", "node":
		return cg.generateJavaScript(pattern, req)
	case "go", "golang":
		return cg.generateGo(pattern, req)
	default:
		// Try to detect language from description
		lang = detectCodeLanguage(req.Description)
		if lang != "" {
			req.Language = lang
			return cg.Generate(req)
		}
		return nil
	}
}

// GenerateFromQuery is the main entry point — takes a natural language
// query and produces code.
func (cg *CodeGenerator) GenerateFromQuery(query string) *GeneratedCode {
	lang := detectCodeLanguage(query)
	if lang == "" {
		lang = "python" // default
	}
	pattern := detectCodePattern(query, lang)
	params := extractCodeParams(query)

	return cg.Generate(CodeRequest{
		Language:    lang,
		Pattern:     pattern,
		Description: query,
		Params:      params,
	})
}

// -----------------------------------------------------------------------
// Python templates
// -----------------------------------------------------------------------

func (cg *CodeGenerator) generatePython(pattern string, req CodeRequest) *GeneratedCode {
	p := req.Params
	funcName := paramOr(p, "function_name", "main")

	switch pattern {
	case "csv_reader":
		fileName := paramOr(p, "file_path", "data.csv")
		return &GeneratedCode{
			Language: "python",
			Pattern:  pattern,
			Code: fmt.Sprintf(`import csv

def read_csv(file_path: str = "%s") -> list[dict]:
    """Read a CSV file and return a list of dictionaries."""
    rows = []
    with open(file_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows

# Usage
if __name__ == "__main__":
    data = read_csv()
    for row in data:
        print(row)
`, fileName),
			Explanation: "Reads a CSV file using DictReader, returning each row as a dictionary.",
		}

	case "http_request":
		url := paramOr(p, "url", "https://api.example.com/data")
		return &GeneratedCode{
			Language: "python",
			Pattern:  pattern,
			Code: fmt.Sprintf(`import urllib.request
import json

def fetch_data(url: str = "%s") -> dict:
    """Fetch JSON data from a URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())

# Usage
if __name__ == "__main__":
    data = fetch_data()
    print(json.dumps(data, indent=2))
`, url),
			Explanation: "Makes an HTTP GET request and parses the JSON response.",
		}

	case "file_processor":
		return &GeneratedCode{
			Language: "python",
			Pattern:  pattern,
			Code: `import os

def process_files(directory: str, extension: str = ".txt") -> list[str]:
    """Process all files with given extension in a directory."""
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                content = f.read()
            results.append(f"{filename}: {len(content)} chars")
    return results

# Usage
if __name__ == "__main__":
    for result in process_files("."):
        print(result)
`,
			Explanation: "Walks a directory, reads files matching an extension, and returns summaries.",
		}

	case "class":
		className := paramOr(p, "class_name", "DataProcessor")
		return &GeneratedCode{
			Language: "python",
			Pattern:  pattern,
			Code: fmt.Sprintf(`class %s:
    """A data processing class."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.data = []

    def add(self, item) -> None:
        """Add an item to the dataset."""
        self.data.append(item)

    def process(self) -> list:
        """Process and return the data."""
        return [str(item).strip() for item in self.data if item]

    def summary(self) -> str:
        """Return a summary of the data."""
        return f"{self.name}: {len(self.data)} items"

    def __repr__(self) -> str:
        return f"%s(name='{self.name}', items={len(self.data)})"

# Usage
if __name__ == "__main__":
    dp = %s("example")
    dp.add("hello")
    dp.add("world")
    print(dp.summary())
    print(dp.process())
`, className, className, className),
			Explanation: fmt.Sprintf("A %s class with add, process, and summary methods.", className),
		}

	case "test":
		return &GeneratedCode{
			Language: "python",
			Pattern:  pattern,
			Code: fmt.Sprintf(`import unittest

def %s(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

class Test%s(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(%s(2, 3), 5)

    def test_negative(self):
        self.assertEqual(%s(-1, 1), 0)

    def test_zero(self):
        self.assertEqual(%s(0, 0), 0)

if __name__ == "__main__":
    unittest.main()
`, funcName, capitalizeFirst(funcName), funcName, funcName, funcName),
			Explanation: "A function with unittest test cases covering basic, negative, and zero inputs.",
		}

	case "pandas":
		return &GeneratedCode{
			Language: "python",
			Pattern:  pattern,
			Code: `import csv
from collections import Counter

def analyze_csv(file_path: str) -> dict:
    """Analyze a CSV file: row count, column stats, value frequencies."""
    with open(file_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"rows": 0, "columns": []}

    columns = list(rows[0].keys())
    stats = {}
    for col in columns:
        values = [row[col] for row in rows if row[col]]
        freq = Counter(values).most_common(5)
        stats[col] = {
            "count": len(values),
            "unique": len(set(values)),
            "top_values": freq,
        }

    return {"rows": len(rows), "columns": columns, "stats": stats}

# Usage
if __name__ == "__main__":
    import json
    result = analyze_csv("data.csv")
    print(json.dumps(result, indent=2, default=str))
`,
			Explanation: "Analyzes a CSV file: counts rows, finds unique values, and shows top frequencies per column.",
		}

	default:
		return cg.generatePythonGeneric(req)
	}
}

func (cg *CodeGenerator) generatePythonGeneric(req CodeRequest) *GeneratedCode {
	funcName := paramOr(req.Params, "function_name", "main")
	return &GeneratedCode{
		Language: "python",
		Pattern:  "generic",
		Code: fmt.Sprintf(`def %s():
    """TODO: Implement — %s"""
    pass

if __name__ == "__main__":
    %s()
`, funcName, req.Description, funcName),
		Explanation: "A skeleton function. Fill in the implementation.",
	}
}

// -----------------------------------------------------------------------
// JavaScript templates
// -----------------------------------------------------------------------

func (cg *CodeGenerator) generateJavaScript(pattern string, req CodeRequest) *GeneratedCode {
	switch pattern {
	case "fetch", "http_request":
		url := paramOr(req.Params, "url", "https://api.example.com/data")
		return &GeneratedCode{
			Language: "javascript",
			Pattern:  "fetch",
			Code: fmt.Sprintf(`async function fetchData(url = '%s') {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('HTTP ' + response.status);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Fetch failed:', error.message);
    throw error;
  }
}

// Usage
fetchData().then(data => console.log(data));
`, url),
			Explanation: "Async fetch with error handling and JSON parsing.",
		}

	case "express", "http_handler":
		return &GeneratedCode{
			Language: "javascript",
			Pattern:  "express",
			Code: `const http = require('http');

const routes = {
  'GET /': (req, res) => {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ message: 'Hello, World!' }));
  },
  'GET /health': (req, res) => {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok' }));
  },
};

const server = http.createServer((req, res) => {
  const key = req.method + ' ' + req.url;
  const handler = routes[key];
  if (handler) {
    handler(req, res);
  } else {
    res.writeHead(404);
    res.end('Not Found');
  }
});

server.listen(3000, () => console.log('Server running on port 3000'));
`,
			Explanation: "A minimal HTTP server with route handling using Node.js built-in http module.",
		}

	case "react", "component":
		name := paramOr(req.Params, "component_name", "DataList")
		return &GeneratedCode{
			Language: "javascript",
			Pattern:  "react",
			Code: fmt.Sprintf(`function %s({ items = [], onSelect }) {
  if (items.length === 0) {
    return <p>No items to display.</p>;
  }

  return (
    <ul>
      {items.map((item, index) => (
        <li key={index} onClick={() => onSelect?.(item)}>
          {item.name || item.toString()}
        </li>
      ))}
    </ul>
  );
}

// Usage:
// <DataList items={data} onSelect={(item) => console.log(item)} />
`, name),
			Explanation: fmt.Sprintf("A %s React component that renders a list with click handlers.", name),
		}

	case "dom":
		return &GeneratedCode{
			Language: "javascript",
			Pattern:  "dom",
			Code: `function createElement(tag, attrs = {}, children = []) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key.startsWith('on')) {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else {
      el.setAttribute(key, value);
    }
  }
  for (const child of children) {
    el.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
  }
  return el;
}

// Usage
const button = createElement('button',
  { class: 'btn', onClick: () => alert('Clicked!') },
  ['Click Me']
);
document.body.appendChild(button);
`,
			Explanation: "A utility for creating DOM elements with attributes and event listeners.",
		}

	case "async":
		return &GeneratedCode{
			Language: "javascript",
			Pattern:  "async",
			Code: `async function processInParallel(urls) {
  const results = await Promise.allSettled(
    urls.map(url => fetch(url).then(r => r.json()))
  );

  const successes = results
    .filter(r => r.status === 'fulfilled')
    .map(r => r.value);

  const failures = results
    .filter(r => r.status === 'rejected')
    .map(r => r.reason.message);

  return { successes, failures };
}

// Usage
processInParallel([
  'https://api.example.com/a',
  'https://api.example.com/b',
]).then(({ successes, failures }) => {
  console.log('Succeeded:', successes.length);
  console.log('Failed:', failures.length);
});
`,
			Explanation: "Parallel async requests with Promise.allSettled for graceful error handling.",
		}

	default:
		return &GeneratedCode{
			Language: "javascript",
			Pattern:  "generic",
			Code: fmt.Sprintf(`// %s
function main() {
  // TODO: implement
  console.log('Not implemented yet');
}

main();
`, req.Description),
			Explanation: "A skeleton function. Fill in the implementation.",
		}
	}
}

// -----------------------------------------------------------------------
// Go templates
// -----------------------------------------------------------------------

func (cg *CodeGenerator) generateGo(pattern string, req CodeRequest) *GeneratedCode {
	switch pattern {
	case "http_handler":
		return &GeneratedCode{
			Language: "go",
			Pattern:  "http_handler",
			Code: `package main

import (
	"encoding/json"
	"log"
	"net/http"
)

func main() {
	http.HandleFunc("/", handleIndex)
	http.HandleFunc("/health", handleHealth)
	log.Println("Listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"message": "Hello, World!"})
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}
`,
			Explanation: "An HTTP server with JSON handlers using the standard library.",
		}

	case "struct":
		name := paramOr(req.Params, "struct_name", "Item")
		return &GeneratedCode{
			Language: "go",
			Pattern:  "struct",
			Code: fmt.Sprintf(`package main

import "fmt"

type %s struct {
	Name  string
	Value int
}

func New%s(name string, value int) *%s {
	return &%s{Name: name, Value: value}
}

func (i *%s) String() string {
	return fmt.Sprintf("%%s: %%d", i.Name, i.Value)
}

func (i *%s) IsValid() bool {
	return i.Name != "" && i.Value >= 0
}

func main() {
	item := New%s("example", 42)
	fmt.Println(item)
	fmt.Println("Valid:", item.IsValid())
}
`, name, name, name, name, name, name, name),
			Explanation: fmt.Sprintf("A %s struct with constructor, String method, and validation.", name),
		}

	case "file_reader":
		return &GeneratedCode{
			Language: "go",
			Pattern:  "file_reader",
			Code: `package main

import (
	"bufio"
	"fmt"
	"os"
)

func readLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var lines []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	return lines, scanner.Err()
}

func main() {
	lines, err := readLines("input.txt")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	for i, line := range lines {
		fmt.Printf("%d: %s\n", i+1, line)
	}
}
`,
			Explanation: "Reads a file line by line using bufio.Scanner.",
		}

	case "goroutine", "concurrent":
		return &GeneratedCode{
			Language: "go",
			Pattern:  "goroutine",
			Code: `package main

import (
	"fmt"
	"sync"
)

func processItems(items []string) []string {
	results := make([]string, len(items))
	var wg sync.WaitGroup

	for i, item := range items {
		wg.Add(1)
		go func(idx int, val string) {
			defer wg.Done()
			// Process each item concurrently
			results[idx] = fmt.Sprintf("processed: %s", val)
		}(i, item)
	}

	wg.Wait()
	return results
}

func main() {
	items := []string{"alpha", "beta", "gamma", "delta"}
	results := processItems(items)
	for _, r := range results {
		fmt.Println(r)
	}
}
`,
			Explanation: "Concurrent processing with goroutines and sync.WaitGroup.",
		}

	case "test":
		funcName := paramOr(req.Params, "function_name", "Add")
		return &GeneratedCode{
			Language: "go",
			Pattern:  "test",
			Code: fmt.Sprintf(`package main

import "testing"

func %s(a, b int) int {
	return a + b
}

func Test%s(t *testing.T) {
	tests := []struct {
		a, b, want int
	}{
		{2, 3, 5},
		{-1, 1, 0},
		{0, 0, 0},
		{100, -50, 50},
	}
	for _, tt := range tests {
		got := %s(tt.a, tt.b)
		if got != tt.want {
			t.Errorf("%s(%%d, %%d) = %%d, want %%d", tt.a, tt.b, got, tt.want)
		}
	}
}
`, funcName, funcName, funcName, funcName),
			Explanation: "A table-driven test following Go conventions.",
		}

	default:
		return &GeneratedCode{
			Language: "go",
			Pattern:  "generic",
			Code: fmt.Sprintf(`package main

import "fmt"

func main() {
	// TODO: %s
	fmt.Println("Not implemented yet")
}
`, req.Description),
			Explanation: "A skeleton main function. Fill in the implementation.",
		}
	}
}

// -----------------------------------------------------------------------
// Pattern and language detection from natural language
// -----------------------------------------------------------------------

func detectCodeLanguage(query string) string {
	lower := strings.ToLower(query)
	if strings.Contains(lower, "python") || strings.Contains(lower, "pandas") || strings.Contains(lower, "csv") {
		return "python"
	}
	if strings.Contains(lower, "javascript") || strings.Contains(lower, "react") ||
		strings.Contains(lower, "node") || strings.Contains(lower, "express") ||
		strings.Contains(lower, "dom") || strings.Contains(lower, "fetch api") {
		return "javascript"
	}
	if strings.Contains(lower, " go ") || strings.Contains(lower, "golang") ||
		strings.Contains(lower, "goroutine") {
		return "go"
	}
	return ""
}

func detectCodePattern(query, lang string) string {
	lower := strings.ToLower(query)

	// CSV/data patterns
	if strings.Contains(lower, "csv") || strings.Contains(lower, "read csv") {
		return "csv_reader"
	}
	if strings.Contains(lower, "pandas") || strings.Contains(lower, "data analysis") ||
		strings.Contains(lower, "analyze") && strings.Contains(lower, "data") {
		return "pandas"
	}

	// HTTP/API patterns
	if strings.Contains(lower, "http") || strings.Contains(lower, "api") ||
		strings.Contains(lower, "server") || strings.Contains(lower, "endpoint") ||
		strings.Contains(lower, "handler") || strings.Contains(lower, "route") {
		if strings.Contains(lower, "request") || strings.Contains(lower, "fetch") ||
			strings.Contains(lower, "call") || strings.Contains(lower, "get") {
			return "http_request"
		}
		return "http_handler"
	}

	// File patterns
	if strings.Contains(lower, "file") && (strings.Contains(lower, "read") || strings.Contains(lower, "process")) {
		return "file_reader"
	}

	// Class/struct patterns
	if strings.Contains(lower, "class") || strings.Contains(lower, "struct") {
		if lang == "go" || lang == "golang" {
			return "struct"
		}
		return "class"
	}

	// Test patterns
	if strings.Contains(lower, "test") || strings.Contains(lower, "unit test") {
		return "test"
	}

	// React/component patterns
	if strings.Contains(lower, "react") || strings.Contains(lower, "component") {
		return "react"
	}

	// DOM patterns
	if strings.Contains(lower, "dom") {
		return "dom"
	}

	// Async patterns
	if strings.Contains(lower, "async") || strings.Contains(lower, "parallel") ||
		strings.Contains(lower, "concurrent") {
		if lang == "go" || lang == "golang" {
			return "goroutine"
		}
		return "async"
	}

	// Express/route patterns
	if strings.Contains(lower, "express") || strings.Contains(lower, "route") {
		return "express"
	}

	return "generic"
}

func extractCodeParams(query string) map[string]string {
	params := make(map[string]string)

	lower := strings.ToLower(query)

	// Extract file path
	if idx := strings.Index(lower, ".csv"); idx > 0 {
		// Walk backwards to find the start of the filename
		start := idx
		for start > 0 && lower[start-1] != ' ' && lower[start-1] != '"' && lower[start-1] != '\'' {
			start--
		}
		params["file_path"] = query[start : idx+4]
	}

	// Extract URL
	if idx := strings.Index(lower, "http"); idx >= 0 {
		end := strings.IndexAny(query[idx:], " \"')")
		if end < 0 {
			end = len(query) - idx
		}
		params["url"] = query[idx : idx+end]
	}

	// Extract function/class name from "called X" or "named X"
	for _, prefix := range []string{"called ", "named "} {
		if idx := strings.Index(lower, prefix); idx >= 0 {
			rest := query[idx+len(prefix):]
			end := strings.IndexAny(rest, " ,.\n")
			if end < 0 {
				end = len(rest)
			}
			name := strings.TrimSpace(rest[:end])
			if name != "" {
				params["function_name"] = name
				params["class_name"] = name
				params["struct_name"] = name
				params["component_name"] = name
			}
		}
	}

	return params
}

func paramOr(params map[string]string, key, fallback string) string {
	if v, ok := params[key]; ok && v != "" {
		return v
	}
	return fallback
}
