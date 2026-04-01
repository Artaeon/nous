package cognitive

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"strings"
	"unicode"
)

// -----------------------------------------------------------------------
// Code Explainer
//
// Analyzes Go source code AST to produce semantic descriptions of what
// functions do — not just what syntax patterns are present. Uses a
// database of known Go stdlib functions, composite pattern detection,
// and data-flow summarization to generate Copilot-quality explanations.
// -----------------------------------------------------------------------

// CodeExplainer produces semantic descriptions of Go functions.
type CodeExplainer struct {
	fset *token.FileSet
}

// NewCodeExplainer creates a code explainer.
func NewCodeExplainer() *CodeExplainer {
	return &CodeExplainer{
		fset: token.NewFileSet(),
	}
}

// ExplainedFunc holds the explanation for a single function.
type ExplainedFunc struct {
	Name       string   // function name
	Signature  string   // parameters and return types
	Summary    string   // coherent paragraph description
	Patterns   []string // detected semantic patterns
	DataFlow   string   // input → transform → output summary
	Composite  []string // higher-level composite patterns
}

// knownFunctions maps well-known Go stdlib and common library
// function/method calls to their semantic meaning. This enables the
// explainer to describe WHAT code does rather than just noting syntax.
var knownFunctions = map[string]string{
	// ---------------------------------------------------------------
	// net/http — server
	// ---------------------------------------------------------------
	"http.ListenAndServe":    "starts an HTTP server",
	"http.ListenAndServeTLS": "starts an HTTPS/TLS server",
	"http.HandleFunc":        "registers an HTTP route handler",
	"http.Handle":            "registers an HTTP handler",
	"http.Serve":             "serves HTTP on a listener",
	"http.NewServeMux":       "creates an HTTP request multiplexer",
	"mux.HandleFunc":         "registers a route on the multiplexer",
	"mux.Handle":             "registers a handler on the multiplexer",
	"http.StripPrefix":       "strips a URL prefix before handling",
	"http.FileServer":        "serves static files from a directory",
	"http.NotFound":          "sends a 404 Not Found response",
	"http.Error":             "sends an HTTP error response",
	"http.Redirect":          "sends an HTTP redirect",
	"http.SetCookie":         "sets an HTTP cookie on the response",
	"http.MaxBytesReader":    "limits the size of the request body",
	"http.TimeoutHandler":    "wraps a handler with a timeout",

	// net/http — client
	"http.Get":            "makes an HTTP GET request",
	"http.Post":           "makes an HTTP POST request",
	"http.PostForm":       "sends a form-encoded POST request",
	"http.Head":           "makes an HTTP HEAD request",
	"http.NewRequest":     "creates a new HTTP request",
	"http.NewRequestWithContext": "creates a new HTTP request with context",
	"client.Do":           "executes an HTTP request",
	"http.DefaultClient":  "uses the default HTTP client",
	"resp.Body.Close":     "closes the HTTP response body",

	// net/http — response writer
	"w.Header":      "sets HTTP response headers",
	"w.WriteHeader":  "sets the HTTP status code",
	"w.Write":        "writes the HTTP response body",
	"r.FormValue":    "reads a form parameter",
	"r.URL.Query":    "reads URL query parameters",
	"r.ParseForm":    "parses the request form data",
	"r.ParseMultipartForm": "parses a multipart form (file upload)",
	"r.Cookie":       "reads a cookie from the request",
	"r.Context":      "gets the request context",
	"r.Body":         "accesses the request body",
	"r.Header.Get":   "reads a request header",
	"r.Method":       "checks the HTTP method",
	"r.URL.Path":     "reads the request URL path",

	// ---------------------------------------------------------------
	// encoding/json
	// ---------------------------------------------------------------
	"json.Marshal":       "serializes data to JSON",
	"json.MarshalIndent": "serializes data to pretty-printed JSON",
	"json.Unmarshal":     "deserializes JSON data",
	"json.NewEncoder":    "creates a JSON encoder for streaming",
	"json.NewDecoder":    "creates a JSON decoder for streaming",
	"enc.Encode":         "writes a JSON-encoded value",
	"dec.Decode":         "reads and decodes a JSON value",
	"json.Valid":         "checks if bytes are valid JSON",
	"json.Compact":       "compacts JSON encoding",
	"json.Indent":        "indents JSON for readability",

	// ---------------------------------------------------------------
	// encoding — other formats
	// ---------------------------------------------------------------
	"xml.Marshal":          "serializes data to XML",
	"xml.Unmarshal":        "deserializes XML data",
	"xml.NewEncoder":       "creates an XML encoder",
	"xml.NewDecoder":       "creates an XML decoder",
	"csv.NewReader":        "creates a CSV reader",
	"csv.NewWriter":        "creates a CSV writer",
	"gob.NewEncoder":       "creates a gob encoder for Go values",
	"gob.NewDecoder":       "creates a gob decoder for Go values",
	"base64.StdEncoding":   "base64 encodes/decodes",
	"base64.URLEncoding":   "URL-safe base64 encodes/decodes",
	"hex.EncodeToString":   "encodes bytes to hexadecimal string",
	"hex.DecodeString":     "decodes hexadecimal string to bytes",
	"binary.Write":         "writes binary data",
	"binary.Read":          "reads binary data",
	"binary.BigEndian":     "uses big-endian byte order",
	"binary.LittleEndian":  "uses little-endian byte order",

	// ---------------------------------------------------------------
	// database/sql
	// ---------------------------------------------------------------
	"sql.Open":        "opens a database connection",
	"db.Ping":         "verifies the database connection",
	"db.PingContext":   "verifies the database connection with context",
	"db.Query":        "executes a database query returning rows",
	"db.QueryContext":  "executes a database query with context",
	"db.QueryRow":     "queries a single database row",
	"db.QueryRowContext": "queries a single database row with context",
	"db.Exec":         "executes a database statement",
	"db.ExecContext":   "executes a database statement with context",
	"db.Prepare":      "prepares a database statement",
	"db.Begin":        "begins a database transaction",
	"db.BeginTx":      "begins a database transaction with options",
	"db.Close":        "closes the database connection",
	"rows.Scan":       "reads database columns into variables",
	"rows.Next":       "advances to the next database row",
	"rows.Close":      "closes the result set",
	"row.Scan":        "reads a single row into variables",
	"tx.Commit":       "commits a database transaction",
	"tx.Rollback":     "rolls back a database transaction",
	"tx.Exec":         "executes within a transaction",
	"tx.Query":        "queries within a transaction",
	"stmt.Exec":       "executes a prepared statement",
	"stmt.Query":      "queries with a prepared statement",
	"stmt.Close":      "closes a prepared statement",

	// ---------------------------------------------------------------
	// context
	// ---------------------------------------------------------------
	"context.Background":    "creates a root context",
	"context.TODO":          "creates a placeholder context",
	"context.WithTimeout":   "creates a context with timeout",
	"context.WithDeadline":  "creates a context with deadline",
	"context.WithCancel":    "creates a cancellable context",
	"context.WithValue":     "creates a context with a key-value pair",
	"ctx.Done":              "checks if context is cancelled",
	"ctx.Err":               "returns context cancellation error",
	"ctx.Value":             "retrieves a value from context",
	"ctx.Deadline":          "returns the context deadline",

	// ---------------------------------------------------------------
	// sync
	// ---------------------------------------------------------------
	"sync.NewMutex":       "creates a mutex",
	"mu.Lock":             "acquires a mutex lock",
	"mu.Unlock":           "releases a mutex lock",
	"mu.RLock":            "acquires a read lock",
	"mu.RUnlock":          "releases a read lock",
	"wg.Add":              "increments the wait group counter",
	"wg.Done":             "decrements the wait group counter",
	"wg.Wait":             "blocks until all goroutines complete",
	"once.Do":             "executes a function exactly once",
	"pool.Get":            "retrieves an object from the sync pool",
	"pool.Put":            "returns an object to the sync pool",
	"cond.Wait":           "waits for a condition signal",
	"cond.Signal":         "wakes one waiting goroutine",
	"cond.Broadcast":      "wakes all waiting goroutines",
	"sync.Map":            "uses a concurrent-safe map",
	"m.Load":              "reads from a concurrent map",
	"m.Store":             "writes to a concurrent map",
	"m.LoadOrStore":       "reads or inserts into a concurrent map",
	"m.Delete":            "removes from a concurrent map",
	"m.Range":             "iterates over a concurrent map",

	// sync/atomic
	"atomic.AddInt64":      "atomically increments a counter",
	"atomic.LoadInt64":     "atomically reads a value",
	"atomic.StoreInt64":    "atomically writes a value",
	"atomic.CompareAndSwapInt64": "atomically compares and swaps a value",

	// ---------------------------------------------------------------
	// os
	// ---------------------------------------------------------------
	"os.Open":           "opens a file for reading",
	"os.Create":         "creates a new file",
	"os.OpenFile":       "opens a file with specific flags",
	"os.ReadFile":       "reads an entire file into memory",
	"os.WriteFile":      "writes data to a file",
	"os.Stat":           "checks file metadata",
	"os.Lstat":          "checks file metadata without following symlinks",
	"os.MkdirAll":       "creates directories recursively",
	"os.Mkdir":          "creates a directory",
	"os.Remove":         "deletes a file",
	"os.RemoveAll":      "deletes a file or directory recursively",
	"os.Rename":         "renames or moves a file",
	"os.Getenv":         "reads an environment variable",
	"os.Setenv":         "sets an environment variable",
	"os.LookupEnv":      "reads an environment variable with existence check",
	"os.Exit":           "terminates the process",
	"os.Getwd":          "gets the current working directory",
	"os.Chdir":          "changes the working directory",
	"os.Args":           "accesses command-line arguments",
	"os.Stdin":          "reads from standard input",
	"os.Stdout":         "writes to standard output",
	"os.Stderr":         "writes to standard error",
	"os.TempDir":        "gets the temp directory path",
	"os.UserHomeDir":    "gets the user home directory",
	"os.Executable":     "gets the current executable path",
	"os.IsNotExist":     "checks if error means file not found",
	"os.IsExist":        "checks if error means file exists",
	"os.IsPermission":   "checks if error is a permission error",

	// ---------------------------------------------------------------
	// io / io/ioutil / bufio
	// ---------------------------------------------------------------
	"io.Copy":             "copies data between reader and writer",
	"io.ReadAll":          "reads all data from a reader",
	"io.WriteString":      "writes a string to a writer",
	"io.Pipe":             "creates a synchronous in-memory pipe",
	"io.TeeReader":        "creates a reader that writes to a writer",
	"io.LimitReader":      "limits the bytes read from a reader",
	"io.MultiReader":      "concatenates multiple readers",
	"io.MultiWriter":      "duplicates writes to multiple writers",
	"io.NopCloser":        "wraps a reader as a ReadCloser",
	"ioutil.ReadAll":      "reads all data from a reader",
	"ioutil.ReadFile":     "reads an entire file",
	"ioutil.WriteFile":    "writes data to a file",
	"ioutil.TempFile":     "creates a temporary file",
	"ioutil.TempDir":      "creates a temporary directory",
	"bufio.NewScanner":    "creates a line-by-line text scanner",
	"bufio.NewReader":     "creates a buffered reader",
	"bufio.NewWriter":     "creates a buffered writer",
	"scanner.Scan":        "reads the next token",
	"scanner.Text":        "returns the current token as text",
	"scanner.Bytes":       "returns the current token as bytes",
	"scanner.Err":         "returns the first scanning error",
	"reader.ReadString":   "reads until a delimiter",
	"reader.ReadLine":     "reads a single line",
	"writer.Flush":        "flushes buffered data to the writer",

	// ---------------------------------------------------------------
	// filepath / path
	// ---------------------------------------------------------------
	"filepath.Walk":      "recursively walks a directory tree",
	"filepath.WalkDir":   "recursively walks a directory tree efficiently",
	"filepath.Join":      "constructs a file path",
	"filepath.Glob":      "finds files matching a pattern",
	"filepath.Abs":       "converts to an absolute path",
	"filepath.Base":      "extracts the file name from a path",
	"filepath.Dir":       "extracts the directory from a path",
	"filepath.Ext":       "extracts the file extension",
	"filepath.Rel":       "computes a relative path",
	"filepath.Match":     "matches a file name pattern",
	"filepath.Clean":     "cleans up a file path",
	"path.Join":          "joins URL path elements",
	"path.Base":          "returns the last element of a path",

	// ---------------------------------------------------------------
	// fmt
	// ---------------------------------------------------------------
	"fmt.Sprintf":   "formats a string",
	"fmt.Fprintf":   "writes formatted output to a writer",
	"fmt.Printf":    "prints formatted output",
	"fmt.Println":   "prints a line to stdout",
	"fmt.Print":     "prints to stdout",
	"fmt.Errorf":    "creates a formatted error",
	"fmt.Sscanf":    "parses formatted input from a string",
	"fmt.Fscanf":    "parses formatted input from a reader",

	// ---------------------------------------------------------------
	// log / log/slog
	// ---------------------------------------------------------------
	"log.Printf":    "logs a formatted message",
	"log.Println":   "logs a message",
	"log.Fatal":     "logs a message and exits",
	"log.Fatalf":    "logs a formatted message and exits",
	"log.Panic":     "logs a message and panics",
	"log.SetOutput": "sets the log output destination",
	"log.SetFlags":  "configures the log format",
	"log.New":       "creates a new logger",
	"slog.Info":     "logs at info level",
	"slog.Warn":     "logs at warning level",
	"slog.Error":    "logs at error level",
	"slog.Debug":    "logs at debug level",
	"slog.With":     "creates a logger with preset fields",
	"slog.NewJSONHandler":   "creates a JSON log handler",
	"slog.NewTextHandler":   "creates a text log handler",

	// ---------------------------------------------------------------
	// strings / bytes
	// ---------------------------------------------------------------
	"strings.Contains":    "checks if a string contains a substring",
	"strings.HasPrefix":   "checks string prefix",
	"strings.HasSuffix":   "checks string suffix",
	"strings.Split":       "splits a string into parts",
	"strings.SplitN":      "splits a string into at most N parts",
	"strings.Join":        "joins strings with a separator",
	"strings.Replace":     "replaces occurrences in a string",
	"strings.ReplaceAll":  "replaces all occurrences in a string",
	"strings.TrimSpace":   "removes leading/trailing whitespace",
	"strings.Trim":        "removes leading/trailing characters",
	"strings.TrimPrefix":  "removes a prefix from a string",
	"strings.TrimSuffix":  "removes a suffix from a string",
	"strings.ToLower":     "converts a string to lowercase",
	"strings.ToUpper":     "converts a string to uppercase",
	"strings.Title":       "title-cases a string",
	"strings.Count":       "counts substring occurrences",
	"strings.Index":       "finds the first occurrence of a substring",
	"strings.LastIndex":   "finds the last occurrence of a substring",
	"strings.Repeat":      "repeats a string N times",
	"strings.Map":         "applies a function to each rune",
	"strings.Fields":      "splits on whitespace",
	"strings.NewReader":   "creates a string reader",
	"strings.NewReplacer": "creates a batch string replacer",
	"strings.Builder":     "efficiently builds a string",
	"bytes.Buffer":        "creates a byte buffer",
	"bytes.Contains":      "checks if bytes contain a subsequence",
	"bytes.Equal":         "compares two byte slices",
	"bytes.Join":          "joins byte slices with a separator",
	"bytes.Split":         "splits a byte slice",
	"bytes.TrimSpace":     "trims whitespace from bytes",

	// ---------------------------------------------------------------
	// strconv
	// ---------------------------------------------------------------
	"strconv.Atoi":          "converts string to integer",
	"strconv.Itoa":          "converts integer to string",
	"strconv.ParseFloat":    "parses a float from a string",
	"strconv.ParseInt":      "parses an integer from a string",
	"strconv.ParseBool":     "parses a boolean from a string",
	"strconv.FormatFloat":   "formats a float as string",
	"strconv.FormatInt":     "formats an integer as string",
	"strconv.FormatBool":    "formats a boolean as string",

	// ---------------------------------------------------------------
	// regexp
	// ---------------------------------------------------------------
	"regexp.Compile":      "compiles a regular expression",
	"regexp.MustCompile":  "compiles a regex (panics on error)",
	"re.MatchString":      "tests if a string matches",
	"re.FindString":       "finds the first match",
	"re.FindAllString":    "finds all matches",
	"re.ReplaceAllString": "replaces all regex matches",
	"re.FindStringSubmatch": "finds match with capture groups",
	"re.Split":            "splits a string by regex",

	// ---------------------------------------------------------------
	// time
	// ---------------------------------------------------------------
	"time.Now":         "gets the current time",
	"time.Since":       "calculates elapsed time",
	"time.Until":       "calculates time remaining",
	"time.After":       "creates a timer channel",
	"time.Sleep":       "pauses execution",
	"time.NewTicker":   "creates a periodic ticker",
	"time.NewTimer":    "creates a one-shot timer",
	"time.Parse":       "parses a time string",
	"time.ParseDuration": "parses a duration string",
	"time.Duration":    "represents a time duration",
	"time.Date":        "creates a specific time",
	"t.Format":         "formats a time as string",
	"t.Add":            "adds a duration to a time",
	"t.Sub":            "subtracts two times",
	"t.Before":         "checks if time is before another",
	"t.After":          "checks if time is after another",
	"t.Unix":           "converts time to Unix timestamp",
	"t.UTC":            "converts time to UTC",
	"t.Local":          "converts time to local timezone",
	"ticker.Stop":      "stops a ticker",

	// ---------------------------------------------------------------
	// crypto / hash
	// ---------------------------------------------------------------
	"sha256.Sum256":    "computes SHA-256 hash",
	"sha256.New":       "creates a SHA-256 hasher",
	"sha512.Sum512":    "computes SHA-512 hash",
	"sha512.New":       "creates a SHA-512 hasher",
	"md5.Sum":          "computes MD5 hash",
	"md5.New":          "creates an MD5 hasher",
	"hmac.New":         "creates an HMAC signer",
	"hmac.Equal":       "compares HMAC signatures safely",
	"rand.Read":        "generates cryptographic random bytes",
	"rand.Int":         "generates a random big integer",
	"rand.Intn":        "generates a random integer",
	"aes.NewCipher":    "creates an AES cipher block",
	"cipher.NewGCM":    "creates an AES-GCM cipher",
	"rsa.GenerateKey":  "generates an RSA key pair",
	"rsa.EncryptPKCS1v15": "encrypts with RSA",
	"rsa.DecryptPKCS1v15": "decrypts with RSA",
	"tls.LoadX509KeyPair": "loads a TLS certificate",
	"x509.ParseCertificate": "parses an X.509 certificate",
	"bcrypt.GenerateFromPassword": "hashes a password with bcrypt",
	"bcrypt.CompareHashAndPassword": "verifies a bcrypt password",

	// ---------------------------------------------------------------
	// sort
	// ---------------------------------------------------------------
	"sort.Slice":       "sorts a slice with a custom comparator",
	"sort.SliceStable": "stable-sorts a slice with a comparator",
	"sort.Strings":     "sorts a string slice",
	"sort.Ints":        "sorts an integer slice",
	"sort.Float64s":    "sorts a float64 slice",
	"sort.Search":      "binary searches a sorted collection",
	"sort.Sort":        "sorts using the sort.Interface",
	"sort.Reverse":     "reverses the sort order",
	"sort.IsSorted":    "checks if a collection is sorted",

	// ---------------------------------------------------------------
	// errors
	// ---------------------------------------------------------------
	"errors.New":    "creates a new error",
	"errors.Is":     "checks if an error matches a target",
	"errors.As":     "extracts a typed error from the chain",
	"errors.Unwrap": "unwraps a wrapped error",
	"errors.Join":   "combines multiple errors",

	// ---------------------------------------------------------------
	// testing
	// ---------------------------------------------------------------
	"t.Run":        "runs a subtest",
	"t.Errorf":     "reports a test failure",
	"t.Error":      "reports a test failure",
	"t.Fatal":      "reports a fatal test failure",
	"t.Fatalf":     "reports a fatal test failure with format",
	"t.Skip":       "skips a test",
	"t.Skipf":      "skips a test with reason",
	"t.Helper":     "marks a test helper function",
	"t.Parallel":   "runs tests in parallel",
	"t.Cleanup":    "registers a test cleanup function",
	"t.TempDir":    "creates a temporary test directory",
	"t.Setenv":     "sets an env var for the test duration",
	"t.Log":        "logs a test message",
	"t.Logf":       "logs a formatted test message",
	"b.ResetTimer": "resets the benchmark timer",
	"b.ReportAllocs": "reports memory allocations in benchmark",
	"b.RunParallel":  "runs benchmark iterations in parallel",
	"b.N":            "benchmark iteration count",

	// ---------------------------------------------------------------
	// net
	// ---------------------------------------------------------------
	"net.Listen":       "starts listening on a network address",
	"net.Dial":         "connects to a network address",
	"net.DialTimeout":  "connects with a timeout",
	"listener.Accept":  "accepts an incoming connection",
	"conn.Read":        "reads from a connection",
	"conn.Write":       "writes to a connection",
	"conn.Close":       "closes a connection",
	"conn.SetDeadline": "sets a connection deadline",
	"net.LookupHost":   "resolves a hostname to IPs",
	"net.ParseCIDR":    "parses a CIDR address range",
	"url.Parse":        "parses a URL string",
	"url.Values":       "builds URL query parameters",

	// ---------------------------------------------------------------
	// reflect
	// ---------------------------------------------------------------
	"reflect.TypeOf":   "gets the runtime type of a value",
	"reflect.ValueOf":  "gets the runtime value wrapper",
	"reflect.DeepEqual": "deeply compares two values",

	// ---------------------------------------------------------------
	// exec / os/exec
	// ---------------------------------------------------------------
	"exec.Command":     "creates an external command",
	"exec.CommandContext": "creates an external command with context",
	"cmd.Run":          "runs an external command",
	"cmd.Output":       "runs a command and captures output",
	"cmd.Start":        "starts a command asynchronously",
	"cmd.Wait":         "waits for a command to finish",
	"cmd.CombinedOutput": "runs command, captures stdout+stderr",

	// ---------------------------------------------------------------
	// flag
	// ---------------------------------------------------------------
	"flag.String":     "defines a string command-line flag",
	"flag.Int":        "defines an integer command-line flag",
	"flag.Bool":       "defines a boolean command-line flag",
	"flag.Parse":      "parses command-line flags",
	"flag.Args":       "returns non-flag arguments",
	"flag.Usage":      "prints usage information",
	"flag.NewFlagSet": "creates a new flag set",

	// ---------------------------------------------------------------
	// template
	// ---------------------------------------------------------------
	"template.New":           "creates a new text template",
	"template.Must":          "wraps a template parse that must succeed",
	"template.ParseFiles":    "parses template files",
	"tmpl.Execute":           "renders a template with data",
	"tmpl.ExecuteTemplate":   "renders a named template",

	// ---------------------------------------------------------------
	// embed
	// ---------------------------------------------------------------
	"embed.FS": "embeds files into the binary",

	// ---------------------------------------------------------------
	// maps / slices (Go 1.21+)
	// ---------------------------------------------------------------
	"slices.Sort":      "sorts a slice",
	"slices.Contains":  "checks if a slice contains a value",
	"slices.Index":     "finds the index of a value in a slice",
	"slices.SortFunc":  "sorts a slice with a custom comparator",
	"maps.Keys":        "returns the keys of a map",
	"maps.Values":      "returns the values of a map",
	"maps.Clone":       "shallow-clones a map",
}

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

// ExplainSource parses Go source code and returns explanations for every
// top-level function and method declaration.
func (ce *CodeExplainer) ExplainSource(src string) ([]ExplainedFunc, error) {
	f, err := parser.ParseFile(ce.fset, "input.go", src, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("parse: %w", err)
	}
	var out []ExplainedFunc
	for _, decl := range f.Decls {
		fd, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		out = append(out, ce.explainFuncDecl(fd))
	}
	return out, nil
}

// ExplainFunc parses source that contains a single function and
// returns its explanation.
func (ce *CodeExplainer) ExplainFunc(src string) (ExplainedFunc, error) {
	all, err := ce.ExplainSource(src)
	if err != nil {
		return ExplainedFunc{}, err
	}
	if len(all) == 0 {
		return ExplainedFunc{}, fmt.Errorf("no function declarations found")
	}
	return all[0], nil
}

// -----------------------------------------------------------------------
// Core analysis
// -----------------------------------------------------------------------

// explainFuncDecl produces an ExplainedFunc for a single function decl.
func (ce *CodeExplainer) explainFuncDecl(fd *ast.FuncDecl) ExplainedFunc {
	name := fd.Name.Name
	sig := ce.formatSignature(fd)
	patterns := ce.analyzeBody(fd)
	composite := ce.detectCompositePatterns(fd)
	dataFlow := ce.summarizeDataFlow(fd)

	ef := ExplainedFunc{
		Name:      name,
		Signature: sig,
		Patterns:  patterns,
		Composite: composite,
		DataFlow:  dataFlow,
	}

	// Merge composite patterns into the pattern list for summary
	allPatterns := make([]string, 0, len(patterns)+len(composite))
	allPatterns = append(allPatterns, composite...) // composite first — higher-level
	allPatterns = append(allPatterns, patterns...)

	ef.Summary = ce.generateSummary(name, allPatterns, sig)
	if dataFlow != "" {
		ef.Summary += " " + dataFlow
	}
	return ef
}

// formatSignature produces a human-readable "params -> returns" string.
func (ce *CodeExplainer) formatSignature(fd *ast.FuncDecl) string {
	var params []string
	if fd.Type.Params != nil {
		for _, field := range fd.Type.Params.List {
			typeStr := ce.nodeString(field.Type)
			for _, n := range field.Names {
				params = append(params, n.Name+" "+typeStr)
			}
			if len(field.Names) == 0 {
				params = append(params, typeStr)
			}
		}
	}

	var results []string
	if fd.Type.Results != nil {
		for _, field := range fd.Type.Results.List {
			typeStr := ce.nodeString(field.Type)
			for _, n := range field.Names {
				results = append(results, n.Name+" "+typeStr)
			}
			if len(field.Names) == 0 {
				results = append(results, typeStr)
			}
		}
	}

	paramStr := strings.Join(params, ", ")
	if len(results) == 0 {
		return "(" + paramStr + ")"
	}
	return "(" + paramStr + ") -> (" + strings.Join(results, ", ") + ")"
}

// analyzeBody walks the function body AST and returns a list of
// semantic pattern descriptions. This is the core method that replaces
// shallow syntax-based detection with rich, contextual explanations.
func (ce *CodeExplainer) analyzeBody(fd *ast.FuncDecl) []string {
	if fd.Body == nil {
		return []string{"Declares an interface method (no body)"}
	}

	var patterns []string
	seen := make(map[string]bool) // deduplicate

	add := func(s string) {
		if s == "" || seen[s] {
			return
		}
		seen[s] = true
		patterns = append(patterns, s)
	}

	ast.Inspect(fd.Body, func(n ast.Node) bool {
		switch node := n.(type) {

		// ---- Range loops: describe WHAT is iterated ----
		case *ast.RangeStmt:
			target := ce.exprString(node.X)
			desc := ce.describeRangeTarget(target)
			add(desc)

		// ---- For loops ----
		case *ast.ForStmt:
			if node.Cond != nil {
				cond := ce.exprString(node.Cond)
				if strings.Contains(cond, "Next") || strings.Contains(cond, "Scan") {
					add("Iterates by reading successive items")
				} else {
					add(fmt.Sprintf("Loops while %s", cond))
				}
			} else {
				add("Runs an infinite loop")
			}

		// ---- Go statements ----
		case *ast.GoStmt:
			callName := ce.callExprName(node.Call)
			if callName != "" {
				add(fmt.Sprintf("Launches goroutine: %s", callName))
			} else {
				add("Launches a goroutine")
			}

		// ---- Defer ----
		case *ast.DeferStmt:
			callName := ce.callExprName(node.Call)
			if desc, ok := knownFunctions[callName]; ok {
				add(fmt.Sprintf("Defers: %s", desc))
			} else if callName != "" {
				add(fmt.Sprintf("Defers %s for cleanup", callName))
			} else {
				add("Defers a cleanup call")
			}

		// ---- Select ----
		case *ast.SelectStmt:
			n := len(node.Body.List)
			add(fmt.Sprintf("Selects between %d channel operations", n))

		// ---- Switch / type switch ----
		case *ast.SwitchStmt:
			if node.Tag != nil {
				tag := ce.exprString(node.Tag)
				add(fmt.Sprintf("Switches on %s", tag))
			} else {
				add("Uses a conditional switch")
			}
		case *ast.TypeSwitchStmt:
			add("Performs a type switch")

		// ---- Return with error check pattern ----
		case *ast.IfStmt:
			ce.analyzeIfStmt(node, add)

		// ---- Function / method calls ----
		case *ast.CallExpr:
			ce.analyzeCallExpr(node, add)

		// ---- Send on channel ----
		case *ast.SendStmt:
			ch := ce.exprString(node.Chan)
			add(fmt.Sprintf("Sends a value on channel %s", ch))

		// ---- Assignments that reveal intent ----
		case *ast.AssignStmt:
			ce.analyzeAssign(node, add)
		}
		return true
	})

	if len(patterns) == 0 {
		add("Performs a simple operation")
	}

	return patterns
}

// describeRangeTarget produces a semantic description of what a range
// loop iterates over, based on the expression name.
func (ce *CodeExplainer) describeRangeTarget(target string) string {
	lower := strings.ToLower(target)

	// Well-known domain terms
	domainPatterns := []struct {
		substr string
		desc   string
	}{
		{"layer", "Processes each layer sequentially"},
		{"arg", "Processes command-line arguments"},
		{"file", "Processes each file"},
		{"row", "Iterates over database rows"},
		{"record", "Iterates over records"},
		{"item", "Processes each item"},
		{"token", "Processes each token"},
		{"node", "Traverses each node"},
		{"child", "Iterates over child elements"},
		{"result", "Iterates over results"},
		{"route", "Iterates over routes"},
		{"handler", "Iterates over handlers"},
		{"header", "Iterates over headers"},
		{"param", "Iterates over parameters"},
		{"field", "Iterates over fields"},
		{"key", "Iterates over keys"},
		{"value", "Iterates over values"},
		{"entry", "Iterates over entries"},
		{"element", "Iterates over elements"},
		{"line", "Processes each line"},
		{"word", "Processes each word"},
		{"byte", "Processes each byte"},
		{"rune", "Processes each rune"},
		{"char", "Processes each character"},
		{"err", "Iterates over errors"},
		{"test", "Iterates over test cases"},
		{"case", "Iterates over cases"},
		{"conn", "Iterates over connections"},
		{"peer", "Iterates over peers"},
		{"message", "Processes each message"},
		{"event", "Processes each event"},
		{"task", "Processes each task"},
		{"job", "Processes each job"},
		{"batch", "Processes each batch"},
		{"chunk", "Processes each chunk"},
		{"weight", "Iterates over weights"},
		{"gradient", "Iterates over gradients"},
		{"embed", "Iterates over embeddings"},
	}

	for _, dp := range domainPatterns {
		if strings.Contains(lower, dp.substr) {
			return dp.desc
		}
	}

	// os.Args special case
	if target == "os.Args" || target == "os.Args[1:]" {
		return "Processes command-line arguments"
	}

	return fmt.Sprintf("Iterates over %s", target)
}

// analyzeIfStmt detects error handling patterns in if statements.
func (ce *CodeExplainer) analyzeIfStmt(node *ast.IfStmt, add func(string)) {
	// Detect: if err != nil { return ... }
	cond := ce.exprString(node.Cond)
	if strings.Contains(cond, "err") && strings.Contains(cond, "nil") {
		// Check what the init does — often if err := foo(); err != nil
		if node.Init != nil {
			initStr := ce.stmtString(node.Init)
			if initStr != "" {
				// Extract the call from the init
				if assign, ok := node.Init.(*ast.AssignStmt); ok {
					for _, rhs := range assign.Rhs {
						if call, ok := rhs.(*ast.CallExpr); ok {
							name := ce.callExprName(call)
							if desc, ok := knownFunctions[name]; ok {
								add(fmt.Sprintf("Checks error after: %s", desc))
								return
							}
						}
					}
				}
			}
		}
		add("Handles errors")
	}
}

// analyzeCallExpr looks up a function call in knownFunctions and adds
// a semantic description.
func (ce *CodeExplainer) analyzeCallExpr(node *ast.CallExpr, add func(string)) {
	name := ce.callExprName(node)
	if name == "" {
		return
	}

	// Direct lookup
	if desc, ok := knownFunctions[name]; ok {
		add(ceCapitalize(desc))
		return
	}

	// Try with common receiver name aliases. For instance, a call like
	// "myDB.Query" should match "db.Query".
	if dot := strings.LastIndex(name, "."); dot > 0 {
		method := name[dot+1:]
		receiver := name[:dot]

		// Try common receiver aliases
		aliases := ce.receiverAliases(receiver, method)
		for _, alias := range aliases {
			if desc, ok := knownFunctions[alias]; ok {
				add(ceCapitalize(desc))
				return
			}
		}

		// If it is a known package call but not in our map, still note it
		knownPkgs := map[string]string{
			"http": "HTTP", "json": "JSON", "sql": "database",
			"os": "OS", "io": "I/O", "fmt": "formatting",
			"log": "logging", "sync": "synchronization",
			"crypto": "cryptographic", "time": "time",
			"strings": "string", "filepath": "file path",
			"context": "context", "net": "network",
			"regexp": "regex", "sort": "sorting",
			"exec": "command execution", "flag": "flag parsing",
			"template": "template", "reflect": "reflection",
			"testing": "testing", "bufio": "buffered I/O",
			"strconv": "string conversion", "errors": "error",
			"encoding": "encoding", "xml": "XML",
		}
		if pkgDesc, ok := knownPkgs[receiver]; ok {
			add(fmt.Sprintf("Calls %s %s operation: %s", pkgDesc, method, name))
			return
		}
	}
}

// receiverAliases generates possible canonical receiver names for a given
// receiver + method combination. For example, "myDB" -> "db", "mu" stays
// as "mu", etc.
func (ce *CodeExplainer) receiverAliases(receiver, method string) []string {
	lower := strings.ToLower(receiver)
	var aliases []string

	// Map common variable name patterns to canonical receivers
	canonicals := []struct {
		contains string
		alias    string
	}{
		{"db", "db"},
		{"conn", "conn"},
		{"client", "client"},
		{"row", "rows"},
		{"row", "row"},
		{"stmt", "stmt"},
		{"tx", "tx"},
		{"mux", "mux"},
		{"wg", "wg"},
		{"mu", "mu"},
		{"lock", "mu"},
		{"once", "once"},
		{"pool", "pool"},
		{"cond", "cond"},
		{"scanner", "scanner"},
		{"reader", "reader"},
		{"writer", "writer"},
		{"ticker", "ticker"},
		{"listener", "listener"},
		{"enc", "enc"},
		{"dec", "dec"},
		{"cmd", "cmd"},
		{"tmpl", "tmpl"},
		{"re", "re"},
		{"ctx", "ctx"},
		{"resp", "resp"},
	}

	for _, c := range canonicals {
		if strings.Contains(lower, c.contains) {
			aliases = append(aliases, c.alias+"."+method)
		}
	}

	// Also try the raw name + method
	aliases = append(aliases, receiver+"."+method)

	return aliases
}

// analyzeAssign checks assignments for patterns that reveal intent.
func (ce *CodeExplainer) analyzeAssign(node *ast.AssignStmt, add func(string)) {
	// Only handle short variable declarations (:=) for things like
	// ctx, cancel := context.WithCancel(...)
	if node.Tok.String() != ":=" {
		return
	}
	// Check if any LHS name is "cancel" (context pattern)
	for _, lhs := range node.Lhs {
		if id, ok := lhs.(*ast.Ident); ok {
			if id.Name == "cancel" {
				add("Creates a cancellable context")
				return
			}
		}
	}
}

// -----------------------------------------------------------------------
// AST helpers
// -----------------------------------------------------------------------

// callExprName returns a dotted name for a call expression.
// For example: http.ListenAndServe, w.Write, db.Query, etc.
func (ce *CodeExplainer) callExprName(call *ast.CallExpr) string {
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		return fun.Name
	case *ast.SelectorExpr:
		return ce.selectorName(fun)
	}
	return ""
}

// selectorName flattens a.b.c into "a.b.c".
func (ce *CodeExplainer) selectorName(sel *ast.SelectorExpr) string {
	switch x := sel.X.(type) {
	case *ast.Ident:
		return x.Name + "." + sel.Sel.Name
	case *ast.SelectorExpr:
		return ce.selectorName(x) + "." + sel.Sel.Name
	}
	return sel.Sel.Name
}

// exprString converts an AST expression to a source string.
func (ce *CodeExplainer) exprString(expr ast.Expr) string {
	if expr == nil {
		return ""
	}
	var buf bytes.Buffer
	if err := printer.Fprint(&buf, ce.fset, expr); err != nil {
		return ""
	}
	return buf.String()
}

// stmtString converts an AST statement to a source string.
func (ce *CodeExplainer) stmtString(stmt ast.Stmt) string {
	if stmt == nil {
		return ""
	}
	var buf bytes.Buffer
	if err := printer.Fprint(&buf, ce.fset, stmt); err != nil {
		return ""
	}
	return buf.String()
}

// nodeString converts any AST node to a source string.
func (ce *CodeExplainer) nodeString(node ast.Node) string {
	if node == nil {
		return ""
	}
	var buf bytes.Buffer
	if err := printer.Fprint(&buf, ce.fset, node); err != nil {
		return ""
	}
	return buf.String()
}

// -----------------------------------------------------------------------
// Summary generation
// -----------------------------------------------------------------------

// generateSummary produces a coherent paragraph from the function name,
// detected patterns, and signature.
func (ce *CodeExplainer) generateSummary(funcName string, patterns []string, sig string) string {
	if len(patterns) == 0 {
		return fmt.Sprintf("%s performs a simple operation.", funcName)
	}

	var sb strings.Builder

	// Opening: derive purpose from function name
	purpose := ce.purposeFromName(funcName)
	if purpose != "" {
		sb.WriteString(fmt.Sprintf("%s %s.", funcName, purpose))
	} else {
		sb.WriteString(fmt.Sprintf("%s %s.", funcName, lowerFirst(patterns[0])))
	}

	// Body: remaining patterns
	remaining := patterns
	if purpose != "" {
		remaining = patterns
	} else if len(patterns) > 1 {
		remaining = patterns[1:]
	} else {
		return sb.String()
	}

	if len(remaining) > 0 {
		sb.WriteString(" It ")
		for i, p := range remaining {
			lower := lowerFirst(p)
			if i == 0 {
				sb.WriteString(lower)
			} else if i == len(remaining)-1 {
				sb.WriteString(", and ")
				sb.WriteString(lower)
			} else {
				sb.WriteString(", ")
				sb.WriteString(lower)
			}
		}
		sb.WriteString(".")
	}

	return sb.String()
}

// purposeFromName infers a verb phrase from common function name patterns.
func (ce *CodeExplainer) purposeFromName(name string) string {
	lower := strings.ToLower(name)

	prefixes := []struct {
		prefix string
		verb   string
	}{
		{"handle", "handles"},
		{"process", "processes"},
		{"parse", "parses"},
		{"validate", "validates"},
		{"create", "creates"},
		{"new", "creates a new"},
		{"init", "initializes"},
		{"setup", "sets up"},
		{"build", "builds"},
		{"make", "constructs"},
		{"get", "retrieves"},
		{"fetch", "fetches"},
		{"load", "loads"},
		{"read", "reads"},
		{"write", "writes"},
		{"save", "saves"},
		{"store", "stores"},
		{"delete", "deletes"},
		{"remove", "removes"},
		{"update", "updates"},
		{"set", "sets"},
		{"register", "registers"},
		{"add", "adds"},
		{"insert", "inserts"},
		{"find", "finds"},
		{"search", "searches for"},
		{"lookup", "looks up"},
		{"check", "checks"},
		{"verify", "verifies"},
		{"test", "tests"},
		{"run", "runs"},
		{"exec", "executes"},
		{"start", "starts"},
		{"stop", "stops"},
		{"close", "closes"},
		{"open", "opens"},
		{"connect", "connects to"},
		{"disconnect", "disconnects from"},
		{"send", "sends"},
		{"receive", "receives"},
		{"publish", "publishes"},
		{"subscribe", "subscribes to"},
		{"listen", "listens for"},
		{"serve", "serves"},
		{"render", "renders"},
		{"format", "formats"},
		{"convert", "converts"},
		{"transform", "transforms"},
		{"encode", "encodes"},
		{"decode", "decodes"},
		{"marshal", "marshals"},
		{"unmarshal", "unmarshals"},
		{"serialize", "serializes"},
		{"deserialize", "deserializes"},
		{"compress", "compresses"},
		{"decompress", "decompresses"},
		{"encrypt", "encrypts"},
		{"decrypt", "decrypts"},
		{"hash", "hashes"},
		{"sign", "signs"},
		{"sort", "sorts"},
		{"filter", "filters"},
		{"map", "maps"},
		{"reduce", "reduces"},
		{"merge", "merges"},
		{"split", "splits"},
		{"join", "joins"},
		{"count", "counts"},
		{"sum", "sums"},
		{"avg", "averages"},
		{"compute", "computes"},
		{"calculate", "calculates"},
		{"measure", "measures"},
		{"log", "logs"},
		{"print", "prints"},
		{"debug", "debugs"},
		{"trace", "traces"},
		{"notify", "notifies"},
		{"alert", "alerts"},
		{"emit", "emits"},
		{"dispatch", "dispatches"},
		{"forward", "forwards"},
		{"retry", "retries"},
		{"reset", "resets"},
		{"clear", "clears"},
		{"flush", "flushes"},
		{"sync", "synchronizes"},
		{"wait", "waits for"},
		{"poll", "polls"},
		{"watch", "watches"},
		{"monitor", "monitors"},
		{"scan", "scans"},
		{"crawl", "crawls"},
		{"index", "indexes"},
		{"cache", "caches"},
		{"refresh", "refreshes"},
		{"clone", "clones"},
		{"copy", "copies"},
		{"backup", "backs up"},
		{"restore", "restores"},
		{"migrate", "migrates"},
		{"compile", "compiles"},
		{"generate", "generates"},
		{"resolve", "resolves"},
		{"extract", "extracts"},
		{"collect", "collects"},
		{"aggregate", "aggregates"},
		{"apply", "applies"},
		{"execute", "executes"},
		{"perform", "performs"},
		{"ensure", "ensures"},
		{"assert", "asserts"},
	}

	for _, p := range prefixes {
		if strings.HasPrefix(lower, p.prefix) {
			rest := name[len(p.prefix):]
			if rest == "" {
				return p.verb
			}
			// Convert camelCase remainder to words
			words := camelToWords(rest)
			if words != "" {
				return p.verb + " " + strings.ToLower(words)
			}
			return p.verb
		}
	}
	return ""
}

// camelToWords splits "CamelCase" into "Camel Case".
func camelToWords(s string) string {
	if s == "" {
		return ""
	}
	var buf strings.Builder
	runes := []rune(s)
	for i, r := range runes {
		if i > 0 && unicode.IsUpper(r) {
			// Don't split consecutive uppercase (acronyms)
			if i+1 < len(runes) && unicode.IsLower(runes[i+1]) {
				buf.WriteRune(' ')
			} else if i > 0 && unicode.IsLower(runes[i-1]) {
				buf.WriteRune(' ')
			}
		}
		buf.WriteRune(r)
	}
	return buf.String()
}

// -----------------------------------------------------------------------
// Composite pattern detection
// -----------------------------------------------------------------------

// compositeRule describes a higher-level pattern recognized from a
// combination of calls/constructs in the function body.
type compositeRule struct {
	name     string   // human description
	requires []string // call names or keywords that must ALL be present
}

// compositeRules lists patterns detected from combinations of calls.
var compositeRules = []compositeRule{
	{
		name:     "Thread-safe with mutex protection",
		requires: []string{"Lock", "Unlock"},
	},
	{
		name:     "Sets up an HTTP server with route handlers",
		requires: []string{"HandleFunc", "ListenAndServe"},
	},
	{
		name:     "Sets up an HTTPS server with route handlers",
		requires: []string{"HandleFunc", "ListenAndServeTLS"},
	},
	{
		name:     "Reads and parses JSON from request body",
		requires: []string{"NewDecoder", "Decode"},
	},
	{
		name:     "Serializes and writes JSON response",
		requires: []string{"NewEncoder", "Encode"},
	},
	{
		name:     "Reads a file line by line",
		requires: []string{"Open", "Scanner"},
	},
	{
		name:     "Opens database, queries, and cleans up",
		requires: []string{"sql.Open", "Query", "Close"},
	},
	{
		name:     "Executes a database transaction",
		requires: []string{"Begin", "Commit"},
	},
	{
		name:     "Uses timeout-protected context",
		requires: []string{"WithTimeout", "cancel"},
	},
	{
		name:     "Uses cancellable context",
		requires: []string{"WithCancel", "cancel"},
	},
	{
		name:     "Coordinates concurrent work with WaitGroup",
		requires: []string{"Add", "Done", "Wait"},
	},
	{
		name:     "Processes items from a channel",
		requires: []string{"range", "chan"},
	},
	{
		name:     "Implements graceful shutdown",
		requires: []string{"signal.Notify", "Shutdown"},
	},
	{
		name:     "Streams JSON data to the response",
		requires: []string{"NewEncoder", "Header", "Encode"},
	},
	{
		name:     "Parses command-line flags and arguments",
		requires: []string{"flag.Parse", "flag."},
	},
	{
		name:     "Executes an external command and captures output",
		requires: []string{"exec.Command", "Output"},
	},
	{
		name:     "Renders a template with data",
		requires: []string{"template.", "Execute"},
	},
	{
		name:     "Hashes data for integrity verification",
		requires: []string{"sha256.", "Sum"},
	},
	{
		name:     "Measures execution time",
		requires: []string{"time.Now", "time.Since"},
	},
	{
		name:     "Retries an operation on failure",
		requires: []string{"for", "err", "Sleep"},
	},
	{
		name:     "Walks a directory tree processing files",
		requires: []string{"Walk", "filepath"},
	},
	{
		name:     "Reads entire file contents",
		requires: []string{"ReadFile"},
	},
	{
		name:     "Creates and writes to a new file",
		requires: []string{"Create", "Write"},
	},
}

// detectCompositePatterns scans the function body for combinations of
// calls/constructs that indicate a higher-level pattern.
func (ce *CodeExplainer) detectCompositePatterns(fd *ast.FuncDecl) []string {
	if fd.Body == nil {
		return nil
	}

	// Collect all call names and keywords present in the body.
	tokens := make(map[string]bool)

	ast.Inspect(fd.Body, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.CallExpr:
			name := ce.callExprName(node)
			if name != "" {
				tokens[name] = true
				// Also store sub-parts for flexible matching
				if dot := strings.LastIndex(name, "."); dot > 0 {
					tokens[name[dot+1:]] = true // method name
					tokens[name[:dot]+"."] = true // "pkg." prefix
				}
			}
		case *ast.GoStmt:
			tokens["go"] = true
		case *ast.DeferStmt:
			callName := ce.callExprName(node.Call)
			if callName != "" {
				tokens["defer:"+callName] = true
				if dot := strings.LastIndex(callName, "."); dot > 0 {
					tokens[callName[dot+1:]] = true
				}
			}
		case *ast.RangeStmt:
			tokens["range"] = true
			target := ce.exprString(node.X)
			if strings.Contains(target, "chan") {
				tokens["chan"] = true
			}
		case *ast.ForStmt:
			tokens["for"] = true
		case *ast.Ident:
			if node.Name == "cancel" {
				tokens["cancel"] = true
			}
		case *ast.AssignStmt:
			for _, lhs := range node.Lhs {
				if id, ok := lhs.(*ast.Ident); ok {
					if id.Name == "cancel" {
						tokens["cancel"] = true
					}
					if id.Name == "err" {
						tokens["err"] = true
					}
				}
			}
		case *ast.UnaryExpr:
			// Detect channel receive in range
			if node.Op.String() == "<-" {
				tokens["chan"] = true
			}
		}
		return true
	})

	var matched []string
	seen := make(map[string]bool)

	for _, rule := range compositeRules {
		allFound := true
		for _, req := range rule.requires {
			if !tokens[req] {
				// Try prefix matching for things like "flag."
				found := false
				for tok := range tokens {
					if strings.Contains(tok, req) {
						found = true
						break
					}
				}
				if !found {
					allFound = false
					break
				}
			}
		}
		if allFound && !seen[rule.name] {
			seen[rule.name] = true
			matched = append(matched, rule.name)
		}
	}

	return matched
}

// -----------------------------------------------------------------------
// Data flow summary
// -----------------------------------------------------------------------

// summarizeDataFlow traces the rough input -> transformation -> output
// pipeline of a function by inspecting parameters, assignments, and
// return statements.
func (ce *CodeExplainer) summarizeDataFlow(fd *ast.FuncDecl) string {
	if fd.Body == nil {
		return ""
	}

	// 1. Identify main input (first non-context, non-error parameter).
	input := ce.mainInput(fd)

	// 2. Collect transformation steps from assignments and calls.
	transforms := ce.collectTransforms(fd)

	// 3. Identify output from return type.
	output := ce.mainOutput(fd)

	// Build the flow string only if we have meaningful information.
	if input == "" && output == "" {
		return ""
	}
	if len(transforms) == 0 && input == "" {
		return ""
	}

	var parts []string
	if input != "" {
		parts = append(parts, "Takes "+input)
	}
	for _, t := range transforms {
		parts = append(parts, t)
	}
	if output != "" {
		parts = append(parts, "returns "+output)
	}

	if len(parts) <= 1 {
		return ""
	}

	return strings.Join(parts, " -> ") + "."
}

// mainInput returns a description of the function's primary input parameter.
func (ce *CodeExplainer) mainInput(fd *ast.FuncDecl) string {
	if fd.Type.Params == nil {
		return ""
	}
	for _, field := range fd.Type.Params.List {
		typeStr := ce.nodeString(field.Type)
		// Skip context, error, and common infrastructure types
		if strings.Contains(typeStr, "context.Context") ||
			strings.Contains(typeStr, "http.ResponseWriter") ||
			strings.Contains(typeStr, "http.Request") ||
			strings.Contains(typeStr, "testing.T") ||
			strings.Contains(typeStr, "testing.B") {
			continue
		}
		for _, n := range field.Names {
			name := n.Name
			return ce.describeParam(name, typeStr)
		}
	}
	return ""
}

// describeParam produces a human-readable description of a parameter.
func (ce *CodeExplainer) describeParam(name, typeStr string) string {
	// Map common parameter names to descriptions
	lower := strings.ToLower(name)
	switch {
	case lower == "id" || strings.HasSuffix(lower, "id"):
		return "an " + camelToReadable(name)
	case lower == "path" || lower == "filepath" || lower == "filename":
		return "a file path"
	case lower == "url" || lower == "uri":
		return "a URL"
	case lower == "query" || lower == "q":
		return "a query"
	case lower == "data" || lower == "payload":
		return "input data"
	case lower == "key":
		return "a key"
	case lower == "name":
		return "a name"
	case lower == "src" || lower == "source":
		return "source data"
	case lower == "dst" || lower == "dest" || lower == "target":
		return "a destination"
	case lower == "msg" || lower == "message":
		return "a message"
	case lower == "cfg" || lower == "config" || lower == "conf":
		return "a configuration"
	case lower == "opts" || lower == "options":
		return "options"
	case lower == "db":
		return "a database connection"
	case lower == "conn":
		return "a connection"
	case lower == "ctx":
		return "a context"
	case lower == "buf" || lower == "buffer":
		return "a buffer"
	case lower == "token":
		return "a token"
	case lower == "input":
		return "input"
	case lower == "output":
		return "output"
	}

	// Fall back to type-based description
	switch {
	case typeStr == "string":
		return "a " + camelToReadable(name) + " string"
	case typeStr == "int" || typeStr == "int64" || typeStr == "int32":
		return "a " + camelToReadable(name) + " integer"
	case typeStr == "bool":
		return "a " + camelToReadable(name) + " flag"
	case typeStr == "[]byte":
		return camelToReadable(name) + " bytes"
	case strings.HasPrefix(typeStr, "[]"):
		inner := typeStr[2:]
		return "a slice of " + inner
	case strings.HasPrefix(typeStr, "map["):
		return "a " + camelToReadable(name) + " map"
	case strings.HasPrefix(typeStr, "*"):
		inner := typeStr[1:]
		return "a " + inner
	}
	return camelToReadable(name)
}

// mainOutput returns a description of the function's primary return value.
func (ce *CodeExplainer) mainOutput(fd *ast.FuncDecl) string {
	if fd.Type.Results == nil {
		return ""
	}
	var resultTypes []string
	for _, field := range fd.Type.Results.List {
		typeStr := ce.nodeString(field.Type)
		if typeStr == "error" {
			continue // skip error returns
		}
		for _, n := range field.Names {
			resultTypes = append(resultTypes, ce.describeReturnType(n.Name, typeStr))
		}
		if len(field.Names) == 0 {
			resultTypes = append(resultTypes, ce.describeReturnType("", typeStr))
		}
	}
	if len(resultTypes) == 0 {
		return ""
	}
	return strings.Join(resultTypes, " and ")
}

// describeReturnType produces a readable return type description.
func (ce *CodeExplainer) describeReturnType(name, typeStr string) string {
	if name != "" {
		return camelToReadable(name)
	}
	switch {
	case typeStr == "string":
		return "a string"
	case typeStr == "int" || typeStr == "int64":
		return "an integer"
	case typeStr == "bool":
		return "a boolean"
	case typeStr == "[]byte":
		return "bytes"
	case strings.HasPrefix(typeStr, "[]"):
		inner := typeStr[2:]
		return "a slice of " + inner
	case strings.HasPrefix(typeStr, "map["):
		return "a map"
	case strings.HasPrefix(typeStr, "*"):
		return "a " + typeStr[1:]
	}
	return typeStr
}

// collectTransforms extracts the key transformation steps from the
// function body by looking at assignments whose RHS are call expressions
// that appear in knownFunctions.
func (ce *CodeExplainer) collectTransforms(fd *ast.FuncDecl) []string {
	if fd.Body == nil {
		return nil
	}

	var transforms []string
	seen := make(map[string]bool)

	// Walk top-level statements only (not deeply nested) for the main
	// pipeline. This keeps the data flow summary at a high level.
	for _, stmt := range fd.Body.List {
		switch s := stmt.(type) {
		case *ast.AssignStmt:
			for _, rhs := range s.Rhs {
				if call, ok := rhs.(*ast.CallExpr); ok {
					name := ce.callExprName(call)
					if desc, ok := knownFunctions[name]; ok && !seen[desc] {
						seen[desc] = true
						transforms = append(transforms, desc)
					}
				}
			}
		case *ast.ExprStmt:
			if call, ok := s.X.(*ast.CallExpr); ok {
				name := ce.callExprName(call)
				if desc, ok := knownFunctions[name]; ok && !seen[desc] {
					seen[desc] = true
					transforms = append(transforms, desc)
				}
			}
		case *ast.IfStmt:
			// Check init statement (if err := foo(); err != nil)
			if s.Init != nil {
				if assign, ok := s.Init.(*ast.AssignStmt); ok {
					for _, rhs := range assign.Rhs {
						if call, ok := rhs.(*ast.CallExpr); ok {
							name := ce.callExprName(call)
							if desc, ok := knownFunctions[name]; ok && !seen[desc] {
								seen[desc] = true
								transforms = append(transforms, desc)
							}
						}
					}
				}
			}
		}
	}

	// Limit to 5 steps to keep it readable.
	if len(transforms) > 5 {
		transforms = transforms[:5]
	}

	return transforms
}

// camelToReadable converts "tokenID" to "token ID", "userName" to
// "user name", etc.
func camelToReadable(s string) string {
	words := camelToWords(s)
	return strings.ToLower(words)
}

// -----------------------------------------------------------------------
// String helpers
// -----------------------------------------------------------------------

// ceCapitalize capitalizes the first letter without colliding with
// the existing capitalizeFirst in composer.go.
func ceCapitalize(s string) string {
	if s == "" {
		return s
	}
	runes := []rune(s)
	runes[0] = unicode.ToUpper(runes[0])
	return string(runes)
}

// lowerFirst is defined in composer.go
