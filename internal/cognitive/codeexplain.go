package cognitive

import (
	"go/token"
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
