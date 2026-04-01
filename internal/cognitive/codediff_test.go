package cognitive

import (
	"strings"
	"testing"
)

func TestExplainDiffEmptyInput(t *testing.T) {
	de := &DiffExplainer{}
	r := de.ExplainDiff("")
	if r.Summary != "No changes detected." {
		t.Errorf("expected no-changes summary, got %q", r.Summary)
	}
	if r.Risk != "low" {
		t.Errorf("expected low risk for empty diff, got %q", r.Risk)
	}
	if len(r.Files) != 0 {
		t.Errorf("expected no files, got %d", len(r.Files))
	}
}

func TestExplainDiffWhitespaceOnly(t *testing.T) {
	de := &DiffExplainer{}
	r := de.ExplainDiff("   \n\n  \t  \n")
	if r.Summary != "No changes detected." {
		t.Errorf("expected no-changes summary, got %q", r.Summary)
	}
}

func TestExplainDiffAddNewFunction(t *testing.T) {
	diff := `diff --git a/server.go b/server.go
index abc1234..def5678 100644
--- a/server.go
+++ b/server.go
@@ -10,6 +10,15 @@ import (
 )

+func NewRateLimiter(rate int) *RateLimiter {
+	return &RateLimiter{
+		rate:   rate,
+		tokens: rate,
+	}
+}
+
 func main() {
 	fmt.Println("hello")
 }
`
	de := &DiffExplainer{}
	r := de.ExplainDiff(diff)

	if len(r.Files) != 1 {
		t.Fatalf("expected 1 file, got %d", len(r.Files))
	}
	f := r.Files[0]
	if f.Path != "server.go" {
		t.Errorf("expected path server.go, got %q", f.Path)
	}
	if f.Added != 7 {
		t.Errorf("expected 7 added lines, got %d", f.Added)
	}
	if f.Removed != 0 {
		t.Errorf("expected 0 removed lines, got %d", f.Removed)
	}

	// Should detect the new function.
	foundAddFunc := false
	for _, c := range f.Changes {
		if c.Type == "add_func" && c.Name == "NewRateLimiter" {
			foundAddFunc = true
		}
	}
	if !foundAddFunc {
		t.Error("expected add_func change for NewRateLimiter")
	}

	if r.Intent != "feature" {
		t.Errorf("expected intent 'feature', got %q", r.Intent)
	}
	if r.Risk != "low" {
		t.Errorf("expected low risk for additive change, got %q", r.Risk)
	}
	if r.Breaking {
		t.Error("additive change should not be breaking")
	}
}

func TestExplainDiffRemoveExportedFunction(t *testing.T) {
	diff := `diff --git a/api.go b/api.go
index abc1234..def5678 100644
--- a/api.go
+++ b/api.go
@@ -15,10 +15,6 @@ import (
 )

-func HandleRequest(w http.ResponseWriter, r *http.Request) {
-	w.WriteHeader(200)
-	w.Write([]byte("ok"))
-}
-
 func internalHelper() {
 	// ...
 }
`
	de := &DiffExplainer{}
	r := de.ExplainDiff(diff)

	if len(r.Files) != 1 {
		t.Fatalf("expected 1 file, got %d", len(r.Files))
	}

	// Should detect removal of exported function.
	foundRemove := false
	for _, c := range r.Files[0].Changes {
		if c.Type == "remove_func" && c.Name == "HandleRequest" {
			foundRemove = true
		}
	}
	if !foundRemove {
		t.Error("expected remove_func change for HandleRequest")
	}

	if r.Risk != "high" {
		t.Errorf("expected high risk for removed exported function, got %q", r.Risk)
	}
	if !r.Breaking {
		t.Error("removing exported function should be a breaking change")
	}
}

func TestExplainDiffRefactorSameFunction(t *testing.T) {
	diff := `diff --git a/handler.go b/handler.go
index abc1234..def5678 100644
--- a/handler.go
+++ b/handler.go
@@ -5,8 +5,9 @@ import (
 )

-func Process(data []byte) error {
-	return json.Unmarshal(data, &result)
+func Process(data []byte) error {
+	if len(data) == 0 {
+		return errors.New("empty data")
+	}
+	return json.Unmarshal(data, &result)
 }
`
	de := &DiffExplainer{}
	r := de.ExplainDiff(diff)

	// Should detect modified function.
	foundModify := false
	for _, c := range r.Files[0].Changes {
		if c.Type == "modify_func" && c.Name == "Process" {
			foundModify = true
		}
	}
	if !foundModify {
		t.Error("expected modify_func change for Process")
	}

	// Refactor: same function with different logic.
	if r.Intent != "refactor" {
		t.Errorf("expected intent 'refactor', got %q", r.Intent)
	}
}

func TestExplainDiffTestFileAddition(t *testing.T) {
	diff := `diff --git a/handler_test.go b/handler_test.go
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/handler_test.go
@@ -0,0 +1,20 @@
+package handler
+
+import "testing"
+
+func TestProcess(t *testing.T) {
+	err := Process([]byte("{}"))
+	if err != nil {
+		t.Fatal(err)
+	}
+}
+
+func TestProcessEmpty(t *testing.T) {
+	err := Process(nil)
+	if err == nil {
+		t.Fatal("expected error for nil data")
+	}
+}
`
	de := &DiffExplainer{}
	r := de.ExplainDiff(diff)

	if r.Intent != "test" {
		t.Errorf("expected intent 'test', got %q", r.Intent)
	}
	if r.Risk != "low" {
		t.Errorf("expected low risk for test addition, got %q", r.Risk)
	}
}

func TestExplainDiffIntentBugfix(t *testing.T) {
	diff := `diff --git a/conn.go b/conn.go
index abc1234..def5678 100644
--- a/conn.go
+++ b/conn.go
@@ -10,6 +10,9 @@ func Connect(addr string) (*Conn, error) {
 	conn, err := net.Dial("tcp", addr)
+	// fix: handle nil connection on timeout
+	if conn == nil {
+		return nil, errors.New("nil connection")
+	}
 	if err != nil {
 		return nil, err
 	}
`
	de := &DiffExplainer{}
	r := de.ExplainDiff(diff)

	if r.Intent != "bugfix" {
		t.Errorf("expected intent 'bugfix', got %q", r.Intent)
	}
}

func TestExplainDiffIntentDocs(t *testing.T) {
	diff := `diff --git a/README.md b/README.md
index abc1234..def5678 100644
--- a/README.md
+++ b/README.md
@@ -1,3 +1,5 @@
 # My Project
+
+## Installation
+
+Run go install to get started.
`
	de := &DiffExplainer{}
	r := de.ExplainDiff(diff)

	if r.Intent != "docs" {
		t.Errorf("expected intent 'docs', got %q", r.Intent)
	}
}

func TestExplainDiffIntentPerf(t *testing.T) {
	diff := `diff --git a/cache.go b/cache.go
index abc1234..def5678 100644
--- a/cache.go
+++ b/cache.go
@@ -5,6 +5,8 @@ import "sync"
-func Lookup(key string) string {
-	return db.Query(key)
+func Lookup(key string) string {
+	// optimize: use in-memory cache to avoid repeated DB lookups
+	if v, ok := cache.Get(key); ok {
+		return v
+	}
+	return db.Query(key)
 }
`
	de := &DiffExplainer{}
	r := de.ExplainDiff(diff)

	if r.Intent != "perf" {
		t.Errorf("expected intent 'perf', got %q", r.Intent)
	}
}

func TestExplainDiffRiskMediumForExportedModification(t *testing.T) {
	diff := `diff --git a/api.go b/api.go
index abc1234..def5678 100644
--- a/api.go
+++ b/api.go
@@ -5,5 +5,6 @@
-func Serve(port int) {
-	http.ListenAndServe(fmt.Sprintf(":%d", port), nil)
+func Serve(port int) {
+	log.Printf("listening on :%d", port)
+	http.ListenAndServe(fmt.Sprintf(":%d", port), nil)
 }
`
	de := &DiffExplainer{}
	r := de.ExplainDiff(diff)

	if r.Risk != "medium" {
		t.Errorf("expected medium risk for modified exported func, got %q", r.Risk)
	}
	if r.Breaking {
		t.Error("modifying (not removing) exported func should not be breaking")
	}
}

func TestExplainDiffRiskHighForDatabaseChanges(t *testing.T) {
	diff := `diff --git a/migration/001_create_users.sql b/migration/001_create_users.sql
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/migration/001_create_users.sql
@@ -0,0 +1,5 @@
+CREATE TABLE users (
+    id SERIAL PRIMARY KEY,
+    name TEXT NOT NULL
+);
`
	de := &DiffExplainer{}
	r := de.ExplainDiff(diff)

	if r.Risk != "high" {
		t.Errorf("expected high risk for database migration, got %q", r.Risk)
	}
}

func TestExplainDiffMultipleFiles(t *testing.T) {
	diff := `diff --git a/server.go b/server.go
index abc1234..def5678 100644
--- a/server.go
+++ b/server.go
@@ -10,6 +10,10 @@
+func NewMiddleware() *Middleware {
+	return &Middleware{}
+}
+
diff --git a/middleware.go b/middleware.go
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/middleware.go
@@ -0,0 +1,8 @@
+package server
+
+type Middleware struct {
+	Name string
+}
+
+func (m *Middleware) Run() {
+}
`
	de := &DiffExplainer{}
	r := de.ExplainDiff(diff)

	if len(r.Files) != 2 {
		t.Fatalf("expected 2 files, got %d", len(r.Files))
	}

	paths := map[string]bool{}
	for _, f := range r.Files {
		paths[f.Path] = true
	}
	if !paths["server.go"] || !paths["middleware.go"] {
		t.Errorf("expected server.go and middleware.go, got %v", paths)
	}
}

func TestExplainDiffSummaryContainsKeyInfo(t *testing.T) {
	diff := `diff --git a/server.go b/server.go
index abc1234..def5678 100644
--- a/server.go
+++ b/server.go
@@ -10,6 +10,10 @@
+func Start() error {
+	return nil
+}
`
	de := &DiffExplainer{}
	r := de.ExplainDiff(diff)

	if !strings.Contains(r.Summary, "feature") {
		t.Errorf("summary should mention feature intent, got %q", r.Summary)
	}
	if !strings.Contains(r.Summary, "1 file") {
		t.Errorf("summary should mention file count, got %q", r.Summary)
	}
	if !strings.Contains(r.Summary, "new function") {
		t.Errorf("summary should mention new function, got %q", r.Summary)
	}
}

func TestExplainDiffBreakingFieldRemoval(t *testing.T) {
	diff := `diff --git a/model.go b/model.go
index abc1234..def5678 100644
--- a/model.go
+++ b/model.go
@@ -3,7 +3,6 @@ type User struct {
 	Name  string
-	Email string
 	Age   int
 }
`
	de := &DiffExplainer{}
	r := de.ExplainDiff(diff)

	foundRemoveField := false
	for _, c := range r.Files[0].Changes {
		if c.Type == "remove_field" && c.Name == "Email" {
			foundRemoveField = true
		}
	}
	if !foundRemoveField {
		t.Error("expected remove_field for Email")
	}
	if !r.Breaking {
		t.Error("removing exported field should be breaking")
	}
}

func TestExplainDiffImportChanges(t *testing.T) {
	diff := `diff --git a/main.go b/main.go
index abc1234..def5678 100644
--- a/main.go
+++ b/main.go
@@ -2,6 +2,7 @@ import (
 	"fmt"
+	"net/http"
 	"os"
 )
`
	de := &DiffExplainer{}
	r := de.ExplainDiff(diff)

	foundImport := false
	for _, c := range r.Files[0].Changes {
		if c.Type == "import_change" {
			foundImport = true
		}
	}
	if !foundImport {
		t.Error("expected import_change in changes")
	}
}

func TestExplainDiffMethodWithReceiver(t *testing.T) {
	diff := `diff --git a/server.go b/server.go
index abc1234..def5678 100644
--- a/server.go
+++ b/server.go
@@ -10,6 +10,10 @@
+func (s *Server) Shutdown() error {
+	return s.listener.Close()
+}
`
	de := &DiffExplainer{}
	r := de.ExplainDiff(diff)

	foundMethod := false
	for _, c := range r.Files[0].Changes {
		if c.Type == "add_func" && c.Name == "Shutdown" {
			foundMethod = true
		}
	}
	if !foundMethod {
		t.Error("expected add_func for method Shutdown")
	}
}

func TestFormatDiffResult(t *testing.T) {
	r := &DiffResult{
		Files: []FileDiff{
			{
				Path:        "server.go",
				Added:       10,
				Removed:     2,
				Description: "added 1 function(s)",
				Changes: []Change{
					{Type: "add_func", Name: "Start", Description: "Added new function Start"},
				},
			},
		},
		Summary:  "This change is a feature affecting 1 file(s).",
		Intent:   "feature",
		Risk:     "low",
		Breaking: false,
	}

	out := FormatDiffResult(r)
	if !strings.Contains(out, "server.go") {
		t.Error("formatted output should contain file path")
	}
	if !strings.Contains(out, "feature") {
		t.Error("formatted output should contain intent")
	}
	if !strings.Contains(out, "low") {
		t.Error("formatted output should contain risk level")
	}
	if strings.Contains(out, "Breaking: yes") {
		t.Error("formatted output should not show breaking for non-breaking change")
	}
}
