package cognitive

import (
	"testing"
)

func TestFastPathClassifier_SimpleQueries(t *testing.T) {
	c := &FastPathClassifier{}

	simple := []string{
		"hi",
		"hello!",
		"hey there",
		"good morning",
		"thanks",
		"bye",
		"who are you?",
		"what is Go?",
		"what are neural networks?",
		"tell me a joke",
		"what is 2 + 2?",
		"42 * 7?",
		"define photosynthesis",
		"who was Einstein?",
		"do you like programming?",
		"what tools do you have?",
		"what can you do?",
		"how are you?",
		"translate hello to french",
	}

	for _, q := range simple {
		if !c.IsSimple(q) {
			t.Errorf("expected simple, got complex: %q", q)
		}
	}
}

func TestFastPathClassifier_ComplexQueries(t *testing.T) {
	c := &FastPathClassifier{}

	complex := []string{
		"read file main.go",
		"find all TODO comments in the codebase",
		"search for the function handleRequest in the code",
		"run the tests",
		"git status",
		"create a new file called test.txt",
		"delete the directory /tmp/old",
		"refactor the server module",
		"debug the login function",
		"compile the project",
		"fetch the API response from https://example.com",
		"analyze the codebase for security issues",
		"deploy the application",
		"build a web scraper project",
		"compare the two file versions",
		"execute the shell command",
		"search the web for Go tutorials",
		"install the dependencies",
		"list files in the current directory",
	}

	for _, q := range complex {
		if c.IsSimple(q) {
			t.Errorf("expected complex, got simple: %q", q)
		}
	}
}

func TestFastPathClassifier_EmptyQuery(t *testing.T) {
	c := &FastPathClassifier{}
	if c.IsSimple("") {
		t.Error("empty query should not be simple")
	}
	if c.IsSimple("   ") {
		t.Error("whitespace-only query should not be simple")
	}
}

func TestFastPathClassifier_ShortAmbiguous(t *testing.T) {
	c := &FastPathClassifier{}

	// Short messages without tool keywords should be simple.
	short := []string{
		"why is the sky blue?",
		"cool",
		"ok thanks",
		"interesting",
		"that makes sense",
	}

	for _, q := range short {
		if !c.IsSimple(q) {
			t.Errorf("short query should be simple: %q", q)
		}
	}
}

func TestClassifyQuery_Fast(t *testing.T) {
	c := &FastPathClassifier{}

	fast := []string{
		"hi",
		"hello!",
		"hey",
		"good morning",
		"thanks",
		"thank you",
		"bye",
		"cheers",
		"yes",
		"no",
		"nope",
		"ok",
		"cool",
		"great",
		"got it",
		"who are you?",
		"what's your name?",
		"how are you?",
		"tell me a joke",
		"what is 2 + 2?",
		"sure thing",
	}

	for _, q := range fast {
		if path := c.ClassifyQuery(q); path != PathFast {
			t.Errorf("expected fast for %q, got %q", q, path)
		}
	}
}

func TestClassifyQuery_Medium(t *testing.T) {
	c := &FastPathClassifier{}

	medium := []string{
		"my name is Raphael",
		"I'm a developer",
		"I work on distributed systems",
		"call me Raph",
		"explain how garbage collection works in Go",
		"what is a neural network and how does it learn",
		"tell me more about that topic please",
		"what do you think about functional programming languages",
		"which is better, Python or Go for web development",
		"can you help me understand monads in Haskell",
		"summarize the main ideas of machine learning algorithms",
		"why is Rust considered memory safe compared to C",
		// Possessive recall patterns
		"what's my favorite food?",
		"what is my name?",
		"what are my interests?",
		"do you remember my name?",
		"do you know who I am?",
		"do you know about me?",
		"tell me my name",
		"we talked about this before",
	}

	for _, q := range medium {
		if path := c.ClassifyQuery(q); path != PathMedium {
			t.Errorf("expected medium for %q, got %q", q, path)
		}
	}
}

func TestClassifyQuery_Full(t *testing.T) {
	c := &FastPathClassifier{}

	full := []string{
		"read file main.go",
		"find all TODO comments in the codebase",
		"run the tests",
		"git status",
		"create a new file called test.txt",
		"refactor the server module",
		"debug the login function",
		"compile the project",
		"fetch the API response from https://example.com",
		"analyze the codebase for security issues",
		"search the web for Go tutorials",
		"list files in the current directory",
		"deploy the application",
	}

	for _, q := range full {
		if path := c.ClassifyQuery(q); path != PathFull {
			t.Errorf("expected full for %q, got %q", q, path)
		}
	}
}

func TestClassifyQuery_Empty(t *testing.T) {
	c := &FastPathClassifier{}
	if path := c.ClassifyQuery(""); path != PathFull {
		t.Errorf("empty query should be full, got %q", path)
	}
	if path := c.ClassifyQuery("   "); path != PathFull {
		t.Errorf("whitespace query should be full, got %q", path)
	}
}

// --- Edge case and stress tests ---

func TestClassifyQuery_LanguageVariants(t *testing.T) {
	c := &FastPathClassifier{}

	// Greetings in different styles
	greetings := []string{"Hi!", "HELLO", "Hey.", "howdy", "Yo", "sup", "Greetings!", "Hola", "Bonjour", "Guten Tag", "Hallo"}
	for _, q := range greetings {
		if path := c.ClassifyQuery(q); path != PathFast {
			t.Errorf("greeting %q should be fast, got %q", q, path)
		}
	}
}

func TestClassifyQuery_YesNoVariants(t *testing.T) {
	c := &FastPathClassifier{}

	yesNo := []string{"yes", "no", "y", "n", "absolutely", "definitely", "of course", "nah", "nope", "maybe"}
	for _, q := range yesNo {
		if path := c.ClassifyQuery(q); path != PathFast {
			t.Errorf("yes/no %q should be fast, got %q", q, path)
		}
	}
}

func TestClassifyQuery_ComplexPatternPriority(t *testing.T) {
	c := &FastPathClassifier{}

	// Complex patterns should take priority even if query is short
	mustBeComplex := []string{
		"run tests",
		"git diff",
		"git status",
		"compile the project",
		"build a container",
		"what files are in internal/cognitive/",
		"find all test files",
		"list go files",
		"show all spec files",
	}
	for _, q := range mustBeComplex {
		if path := c.ClassifyQuery(q); path != PathFull {
			t.Errorf("complex query %q should be full, got %q", q, path)
		}
	}
}

func TestClassifyQuery_MathExpressions(t *testing.T) {
	c := &FastPathClassifier{}

	math := []string{"what is 100 + 200?", "42 * 7?", "100 / 5?", "99 - 1"}
	for _, q := range math {
		if path := c.ClassifyQuery(q); path != PathFast {
			t.Errorf("math %q should be fast, got %q", q, path)
		}
	}
}

func TestClassifyQuery_FollowUp(t *testing.T) {
	c := &FastPathClassifier{}

	followups := []string{
		"tell me more about that topic please",
		"go on and continue explaining that idea",
		"can you explain that in more detail please",
		"what about the performance implications of this approach",
	}
	for _, q := range followups {
		if path := c.ClassifyQuery(q); path != PathMedium {
			t.Errorf("follow-up %q should be medium, got %q", q, path)
		}
	}
}

func TestClassifyQuery_SecurityTools(t *testing.T) {
	c := &FastPathClassifier{}

	// Docker/sandbox queries should be full pipeline
	security := []string{
		"start a docker container for the test environment",
		"kill the running process on port 8080",
		"restart the sandbox environment for testing",
	}
	for _, q := range security {
		if path := c.ClassifyQuery(q); path != PathFull {
			t.Errorf("security/system query %q should be full, got %q", q, path)
		}
	}
}

func TestIsSimpleBackwardCompat(t *testing.T) {
	c := &FastPathClassifier{}

	// IsSimple should return true for both fast and medium
	if !c.IsSimple("hi") {
		t.Error("'hi' (fast) should be simple")
	}
	if !c.IsSimple("explain how TCP works in detail for networking") {
		t.Error("explanatory query (medium) should be simple")
	}
	if c.IsSimple("read file main.go") {
		t.Error("tool query (full) should not be simple")
	}
}

func TestQuickGreetingResponses(t *testing.T) {
	// Trivial greetings should get instant canned responses
	instant := []string{"hello", "hi", "hey", "thanks", "thanks!", "bye", "ok", "cool"}
	for _, q := range instant {
		resp := tryQuickResponse(q)
		if resp == "" {
			t.Errorf("expected quick response for %q, got empty", q)
		}
	}

	// Non-trivial queries should NOT get canned responses
	notInstant := []string{"what is Go", "read go.mod", "help me debug", "explain this code"}
	for _, q := range notInstant {
		resp := tryQuickResponse(q)
		if resp != "" {
			t.Errorf("unexpected quick response for %q: %s", q, resp)
		}
	}
}

func TestPathConstants(t *testing.T) {
	if PathFast != "fast" {
		t.Errorf("PathFast = %q, want 'fast'", PathFast)
	}
	if PathMedium != "medium" {
		t.Errorf("PathMedium = %q, want 'medium'", PathMedium)
	}
	if PathFull != "full" {
		t.Errorf("PathFull = %q, want 'full'", PathFull)
	}
}
