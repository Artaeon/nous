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
		"my name is Raphael",
		"I'm a developer",
		"I work on distributed systems",
		"call me Raph",
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
		"explain how garbage collection works in Go",
		"what is a neural network and how does it learn",
		"tell me more about that topic please",
		"what do you think about functional programming languages",
		"which is better, Python or Go for web development",
		"can you help me understand monads in Haskell",
		"summarize the main ideas of machine learning algorithms",
		"why is Rust considered memory safe compared to C",
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
