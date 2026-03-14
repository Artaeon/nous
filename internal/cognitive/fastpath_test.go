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
