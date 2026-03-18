package cognitive

import (
	"testing"
)

func TestNLUGreetings(t *testing.T) {
	nlu := NewNLU()
	cases := []string{
		"hi", "hello", "hey", "howdy", "good morning",
		"good afternoon", "Hi there!", "Hello!",
		"Hey, how are you?", "Greetings",
	}
	for _, input := range cases {
		r := nlu.Understand(input)
		if r.Intent != "greeting" {
			t.Errorf("Understand(%q): want intent=greeting, got %q", input, r.Intent)
		}
		if r.Action != "respond" {
			t.Errorf("Understand(%q): want action=respond, got %q", input, r.Action)
		}
		if r.Confidence < 0.9 {
			t.Errorf("Understand(%q): want confidence>=0.9, got %.2f", input, r.Confidence)
		}
	}
}

func TestNLUFarewells(t *testing.T) {
	nlu := NewNLU()
	cases := []string{
		"bye", "goodbye", "see ya", "later", "farewell",
		"take care", "ciao", "good night",
	}
	for _, input := range cases {
		r := nlu.Understand(input)
		if r.Intent != "farewell" {
			t.Errorf("Understand(%q): want intent=farewell, got %q", input, r.Intent)
		}
		if r.Action != "respond" {
			t.Errorf("Understand(%q): want action=respond, got %q", input, r.Action)
		}
	}
}

func TestNLUAffirmations(t *testing.T) {
	nlu := NewNLU()
	cases := []string{
		"yes", "ok", "thanks", "great", "cool",
		"no", "nope", "got it",
	}
	for _, input := range cases {
		r := nlu.Understand(input)
		if r.Intent != "affirmation" {
			t.Errorf("Understand(%q): want intent=affirmation, got %q", input, r.Intent)
		}
		if r.Action != "respond" {
			t.Errorf("Understand(%q): want action=respond, got %q", input, r.Action)
		}
	}
}

func TestNLUMeta(t *testing.T) {
	nlu := NewNLU()
	cases := []string{
		"what can you do", "who are you", "help",
		"what are you", "tell me about yourself",
	}
	for _, input := range cases {
		r := nlu.Understand(input)
		if r.Intent != "meta" {
			t.Errorf("Understand(%q): want intent=meta, got %q", input, r.Intent)
		}
		if r.Action != "respond" {
			t.Errorf("Understand(%q): want action=respond, got %q", input, r.Action)
		}
	}
}

func TestNLUQuestions(t *testing.T) {
	nlu := NewNLU()
	cases := []struct {
		input string
		topic string
	}{
		{"What is Go?", "go"},
		{"How does TCP work?", "tcp work"},
		{"Why is the sky blue?", "the sky blue"},
		{"Is Python better than Go?", ""},
		{"Who invented the telephone?", "invented the telephone"},
	}
	for _, c := range cases {
		r := nlu.Understand(c.input)
		if r.Intent != "question" && r.Intent != "explain" {
			t.Errorf("Understand(%q): want intent=question|explain, got %q", c.input, r.Intent)
		}
		if r.Confidence < 0.5 {
			t.Errorf("Understand(%q): want confidence>=0.5, got %.2f", c.input, r.Confidence)
		}
		if c.topic != "" {
			if got, ok := r.Entities["topic"]; !ok || got != c.topic {
				t.Errorf("Understand(%q): want topic=%q, got %q", c.input, c.topic, got)
			}
		}
	}
}

func TestNLUCommands(t *testing.T) {
	nlu := NewNLU()
	cases := []string{
		"run the tests", "deploy to production",
		"build the project", "restart the server",
		"install docker", "delete the cache",
	}
	for _, input := range cases {
		r := nlu.Understand(input)
		if r.Intent != "command" {
			t.Errorf("Understand(%q): want intent=command, got %q", input, r.Intent)
		}
		if r.Confidence < 0.7 {
			t.Errorf("Understand(%q): want confidence>=0.7, got %.2f", input, r.Confidence)
		}
	}
}

func TestNLUSearch(t *testing.T) {
	nlu := NewNLU()
	cases := []struct {
		input string
		query string
	}{
		{"search for quantum computing", "quantum computing"},
		{"find information about black holes", "information about black holes"},
		{"look up Go concurrency patterns", "go concurrency patterns"},
		{"google best pizza in NYC", "best pizza in nyc"},
	}
	for _, c := range cases {
		r := nlu.Understand(c.input)
		if r.Intent != "search" {
			t.Errorf("Understand(%q): want intent=search, got %q", c.input, r.Intent)
		}
		if r.Action != "web_search" {
			t.Errorf("Understand(%q): want action=web_search, got %q", c.input, r.Action)
		}
	}
}

func TestNLUWebLookup(t *testing.T) {
	nlu := NewNLU()
	cases := []struct {
		input  string
		action string
	}{
		{"What's the weather in Berlin?", "web_search"},
		{"Who won the Super Bowl?", "web_search"},
		{"Latest news about AI", "web_search"},
		{"What's the stock price of Apple?", "web_search"},
		{"What is happening in Ukraine?", "web_search"},
	}
	for _, c := range cases {
		r := nlu.Understand(c.input)
		if r.Action != c.action {
			t.Errorf("Understand(%q): want action=%s, got %s (intent=%s)", c.input, c.action, r.Action, r.Intent)
		}
	}
}

func TestNLUExplain(t *testing.T) {
	nlu := NewNLU()
	cases := []struct {
		input  string
		action string
		topic  string
	}{
		{"explain quantum entanglement", "lookup_knowledge", "quantum entanglement"},
		{"what is photosynthesis", "lookup_knowledge", "photosynthesis"},
		{"how does a CPU work", "lookup_knowledge", "a cpu work"},
		{"tell me about machine learning", "lookup_knowledge", "machine learning"},
	}
	for _, c := range cases {
		r := nlu.Understand(c.input)
		if r.Intent != "explain" {
			t.Errorf("Understand(%q): want intent=explain, got %q", c.input, r.Intent)
		}
		if r.Action != c.action {
			t.Errorf("Understand(%q): want action=%s, got %s", c.input, c.action, r.Action)
		}
	}
}

func TestNLUFileOp(t *testing.T) {
	nlu := NewNLU()
	cases := []string{
		"read the config file",
		"open main.go",
		"edit the Dockerfile",
		"create a new file",
	}
	for _, input := range cases {
		r := nlu.Understand(input)
		if r.Intent != "file_op" {
			t.Errorf("Understand(%q): want intent=file_op, got %q", input, r.Intent)
		}
		if r.Action != "file_op" {
			t.Errorf("Understand(%q): want action=file_op, got %q", input, r.Action)
		}
	}
}

func TestNLURemember(t *testing.T) {
	nlu := NewNLU()
	cases := []string{
		"remember that my favorite color is blue",
		"my name is Raphael",
		"i work at Stoicera",
		"note that the server IP is 192.168.1.1",
	}
	for _, input := range cases {
		r := nlu.Understand(input)
		if r.Intent != "remember" {
			t.Errorf("Understand(%q): want intent=remember, got %q", input, r.Intent)
		}
		if r.Action != "lookup_memory" {
			t.Errorf("Understand(%q): want action=lookup_memory, got %q", input, r.Action)
		}
	}
}

func TestNLURecall(t *testing.T) {
	nlu := NewNLU()
	cases := []string{
		"do you remember my favorite color",
		"what's my name",
		"what did i tell you about the server",
	}
	for _, input := range cases {
		r := nlu.Understand(input)
		if r.Intent != "recall" {
			t.Errorf("Understand(%q): want intent=recall, got %q", input, r.Intent)
		}
		if r.Action != "lookup_memory" {
			t.Errorf("Understand(%q): want action=lookup_memory, got %q", input, r.Action)
		}
	}
}

func TestNLUPlan(t *testing.T) {
	nlu := NewNLU()
	cases := []string{
		"plan my day tomorrow",
		"remind me to buy groceries",
		"schedule a meeting for next Monday",
		"add a todo for code review",
	}
	for _, input := range cases {
		r := nlu.Understand(input)
		if r.Intent != "plan" {
			t.Errorf("Understand(%q): want intent=plan, got %q", input, r.Intent)
		}
		if r.Action != "schedule" {
			t.Errorf("Understand(%q): want action=schedule, got %q", input, r.Action)
		}
	}
}

func TestNLUCompute(t *testing.T) {
	nlu := NewNLU()
	cases := []struct {
		input      string
		hasExpr    bool
		expression string
	}{
		{"what is 2 + 2", true, "2 + 2"},
		{"calculate 15 * 3", true, "15 * 3"},
		{"how much is 100 / 4", true, "100 / 4"},
	}
	for _, c := range cases {
		r := nlu.Understand(c.input)
		if r.Intent != "compute" {
			t.Errorf("Understand(%q): want intent=compute, got %q", c.input, r.Intent)
		}
		if r.Action != "compute" {
			t.Errorf("Understand(%q): want action=compute, got %q", c.input, r.Action)
		}
		if c.hasExpr {
			if expr, ok := r.Entities["expression"]; !ok || expr != c.expression {
				t.Errorf("Understand(%q): want expression=%q, got %q", c.input, c.expression, expr)
			}
		}
	}
}

func TestNLUEntityExtraction(t *testing.T) {
	nlu := NewNLU()

	t.Run("URL", func(t *testing.T) {
		r := nlu.Understand("fetch https://example.com/api/data")
		if url, ok := r.Entities["url"]; !ok || url != "https://example.com/api/data" {
			t.Errorf("want url=https://example.com/api/data, got %q", r.Entities["url"])
		}
	})

	t.Run("FilePath", func(t *testing.T) {
		r := nlu.Understand("read ./internal/main.go")
		if path, ok := r.Entities["path"]; !ok || path != "./internal/main.go" {
			t.Errorf("want path=./internal/main.go, got %q", r.Entities["path"])
		}
	})

	t.Run("MathExpression", func(t *testing.T) {
		r := nlu.Understand("compute 42 + 58")
		if expr, ok := r.Entities["expression"]; !ok || expr != "42 + 58" {
			t.Errorf("want expression=42 + 58, got %q", r.Entities["expression"])
		}
	})

	t.Run("QuotedString", func(t *testing.T) {
		r := nlu.Understand(`search for "exact phrase match"`)
		if q, ok := r.Entities["quoted"]; !ok || q != "exact phrase match" {
			t.Errorf("want quoted=exact phrase match, got %q", r.Entities["quoted"])
		}
	})

	t.Run("DateWord", func(t *testing.T) {
		r := nlu.Understand("schedule a meeting for tomorrow")
		if d, ok := r.Entities["date"]; !ok || d != "tomorrow" {
			t.Errorf("want date=tomorrow, got %q", r.Entities["date"])
		}
	})

	t.Run("DateFormal", func(t *testing.T) {
		r := nlu.Understand("remind me on March 5th")
		if d, ok := r.Entities["date"]; !ok || d != "March 5th" {
			t.Errorf("want date=March 5th, got %q", r.Entities["date"])
		}
	})
}

func TestNLUConfidence(t *testing.T) {
	nlu := NewNLU()

	t.Run("HighConfidence", func(t *testing.T) {
		r := nlu.Understand("hello")
		if r.Confidence < 0.9 {
			t.Errorf("greeting 'hello' should have high confidence, got %.2f", r.Confidence)
		}
	})

	t.Run("MediumConfidence", func(t *testing.T) {
		r := nlu.Understand("how does quantum mechanics relate to consciousness?")
		if r.Confidence < 0.5 || r.Confidence > 0.95 {
			t.Errorf("complex question should have medium confidence, got %.2f", r.Confidence)
		}
	})

	t.Run("LowConfidence", func(t *testing.T) {
		r := nlu.Understand("hmm well you see the thing is")
		if r.Confidence >= 0.5 {
			t.Errorf("ambiguous input should have low confidence (<0.5), got %.2f", r.Confidence)
		}
	})
}

func TestNLUEdgeCases(t *testing.T) {
	nlu := NewNLU()

	t.Run("EmptyInput", func(t *testing.T) {
		r := nlu.Understand("")
		if r.Intent != "unknown" {
			t.Errorf("empty input: want intent=unknown, got %q", r.Intent)
		}
		if r.Confidence != 0.0 {
			t.Errorf("empty input: want confidence=0.0, got %.2f", r.Confidence)
		}
	})

	t.Run("WhitespaceOnly", func(t *testing.T) {
		r := nlu.Understand("   \t\n  ")
		if r.Intent != "unknown" {
			t.Errorf("whitespace input: want intent=unknown, got %q", r.Intent)
		}
	})

	t.Run("SingleWord", func(t *testing.T) {
		r := nlu.Understand("photosynthesis")
		if r.Intent == "unknown" {
			t.Errorf("single word should get some intent, got unknown")
		}
	})

	t.Run("VeryLongInput", func(t *testing.T) {
		long := "tell me about " + repeatWord("very ", 100) + "interesting topic"
		r := nlu.Understand(long)
		if r.Intent == "" {
			t.Error("long input should still produce an intent")
		}
	})

	t.Run("URLAsMainInput", func(t *testing.T) {
		r := nlu.Understand("https://golang.org/doc/effective_go")
		if _, ok := r.Entities["url"]; !ok {
			t.Error("bare URL should be extracted as entity")
		}
	})

	t.Run("MixedIntent", func(t *testing.T) {
		// "search" verb + compute expression: search verb should win since it comes first
		r := nlu.Understand("search for results about 2 + 2")
		if r.Intent != "search" {
			t.Errorf("mixed intent should resolve to search, got %q", r.Intent)
		}
		// But math entity should still be extracted
		if _, ok := r.Entities["expression"]; !ok {
			t.Error("math expression should still be extracted even when intent is search")
		}
	})
}

func TestNLUActionMapping(t *testing.T) {
	nlu := NewNLU()

	tests := []struct {
		input  string
		action string
	}{
		// Instant responses
		{"hi", "respond"},
		{"bye", "respond"},
		{"thanks", "respond"},
		{"who are you", "respond"},
		// Web search
		{"what's the weather in London", "web_search"},
		{"latest news about AI", "web_search"},
		{"search for Go tutorials", "web_search"},
		// Knowledge lookup
		{"explain quantum computing", "lookup_knowledge"},
		{"what is photosynthesis", "lookup_knowledge"},
		// Memory
		{"remember my birthday is March 5th", "lookup_memory"},
		{"what's my name", "lookup_memory"},
		// File ops
		{"read main.go", "file_op"},
		// Compute
		{"what is 42 + 58", "compute"},
		// Schedule
		{"remind me to buy milk tomorrow", "schedule"},
	}

	for _, tt := range tests {
		r := nlu.Understand(tt.input)
		if r.Action != tt.action {
			t.Errorf("Understand(%q): want action=%s, got %s (intent=%s)", tt.input, tt.action, r.Action, r.Intent)
		}
	}
}

func TestNLURawPreserved(t *testing.T) {
	nlu := NewNLU()
	input := "  Hello, World!  "
	r := nlu.Understand(input)
	if r.Raw != input {
		t.Errorf("Raw should preserve original input exactly, got %q", r.Raw)
	}
}

func TestNLUKnowledgeVsWeb(t *testing.T) {
	nlu := NewNLU()

	// General concepts should go to knowledge base first
	r := nlu.Understand("what is photosynthesis")
	if r.Action != "lookup_knowledge" {
		t.Errorf("general concept should use lookup_knowledge, got %s", r.Action)
	}

	// Current events should go to web
	r = nlu.Understand("who won the election")
	if r.Action != "web_search" {
		t.Errorf("current event should use web_search, got %s", r.Action)
	}

	// Weather should go to web
	r = nlu.Understand("what is the weather today")
	if r.Action != "web_search" {
		t.Errorf("weather should use web_search, got %s", r.Action)
	}
}

// repeatWord repeats a word n times.
func repeatWord(word string, n int) string {
	result := ""
	for i := 0; i < n; i++ {
		result += word
	}
	return result
}

// --- Follow-up resolution tests ---

// newConvWithHistory creates a Conversation with a prior user+assistant exchange.
func newConvWithHistory(userMsg, assistantMsg string) *Conversation {
	conv := NewConversation(20)
	conv.User(userMsg)
	conv.Assistant(assistantMsg)
	return conv
}

func TestNLUFollowUp_ExplainFurther(t *testing.T) {
	nlu := NewNLU()
	conv := newConvWithHistory(
		"explain quantum entanglement",
		"Quantum entanglement is a phenomenon where two particles become correlated.",
	)

	cases := []string{
		"explain further",
		"explain more",
		"elaborate",
		"more details",
	}
	for _, input := range cases {
		r := nlu.UnderstandWithContext(input, conv)
		if r.Entities["follow_up"] != "true" {
			t.Errorf("UnderstandWithContext(%q): expected follow_up=true, got %q", input, r.Entities["follow_up"])
		}
		if r.Entities["previous_topic"] == "" {
			t.Errorf("UnderstandWithContext(%q): expected previous_topic to be set", input)
		}
		if r.Intent != "explain" {
			t.Errorf("UnderstandWithContext(%q): want intent=explain, got %q", input, r.Intent)
		}
		if r.Confidence < 0.7 {
			t.Errorf("UnderstandWithContext(%q): want confidence>=0.7, got %.2f", input, r.Confidence)
		}
	}
}

func TestNLUFollowUp_TellMeMore(t *testing.T) {
	nlu := NewNLU()
	conv := newConvWithHistory(
		"what is photosynthesis",
		"Photosynthesis is the process by which plants convert sunlight into energy.",
	)

	cases := []string{
		"tell me more",
		"go on",
		"continue",
		"what else?",
		"and then?",
	}
	for _, input := range cases {
		r := nlu.UnderstandWithContext(input, conv)
		if r.Entities["follow_up"] != "true" {
			t.Errorf("UnderstandWithContext(%q): expected follow_up=true", input)
		}
		topic := r.Entities["topic"]
		if topic == "" {
			t.Errorf("UnderstandWithContext(%q): expected topic to be resolved from prior turn", input)
		}
		if r.Confidence < 0.7 {
			t.Errorf("UnderstandWithContext(%q): want confidence>=0.7, got %.2f", input, r.Confidence)
		}
	}
}

func TestNLUFollowUp_SingleWordQuestion(t *testing.T) {
	nlu := NewNLU()
	conv := newConvWithHistory(
		"explain how black holes form",
		"Black holes form when massive stars collapse at the end of their life cycle.",
	)

	cases := []string{
		"why?",
		"how?",
	}
	for _, input := range cases {
		r := nlu.UnderstandWithContext(input, conv)
		if r.Entities["follow_up"] != "true" {
			t.Errorf("UnderstandWithContext(%q): expected follow_up=true", input)
		}
		if r.Entities["previous_topic"] == "" {
			t.Errorf("UnderstandWithContext(%q): expected previous_topic to be set", input)
		}
	}
}

func TestNLUFollowUp_WithNewAngle(t *testing.T) {
	nlu := NewNLU()
	conv := newConvWithHistory(
		"explain quantum physics",
		"Quantum physics deals with the behavior of matter at the subatomic level.",
	)

	r := nlu.UnderstandWithContext("what about entanglement?", conv)
	if r.Entities["follow_up"] != "true" {
		t.Errorf("expected follow_up=true, got %q", r.Entities["follow_up"])
	}
	if r.Entities["previous_topic"] == "" {
		t.Error("expected previous_topic to be set")
	}
	if r.Entities["new_angle"] == "" {
		t.Error("expected new_angle to be set for 'what about' pattern")
	}
	if r.Entities["new_angle"] != "entanglement" {
		t.Errorf("expected new_angle=entanglement, got %q", r.Entities["new_angle"])
	}
}

func TestNLUFollowUp_NoHistoryNoChange(t *testing.T) {
	nlu := NewNLU()
	conv := NewConversation(20) // empty conversation

	r := nlu.UnderstandWithContext("tell me more", conv)
	// Without history, follow-up detection should still trigger but no topic resolved
	if r.Entities["follow_up"] == "true" {
		t.Error("should not mark follow_up without conversation history to resolve from")
	}
}

func BenchmarkNLUUnderstand(b *testing.B) {
	nlu := NewNLU()
	inputs := []string{
		"hello",
		"what is quantum computing?",
		"search for Go concurrency patterns",
		"remember my name is Raphael",
		"what's the weather in Berlin?",
		"calculate 42 + 58",
		"read ./internal/main.go",
		"explain how photosynthesis works",
		"plan my day tomorrow",
		"https://example.com/api/data",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nlu.Understand(inputs[i%len(inputs)])
	}
}
