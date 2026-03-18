package cognitive

import (
	"strings"
	"testing"

	"github.com/artaeon/nous/internal/memory"
	"github.com/artaeon/nous/internal/tools"
)

func TestActionRouter_Respond(t *testing.T) {
	ar := NewActionRouter()

	tests := []struct {
		name     string
		raw      string
		wantDR   bool   // expect DirectResponse
		contains string // substring in DirectResponse
	}{
		{"hello", "hello", true, ""},
		{"hi", "hi", true, ""},
		{"thanks", "thanks", true, ""},
		{"bye", "bye", true, ""},
		{"unknown greeting", "salutations dear friend", false, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nlu := &NLUResult{Action: "respond", Raw: tt.raw}
			result := ar.Execute(nlu, NewConversation(10))

			if tt.wantDR && result.DirectResponse == "" {
				t.Errorf("expected DirectResponse for %q, got empty", tt.raw)
			}
			if !tt.wantDR && result.DirectResponse != "" {
				t.Errorf("did not expect DirectResponse for %q, got %q", tt.raw, result.DirectResponse)
			}
			if tt.wantDR {
				if result.NeedsLLM {
					t.Errorf("canned response should not need LLM")
				}
			}
			if tt.contains != "" && !strings.Contains(result.DirectResponse, tt.contains) {
				t.Errorf("DirectResponse %q should contain %q", result.DirectResponse, tt.contains)
			}
		})
	}
}

func TestActionRouter_Compute(t *testing.T) {
	ar := NewActionRouter()

	tests := []struct {
		name   string
		expr   string
		want   string
		wantOk bool
	}{
		{"simple add", "2+2", "4", true},
		{"multiply", "3*7", "21", true},
		{"division", "10/3", "", true}, // just check it doesn't error
		{"power", "2^10", "1024", true},
		{"parens", "(2+3)*4", "20", true},
		{"nested", "((1+2)*3)+4", "13", true},
		{"negative", "-5+3", "-2", true},
		{"modulo", "10%3", "1", true},
		{"percent of", "15% of 200", "30", true},
		{"sqrt", "sqrt(16)", "4", true},
		{"abs", "abs(-42)", "42", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nlu := &NLUResult{
				Action:   "compute",
				Entities: map[string]string{"expr": tt.expr},
				Raw:      tt.expr,
			}
			result := ar.Execute(nlu, NewConversation(10))
			if result.DirectResponse == "" && tt.wantOk {
				t.Errorf("expected DirectResponse for %q, got Data=%q", tt.expr, result.Data)
				return
			}
			if tt.want != "" && result.DirectResponse != tt.want {
				t.Errorf("compute(%q) = %q, want %q", tt.expr, result.DirectResponse, tt.want)
			}
		})
	}
}

func TestActionRouter_ComputeDate(t *testing.T) {
	ar := NewActionRouter()

	tests := []struct {
		name   string
		expr   string
		wantOk bool
	}{
		{"today", "what day is today", true},
		{"tomorrow", "what day is tomorrow", true},
		{"yesterday", "what day was yesterday", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nlu := &NLUResult{
				Action:   "compute",
				Entities: map[string]string{"expr": tt.expr},
				Raw:      tt.expr,
			}
			result := ar.Execute(nlu, NewConversation(10))
			if tt.wantOk && result.DirectResponse == "" {
				t.Errorf("expected DirectResponse for date %q, got Data=%q", tt.expr, result.Data)
			}
			if tt.wantOk && result.NeedsLLM {
				t.Errorf("date computation should not need LLM")
			}
		})
	}
}

func TestActionRouter_LookupMemory(t *testing.T) {
	ar := NewActionRouter()
	ar.WorkingMem = memory.NewWorkingMemory(10)
	ar.WorkingMem.Store("user_name", "Raphael", 0.9)

	nlu := &NLUResult{
		Action:   "lookup_memory",
		Entities: map[string]string{"query": "name"},
		Raw:      "what is my name",
	}
	result := ar.Execute(nlu, NewConversation(10))

	if !result.NeedsLLM {
		t.Error("memory lookup should need LLM for formatting")
	}
	if result.Source != "memory" {
		t.Errorf("source = %q, want memory", result.Source)
	}
	if !strings.Contains(result.Data, "user_name") {
		t.Errorf("Data should contain 'user_name', got %q", result.Data)
	}
}

func TestActionRouter_LookupMemoryEmpty(t *testing.T) {
	ar := NewActionRouter()

	nlu := &NLUResult{
		Action:   "lookup_memory",
		Entities: map[string]string{"query": "anything"},
		Raw:      "do you remember anything",
	}
	result := ar.Execute(nlu, NewConversation(10))

	if !strings.Contains(result.Data, "no relevant memories") {
		t.Errorf("expected 'no relevant memories', got %q", result.Data)
	}
}

func TestActionRouter_WebSearch_NoTools(t *testing.T) {
	ar := NewActionRouter()

	nlu := &NLUResult{
		Action:   "web_search",
		Entities: map[string]string{"query": "golang generics"},
		Raw:      "search for golang generics",
	}
	result := ar.Execute(nlu, NewConversation(10))

	if !result.NeedsLLM {
		t.Error("web search should need LLM")
	}
	if result.Source != "web" {
		t.Errorf("source = %q, want web", result.Source)
	}
}

func TestActionRouter_FileOp(t *testing.T) {
	ar := NewActionRouter()
	reg := tools.NewRegistry()
	// Register a mock read tool.
	reg.Register(tools.Tool{
		Name:        "read",
		Description: "mock read",
		Execute: func(args map[string]string) (string, error) {
			return "file content: hello world", nil
		},
	})
	ar.Tools = reg

	nlu := &NLUResult{
		Action:   "file_op",
		Entities: map[string]string{"op": "read", "path": "test.txt"},
		Raw:      "read test.txt",
	}
	result := ar.Execute(nlu, NewConversation(10))

	if result.Source != "file" {
		t.Errorf("source = %q, want file", result.Source)
	}
	if !strings.Contains(result.Data, "hello world") {
		t.Errorf("Data = %q, should contain mock output", result.Data)
	}
}

func TestActionRouter_LLMChat(t *testing.T) {
	ar := NewActionRouter()

	nlu := &NLUResult{
		Action: "llm_chat",
		Raw:    "tell me about quantum physics",
	}
	result := ar.Execute(nlu, NewConversation(10))

	if !result.NeedsLLM {
		t.Error("llm_chat should need LLM")
	}
	if result.Source != "conversation" {
		t.Errorf("source = %q, want conversation", result.Source)
	}
}

func TestActionRouter_Schedule(t *testing.T) {
	ar := NewActionRouter()

	nlu := &NLUResult{
		Action: "schedule",
		Entities: map[string]string{
			"task": "review PR",
			"when": "tomorrow",
		},
		Raw: "remind me to review PR tomorrow",
	}
	result := ar.Execute(nlu, NewConversation(10))

	if result.Structured == nil {
		t.Fatal("expected Structured data")
	}
	if result.Structured["task"] != "review PR" {
		t.Errorf("task = %q, want 'review PR'", result.Structured["task"])
	}
	if result.Structured["parsed_time"] == "" {
		t.Error("expected parsed_time in structured data")
	}
}

func TestActionRouter_UnknownAction(t *testing.T) {
	ar := NewActionRouter()

	nlu := &NLUResult{
		Action: "unknown_action",
		Raw:    "do something weird",
	}
	result := ar.Execute(nlu, NewConversation(10))

	if !result.NeedsLLM {
		t.Error("unknown action should need LLM")
	}
	if result.Source != "fallback" {
		t.Errorf("source = %q, want fallback", result.Source)
	}
}

func TestEvaluateMath(t *testing.T) {
	tests := []struct {
		expr string
		want string
		err  bool
	}{
		{"2+2", "4", false},
		{"10-3", "7", false},
		{"3*4", "12", false},
		{"15/3", "5", false},
		{"2^8", "256", false},
		{"(1+2)*3", "9", false},
		{"sqrt(144)", "12", false},
		{"abs(-7)", "7", false},
		{"10%3", "1", false},
		{"15% of 200", "30", false},
		{"2+3*4", "14", false},   // operator precedence
		{"(2+3)*4", "20", false}, // parentheses
		{"-5", "-5", false},
		{"", "", true},
		{"1/0", "", true}, // division by zero
	}

	for _, tt := range tests {
		t.Run(tt.expr, func(t *testing.T) {
			got, err := evaluateMath(tt.expr)
			if tt.err {
				if err == nil {
					t.Errorf("evaluateMath(%q) = %q, wanted error", tt.expr, got)
				}
				return
			}
			if err != nil {
				t.Errorf("evaluateMath(%q) error: %v", tt.expr, err)
				return
			}
			if got != tt.want {
				t.Errorf("evaluateMath(%q) = %q, want %q", tt.expr, got, tt.want)
			}
		})
	}
}

func TestEvaluateDate(t *testing.T) {
	tests := []struct {
		expr   string
		wantOk bool
	}{
		{"what day is today", true},
		{"tomorrow", true},
		{"yesterday", true},
		{"random text", false},
	}

	for _, tt := range tests {
		t.Run(tt.expr, func(t *testing.T) {
			result, ok := evaluateDate(tt.expr)
			if ok != tt.wantOk {
				t.Errorf("evaluateDate(%q) ok=%v, want %v (result=%q)", tt.expr, ok, tt.wantOk, result)
			}
		})
	}
}

func TestResponseFormatter_DirectResponse(t *testing.T) {
	rf := &ResponseFormatter{} // no LLM

	result := &ActionResult{DirectResponse: "Hello!"}
	got, err := rf.Format("hi", result, NewConversation(10))
	if err != nil {
		t.Fatal(err)
	}
	if got != "Hello!" {
		t.Errorf("Format = %q, want 'Hello!'", got)
	}
}

func TestResponseFormatter_NoLLM_RawData(t *testing.T) {
	rf := &ResponseFormatter{} // no LLM client

	result := &ActionResult{Data: "raw facts here", NeedsLLM: true}
	got, err := rf.Format("question", result, NewConversation(10))
	if err != nil {
		t.Fatal(err)
	}
	if got != "raw facts here" {
		t.Errorf("Format without LLM = %q, want raw data", got)
	}
}

func TestNeedsLLM_Flags(t *testing.T) {
	ar := NewActionRouter()

	// Canned responses should NOT need LLM.
	nlu := &NLUResult{Action: "respond", Raw: "hello"}
	result := ar.Execute(nlu, NewConversation(10))
	if result.NeedsLLM {
		t.Error("canned 'hello' should not need LLM")
	}

	// Computed math should NOT need LLM.
	nlu = &NLUResult{Action: "compute", Entities: map[string]string{"expr": "2+2"}, Raw: "2+2"}
	result = ar.Execute(nlu, NewConversation(10))
	if result.NeedsLLM {
		t.Error("computed '2+2' should not need LLM")
	}

	// Chat should need LLM.
	nlu = &NLUResult{Action: "llm_chat", Raw: "what is life"}
	result = ar.Execute(nlu, NewConversation(10))
	if !result.NeedsLLM {
		t.Error("llm_chat should need LLM")
	}

	// Web search should need LLM (for formatting results).
	nlu = &NLUResult{Action: "web_search", Entities: map[string]string{"query": "test"}, Raw: "search test"}
	result = ar.Execute(nlu, NewConversation(10))
	if !result.NeedsLLM {
		t.Error("web_search should need LLM")
	}
}

func TestParseRelativeTime(t *testing.T) {
	tests := []struct {
		input  string
		wantOk bool
	}{
		{"tomorrow", true},
		{"today", true},
		{"now", true},
		{"in 2 hours", true},
		{"in 30 minutes", true},
		{"in 5 days", true},
		{"in 1 week", true},
		{"next month", false},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			_, ok := parseRelativeTime(tt.input)
			if ok != tt.wantOk {
				t.Errorf("parseRelativeTime(%q) ok=%v, want %v", tt.input, ok, tt.wantOk)
			}
		})
	}
}
