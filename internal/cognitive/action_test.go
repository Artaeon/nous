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
		{"unknown greeting", "salutations dear friend", true, ""},
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
				if false /* NeedsLLM removed */ {
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
			if tt.wantOk && false /* NeedsLLM removed */ {
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

	if result.Source != "memory" {
		t.Errorf("source = %q, want memory", result.Source)
	}
	// With only working memory, should still produce a response mentioning the name.
	output := result.Data + result.DirectResponse
	if !strings.Contains(output, "Raphael") && !strings.Contains(output, "user_name") {
		t.Errorf("output should contain name or key, got %q", output)
	}
}

func TestActionRouter_LookupMemory_DirectFact(t *testing.T) {
	ar := NewActionRouter()
	ar.LongTermMem = memory.NewLongTermMemory("")

	ar.LongTermMem.Store("user_name", "Raphael", "personal")

	nlu := &NLUResult{
		Action:   "lookup_memory",
		Entities: map[string]string{"query": "user_name"},
		Raw:      "what is my name",
	}
	result := ar.Execute(nlu, NewConversation(10))

	if result.Source != "memory" {
		t.Errorf("source = %q, want memory", result.Source)
	}
	// Single longterm fact should return directly without LLM.
	if false /* NeedsLLM removed */ {
		t.Error("single longterm fact should not need LLM")
	}
	if !strings.Contains(result.DirectResponse, "Raphael") {
		t.Errorf("DirectResponse = %q, should contain 'Raphael'", result.DirectResponse)
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

	output := result.Data + result.DirectResponse
	if !strings.Contains(output, "no relevant memories") {
		t.Errorf("expected 'no relevant memories', got %q", output)
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

	if result.DirectResponse == "" {
		t.Error("web search with no tools should produce a DirectResponse")
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
	combined := result.Data + result.DirectResponse
	if !strings.Contains(combined, "hello world") {
		t.Errorf("response = %q, should contain mock output", combined)
	}
}

func TestActionRouter_LLMChat(t *testing.T) {
	ar := NewActionRouter()

	nlu := &NLUResult{
		Action: "llm_chat",
		Raw:    "tell me about quantum physics",
	}
	result := ar.Execute(nlu, NewConversation(10))

	// With LLM removed, llm_chat falls through to Composer or fallback.
	// It should never set NeedsLLM.
	if false /* NeedsLLM removed */ {
		t.Error("llm_chat should NOT need LLM — engine handles all responses")
	}
	if result.DirectResponse == "" {
		t.Error("should produce a direct response from composer or fallback")
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
	// Schedule now returns a direct response — no LLM needed.
	if false /* NeedsLLM removed */ {
		t.Error("schedule should not need LLM")
	}
	if result.DirectResponse == "" {
		t.Error("schedule should return DirectResponse")
	}
	if !strings.Contains(result.DirectResponse, "review PR") {
		t.Errorf("DirectResponse should mention the task, got %q", result.DirectResponse)
	}
}

func TestActionRouter_UnknownAction(t *testing.T) {
	ar := NewActionRouter()

	nlu := &NLUResult{
		Action: "unknown_action",
		Raw:    "do something weird",
	}
	result := ar.Execute(nlu, NewConversation(10))

	// Unknown actions now get handled by the cognitive engine
	if result.DirectResponse == "" && result.Data == "" {
		t.Error("unknown action should produce some response")
	}
	// Source could be fallback, composer, or thinking
	if result.Source == "" {
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

	result := &ActionResult{Data: "raw facts here"}
	got, err := rf.Format("question", result, NewConversation(10))
	if err != nil {
		t.Fatal(err)
	}
	if got != "raw facts here" {
		t.Errorf("Format without LLM = %q, want raw data", got)
	}
}

func TestNoLLM_AllDirectResponse(t *testing.T) {
	ar := NewActionRouter()

	// All actions should produce DirectResponse — no LLM needed.
	tests := []struct {
		action string
		raw    string
	}{
		{"respond", "hello"},
		{"compute", "2+2"},
		{"llm_chat", "what is life"},
	}
	for _, tt := range tests {
		nlu := &NLUResult{Action: tt.action, Entities: map[string]string{"expr": tt.raw}, Raw: tt.raw}
		result := ar.Execute(nlu, NewConversation(10))
		if result.DirectResponse == "" && result.Data == "" {
			t.Errorf("action %q for %q produced no response", tt.action, tt.raw)
		}
	}
}

func TestActionChain_ResearchAndWrite(t *testing.T) {
	ar := NewActionRouter()
	// No tools registered — web search and knowledge both return "unavailable" style data.

	nlu := &NLUResult{
		Action: "chain",
		Entities: map[string]string{
			"chain_type": "research_and_write",
			"topic":      "quantum physics",
		},
		Raw: "research quantum physics",
	}
	result := ar.Execute(nlu, NewConversation(10))

	if !strings.HasPrefix(result.Source, "chain:") {
		t.Errorf("source should start with 'chain:', got %q", result.Source)
	}
	// Chain now returns DirectResponse (not Data).
	if result.DirectResponse == "" {
		t.Error("chain result DirectResponse should not be empty")
	}
}

func TestActionChain_SearchAndSave(t *testing.T) {
	ar := NewActionRouter()
	reg := tools.NewRegistry()
	// Register a mock websearch tool.
	reg.Register(tools.Tool{
		Name:        "websearch",
		Description: "mock search",
		Execute: func(args map[string]string) (string, error) {
			return "search results for: " + args["query"], nil
		},
	})
	// Register a mock write tool.
	var writtenContent string
	reg.Register(tools.Tool{
		Name:        "write",
		Description: "mock write",
		Execute: func(args map[string]string) (string, error) {
			writtenContent = args["content"]
			return "wrote to " + args["path"], nil
		},
	})
	ar.Tools = reg

	nlu := &NLUResult{
		Action: "chain",
		Entities: map[string]string{
			"chain_type": "search_and_save",
			"topic":      "AI news",
			"path":       "ai_news.txt",
		},
		Raw: "search for AI news and save it to a file",
	}
	_ = ar.Execute(nlu, NewConversation(10))

	// The write step should have received the search output via dependency.
	if writtenContent == "" {
		t.Error("write step should have received content from search step via dependency")
	}
	if !strings.Contains(writtenContent, "search results for: AI news") {
		t.Errorf("written content should contain search results, got %q", writtenContent)
	}
}

func TestActionChain_StepDependency(t *testing.T) {
	ar := NewActionRouter()
	reg := tools.NewRegistry()
	reg.Register(tools.Tool{
		Name:        "websearch",
		Description: "mock search",
		Execute: func(args map[string]string) (string, error) {
			return "SEARCH_OUTPUT_DATA", nil
		},
	})
	var capturedInput string
	reg.Register(tools.Tool{
		Name:        "write",
		Description: "mock write",
		Execute: func(args map[string]string) (string, error) {
			capturedInput = args["content"]
			return "ok", nil
		},
	})
	ar.Tools = reg

	chain := &ActionChain{
		Steps: []ChainStep{
			{
				Action:    "web_search",
				Entities:  map[string]string{"query": "test"},
				DependsOn: -1,
			},
			{
				Action:    "file_op",
				Entities:  map[string]string{"op": "write", "path": "out.txt"},
				DependsOn: 0, // depends on step 0
			},
		},
	}

	nlu := &NLUResult{
		Action:   "chain",
		Entities: map[string]string{},
		Raw:      "test",
	}
	ar.ExecuteChain(chain, nlu, NewConversation(10))

	// Verify step 1 received step 0's output.
	if capturedInput != "SEARCH_OUTPUT_DATA" {
		t.Errorf("step 1 should receive step 0 output via dependency, got %q", capturedInput)
	}

	// Verify Results were populated.
	if len(chain.Results) != 2 {
		t.Fatalf("chain should have 2 results, got %d", len(chain.Results))
	}
	step0Output := chain.Results[0].Data
	if step0Output == "" {
		step0Output = chain.Results[0].DirectResponse
	}
	if !strings.Contains(step0Output, "SEARCH_OUTPUT_DATA") {
		t.Errorf("step 0 result should contain SEARCH_OUTPUT_DATA, got %q", step0Output)
	}
}

func TestActionRouter_GenerateDoc(t *testing.T) {
	ar := NewActionRouter()

	nlu := &NLUResult{
		Action:   "generate_doc",
		Entities: map[string]string{"topic": "quantum computing"},
		Raw:      "create a document about quantum computing",
	}
	result := ar.Execute(nlu, NewConversation(10))

	if !strings.Contains(result.DirectResponse, "[Document Request: quantum computing]") {
		t.Errorf("generate_doc DirectResponse should contain document request header, got %q", result.DirectResponse)
	}
	if result.Structured == nil || result.Structured["format"] != "document" {
		t.Error("generate_doc should set Structured format=document")
	}
	if result.Structured["topic"] != "quantum computing" {
		t.Errorf("generate_doc topic = %q, want 'quantum computing'", result.Structured["topic"])
	}
}

func TestActionRouter_CompareParsesVsItems(t *testing.T) {
	ar := NewActionRouter()

	nlu := &NLUResult{
		Action: "compare",
		Raw:    "compare go vs rust",
	}
	result := ar.Execute(nlu, NewConversation(10))

	if result == nil || result.DirectResponse == "" {
		t.Fatal("expected non-empty compare response")
	}
	lower := strings.ToLower(result.DirectResponse)
	if strings.Contains(lower, "compare compare go and rust") || strings.Contains(lower, "about compare go and rust") {
		t.Fatalf("compare parser should strip command prefix, got %q", result.DirectResponse)
	}
	if !strings.Contains(lower, "go") || !strings.Contains(lower, "rust") {
		t.Fatalf("expected response to reference both go and rust, got %q", result.DirectResponse)
	}
}

func TestActionRouter_CompareSparseFallbackIsStructured(t *testing.T) {
	ar := NewActionRouter()

	nlu := &NLUResult{
		Action: "compare",
		Raw:    "compare go vs rust",
	}
	result := ar.Execute(nlu, NewConversation(10))

	if result == nil || result.DirectResponse == "" {
		t.Fatal("expected non-empty compare response")
	}
	lower := strings.ToLower(result.DirectResponse)
	if !strings.Contains(lower, "comparison framework") {
		t.Fatalf("expected structured sparse comparison fallback, got %q", result.DirectResponse)
	}
	if !strings.Contains(lower, "learning curve") {
		t.Fatalf("expected criteria list in sparse fallback, got %q", result.DirectResponse)
	}
}

func TestActionRouter_LookupKnowledgeSparseFallbackIsInformative(t *testing.T) {
	ar := NewActionRouter()

	nlu := &NLUResult{
		Action:   "lookup_knowledge",
		Intent:   "explain",
		Raw:      "give me an overview of operating systems",
		Entities: map[string]string{"topic": "operating systems"},
	}
	result := ar.Execute(nlu, NewConversation(10))

	if result == nil || result.DirectResponse == "" {
		t.Fatal("expected non-empty knowledge response")
	}
	if strings.TrimSpace(result.DirectResponse) == "operating systems" {
		t.Fatalf("expected informative sparse fallback, got raw topic %q", result.DirectResponse)
	}
	lower := strings.ToLower(result.DirectResponse)
	if !strings.Contains(lower, "don't have detailed knowledge") {
		t.Fatalf("expected honest sparse fallback message, got %q", result.DirectResponse)
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
