package cognitive

import (
	"strings"
	"testing"

	"github.com/artaeon/nous/internal/memory"
	"github.com/artaeon/nous/internal/tools"
)

// TestExtractiveFlowEndToEnd simulates: fetch URL → ask follow-ups → get answers without LLM.
func TestExtractiveFlowEndToEnd(t *testing.T) {
	ar := NewActionRouter()
	reg := tools.NewRegistry()

	// Mock summarize tool that returns page content
	reg.Register(tools.Tool{
		Name:        "summarize",
		Description: "mock summarize",
		Execute: func(args map[string]string) (string, error) {
			return `Stoicera is a philosophy company based in Vienna, Austria.
The company was founded in 2023 by Raphael Lugmayr.
Stoicera creates tools and products for modern stoics who want to live a meaningful life.
Their product line includes guided journals, meditation timers, and daily reflection cards.
The stoic philosophy emphasizes virtue, resilience, and living according to nature.
Stoicera's mission is to make ancient wisdom accessible to everyone through beautiful design.
The website features a dark theme with elegant green accents.
Customers can purchase products directly from the online store.
Stoicera also offers a free daily stoic quote newsletter.
The company is fully bootstrapped with no outside funding.`, nil
		},
	})
	ar.Tools = reg
	ar.WorkingMem = memory.NewWorkingMemory(64)
	ar.Tracker = NewConversationTracker()

	conv := NewConversation(10)

	// Step 1: User fetches the URL
	fetchNLU := &NLUResult{
		Action:   "fetch_url",
		Entities: map[string]string{"url": "https://stoicera.com"},
		Raw:      "scrape https://stoicera.com",
	}
	result := ar.Execute(fetchNLU, conv)

	if result.DirectResponse == "" {
		t.Fatalf("fetch should return DirectResponse, got Data=%q", result.Data)
	}
	if false /* NeedsLLM removed */ {
		t.Error("fetch should NOT need LLM — extractive summary should work")
	}
	if !strings.Contains(result.DirectResponse, "facts") {
		t.Logf("fetch response: %s", result.DirectResponse)
	}
	t.Logf("Fetch response (first 200 chars): %.200s", result.DirectResponse)

	// Step 2: User asks "what is Stoicera?"
	questionNLU := &NLUResult{
		Action:   "llm_chat",
		Entities: map[string]string{"topic": "stoicera"},
		Raw:      "what is Stoicera",
	}
	result = ar.Execute(questionNLU, conv)

	if false /* NeedsLLM removed */ {
		t.Error("follow-up should be answered extractively, not need LLM")
	}
	if result.DirectResponse == "" {
		t.Fatal("should have an extractive answer about Stoicera")
	}
	if !strings.Contains(strings.ToLower(result.DirectResponse), "philosophy") {
		t.Errorf("answer should mention philosophy, got %q", result.DirectResponse)
	}
	t.Logf("Q: 'what is Stoicera' → A: %s", result.DirectResponse)

	// Step 3: User asks "who founded it?"
	founderNLU := &NLUResult{
		Action:   "llm_chat",
		Entities: map[string]string{},
		Raw:      "who founded it",
	}
	result = ar.Execute(founderNLU, conv)

	if false /* NeedsLLM removed */ {
		t.Error("founder question should be answered extractively")
	}
	if result.DirectResponse == "" {
		t.Fatal("should answer the founder question")
	}
	if !strings.Contains(result.DirectResponse, "Raphael") {
		t.Errorf("answer should mention Raphael, got %q", result.DirectResponse)
	}
	t.Logf("Q: 'who founded it' → A: %s", result.DirectResponse)

	// Step 4: User asks "what products do they offer?"
	productsNLU := &NLUResult{
		Action:   "llm_chat",
		Entities: map[string]string{},
		Raw:      "what products do they offer",
	}
	result = ar.Execute(productsNLU, conv)

	if result.DirectResponse == "" {
		t.Fatal("should answer the products question")
	}
	t.Logf("Q: 'what products do they offer' → A: %s", result.DirectResponse)

	// Step 5: "tell me more"
	moreNLU := &NLUResult{
		Action: "llm_chat",
		Raw:    "tell me more",
	}
	result = ar.Execute(moreNLU, conv)

	if result.DirectResponse == "" {
		t.Fatal("should have more to tell")
	}
	if false /* NeedsLLM removed */ {
		t.Error("continuation should be extractive")
	}
	t.Logf("Q: 'tell me more' → A: %s", result.DirectResponse)
}

// TestExtractiveFlowWithKnowledgeLookup tests that "what is X" queries
// use extractive QA when facts are available.
func TestExtractiveFlowWithKnowledgeLookup(t *testing.T) {
	ar := NewActionRouter()
	ar.Tracker = NewConversationTracker()

	// Pre-ingest some knowledge
	ar.Tracker.IngestContent(
		`Go is a programming language created by Google in 2009.
Go compiles to native machine code and runs without a VM.
Go has built-in concurrency primitives called goroutines and channels.
The Go standard library is comprehensive and includes an HTTP server.
Go is known for its simplicity, fast compilation, and excellent tooling.`,
		"golang.org", "Go",
	)

	conv := NewConversation(10)

	// This would normally go to lookup_knowledge → LLM
	nlu := &NLUResult{
		Action:   "lookup_knowledge",
		Entities: map[string]string{"topic": "Go programming"},
		Raw:      "tell me about Go programming",
	}
	result := ar.Execute(nlu, conv)

	if false /* NeedsLLM removed */ {
		t.Error("should answer from extracted facts, not LLM")
	}
	if result.DirectResponse == "" {
		t.Fatal("should have extractive answer about Go")
	}
	if !strings.Contains(result.DirectResponse, "Google") {
		t.Errorf("answer should mention Google, got %q", result.DirectResponse)
	}
	t.Logf("Extractive knowledge answer: %s", result.DirectResponse)
}

// TestExtractiveNoFacts verifies graceful response when no facts available.
func TestExtractiveNoFacts(t *testing.T) {
	ar := NewActionRouter()
	ar.Tracker = NewConversationTracker()

	conv := NewConversation(10)

	nlu := &NLUResult{
		Action: "llm_chat",
		Raw:    "explain quantum entanglement",
	}
	result := ar.Execute(nlu, conv)

	// Engine handles all responses — should produce a direct response, not need LLM.
	if false /* NeedsLLM removed */ {
		t.Error("should NOT need LLM — engine handles all responses")
	}
	if result.DirectResponse == "" {
		t.Error("should produce a direct response from composer or fallback")
	}
}
