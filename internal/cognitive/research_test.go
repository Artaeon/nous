package cognitive

import (
	"strings"
	"testing"

	"github.com/artaeon/nous/internal/tools"
)

func TestInlineResearcher_NoTools(t *testing.T) {
	ir := &InlineResearcher{Tools: nil}
	result := ir.Research("quantum physics")

	if !result.NeedsLLM {
		t.Error("expected NeedsLLM=true when tools are nil")
	}
	if result.Source != "research" {
		t.Errorf("source = %q, want research", result.Source)
	}
	if !strings.Contains(result.Data, "unavailable") {
		t.Errorf("expected 'unavailable' in Data, got %q", result.Data)
	}
}

func TestInlineResearcher_CombinesResults(t *testing.T) {
	reg := tools.NewRegistry()

	// Mock websearch tool
	reg.Register(tools.Tool{
		Name:        "websearch",
		Description: "mock websearch",
		Execute: func(args map[string]string) (string, error) {
			query := args["query"]
			return "1. Result about " + query + "\n   https://example.com\n   A snippet about " + query, nil
		},
	})

	// Mock wikipedia tool
	reg.Register(tools.Tool{
		Name:        "wikipedia",
		Description: "mock wikipedia",
		Execute: func(args map[string]string) (string, error) {
			topic := args["topic"]
			return "# " + topic + "\n\nWikipedia summary about " + topic, nil
		},
	})

	ir := &InlineResearcher{Tools: reg}
	result := ir.Research("quantum physics")

	if !result.NeedsLLM {
		t.Error("expected NeedsLLM=true")
	}
	if result.Source != "research" {
		t.Errorf("source = %q, want research", result.Source)
	}

	// Should contain both web search and wikipedia sections
	if !strings.Contains(result.Data, "[Web Search Results]") {
		t.Error("expected [Web Search Results] section in Data")
	}
	if !strings.Contains(result.Data, "[Wikipedia]") {
		t.Error("expected [Wikipedia] section in Data")
	}
	if !strings.Contains(result.Data, "quantum physics") {
		t.Error("expected topic name in Data")
	}

	// Should have structured metadata
	if result.Structured == nil {
		t.Fatal("expected Structured data")
	}
	if result.Structured["topic"] != "quantum physics" {
		t.Errorf("Structured[topic] = %q, want 'quantum physics'", result.Structured["topic"])
	}
	if result.Structured["source_count"] != "2" {
		t.Errorf("Structured[source_count] = %q, want '2'", result.Structured["source_count"])
	}
}

func TestActionRouter_Research(t *testing.T) {
	ar := NewActionRouter()
	reg := tools.NewRegistry()
	reg.Register(tools.Tool{
		Name:        "websearch",
		Description: "mock",
		Execute: func(args map[string]string) (string, error) {
			return "search results for " + args["query"], nil
		},
	})
	ar.Tools = reg

	nlu := &NLUResult{
		Action:   "research",
		Entities: map[string]string{"topic": "machine learning"},
		Raw:      "research machine learning",
	}
	result := ar.Execute(nlu, NewConversation(10))

	if result.Source != "research" {
		t.Errorf("source = %q, want research", result.Source)
	}
	if !result.NeedsLLM {
		t.Error("research should need LLM for formatting")
	}
}

// TestActionRouter_GenerateDoc is in action_test.go (more thorough version).
