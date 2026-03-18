package cognitive

import (
	"fmt"
	"strings"

	"github.com/artaeon/nous/internal/tools"
)

// InlineResearcher performs quick research within the request pipeline.
// Unlike hands (which run async), this runs synchronously and returns
// results for the ResponseFormatter to present.
type InlineResearcher struct {
	Tools *tools.Registry
}

// Research gathers information from multiple sources about a topic.
// Returns combined data from web search + wikipedia + any fetched URLs.
func (ir *InlineResearcher) Research(topic string) *ActionResult {
	if ir.Tools == nil {
		return &ActionResult{
			Data:     "research tools unavailable",
			Source:   "research",
			NeedsLLM: true,
		}
	}

	var sections []string

	// 1. Web search the topic (get top 5 results)
	if searchTool, err := ir.Tools.Get("websearch"); err == nil {
		result, err := searchTool.Execute(map[string]string{
			"query":       topic,
			"max_results": "5",
		})
		if err == nil && result != "" {
			sections = append(sections, fmt.Sprintf("[Web Search Results]\n%s", result))
		}
	}

	// 2. Wikipedia lookup
	if wikiTool, err := ir.Tools.Get("wikipedia"); err == nil {
		result, err := wikiTool.Execute(map[string]string{
			"topic": topic,
		})
		if err == nil && result != "" {
			sections = append(sections, fmt.Sprintf("[Wikipedia]\n%s", result))
		}
	}

	if len(sections) == 0 {
		return &ActionResult{
			Data:     fmt.Sprintf("no research data found for %q", topic),
			Source:   "research",
			NeedsLLM: true,
		}
	}

	return &ActionResult{
		Data:   strings.Join(sections, "\n\n"),
		Source: "research",
		Structured: map[string]string{
			"topic":        topic,
			"source_count": fmt.Sprintf("%d", len(sections)),
		},
		NeedsLLM: true,
	}
}
