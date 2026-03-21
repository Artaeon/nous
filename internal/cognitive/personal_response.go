package cognitive

import (
	"fmt"
	"strings"
	"time"
)

// PersonalResponseGenerator creates personalized responses without LLM.
// Uses PersonalGrowth profile + templates + fact store to generate
// contextual, human-feeling output. Pure code, microsecond latency.
type PersonalResponseGenerator struct {
	Growth  *PersonalGrowth
	Tracker *ConversationTracker
}

// DailyBriefing generates a personalized morning briefing from all available data.
// Combines: weather, habits, todos, expenses, journal, calendar — all deterministic.
func (prg *PersonalResponseGenerator) DailyBriefing(toolResults map[string]string) string {
	var b strings.Builder
	now := time.Now()

	// Greeting based on time of day
	greeting := "Good morning"
	hour := now.Hour()
	switch {
	case hour < 5:
		greeting = "You're up late"
	case hour < 12:
		greeting = "Good morning"
	case hour < 17:
		greeting = "Good afternoon"
	case hour < 21:
		greeting = "Good evening"
	default:
		greeting = "Good night"
	}

	// Personalize with user name if known
	userName := ""
	if prg.Growth != nil {
		for _, f := range prg.Growth.Profile().Facts {
			if f.Category == "identity" || f.Category == "personal" {
				lower := strings.ToLower(f.Fact)
				if strings.Contains(lower, "name is") || strings.Contains(lower, "i'm ") ||
					strings.Contains(lower, "call me") {
					// Extract name from "my name is X" or "I'm X" or "call me X"
					for _, prefix := range []string{"name is ", "i'm ", "call me "} {
						if idx := strings.Index(lower, prefix); idx >= 0 {
							userName = strings.TrimSpace(f.Fact[idx+len(prefix):])
							userName = strings.TrimRight(userName, ".!,")
							break
						}
					}
				}
			}
		}
	}

	if userName != "" {
		b.WriteString(fmt.Sprintf("%s, %s! ", greeting, userName))
	} else {
		b.WriteString(greeting + "! ")
	}
	b.WriteString(fmt.Sprintf("It's %s.\n\n", now.Format("Monday, January 2")))

	// Weather
	if weather, ok := toolResults["weather"]; ok && weather != "" {
		b.WriteString("Weather: ")
		b.WriteString(weather)
		b.WriteString("\n\n")
	}

	// Habits
	if habits, ok := toolResults["habits"]; ok && habits != "" {
		b.WriteString("Habits:\n")
		b.WriteString(habits)
		b.WriteString("\n\n")
	}

	// Todos
	if todos, ok := toolResults["todos"]; ok && todos != "" {
		b.WriteString("Tasks:\n")
		b.WriteString(todos)
		b.WriteString("\n\n")
	}

	// Expenses summary
	if expenses, ok := toolResults["expenses"]; ok && expenses != "" {
		b.WriteString("Spending: ")
		b.WriteString(expenses)
		b.WriteString("\n\n")
	}

	// Calendar
	if calendar, ok := toolResults["calendar"]; ok && calendar != "" {
		b.WriteString("Schedule:\n")
		b.WriteString(calendar)
		b.WriteString("\n\n")
	}

	// Top interests / what the user has been working on
	if prg.Growth != nil {
		interests := prg.Growth.TopInterests(3)
		if len(interests) > 0 {
			names := make([]string, len(interests))
			for i, t := range interests {
				names[i] = t.Name
			}
			b.WriteString("Recent interests: ")
			b.WriteString(strings.Join(names, ", "))
			b.WriteString("\n")
		}
	}

	return strings.TrimRight(b.String(), "\n")
}

// PersonalizeResponse adapts a response based on user style preferences.
func (prg *PersonalResponseGenerator) PersonalizeResponse(response string) string {
	if prg.Growth == nil || response == "" {
		return response
	}

	style := prg.Growth.Profile().Style

	// Concise preference: trim to first 2 sentences if response is long
	if style.PrefersConcise && len(response) > 300 {
		sentences := splitSentences(response)
		if len(sentences) > 3 {
			response = strings.Join(sentences[:3], ". ") + "."
		}
	}

	return response
}

// EnrichWithContext adds relevant personal context to an extractive response.
// If the user has personal facts related to the topic, weave them in.
func (prg *PersonalResponseGenerator) EnrichWithContext(response, topic string) string {
	if prg.Growth == nil || topic == "" {
		return response
	}

	topicLower := strings.ToLower(topic)
	var relevant []string

	for _, f := range prg.Growth.Profile().Facts {
		factLower := strings.ToLower(f.Fact)
		if strings.Contains(factLower, topicLower) {
			relevant = append(relevant, f.Fact)
		}
	}

	if len(relevant) == 0 {
		return response
	}

	// Add a personal touch
	var b strings.Builder
	b.WriteString(response)
	b.WriteString("\n\nYou've mentioned: ")
	b.WriteString(strings.Join(relevant, "; "))
	b.WriteString(".")

	return b.String()
}

// SmartLLMPrompt creates an optimized LLM prompt that uses extractive facts
// as pre-filled context, so the LLM only needs to rephrase/synthesize,
// not think from scratch. This makes LLM calls faster and better.
func (prg *PersonalResponseGenerator) SmartLLMPrompt(query string, rawData string) string {
	var b strings.Builder

	// Pre-fill with extractive facts if available
	if prg.Tracker != nil {
		answer := prg.Tracker.AnswerQuestion(query)
		if answer != "" {
			b.WriteString("[Known facts]\n")
			b.WriteString(answer)
			b.WriteString("\n\n")
		}
	}

	// Add personal context
	if prg.Growth != nil {
		ctx := prg.Growth.ContextForQuery(query)
		if ctx != "" {
			b.WriteString(ctx)
			b.WriteString("\n\n")
		}
	}

	// Add any raw data
	if rawData != "" {
		b.WriteString("[Data]\n")
		b.WriteString(rawData)
		b.WriteString("\n\n")
	}

	b.WriteString("[Question]\n")
	b.WriteString(query)
	b.WriteString("\n\n[Instruction]\nUsing the facts and data above, answer the question naturally and concisely. Do not add information beyond what is provided.")

	return b.String()
}
