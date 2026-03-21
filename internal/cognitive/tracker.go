package cognitive

import (
	"strings"
	"time"
)

// ConversationTracker maintains conversational state for follow-up questions
// and deep conversation without LLM. Pure code, microsecond latency.
type ConversationTracker struct {
	// Topic stack — most recent first
	topics []TrackedTopic

	// Entity memory — things mentioned in conversation
	entities map[string]string // "it" → "https://stoicera.com", "that" → "Stoicera"

	// Last content — for "tell me more" / "go on" / "continue"
	lastSource   string
	lastFactIdx  int // how many facts from this source we've shown
	lastResponse string

	// Conversation turns
	turnCount    int
	lastActivity time.Time

	// Fact store — shared across conversation (in-memory)
	Facts *FactStore

	// Persistent fact store — survives across sessions
	PersistentFacts *PersistentFactStore

	// Extractive QA engine
	QA *ExtractiveQA
}

// TrackedTopic represents a topic being discussed.
type TrackedTopic struct {
	Name      string
	Source    string // URL, tool, or "user"
	MentionedAt time.Time
	TurnNumber  int
	FactCount   int // how many facts we have about this
}

// NewConversationTracker creates a new tracker with optional persistent storage.
func NewConversationTracker() *ConversationTracker {
	return &ConversationTracker{
		entities:     make(map[string]string),
		Facts:        NewFactStore(),
		QA:           NewExtractiveQA(),
		lastActivity: time.Now(),
	}
}

// NewConversationTrackerPersistent creates a tracker with persistent fact storage.
func NewConversationTrackerPersistent(factsPath string) *ConversationTracker {
	return &ConversationTracker{
		entities:        make(map[string]string),
		Facts:           NewFactStore(),
		PersistentFacts: NewPersistentFactStore(factsPath),
		QA:              NewExtractiveQA(),
		lastActivity:    time.Now(),
	}
}

// TrackTopic adds or promotes a topic to the top of the stack.
func (ct *ConversationTracker) TrackTopic(name, source string) {
	ct.turnCount++
	ct.lastActivity = time.Now()

	// Remove existing entry for this topic
	for i, t := range ct.topics {
		if strings.EqualFold(t.Name, name) {
			ct.topics = append(ct.topics[:i], ct.topics[i+1:]...)
			break
		}
	}

	// Push to top
	ct.topics = append([]TrackedTopic{{
		Name:        name,
		Source:      source,
		MentionedAt: time.Now(),
		TurnNumber:  ct.turnCount,
		FactCount:   len(ct.Facts.FactsAbout(name)),
	}}, ct.topics...)

	// Keep stack manageable
	if len(ct.topics) > 10 {
		ct.topics = ct.topics[:10]
	}

	// Update entity references
	ct.entities["it"] = name
	ct.entities["that"] = name
	ct.entities["this"] = name
	if source != "" {
		ct.entities["there"] = source
	}
}

// TrackEntity sets a pronoun/reference resolution.
func (ct *ConversationTracker) TrackEntity(ref, value string) {
	ct.entities[strings.ToLower(ref)] = value
}

// CurrentTopic returns the most recent topic, or empty string.
func (ct *ConversationTracker) CurrentTopic() string {
	if len(ct.topics) == 0 {
		return ""
	}
	return ct.topics[0].Name
}

// CurrentSource returns the source of the current topic.
func (ct *ConversationTracker) CurrentSource() string {
	if len(ct.topics) == 0 {
		return ""
	}
	return ct.topics[0].Source
}

// ResolveReference resolves pronouns and references to actual entities.
func (ct *ConversationTracker) ResolveReference(input string) string {
	lower := strings.ToLower(input)

	// Replace pronouns with resolved entities
	for ref, value := range ct.entities {
		// Only replace standalone words (not substrings)
		patterns := []string{
			" " + ref + " ",
			" " + ref + "?",
			" " + ref + ".",
			" " + ref + "!",
			" " + ref + ",",
		}
		for _, pattern := range patterns {
			if strings.Contains(lower, pattern) {
				input = strings.Replace(strings.ToLower(input), ref, value, 1)
				break
			}
		}
	}
	return input
}

// IsFollowUp detects whether the input is a follow-up to the current conversation.
func (ct *ConversationTracker) IsFollowUp(input string) bool {
	if len(ct.topics) == 0 {
		return false
	}

	lower := strings.ToLower(input)

	// Explicit follow-up markers
	followUpMarkers := []string{
		"tell me more", "more about", "go on", "continue",
		"what else", "anything else", "elaborate", "expand on",
		"can you explain", "why is that", "how does that",
		"what about", "and what", "also", "furthermore",
		"in more detail", "deeper", "dig deeper",
	}
	for _, marker := range followUpMarkers {
		if strings.Contains(lower, marker) {
			return true
		}
	}

	// Pronoun references to current topic
	pronouns := []string{"it", "that", "this", "they", "them"}
	for _, p := range pronouns {
		if containsWord(lower, p) {
			return true
		}
	}

	// Short question about current topic
	if len(strings.Fields(input)) <= 5 && ct.CurrentTopic() != "" {
		topicLower := strings.ToLower(ct.CurrentTopic())
		if strings.Contains(lower, topicLower) {
			return true
		}
	}

	return false
}

// IsContinuation detects "tell me more" / "go on" type requests.
func (ct *ConversationTracker) IsContinuation(input string) bool {
	lower := strings.ToLower(input)
	continuations := []string{
		"tell me more", "more", "go on", "continue",
		"keep going", "what else", "anything else",
		"next", "more details", "elaborate",
	}
	for _, c := range continuations {
		if strings.Contains(lower, c) {
			return true
		}
	}
	return false
}

// IngestContent extracts facts from content and stores them.
// Stores in both session (fast) and persistent (survives restarts) stores.
func (ct *ConversationTracker) IngestContent(content, source, topic string) int {
	facts := ExtractFacts(content, source, topic)
	for _, f := range facts {
		ct.Facts.Add(f)
	}

	// Also persist for cross-session knowledge
	if ct.PersistentFacts != nil {
		ct.PersistentFacts.AddMany(facts)
		ct.PersistentFacts.Save()
	}

	if topic != "" {
		ct.TrackTopic(topic, source)
	}

	return len(facts)
}

// AnswerQuestion tries to answer from stored facts. Returns empty if insufficient.
func (ct *ConversationTracker) AnswerQuestion(question string) string {
	// Resolve pronouns first
	resolved := ct.ResolveReference(question)

	// Get all relevant facts from session store
	var candidates []Fact

	// 1. Facts about the current topic
	if topic := ct.CurrentTopic(); topic != "" {
		candidates = append(candidates, ct.Facts.FactsAbout(topic)...)
	}

	// 2. Facts from the current source
	if source := ct.CurrentSource(); source != "" {
		candidates = append(candidates, ct.Facts.FactsFromSource(source)...)
	}

	// 3. Search all session facts if we have few candidates
	if len(candidates) < 3 {
		candidates = ct.Facts.AllFacts()
	}

	// 4. Also search persistent facts (cross-session knowledge)
	if ct.PersistentFacts != nil {
		// Search by topic
		queryWords := strings.Fields(strings.ToLower(resolved))
		for _, w := range queryWords {
			if len(w) > 3 && !isExtractiveStop(w) {
				candidates = append(candidates, ct.PersistentFacts.FactsAbout(w)...)
			}
		}
		// If still few candidates, add all persistent facts
		if len(candidates) < 5 {
			candidates = append(candidates, ct.PersistentFacts.AllFacts()...)
		}
	}

	// Deduplicate
	candidates = deduplicateFacts(candidates)

	if len(candidates) == 0 {
		return ""
	}

	// Run extractive QA
	scored := ct.QA.Answer(resolved, candidates, 5)
	if len(scored) == 0 || scored[0].Relevance < 0.1 {
		return ""
	}

	// Compose response
	response := ComposeResponse(resolved, scored, ct.CurrentSource())
	if response != "" {
		ct.lastResponse = response
		ct.lastSource = ct.CurrentSource()
	}

	return response
}

// ContinueResponse returns more facts about the current topic.
func (ct *ConversationTracker) ContinueResponse() string {
	topic := ct.CurrentTopic()
	if topic == "" {
		return ""
	}

	facts := ct.Facts.FactsAbout(topic)
	if len(facts) == 0 {
		return ""
	}

	// Skip facts we've already shown (tracked by lastFactIdx)
	ct.lastFactIdx += 5
	start := ct.lastFactIdx
	if start >= len(facts) {
		return "That's everything I know about " + topic + "."
	}

	end := start + 5
	if end > len(facts) {
		end = len(facts)
	}

	var parts []string
	for _, f := range facts[start:end] {
		parts = append(parts, "- "+f.Text)
	}

	remaining := len(facts) - end
	response := strings.Join(parts, "\n")
	if remaining > 0 {
		response += "\n\n" + intToWord(remaining) + " more facts available."
	}

	return response
}

// TopicSummary returns a summary of what we know about the current topic.
func (ct *ConversationTracker) TopicSummary() string {
	topic := ct.CurrentTopic()
	if topic == "" {
		return ""
	}

	facts := ct.Facts.FactsAbout(topic)
	if len(facts) == 0 {
		return ""
	}

	ct.lastFactIdx = 0 // reset continuation pointer
	return ComposeTopicSummary(topic, facts)
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

// containsWord checks if a word appears as a standalone word in text.
func containsWord(text, word string) bool {
	fields := strings.Fields(text)
	for _, f := range fields {
		cleaned := strings.Trim(f, "?.,!;:")
		if cleaned == word {
			return true
		}
	}
	return false
}

// deduplicateFacts removes duplicate facts by text.
func deduplicateFacts(facts []Fact) []Fact {
	seen := make(map[string]bool)
	var unique []Fact
	for _, f := range facts {
		if !seen[f.Text] {
			seen[f.Text] = true
			unique = append(unique, f)
		}
	}
	return unique
}
