package cognitive

import (
	"fmt"
	"strings"
	"time"
)

// -----------------------------------------------------------------------
// Conversation State — explicit tracking of dialogue state at any point.
//
// This goes beyond DialogueManager (which tracks state-machine transitions)
// to capture the FULL conversational context:
//   - Active topic and user objective
//   - Unresolved slots that need filling
//   - Entity mentions and coreference chains
//   - Satisfaction signals (clarifications, corrections)
//   - Topic stack for nested conversations
//
// Used by FollowUpResolver and PreferenceModel to make context-aware
// decisions without any LLM.
// -----------------------------------------------------------------------

// ConversationState captures the full dialogue state at any point.
type ConversationState struct {
	// Active tracking
	ActiveTopic     string            // current topic being discussed
	UserObjective   string            // what the user is trying to accomplish
	UnresolvedSlots map[string]string // slots that need filling (slot_name -> description)
	Assumptions     []string          // assumptions the system is making

	// History
	TopicStack []string  // stack of topics (most recent first)
	TurnCount  int       // number of turns in this conversation
	LastIntent string    // last classified intent
	LastTask   ThinkTask // last think task type

	// Entity tracking
	MentionedEntities map[string]string // entity -> last value (e.g., "city" -> "Paris")
	Coreferences      map[string]string // pronoun -> resolved entity ("it" -> "Python")

	// Satisfaction tracking
	ClarificationCount int // how many times user asked for clarification
	CorrectionCount    int // how many times user corrected the system
	FollowUpCount      int // how many follow-ups in current thread

	// Timestamps
	StartedAt  time.Time
	LastTurnAt time.Time

	// Internal: recent turns for reference resolution
	recentInputs    []string
	recentResponses []string
}

// NewConversationState creates a fresh conversation state.
func NewConversationState() *ConversationState {
	return &ConversationState{
		UnresolvedSlots:   make(map[string]string),
		MentionedEntities: make(map[string]string),
		Coreferences:      make(map[string]string),
		TopicStack:        make([]string, 0, 16),
		Assumptions:       make([]string, 0, 8),
		recentInputs:      make([]string, 0, 20),
		recentResponses:   make([]string, 0, 20),
		StartedAt:         time.Now(),
		LastTurnAt:        time.Now(),
	}
}

// Update processes a new turn and updates all state.
func (cs *ConversationState) Update(input string, nlu *NLUResult, response string) {
	cs.TurnCount++
	cs.LastTurnAt = time.Now()

	lower := strings.ToLower(input)

	// Track recent inputs/responses (keep last 20)
	cs.recentInputs = append(cs.recentInputs, input)
	if len(cs.recentInputs) > 20 {
		cs.recentInputs = cs.recentInputs[1:]
	}
	cs.recentResponses = append(cs.recentResponses, response)
	if len(cs.recentResponses) > 20 {
		cs.recentResponses = cs.recentResponses[1:]
	}

	// Update intent tracking from NLU
	if nlu != nil {
		cs.LastIntent = nlu.Intent
		if nlu.Action != "" && cs.LastIntent == "" {
			cs.LastIntent = nlu.Action
		}

		// Extract entities from NLU result
		for k, v := range nlu.Entities {
			cs.TrackEntity(k, v)
		}

		// Detect topic from NLU entities
		if topic, ok := nlu.Entities["topic"]; ok && topic != "" {
			cs.updateTopic(topic)
		} else if subject, ok := nlu.Entities["subject"]; ok && subject != "" {
			cs.updateTopic(subject)
		} else {
			// Try to extract topic from input
			extracted := cs.extractTopicFromInput(lower)
			if extracted != "" {
				cs.updateTopic(extracted)
			}
		}

		// Infer user objective from intent patterns
		cs.inferObjective(nlu, lower)
	}

	// Detect satisfaction signals
	cs.detectClarification(lower)
	cs.detectCorrection(lower)
	cs.detectFollowUp(lower)

	// Update coreferences based on current state
	cs.updateCoreferences(lower)

	// Try to fill unresolved slots from the input
	if nlu != nil {
		cs.FillUnresolvedSlots(input, nlu.Entities)
	}
}

// updateTopic sets a new active topic and manages the stack.
func (cs *ConversationState) updateTopic(topic string) {
	topic = strings.TrimSpace(topic)
	if topic == "" {
		return
	}
	// Don't push duplicates at the top of the stack
	if cs.ActiveTopic != topic {
		cs.PushTopic(topic)
	}
	cs.ActiveTopic = topic
}

// PushTopic adds a new topic to the stack.
func (cs *ConversationState) PushTopic(topic string) {
	topic = strings.TrimSpace(topic)
	if topic == "" {
		return
	}
	// Push to front (most recent first)
	cs.TopicStack = append([]string{topic}, cs.TopicStack...)
	// Keep stack bounded
	if len(cs.TopicStack) > 20 {
		cs.TopicStack = cs.TopicStack[:20]
	}
}

// ActiveContext returns a summary of current conversation context for the generator.
func (cs *ConversationState) ActiveContext() string {
	var parts []string

	if cs.ActiveTopic != "" {
		parts = append(parts, fmt.Sprintf("topic: %s", cs.ActiveTopic))
	}
	if cs.UserObjective != "" {
		parts = append(parts, fmt.Sprintf("objective: %s", cs.UserObjective))
	}
	if cs.TurnCount > 0 {
		parts = append(parts, fmt.Sprintf("turns: %d", cs.TurnCount))
	}

	// Include mentioned entities
	if len(cs.MentionedEntities) > 0 {
		entParts := make([]string, 0, len(cs.MentionedEntities))
		for k, v := range cs.MentionedEntities {
			entParts = append(entParts, fmt.Sprintf("%s=%s", k, v))
		}
		parts = append(parts, fmt.Sprintf("entities: [%s]", strings.Join(entParts, ", ")))
	}

	// Include unresolved slots
	if len(cs.UnresolvedSlots) > 0 {
		slotParts := make([]string, 0, len(cs.UnresolvedSlots))
		for k, v := range cs.UnresolvedSlots {
			slotParts = append(slotParts, fmt.Sprintf("%s(%s)", k, v))
		}
		parts = append(parts, fmt.Sprintf("needs: [%s]", strings.Join(slotParts, ", ")))
	}

	// Include topic history if we've covered multiple topics
	if len(cs.TopicStack) > 1 {
		maxHistory := 5
		if len(cs.TopicStack) < maxHistory {
			maxHistory = len(cs.TopicStack)
		}
		parts = append(parts, fmt.Sprintf("topic_history: [%s]", strings.Join(cs.TopicStack[:maxHistory], " <- ")))
	}

	// Satisfaction signals
	if cs.ClarificationCount > 0 {
		parts = append(parts, fmt.Sprintf("clarifications: %d", cs.ClarificationCount))
	}
	if cs.CorrectionCount > 0 {
		parts = append(parts, fmt.Sprintf("corrections: %d", cs.CorrectionCount))
	}

	if len(parts) == 0 {
		return "no active context"
	}
	return strings.Join(parts, "; ")
}

// ResolveReference resolves a pronoun or reference using conversation state.
func (cs *ConversationState) ResolveReference(ref string) string {
	ref = strings.ToLower(strings.TrimSpace(ref))

	// Check explicit coreference map first
	if resolved, ok := cs.Coreferences[ref]; ok {
		return resolved
	}

	// Common pronouns resolve to active topic or most recent entity
	switch ref {
	case "it", "this", "that":
		if cs.ActiveTopic != "" {
			return cs.ActiveTopic
		}
	case "they", "them":
		// Try to find a plural entity or the active topic
		if cs.ActiveTopic != "" {
			return cs.ActiveTopic
		}
	case "he", "him":
		if v, ok := cs.MentionedEntities["person"]; ok {
			return v
		}
		if v, ok := cs.MentionedEntities["name"]; ok {
			return v
		}
	case "she", "her":
		if v, ok := cs.MentionedEntities["person"]; ok {
			return v
		}
		if v, ok := cs.MentionedEntities["name"]; ok {
			return v
		}
	case "there":
		if v, ok := cs.MentionedEntities["city"]; ok {
			return v
		}
		if v, ok := cs.MentionedEntities["location"]; ok {
			return v
		}
		if v, ok := cs.MentionedEntities["place"]; ok {
			return v
		}
	case "the former":
		if len(cs.TopicStack) >= 2 {
			return cs.TopicStack[1]
		}
	case "the latter":
		if len(cs.TopicStack) >= 1 {
			return cs.TopicStack[0]
		}
	}

	// "one" as a pronoun — resolve to active topic
	if ref == "one" && cs.ActiveTopic != "" {
		return cs.ActiveTopic
	}

	return ref
}

// TrackEntity records an entity mention for future reference.
func (cs *ConversationState) TrackEntity(entityType, value string) {
	entityType = strings.TrimSpace(entityType)
	value = strings.TrimSpace(value)
	if entityType == "" || value == "" {
		return
	}
	cs.MentionedEntities[entityType] = value
}

// RecordAssumption notes an assumption being made.
func (cs *ConversationState) RecordAssumption(assumption string) {
	assumption = strings.TrimSpace(assumption)
	if assumption == "" {
		return
	}
	cs.Assumptions = append(cs.Assumptions, assumption)
	// Keep bounded
	if len(cs.Assumptions) > 20 {
		cs.Assumptions = cs.Assumptions[1:]
	}
}

// NeedsSlot checks if a required slot is still unfilled.
func (cs *ConversationState) NeedsSlot(slotName string) bool {
	_, exists := cs.UnresolvedSlots[slotName]
	return exists
}

// SetSlot fills a slot value and removes it from unresolved.
func (cs *ConversationState) SetSlot(slotName, value string) {
	delete(cs.UnresolvedSlots, slotName)
	// Also track it as an entity
	cs.TrackEntity(slotName, value)
}

// FillUnresolvedSlots attempts to fill unresolved slots from a new input.
func (cs *ConversationState) FillUnresolvedSlots(input string, entities map[string]string) {
	lower := strings.ToLower(input)

	for slot := range cs.UnresolvedSlots {
		// Check if the entity map has a matching value
		if val, ok := entities[slot]; ok && val != "" {
			cs.SetSlot(slot, val)
			continue
		}

		// Try simple keyword extraction for common slot types
		switch slot {
		case "city", "location", "place":
			// Check if any capitalized word in input could be a place
			words := strings.Fields(input)
			for _, w := range words {
				clean := strings.Trim(w, ".,!?;:'\"()")
				if len(clean) > 1 && clean[0] >= 'A' && clean[0] <= 'Z' {
					// Capitalize proper noun — could be a place
					if !isCommonWord(strings.ToLower(clean)) {
						cs.SetSlot(slot, clean)
						break
					}
				}
			}
		case "language", "programming_language":
			for _, lang := range []string{"python", "go", "java", "javascript", "rust", "c++", "ruby", "typescript"} {
				if strings.Contains(lower, lang) {
					cs.SetSlot(slot, lang)
					break
				}
			}
		case "number", "count", "amount":
			// Look for a number
			words := strings.Fields(input)
			for _, w := range words {
				clean := strings.Trim(w, ".,!?;:'\"()")
				if isNumeric(clean) {
					cs.SetSlot(slot, clean)
					break
				}
			}
		}
	}
}

// -----------------------------------------------------------------------
// Internal helper methods
// -----------------------------------------------------------------------

// extractTopicFromInput tries to extract a topic from the input text.
func (cs *ConversationState) extractTopicFromInput(lower string) string {
	// "about X" pattern
	if idx := strings.Index(lower, "about "); idx != -1 {
		rest := strings.TrimSpace(lower[idx+6:])
		rest = trimTrailingPunctuation(rest)
		if rest != "" && len(rest) < 80 {
			return rest
		}
	}

	// "regarding X" pattern
	if idx := strings.Index(lower, "regarding "); idx != -1 {
		rest := strings.TrimSpace(lower[idx+10:])
		rest = trimTrailingPunctuation(rest)
		if rest != "" && len(rest) < 80 {
			return rest
		}
	}

	// "of X" at end of phrase like "tell me of X"
	// (less reliable, skip for now)

	return ""
}

// inferObjective guesses the user's objective from the intent.
func (cs *ConversationState) inferObjective(nlu *NLUResult, lower string) {
	intent := nlu.Intent
	if intent == "" {
		intent = nlu.Action
	}

	switch {
	case strings.Contains(intent, "explain") || strings.Contains(intent, "teach"):
		cs.UserObjective = "understand " + cs.ActiveTopic
	case strings.Contains(intent, "compare"):
		if a, ok := nlu.Entities["item_a"]; ok {
			if b, ok := nlu.Entities["item_b"]; ok {
				cs.UserObjective = fmt.Sprintf("compare %s and %s", a, b)
			}
		}
		if cs.UserObjective == "" {
			cs.UserObjective = "compare options"
		}
	case strings.Contains(intent, "advise") || strings.Contains(intent, "recommend"):
		cs.UserObjective = "get advice"
	case strings.Contains(intent, "create") || strings.Contains(intent, "compose"):
		cs.UserObjective = "create content"
	case strings.Contains(lower, "how do i") || strings.Contains(lower, "how to"):
		cs.UserObjective = "learn how to " + cs.ActiveTopic
	case strings.Contains(lower, "what is") || strings.Contains(lower, "what are"):
		cs.UserObjective = "understand " + cs.ActiveTopic
	case strings.Contains(lower, "why"):
		cs.UserObjective = "understand why " + cs.ActiveTopic
	}
}

// detectClarification checks if the user is asking for clarification.
func (cs *ConversationState) detectClarification(lower string) {
	clarifySignals := []string{
		"what do you mean",
		"can you clarify",
		"i don't understand",
		"i don't get it",
		"what does that mean",
		"huh?",
		"unclear",
		"could you explain",
		"i'm confused",
		"what?",
		"say that again",
		"rephrase",
	}
	for _, sig := range clarifySignals {
		if strings.Contains(lower, sig) {
			cs.ClarificationCount++
			return
		}
	}
}

// detectCorrection checks if the user is correcting the system.
func (cs *ConversationState) detectCorrection(lower string) {
	correctionSignals := []string{
		"no, i meant",
		"that's not what i",
		"that's wrong",
		"actually,",
		"no, it's",
		"i said",
		"not what i asked",
		"incorrect",
		"you're wrong",
		"that's not right",
		"no no",
		"wrong",
	}
	for _, sig := range correctionSignals {
		if strings.Contains(lower, sig) {
			cs.CorrectionCount++
			return
		}
	}
}

// detectFollowUp checks if the user is asking a follow-up.
func (cs *ConversationState) detectFollowUp(lower string) {
	if cs.TurnCount <= 1 {
		return
	}
	followUpSignals := []string{
		"tell me more",
		"go deeper",
		"elaborate",
		"more detail",
		"what about",
		"how about",
		"and also",
		"additionally",
		"what else",
		"keep going",
		"continue",
		"go on",
		"expand on",
	}
	for _, sig := range followUpSignals {
		if strings.Contains(lower, sig) {
			cs.FollowUpCount++
			return
		}
	}
}

// updateCoreferences updates pronoun → entity mappings based on context.
func (cs *ConversationState) updateCoreferences(lower string) {
	// The active topic is the most likely referent for "it", "this", "that"
	if cs.ActiveTopic != "" {
		cs.Coreferences["it"] = cs.ActiveTopic
		cs.Coreferences["this"] = cs.ActiveTopic
		cs.Coreferences["that"] = cs.ActiveTopic
		cs.Coreferences["one"] = cs.ActiveTopic
	}

	// Named entities update specific pronouns
	if person, ok := cs.MentionedEntities["person"]; ok {
		cs.Coreferences["they"] = person
		cs.Coreferences["them"] = person
	}

	if place, ok := cs.MentionedEntities["city"]; ok {
		cs.Coreferences["there"] = place
	} else if place, ok := cs.MentionedEntities["location"]; ok {
		cs.Coreferences["there"] = place
	} else if place, ok := cs.MentionedEntities["place"]; ok {
		cs.Coreferences["there"] = place
	}

	// Topic stack provides "the former" / "the latter"
	if len(cs.TopicStack) >= 2 {
		cs.Coreferences["the former"] = cs.TopicStack[1]
		cs.Coreferences["the latter"] = cs.TopicStack[0]
	}
}

// trimTrailingPunctuation removes trailing punctuation from a string.
func trimTrailingPunctuation(s string) string {
	return strings.TrimRight(s, ".,!?;:'\"")
}

// isCommonWord checks if a word is too common to be a proper noun topic.
func isCommonWord(w string) bool {
	common := map[string]bool{
		"i": true, "me": true, "my": true, "we": true, "you": true,
		"he": true, "she": true, "it": true, "they": true, "them": true,
		"the": true, "a": true, "an": true, "is": true, "are": true,
		"was": true, "were": true, "be": true, "been": true, "being": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
		"may": true, "might": true, "can": true, "shall": true, "must": true,
		"that": true, "this": true, "these": true, "those": true, "what": true,
		"which": true, "who": true, "whom": true, "whose": true, "where": true,
		"when": true, "why": true, "how": true, "not": true, "no": true,
		"yes": true, "but": true, "and": true, "or": true, "if": true,
		"then": true, "so": true, "than": true, "too": true, "very": true,
		"just": true, "about": true, "also": true, "well": true, "here": true,
		"there": true, "now": true, "some": true, "any": true,
		"all": true, "each": true, "every": true, "both": true, "few": true,
		"more": true, "most": true, "other": true, "into": true, "with": true,
		"from": true, "for": true, "on": true, "in": true, "at": true,
		"to": true, "of": true, "by": true, "up": true, "out": true,
	}
	return common[w]
}

// isNumeric checks if a string is a number.
func isNumeric(s string) bool {
	if s == "" {
		return false
	}
	for i, c := range s {
		if c == '.' || c == '-' {
			if i == 0 {
				continue
			}
			if c == '.' {
				continue
			}
			return false
		}
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}
