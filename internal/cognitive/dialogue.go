package cognitive

import (
	"fmt"
	"strings"
	"sync"
)

// DialogueState represents the current state of the conversation state machine.
type DialogueState int

const (
	StateGreeting   DialogueState = iota
	StateExploration              // user is browsing/asking general questions
	StateDeepDive                 // user is drilling into a specific topic
	StateClarifying               // system needs more info from user
	StateActionable               // user wants something done (search, calculate, etc.)
	StateReflection               // philosophical/open-ended discussion
	StateFarewell                 // conversation ending
)

// String returns a human-readable name for the dialogue state.
func (s DialogueState) String() string {
	switch s {
	case StateGreeting:
		return "greeting"
	case StateExploration:
		return "exploration"
	case StateDeepDive:
		return "deep_dive"
	case StateClarifying:
		return "clarifying"
	case StateActionable:
		return "actionable"
	case StateReflection:
		return "reflection"
	case StateFarewell:
		return "farewell"
	default:
		return "unknown"
	}
}

// DialogueContext is the context returned by ProcessTurn for response generation.
type DialogueContext struct {
	State              DialogueState
	Topic              string
	SuggestedFollowUps []string
	TransitionPhrase   string // e.g., "Diving deeper into..." or "Switching gears..."
	NeedsClarification bool
	ClarificationQ     string
}

// DialogueManager tracks conversation state and provides context-aware behavior.
type DialogueManager struct {
	CurrentState  DialogueState
	PreviousState DialogueState
	TopicStack    []string          // stack of topics discussed
	TurnCount     int
	LastIntent    string            // last NLU action
	Entities      map[string]string // accumulated entities across turns
	Ambiguities   []string          // detected ambiguities to resolve
	FollowUps     []string          // suggested follow-up questions
	mu            sync.Mutex

	// internal tracking for state transitions
	consecutiveTopicTurns int    // how many turns on the same topic
	lastTopic             string // topic from previous turn
}

// greeting/farewell/reflection patterns
var (
	greetingPatterns = []string{
		"hi", "hello", "hey", "hey there", "howdy", "hiya", "yo",
		"good morning", "good afternoon", "good evening",
		"morning", "evening", "afternoon",
		"what's up", "whats up", "sup", "greetings", "salutations",
	}
	farewellPatterns = []string{
		"bye", "goodbye", "good bye", "see ya", "see you", "later",
		"farewell", "ciao", "adios", "peace", "take care",
		"good night", "gn", "ttyl", "talk later",
		"gotta go", "catch you later", "ok bye", "okay bye",
		"bye bye", "night", "nite", "thanks", "thank you",
	}
	reflectionPrefixes = []string{
		"what do you think about",
		"why is",
		"why are",
		"why do",
		"what does it mean",
		"what is the meaning of",
		"how important is",
		"do you believe",
		"is it true that",
		"what's the point of",
		"what if",
		"should we",
		"can you reflect on",
		"philosophically",
	}
	// Actions that indicate the user wants something done
	toolActions = map[string]bool{
		"web_search":      true,
		"fetch_url":       true,
		"compute":         true,
		"calculate":       true,
		"file_op":         true,
		"weather":         true,
		"convert":         true,
		"reminder":        true,
		"sysinfo":         true,
		"clipboard":       true,
		"notes":           true,
		"todos":           true,
		"find_files":      true,
		"summarize_url":   true,
		"news":            true,
		"run_code":        true,
		"calendar":        true,
		"check_email":     true,
		"screenshot":      true,
		"password":        true,
		"bookmark":        true,
		"journal":         true,
		"habit":           true,
		"expense":         true,
		"translate":       true,
		"timer":           true,
		"hash":            true,
		"dict":            true,
		"network":         true,
		"volume":          true,
		"brightness":      true,
		"archive":         true,
		"disk_usage":      true,
		"process":         true,
		"qrcode":          true,
		"daily_briefing":  true,
		"lookup_knowledge": true,
		"lookup_memory":   true,
		"research":        true,
	}
	// Reference pronouns to resolve
	referenceWords = []string{"it", "that", "this", "they", "them", "those"}
)

// NewDialogueManager creates a new DialogueManager in the greeting state.
func NewDialogueManager() *DialogueManager {
	return &DialogueManager{
		CurrentState: StateGreeting,
		Entities:     make(map[string]string),
	}
}

// ProcessTurn advances the state machine based on the NLU result and action result,
// returning a DialogueContext for response generation.
func (dm *DialogueManager) ProcessTurn(nlu *NLUResult, response *ActionResult) *DialogueContext {
	dm.mu.Lock()
	defer dm.mu.Unlock()

	dm.TurnCount++
	if nlu == nil {
		return &DialogueContext{State: dm.CurrentState, Topic: dm.currentTopic()}
	}

	dm.LastIntent = nlu.Action
	input := strings.ToLower(strings.TrimSpace(nlu.Raw))

	// Accumulate entities
	for k, v := range nlu.Entities {
		dm.Entities[k] = v
	}

	// Determine the new topic from entities
	newTopic := dm.extractTopic(nlu)

	// Track topic continuity
	if newTopic != "" {
		if newTopic == dm.lastTopic {
			dm.consecutiveTopicTurns++
		} else {
			dm.consecutiveTopicTurns = 1
			dm.lastTopic = newTopic
		}
	}

	// --- State transitions ---
	dm.PreviousState = dm.CurrentState

	switch {
	case dm.matchesGreeting(input):
		dm.CurrentState = StateGreeting

	case dm.matchesFarewell(input):
		dm.CurrentState = StateFarewell

	case dm.matchesReflection(input):
		dm.CurrentState = StateReflection

	case toolActions[nlu.Action]:
		dm.CurrentState = StateActionable

	case dm.CurrentState == StateGreeting:
		// After greeting, move to exploration
		dm.CurrentState = StateExploration

	case dm.CurrentState == StateClarifying:
		// After clarification, return to previous state
		dm.CurrentState = dm.PreviousState

	case dm.consecutiveTopicTurns >= 2 && dm.CurrentState == StateExploration:
		dm.CurrentState = StateDeepDive

	case dm.CurrentState == StateDeepDive && newTopic != "" && newTopic != dm.lastTopic:
		dm.CurrentState = StateExploration
	}

	// Push topic if new
	if newTopic != "" {
		dm.pushTopic(newTopic)
	}

	// Build context
	ctx := &DialogueContext{
		State:            dm.CurrentState,
		Topic:            dm.currentTopic(),
		TransitionPhrase: dm.transitionPhrase(),
	}

	// Generate follow-ups if we have a topic
	if topic := dm.currentTopic(); topic != "" {
		var facts []string
		if response != nil {
			for k, v := range response.Structured {
				facts = append(facts, k+": "+v)
			}
			if response.Data != "" {
				facts = append(facts, response.Data)
			}
		}
		dm.FollowUps = dm.suggestFollowUps(topic, facts)
		ctx.SuggestedFollowUps = dm.FollowUps
	}

	return ctx
}

// SuggestFollowUps generates 2-3 follow-up questions based on topic and known facts.
func (dm *DialogueManager) SuggestFollowUps(topic string, facts []string) []string {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	return dm.suggestFollowUps(topic, facts)
}

func (dm *DialogueManager) suggestFollowUps(topic string, facts []string) []string {
	if topic == "" {
		return nil
	}

	var followUps []string

	// Extract entities from facts to use in follow-up templates
	entities := dm.extractEntitiesFromFacts(facts)

	// Template-based follow-up generation
	templates := []struct {
		format    string
		needsArgs bool
	}{
		{"Would you like to know more about %s?", false},
		{"What aspects of %s interest you most?", false},
		{"How does %s relate to %s?", true},
		{"What about the history of %s?", false},
		{"Can you tell me more about the implications of %s?", false},
	}

	for _, tmpl := range templates {
		if len(followUps) >= 3 {
			break
		}
		if tmpl.needsArgs && len(entities) > 0 {
			for _, ent := range entities {
				if ent != topic {
					followUps = append(followUps, fmt.Sprintf(tmpl.format, topic, ent))
					break
				}
			}
		} else if !tmpl.needsArgs {
			followUps = append(followUps, fmt.Sprintf(tmpl.format, topic))
		}
	}

	// Cap at 3
	if len(followUps) > 3 {
		followUps = followUps[:3]
	}
	return followUps
}

// DetectAmbiguity checks if input is ambiguous given multiple candidates.
// Returns a clarification question if ambiguous, empty string otherwise.
func (dm *DialogueManager) DetectAmbiguity(input string, candidates []string) string {
	dm.mu.Lock()
	defer dm.mu.Unlock()

	if len(candidates) < 2 {
		return ""
	}

	lower := strings.ToLower(input)

	// Check if input is too short or vague
	words := strings.Fields(lower)
	isVague := len(words) <= 2

	// Count how many candidates match
	matches := 0
	var matchedCandidates []string
	for _, c := range candidates {
		if strings.Contains(lower, strings.ToLower(c)) || strings.Contains(strings.ToLower(c), lower) {
			matches++
			matchedCandidates = append(matchedCandidates, c)
		}
	}

	if matches > 1 || (isVague && len(candidates) > 1) {
		dm.Ambiguities = append(dm.Ambiguities, input)
		dm.PreviousState = dm.CurrentState
		dm.CurrentState = StateClarifying

		if len(matchedCandidates) > 0 {
			return fmt.Sprintf("Did you mean %s? Please clarify.", strings.Join(matchedCandidates, " or "))
		}
		return fmt.Sprintf("Could you clarify? Did you mean %s?", strings.Join(candidates, ", "))
	}

	return ""
}

// ResolveReference resolves pronouns like "it", "that", "this" to the current topic.
func (dm *DialogueManager) ResolveReference(input string) string {
	dm.mu.Lock()
	defer dm.mu.Unlock()

	topic := dm.currentTopic()
	if topic == "" {
		return input
	}

	words := strings.Fields(input)
	resolved := make([]string, 0, len(words))
	changed := false

	for _, w := range words {
		lower := strings.ToLower(w)
		isRef := false
		for _, ref := range referenceWords {
			if lower == ref {
				isRef = true
				break
			}
		}
		if isRef {
			resolved = append(resolved, topic)
			changed = true
		} else {
			resolved = append(resolved, w)
		}
	}

	if changed {
		return strings.Join(resolved, " ")
	}
	return input
}

// CurrentTopic returns the top of the topic stack.
func (dm *DialogueManager) CurrentTopic() string {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	return dm.currentTopic()
}

func (dm *DialogueManager) currentTopic() string {
	if len(dm.TopicStack) == 0 {
		return ""
	}
	return dm.TopicStack[len(dm.TopicStack)-1]
}

// PushTopic adds a topic to the stack.
func (dm *DialogueManager) PushTopic(topic string) {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	dm.pushTopic(topic)
}

func (dm *DialogueManager) pushTopic(topic string) {
	// Avoid duplicate consecutive topics
	if len(dm.TopicStack) > 0 && dm.TopicStack[len(dm.TopicStack)-1] == topic {
		return
	}
	dm.TopicStack = append(dm.TopicStack, topic)
	// Keep stack bounded
	if len(dm.TopicStack) > 20 {
		dm.TopicStack = dm.TopicStack[len(dm.TopicStack)-20:]
	}
}

// PopTopic removes and returns the top of the topic stack.
func (dm *DialogueManager) PopTopic() string {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	if len(dm.TopicStack) == 0 {
		return ""
	}
	top := dm.TopicStack[len(dm.TopicStack)-1]
	dm.TopicStack = dm.TopicStack[:len(dm.TopicStack)-1]
	return top
}

// GetTransitionPhrase returns a natural language transition for the current state change.
func (dm *DialogueManager) GetTransitionPhrase() string {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	return dm.transitionPhrase()
}

func (dm *DialogueManager) transitionPhrase() string {
	topic := dm.currentTopic()

	switch dm.CurrentState {
	case StateGreeting:
		return "Welcome!"
	case StateFarewell:
		return "Until next time!"
	case StateDeepDive:
		if topic != "" {
			return fmt.Sprintf("Diving deeper into %s...", topic)
		}
		return "Let's dig deeper..."
	case StateExploration:
		if dm.PreviousState == StateDeepDive && topic != "" {
			return fmt.Sprintf("Switching gears from %s...", topic)
		}
		if dm.PreviousState == StateGreeting {
			return "What can I help you with?"
		}
		return "Exploring further..."
	case StateClarifying:
		return "Let me make sure I understand..."
	case StateActionable:
		return "On it!"
	case StateReflection:
		if topic != "" {
			return fmt.Sprintf("Reflecting on %s...", topic)
		}
		return "That's an interesting thought..."
	}
	return ""
}

// --- internal helpers ---

func (dm *DialogueManager) matchesGreeting(input string) bool {
	for _, g := range greetingPatterns {
		if input == g || strings.HasPrefix(input, g+" ") {
			return true
		}
	}
	return false
}

func (dm *DialogueManager) matchesFarewell(input string) bool {
	for _, f := range farewellPatterns {
		if input == f || strings.HasPrefix(input, f+" ") || strings.HasSuffix(input, " "+f) {
			return true
		}
	}
	return false
}

func (dm *DialogueManager) matchesReflection(input string) bool {
	for _, prefix := range reflectionPrefixes {
		if strings.HasPrefix(input, prefix) {
			return true
		}
	}
	return false
}

func (dm *DialogueManager) extractTopic(nlu *NLUResult) string {
	// Try query/topic/subject entities first
	for _, key := range []string{"topic", "subject", "query", "term"} {
		if v, ok := nlu.Entities[key]; ok && v != "" {
			return v
		}
	}
	// Fall back to the raw input, cleaned up
	raw := strings.TrimSpace(nlu.Raw)
	if len(raw) > 0 && len(raw) <= 60 {
		return raw
	}
	return ""
}

func (dm *DialogueManager) extractEntitiesFromFacts(facts []string) []string {
	var entities []string
	seen := make(map[string]bool)
	for _, fact := range facts {
		// Simple extraction: take words before ":" in key-value facts
		if idx := strings.Index(fact, ":"); idx > 0 {
			key := strings.TrimSpace(fact[:idx])
			if !seen[key] {
				entities = append(entities, key)
				seen[key] = true
			}
			val := strings.TrimSpace(fact[idx+1:])
			// Extract notable words from value (capitalized, longer than 3 chars)
			for _, word := range strings.Fields(val) {
				clean := strings.Trim(word, ".,;:!?()[]\"'")
				if len(clean) > 3 && clean[0] >= 'A' && clean[0] <= 'Z' && !seen[clean] {
					entities = append(entities, clean)
					seen[clean] = true
				}
			}
		}
	}
	return entities
}
