package cognitive

import (
	"strings"
	"sync"
	"time"
)

// -----------------------------------------------------------------------
// Context Window — sliding window of recent conversation turns for NLG.
//
// The NLG engine generates each response in isolation. ContextWindow gives
// it short-term memory: it knows what was said, which topics were covered,
// and can strip duplicate sentences from candidate responses so the bot
// never sounds like a broken record.
// -----------------------------------------------------------------------

// ContextTurn is one turn in the conversation.
type ContextTurn struct {
	Input     string
	Response  string
	Topics    []string
	Timestamp time.Time
}

// ContextWindow maintains a sliding window of recent conversation turns
// that the NLG engine can use to avoid repetition and build on context.
type ContextWindow struct {
	turns    []ContextTurn
	maxTurns int
	mu       sync.RWMutex
}

// NewContextWindow creates a window that retains up to maxTurns turns.
// If maxTurns <= 0 it defaults to 5.
func NewContextWindow(maxTurns int) *ContextWindow {
	if maxTurns <= 0 {
		maxTurns = 5
	}
	return &ContextWindow{
		maxTurns: maxTurns,
	}
}

// Record adds a completed turn to the window. If the window exceeds
// maxTurns the oldest turn is dropped.
func (cw *ContextWindow) Record(input, response string, topics []string) {
	cw.mu.Lock()
	defer cw.mu.Unlock()

	cw.turns = append(cw.turns, ContextTurn{
		Input:     input,
		Response:  response,
		Topics:    topics,
		Timestamp: time.Now(),
	})

	if len(cw.turns) > cw.maxTurns {
		cw.turns = cw.turns[len(cw.turns)-cw.maxTurns:]
	}
}

// RecentTopics returns all topics from the window, deduplicated,
// most-recent first.
func (cw *ContextWindow) RecentTopics() []string {
	cw.mu.RLock()
	defer cw.mu.RUnlock()

	seen := make(map[string]bool)
	var out []string

	// Walk newest to oldest so the first occurrence wins.
	for i := len(cw.turns) - 1; i >= 0; i-- {
		for _, t := range cw.turns[i].Topics {
			lower := strings.ToLower(t)
			if !seen[lower] {
				seen[lower] = true
				out = append(out, t)
			}
		}
	}
	return out
}

// WasMentioned returns true if topic (case-insensitive) appears in any
// turn's topic list.
func (cw *ContextWindow) WasMentioned(topic string) bool {
	cw.mu.RLock()
	defer cw.mu.RUnlock()

	lower := strings.ToLower(topic)
	for _, turn := range cw.turns {
		for _, t := range turn.Topics {
			if strings.ToLower(t) == lower {
				return true
			}
		}
	}
	return false
}

// PreviousResponse returns the most recent response, or "" if the
// window is empty.
func (cw *ContextWindow) PreviousResponse() string {
	cw.mu.RLock()
	defer cw.mu.RUnlock()

	if len(cw.turns) == 0 {
		return ""
	}
	return cw.turns[len(cw.turns)-1].Response
}

// AvoidRepetition removes sentences from candidate that already appeared
// in earlier responses. A sentence is considered duplicate when its
// lowercased, trimmed form matches exactly. If all sentences would be
// removed the candidate is returned unchanged to avoid blank output.
func (cw *ContextWindow) AvoidRepetition(candidate string) string {
	cw.mu.RLock()
	defer cw.mu.RUnlock()

	// Build a set of previously-seen sentences.
	prev := make(map[string]bool)
	for _, turn := range cw.turns {
		for _, s := range cwSplitSentences(turn.Response) {
			norm := strings.ToLower(strings.TrimSpace(s))
			if norm != "" {
				prev[norm] = true
			}
		}
	}

	if len(prev) == 0 {
		return candidate
	}

	sentences := cwSplitSentences(candidate)
	var kept []string
	for _, s := range sentences {
		norm := strings.ToLower(strings.TrimSpace(s))
		if norm == "" {
			continue
		}
		if !prev[norm] {
			kept = append(kept, s)
		}
	}

	if len(kept) == 0 {
		return candidate // don't return empty
	}
	return strings.Join(kept, " ")
}

// BuildContextSummary generates a human-readable one-liner summarising
// recent conversation topics, e.g. "We've been discussing X, Y, and Z."
// Returns "" if there are no topics.
func (cw *ContextWindow) BuildContextSummary() string {
	topics := cw.RecentTopics()
	if len(topics) == 0 {
		return ""
	}

	switch len(topics) {
	case 1:
		return "We've been discussing " + topics[0] + "."
	case 2:
		return "We've been discussing " + topics[0] + " and " + topics[1] + "."
	default:
		return "We've been discussing " + strings.Join(topics[:len(topics)-1], ", ") +
			", and " + topics[len(topics)-1] + "."
	}
}

// cwSplitSentences does a lightweight sentence split on period, exclamation
// mark, or question mark followed by a space or end-of-string. It keeps
// the punctuation attached to the sentence.
func cwSplitSentences(text string) []string {
	var sentences []string
	var buf strings.Builder

	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		buf.WriteRune(runes[i])

		if runes[i] == '.' || runes[i] == '!' || runes[i] == '?' {
			// End of sentence if next char is space, newline, or end.
			if i+1 >= len(runes) || runes[i+1] == ' ' || runes[i+1] == '\n' {
				s := strings.TrimSpace(buf.String())
				if s != "" {
					sentences = append(sentences, s)
				}
				buf.Reset()
			}
		}
	}
	// Trailing fragment without terminal punctuation.
	if s := strings.TrimSpace(buf.String()); s != "" {
		sentences = append(sentences, s)
	}
	return sentences
}
