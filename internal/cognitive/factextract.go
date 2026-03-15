package cognitive

import (
	"regexp"
	"strings"

	"github.com/artaeon/nous/internal/memory"
)

// FactExtractor automatically extracts personal facts from user messages
// and stores them in long-term memory. This enables cross-session recall
// without needing the LLM to explicitly "remember" things.
//
// Pattern-based extraction is fast (sub-ms) and runs on every user input.
// It captures: name, role, interests, current work, preferences.
type FactExtractor struct {
	LTM        *memory.LongTermMemory
	WorkingMem *memory.WorkingMemory
}

// factPattern maps a regex to a (key, category) pair for LTM storage.
type factPattern struct {
	Pattern  *regexp.Regexp
	Key      string
	Category string
	// Extract returns the value to store from the regex match groups.
	Extract func(matches []string) string
}

var factPatterns = []factPattern{
	// Name extraction
	{
		Pattern:  regexp.MustCompile(`(?i:(?:my name is|i'?m called|call me|i go by))\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)`),
		Key:      "user.name",
		Category: "personal",
		Extract:  func(m []string) string { return m[1] },
	},
	// Role extraction
	{
		Pattern:  regexp.MustCompile(`(?i)i(?:'m| am) (?:a |an )?(\w+(?:\s+\w+){0,3})\s+(?:developer|engineer|designer|scientist|researcher|student|teacher|manager|analyst|architect|consultant)`),
		Key:      "user.role",
		Category: "personal",
		Extract: func(m []string) string {
			return strings.TrimSpace(m[0][strings.Index(strings.ToLower(m[0]), "i'm ")+4:])
		},
	},
	{
		Pattern:  regexp.MustCompile(`(?i)i(?:'m| am) (?:a |an )([\w\s]+(?:developer|engineer|designer|scientist|researcher|student|teacher|manager|analyst|architect|consultant))`),
		Key:      "user.role",
		Category: "personal",
		Extract:  func(m []string) string { return m[1] },
	},
	// Work/project
	{
		Pattern:  regexp.MustCompile(`(?i)(?:i'?m |i am )?(?:working on|building|developing|creating|making)\s+(.{5,80}?)(?:\.|!|$)`),
		Key:      "user.current_work",
		Category: "work",
		Extract:  func(m []string) string { return strings.TrimSpace(m[1]) },
	},
	// Interests
	{
		Pattern:  regexp.MustCompile(`(?i)i (?:love|enjoy|like|am interested in|am passionate about|am fascinated by)\s+(.{3,60}?)(?:\.|!|,|\s+and\s+|$)`),
		Key:      "user.interests",
		Category: "personal",
		Extract:  func(m []string) string { return strings.TrimSpace(m[1]) },
	},
	// Location
	{
		Pattern:  regexp.MustCompile(`(?i)i (?:live in|am from|am based in|work in)\s+([A-Z][\w\s,]+?)(?:\.|!|$)`),
		Key:      "user.location",
		Category: "personal",
		Extract:  func(m []string) string { return strings.TrimSpace(m[1]) },
	},
}

// Extract scans a user message for personal facts and stores them.
// Returns the number of new facts extracted.
func (fe *FactExtractor) Extract(input string) int {
	if fe.LTM == nil {
		return 0
	}

	extracted := 0
	for _, fp := range factPatterns {
		m := fp.Pattern.FindStringSubmatch(input)
		if m == nil {
			continue
		}

		value := fp.Extract(m)
		if value == "" || len(value) < 2 {
			continue
		}

		// Store in long-term memory (persists to disk)
		fe.LTM.Store(fp.Key, value, fp.Category)

		// Also store in working memory for immediate conversation access
		if fe.WorkingMem != nil {
			fe.WorkingMem.Store(fp.Key, value, 1.0)
		}

		extracted++
	}

	return extracted
}
