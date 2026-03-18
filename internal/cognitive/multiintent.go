package cognitive

import (
	"strings"
)

// conjunctions is the set of conjunction phrases that may separate distinct intents.
// Ordered longest-first so longer phrases match before shorter substrings.
var conjunctions = []string{
	"and also",
	"and then",
	"as well as",
	"also",
	"plus",
	"then",
	"and",
}

// actionVerbs is a broad set of verbs/action words that signal the start of a
// distinct user intent. Used to decide whether both sides of a conjunction
// contain an independent action.
var actionVerbs = []string{
	// assistant features
	"remind", "set", "add", "create", "start", "stop", "open", "close",
	"launch", "check", "show", "list", "find", "search", "look",
	"translate", "convert", "calculate", "compute", "solve",
	"save", "delete", "remove", "kill", "run", "execute",
	"play", "pause", "skip", "mute", "unmute",
	"zip", "unzip", "compress", "extract", "archive",
	"hash", "encode", "decode",
	"ping", "scan",
	"define", "explain", "describe", "tell",
	"schedule", "plan", "book",
	"send", "email", "text", "call",
	"download", "upload", "fetch",
	"turn", "toggle", "enable", "disable",
	"generate", "make", "build", "write", "draft",
	"read", "edit", "view", "cat",
	"note",

	// question starters that act as implicit actions
	"what", "what's", "whats", "how", "who", "where", "when", "why",
	"is", "are", "can", "could", "does", "do", "will", "would",
}

// compoundPhrases is a set of well-known multi-word phrases where "and"
// connects parts of a single concept (not two separate actions).
var compoundPhrases = []string{
	"search and replace",
	"find and replace",
	"copy and paste",
	"cut and paste",
	"drag and drop",
	"rock and roll",
	"bread and butter",
	"salt and pepper",
	"pros and cons",
	"trial and error",
	"back and forth",
	"up and running",
	"give and take",
	"come and go",
	"rise and fall",
	"dos and don'ts",
	"hit and run",
	"hide and seek",
	"mix and match",
	"pick and choose",
	"wait and see",
	"eggs and milk",
	"eggs and cheese",
	"peanut butter and jelly",
	"mac and cheese",
	"lock and key",
	"nuts and bolts",
	"black and white",
	"left and right",
}

// SplitIntents splits a compound query into individual intent strings.
// It only splits on conjunctions when both sides appear to contain
// a distinct action (verb/action word). Single-concept phrases like
// "search and replace" or "bread and butter" are never split.
func SplitIntents(input string) []string {
	trimmed := strings.TrimSpace(input)
	if trimmed == "" {
		return nil
	}

	lower := strings.ToLower(trimmed)

	// Check if the entire input is (or contains) a known compound phrase.
	// If so, never split.
	for _, cp := range compoundPhrases {
		if strings.Contains(lower, cp) {
			return []string{trimmed}
		}
	}

	// Try each conjunction (longest first) to find a valid split point.
	for _, conj := range conjunctions {
		parts := splitOnConjunction(trimmed, lower, conj)
		if len(parts) > 1 {
			// Recursively try to split both halves further.
			var result []string
			for _, p := range parts {
				subParts := SplitIntents(p)
				result = append(result, subParts...)
			}
			return result
		}
	}

	return []string{trimmed}
}

// splitOnConjunction tries to split input on the given conjunction.
// Returns the parts if both sides contain an action verb, otherwise nil.
func splitOnConjunction(original, lower, conj string) []string {
	conjWithSpaces := " " + conj + " "
	idx := strings.Index(lower, conjWithSpaces)
	if idx < 0 {
		return nil
	}

	leftRaw := strings.TrimSpace(original[:idx])
	rightRaw := strings.TrimSpace(original[idx+len(conjWithSpaces):])

	if leftRaw == "" || rightRaw == "" {
		return nil
	}

	leftLower := strings.ToLower(leftRaw)
	rightLower := strings.ToLower(rightRaw)

	// Check if this "and" sits inside a list context (e.g., "add eggs and milk to my list").
	// Heuristic: if the right side does NOT start with an action verb but the left
	// side's action verb scope extends over the right side, don't split.
	if !startsWithActionVerb(rightLower) {
		return nil
	}

	// Both sides should have action content.
	if !containsActionVerb(leftLower) {
		return nil
	}

	return []string{leftRaw, rightRaw}
}

// startsWithActionVerb returns true if the text starts with a known action verb.
func startsWithActionVerb(lower string) bool {
	for _, v := range actionVerbs {
		if strings.HasPrefix(lower, v+" ") || strings.HasPrefix(lower, v+",") || lower == v {
			return true
		}
	}
	return false
}

// containsActionVerb returns true if the text contains a known action verb
// as a distinct word.
func containsActionVerb(lower string) bool {
	words := strings.Fields(lower)
	for _, w := range words {
		// Strip trailing punctuation from the word.
		clean := strings.TrimRight(w, ".,!?;:'\"")
		for _, v := range actionVerbs {
			if clean == v {
				return true
			}
		}
	}
	return false
}

// UnderstandMulti parses a (possibly compound) input into one or more NLUResults.
// For single-intent input, returns a slice with one element.
// For multi-intent input, returns one result per sub-intent.
func (n *NLU) UnderstandMulti(input string) []*NLUResult {
	parts := SplitIntents(input)
	if len(parts) <= 1 {
		return []*NLUResult{n.Understand(input)}
	}

	results := make([]*NLUResult, 0, len(parts))
	for _, part := range parts {
		results = append(results, n.Understand(part))
	}
	return results
}

// UnderstandMultiWithContext is like UnderstandWithContext but handles
// multi-intent queries. If the primary parse has low confidence (<0.5),
// it attempts to split into multiple intents. If the individual parts
// have higher average confidence, it returns the multi-intent result
// with SubResults populated.
func (n *NLU) UnderstandMultiWithContext(input string, conv *Conversation) *NLUResult {
	result := n.UnderstandWithContext(input, conv)

	// Only try multi-intent if confidence is low
	if result.Confidence >= 0.5 {
		return result
	}

	parts := SplitIntents(input)
	if len(parts) <= 1 {
		return result
	}

	// Parse each part individually
	subResults := make([]*NLUResult, 0, len(parts))
	var totalConf float64
	for _, part := range parts {
		sub := n.Understand(part)
		subResults = append(subResults, sub)
		totalConf += sub.Confidence
	}

	avgConf := totalConf / float64(len(subResults))

	// If the sub-results average higher confidence, use multi-intent
	if avgConf > result.Confidence {
		result.Intent = "multi"
		result.Action = "multi"
		result.Confidence = avgConf
		result.SubResults = subResults
	}

	return result
}
