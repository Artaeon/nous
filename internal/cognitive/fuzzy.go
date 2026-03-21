package cognitive

import "strings"

// -----------------------------------------------------------------------
// Fuzzy matching and synonym expansion for the NLU engine.
// Zero external dependencies — Levenshtein implemented from scratch.
// -----------------------------------------------------------------------

// commonStopword contains very common English words that should never
// fuzzy-match against tool keywords to prevent false positives like
// "feel"→"feed" (news), "have"→"hash", "this"→"disk", etc.
var commonStopword = map[string]bool{
	"feel": true, "felt": true, "fill": true, "fall": true, "fell": true,
	"have": true, "gave": true, "give": true, "live": true,
	"like": true, "make": true, "take": true, "made": true, "came": true,
	"come": true, "some": true, "more": true, "were": true, "here": true,
	"there": true, "their": true, "where": true, "these": true, "those": true,
	"this": true, "that": true, "than": true, "then": true, "them": true,
	"they": true, "when": true, "what": true, "with": true, "will": true,
	"been": true, "does": true, "done": true, "good": true, "just": true,
	"much": true, "most": true, "also": true, "very": true, "only": true,
	"even": true, "well": true, "back": true, "down": true, "from": true,
	"each": true, "said": true, "into": true, "over": true, "such": true,
	"your": true, "know": true, "want": true, "need": true, "tell": true,
	"happy": true, "today": true, "about": true, "would": true, "could": true,
	"being": true, "still": true, "after": true, "great": true, "think": true,
	"shall": true, "might": true, "never": true, "under": true, "since": true,
}

// levenshtein computes the edit distance between two strings.
// Uses O(min(m,n)) space via a single-row dynamic programming approach.
func levenshtein(a, b string) int {
	if len(a) < len(b) {
		a, b = b, a
	}
	if len(b) == 0 {
		return len(a)
	}

	prev := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}

	for i := 1; i <= len(a); i++ {
		curr := make([]int, len(b)+1)
		curr[0] = i
		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			del := prev[j] + 1
			ins := curr[j-1] + 1
			sub := prev[j-1] + cost
			curr[j] = minInt(del, minInt(ins, sub))
		}
		prev = curr
	}
	return prev[len(b)]
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// fuzzyThreshold returns the maximum allowed edit distance for a word of the
// given length.  Words >= 5 chars allow distance 2; shorter words allow 1.
// Words of length 0-1 require exact match (threshold 0).
func fuzzyThreshold(wordLen int) int {
	if wordLen >= 5 {
		return 2
	}
	if wordLen >= 2 {
		return 1
	}
	return 0
}

// fuzzyWordMatch returns true if candidate is within the Levenshtein threshold
// of target.  Both strings should already be lowercase.
func fuzzyWordMatch(candidate, target string) bool {
	if candidate == target {
		return true
	}
	threshold := fuzzyThreshold(len(target))
	if threshold == 0 {
		return false
	}
	// Both candidate and target must be at least 4 chars for fuzzy matching
	// to avoid false positives on short common words ("new"→"news", "a"→"an").
	if len(candidate) < 4 || len(target) < 4 {
		return false
	}
	// Skip fuzzy matching when the candidate is a very common English word
	// that would false-positive against tool keywords (e.g. "feel"→"feed").
	if commonStopword[candidate] {
		return false
	}
	// Quick length-difference check to skip obvious mismatches.
	diff := len(candidate) - len(target)
	if diff < 0 {
		diff = -diff
	}
	if diff > threshold {
		return false
	}
	return levenshtein(candidate, target) <= threshold
}

// fuzzyContains checks whether any word in `input` fuzzy-matches any word in
// the multi-word `phrase`.  For multi-word phrases, it checks a sliding window
// of consecutive words against the phrase words.
//
// This is called ONLY when the exact strings.Contains has already failed,
// so it acts as a fallback — never overriding exact matches.
func fuzzyContains(input, phrase string) bool {
	phraseWords := strings.Fields(phrase)
	inputWords := strings.Fields(input)

	if len(phraseWords) == 0 || len(inputWords) == 0 {
		return false
	}

	if len(phraseWords) == 1 {
		target := phraseWords[0]
		for _, w := range inputWords {
			if fuzzyWordMatch(w, target) {
				return true
			}
		}
		return false
	}

	// Multi-word phrase: sliding window over input words.
	// Use stricter threshold (max 1 edit per word) to avoid false positives
	// like "how does" matching "show notes".
	pLen := len(phraseWords)
	if len(inputWords) < pLen {
		return false
	}
	for i := 0; i <= len(inputWords)-pLen; i++ {
		allMatch := true
		fuzzyCount := 0
		for j := 0; j < pLen; j++ {
			cand := inputWords[i+j]
			targ := phraseWords[j]
			if cand == targ {
				continue
			}
			// For multi-word phrases, require both words >= 4 chars and distance <= 1.
			if len(cand) < 4 || len(targ) < 4 || levenshtein(cand, targ) > 1 {
				allMatch = false
				break
			}
			fuzzyCount++
		}
		if allMatch {
			return true
		}
	}
	return false
}

// -----------------------------------------------------------------------
// Synonym map — maps synonym phrases to a canonical word/phrase that
// already exists in the NLU word lists.
// -----------------------------------------------------------------------

// synonymEntry maps a trigger phrase to a canonical form and the intent
// it should route to.  The canonical form must appear in the NLU's
// existing word lists so the normal matching picks it up.
type synonymEntry struct {
	canonical string
	intent    string
}

// synonymMap is built once and reused.  Keys are lowercase trigger phrases.
var synonymMap = map[string]synonymEntry{
	// Weather
	"temp":        {canonical: "temperature", intent: "weather"},
	"temperature": {canonical: "temperature", intent: "weather"},

	// Volume
	"crank up":  {canonical: "louder", intent: "volume"},
	"raise":     {canonical: "louder", intent: "volume"},
	"increase":  {canonical: "louder", intent: "volume"},
	"louder":    {canonical: "louder", intent: "volume"},
	"turn it up": {canonical: "turn up", intent: "volume"},

	// Dictionary
	"define":          {canonical: "define ", intent: "dict"},
	"meaning of":      {canonical: "meaning of", intent: "dict"},
	"what does x mean": {canonical: "definition of", intent: "dict"},

	// Compute
	"calculate": {canonical: "calculate", intent: "compute"},
	"compute":   {canonical: "compute", intent: "compute"},
	"solve":     {canonical: "solve", intent: "compute"},
	"evaluate":  {canonical: "evaluate", intent: "compute"},

	// Password (maps to hash for encoding tools)
	"generate password": {canonical: "hash", intent: "hash"},
	"random password":   {canonical: "hash", intent: "hash"},
	"new password":      {canonical: "hash", intent: "hash"},

	// Bookmarks (maps to notes as a save mechanism)
	"bookmark":  {canonical: "save a note", intent: "note"},
	"save link": {canonical: "save a note", intent: "note"},
	"save url":  {canonical: "save a note", intent: "note"},

	// Journal (maps to notes)
	"journal":    {canonical: "save a note", intent: "note"},
	"diary":      {canonical: "save a note", intent: "note"},
	"log entry":  {canonical: "save a note", intent: "note"},
	"dear diary": {canonical: "save a note", intent: "note"},

	// Habits (maps to todos)
	"habit":       {canonical: "add todo", intent: "todo"},
	"streak":      {canonical: "show todos", intent: "todo"},
	"daily check": {canonical: "show todos", intent: "todo"},
	"did i":       {canonical: "show todos", intent: "todo"},

	// Expenses (maps to notes for logging)
	"expense":  {canonical: "save a note", intent: "note"},
	"spent":    {canonical: "save a note", intent: "note"},
	"cost":     {canonical: "save a note", intent: "note"},
	"purchase": {canonical: "save a note", intent: "note"},
	"bought":   {canonical: "save a note", intent: "note"},

	// Translate
	"trasnlate":  {canonical: "translate", intent: "translate"},
	"transalte":  {canonical: "translate", intent: "translate"},
	"translte":   {canonical: "translate", intent: "translate"},
}

// expandSynonyms scans the input for synonym triggers and returns the
// intent override if a synonym is found.  Returns ("", "") if no synonym
// matches.  The caller can use the intent to short-circuit classification.
func expandSynonyms(lower string) (intent string, canonical string) {
	// Check multi-word synonyms first (longer phrases take priority).
	// Sort by descending length for greedy matching.
	for trigger, entry := range synonymMap {
		if strings.Contains(trigger, " ") {
			if strings.Contains(lower, trigger) {
				return entry.intent, entry.canonical
			}
		}
	}
	// Single-word synonyms.
	for trigger, entry := range synonymMap {
		if !strings.Contains(trigger, " ") {
			if matchWord(lower, trigger) {
				return entry.intent, entry.canonical
			}
		}
	}
	return "", ""
}

// fuzzyMatchWordList checks if the input fuzzy-matches any phrase in the
// given word list.  This is used as a fallback after exact matching fails.
// To keep performance bounded, we only fuzzy-match individual words from
// the input against single-word entries, and use sliding-window for
// multi-word entries.
func fuzzyMatchWordList(lower string, wordList []string) bool {
	for _, phrase := range wordList {
		if fuzzyContains(lower, phrase) {
			return true
		}
	}
	return false
}

// -----------------------------------------------------------------------
// Integration helpers — used by the NLU classifyIntent method.
// -----------------------------------------------------------------------

// fuzzyClassifyTools tries synonym expansion and fuzzy word-list matching
// as a fallback when exact matching in classifyIntent finds nothing.
// Returns true and populates the result if a match is found.
func (n *NLU) fuzzyClassifyTools(lower string, r *NLUResult) bool {
	// Step 1: Try synonym expansion.
	if intent, _ := expandSynonyms(lower); intent != "" {
		// Check for "how much"/"how many" ambiguity — could be compute or convert.
		if intent == "compute" {
			if strings.Contains(lower, "how much") || strings.Contains(lower, "how many") || strings.Contains(lower, "what percent") {
				// If there's a math operator, it's compute; otherwise check for unit conversion.
				if n.mathRe.MatchString(lower) {
					r.Intent = "compute"
					r.Confidence = 0.80
					r.Action = ""
					return true
				}
				for _, cw := range n.convertWords {
					if strings.Contains(lower, cw) {
						r.Intent = "convert"
						r.Confidence = 0.85
						r.Action = ""
						return true
					}
				}
			}
		}
		r.Intent = intent
		r.Confidence = 0.80
		r.Action = ""
		r.Entities["topic"] = n.extractTopicGeneral(lower)
		return true
	}

	// Step 2: Fuzzy match against tool word lists (most specific first).
	type wordListEntry struct {
		words  []string
		intent string
	}
	toolLists := []wordListEntry{
		{n.weatherWords, "weather"},
		{n.volumeWords, "volume"},
		{n.brightnessWords, "brightness"},
		{n.timerWords, "timer"},
		{n.translateWords, "translate"},
		{n.dictWords, "dict"},
		{n.hashWords, "hash"},
		{n.networkWords, "network"},
		{n.archiveWords, "archive"},
		{n.diskUsageWords, "disk_usage"},
		{n.processWords, "process"},
		{n.qrcodeWords, "qrcode"},
		{n.clipboardWords, "clipboard"},
		{n.screenshotWords, "screenshot"},
		{n.noteWords, "note"},
		{n.todoWords, "todo"},
		{n.newsWords, "news"},
		{n.calendarWords, "calendar"},
		{n.emailWords, "email"},
		{n.appWords, "app"},
	}

	for _, tl := range toolLists {
		if fuzzyMatchWordList(lower, tl.words) {
			r.Intent = tl.intent
			r.Confidence = 0.75 // slightly lower than exact match
			r.Action = ""
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return true
		}
	}

	return false
}
