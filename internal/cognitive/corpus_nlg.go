package cognitive

import (
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unicode"
)

// -----------------------------------------------------------------------
// Corpus-Mined Natural Language Generation
//
// Instead of hardcoded templates like "%s is a %s.", this system mines
// real sentence structures from the Wikipedia knowledge corpus. Every
// output pattern was written by a human — the system learns to reuse
// them by:
//
//   1. Reading knowledge files and splitting into sentences
//   2. Identifying entities (proper nouns, technical terms, dates, places)
//   3. Replacing entities with typed slots: [SUBJECT], [OBJECT], etc.
//   4. Classifying discourse function (definition, origin, property, ...)
//   5. Building a word bigram model for fluent transitions
//   6. Generating text by selecting structures, filling slots, and
//      smoothing with the bigram model
//
// The result: every sentence structure in the output was originally
// crafted by a real writer. Nous just learns which structures work
// and how to recombine them with current facts.
// -----------------------------------------------------------------------

// CorpusNLG generates natural language by mining and recombining patterns
// from a real text corpus. Unlike template-based NLG, every sentence
// structure was written by a human — the system just learns to reuse them.
type CorpusNLG struct {
	patterns     []MinedPattern            // sentence structures extracted from corpus
	byFunction   map[string][]int          // discourse function → pattern indices
	bigrams      map[string]map[string]int // word → next_word → count
	totalBigrams map[string]int            // word → total next count (for probability)
	vocab        map[string]bool           // known vocabulary
	lastUsed     map[int]int               // pattern index → generation counter (anti-repetition)
	genCounter   int                       // incremented each generation call
	mu           sync.RWMutex
}

// MinedPattern is a sentence structure extracted from real text.
type MinedPattern struct {
	Structure string  // e.g. "[SUBJECT] is [CATEGORY] that [VERB_PHRASE] [OBJECT]"
	Function  string  // "definition", "origin", "property", "usage", "relation", "evaluation"
	Original  string  // the source sentence
	SlotCount int     // number of slots to fill
	Quality   float64 // usage-weighted quality score
	WordCount int
}

// discourseFunctions and their detection signals.
var corpusDiscourseFunctions = []struct {
	Name    string
	Signals []string
}{
	{"definition", []string{"is a", "is an", "is the", "are the", "are a", "refers to", "describes"}},
	{"origin", []string{"created by", "founded by", "developed by", "established by", "born in", "invented by", "proposed by", "discovered by", "built by", "designed by", "written by", "composed by"}},
	{"property", []string{"has", "features", "includes", "contains", "offers", "provides", "exhibits", "displays", "possesses"}},
	{"usage", []string{"used for", "used in", "applied to", "enables", "powers", "employed in", "utilized in", "serves as"}},
	{"relation", []string{"related to", "connected to", "associated with", "linked to", "part of", "belongs to", "derives from"}},
	{"evaluation", []string{"important", "significant", "crucial", "essential", "fundamental", "remarkable", "influential", "critical", "pivotal", "transformative"}},
}

// corpusOpenerSeeds maps discourse function to seed words for generating openers.
var corpusOpenerSeeds = map[string][]string{
	"definition": {"is", "are", "describes", "refers"},
	"origin":     {"developed", "created", "established", "founded"},
	"property":   {"features", "includes", "has", "offers"},
	"usage":      {"used", "applied", "enables", "powers"},
	"relation":   {"related", "connected", "associated", "linked"},
	"evaluation": {"remains", "stands", "represents", "proves"},
}

// NewCorpusNLG creates a new corpus-mined NLG system.
func NewCorpusNLG() *CorpusNLG {
	return &CorpusNLG{
		byFunction:   make(map[string][]int),
		bigrams:      make(map[string]map[string]int),
		totalBigrams: make(map[string]int),
		vocab:        make(map[string]bool),
		lastUsed:     make(map[int]int),
	}
}

// -----------------------------------------------------------------------
// IngestCorpus — read all .txt files and mine patterns + bigrams
// -----------------------------------------------------------------------

// IngestCorpus reads all .txt files from the given directory, extracts
// sentence structures (replacing entities with typed slots), classifies
// discourse functions, and builds a word bigram model.
func (c *CorpusNLG) IngestCorpus(dir string) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}

	var allText strings.Builder
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".txt") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		allText.WriteString(string(data))
		allText.WriteString(" ")
	}

	corpus := allText.String()
	if len(corpus) == 0 {
		return nil
	}

	sentences := corpusSplitSentences(corpus)

	c.mu.Lock()
	defer c.mu.Unlock()

	// Build bigram model from the full corpus.
	c.buildBigrams(corpus)

	// Extract patterns from each sentence.
	for _, sent := range sentences {
		sent = strings.TrimSpace(sent)
		if len(sent) < 20 || len(sent) > 500 {
			continue // skip trivially short or excessively long
		}
		wordCount := len(strings.Fields(sent))
		if wordCount < 5 || wordCount > 60 {
			continue
		}

		structure, slotCount := c.extractStructure(sent)
		if slotCount == 0 {
			continue // no entities found — not useful as a pattern
		}

		function := corpusClassifyFunction(sent)

		pattern := MinedPattern{
			Structure: structure,
			Function:  function,
			Original:  sent,
			SlotCount: slotCount,
			Quality:   1.0,
			WordCount: wordCount,
		}

		idx := len(c.patterns)
		c.patterns = append(c.patterns, pattern)
		c.byFunction[function] = append(c.byFunction[function], idx)
	}

	return nil
}

// corpusSplitSentences splits text into sentences at period + space +
// capital letter boundaries, handling abbreviations gracefully.
func corpusSplitSentences(text string) []string {
	var sentences []string
	var current strings.Builder

	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		current.WriteRune(runes[i])

		if runes[i] != '.' {
			continue
		}

		// Look ahead: period + space + capital letter = sentence boundary.
		if i+2 < len(runes) && runes[i+1] == ' ' && unicode.IsUpper(runes[i+2]) {
			s := strings.TrimSpace(current.String())
			if len(s) > 0 {
				sentences = append(sentences, s)
			}
			current.Reset()
			i++ // skip the space; the capital letter starts the next iteration
		}
	}

	// Last sentence.
	s := strings.TrimSpace(current.String())
	if len(s) > 10 {
		sentences = append(sentences, s)
	}

	return sentences
}

// buildBigrams counts word pairs across the full corpus text.
func (c *CorpusNLG) buildBigrams(text string) {
	words := corpusTokenize(text)
	for i := 0; i < len(words)-1; i++ {
		w := strings.ToLower(words[i])
		next := strings.ToLower(words[i+1])
		c.vocab[w] = true
		c.vocab[next] = true

		if c.bigrams[w] == nil {
			c.bigrams[w] = make(map[string]int)
		}
		c.bigrams[w][next]++
		c.totalBigrams[w]++
	}
}

// corpusTokenize splits text into words, stripping punctuation edges
// but keeping all words (no stop-word filtering — needed for bigrams).
func corpusTokenize(text string) []string {
	raw := strings.Fields(text)
	words := make([]string, 0, len(raw))
	for _, w := range raw {
		w = strings.TrimFunc(w, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '-' && r != '\''
		})
		if len(w) > 0 {
			words = append(words, w)
		}
	}
	return words
}

// extractStructure replaces entities with typed slots, returning the
// templated structure and the number of slots inserted.
func (c *CorpusNLG) extractStructure(sentence string) (string, int) {
	result := sentence
	slots := 0

	// Order matters: do longer/more specific patterns first.

	// Person: words after "created by", "founded by", "developed by", etc.
	personTriggers := []string{
		"created by ", "founded by ", "developed by ", "established by ",
		"invented by ", "proposed by ", "designed by ", "written by ",
		"composed by ", "built by ", "directed by ", "produced by ",
		"discovered by ",
	}
	for _, trigger := range personTriggers {
		idx := strings.Index(strings.ToLower(result), trigger)
		if idx >= 0 {
			afterTrigger := idx + len(trigger)
			person := corpusExtractPhrase(result[afterTrigger:])
			if len(person) > 1 {
				result = result[:afterTrigger] + "[PERSON]" + result[afterTrigger+len(person):]
				slots++
				break
			}
		}
	}

	// Date/year: four-digit numbers or phrases like "the early twentieth century".
	result, n := corpusReplaceYearsAndDates(result)
	slots += n

	// Place: capitalized word after " in " (but not at sentence start).
	result, n = corpusReplacePlaces(result)
	slots += n

	// Category: noun phrase after "is a/an/the".
	result, n = corpusReplaceCategories(result)
	slots += n

	// Subject: first capitalized multi-word phrase or technical term.
	result, n = corpusReplaceSubject(result)
	slots += n

	// Object: remaining capitalized phrases not at sentence start.
	result, n = corpusReplaceObjects(result)
	slots += n

	return result, slots
}

// corpusExtractPhrase pulls a noun phrase from the start of text: sequence
// of capitalized words, possibly with "and", "of", "the".
func corpusExtractPhrase(text string) string {
	words := strings.Fields(text)
	var phrase []string
	connectors := map[string]bool{"and": true, "of": true, "the": true, "for": true, "in": true, "de": true, "von": true, "van": true}

	for _, w := range words {
		clean := strings.TrimRight(w, ".,;:!?)")
		if len(clean) == 0 {
			break
		}
		firstRune := []rune(clean)[0]
		if unicode.IsUpper(firstRune) || connectors[strings.ToLower(clean)] {
			phrase = append(phrase, clean)
		} else {
			break
		}
	}
	if len(phrase) == 0 {
		return ""
	}
	return strings.Join(phrase, " ")
}

// corpusReplaceYearsAndDates replaces year patterns (e.g. "1905", "1860s")
// with [DATE].
func corpusReplaceYearsAndDates(s string) (string, int) {
	count := 0
	words := strings.Fields(s)
	var out []string
	for _, w := range words {
		clean := strings.TrimRight(w, ".,;:!?)")
		suffix := w[len(clean):]
		if corpusIsYear(clean) {
			out = append(out, "[DATE]"+suffix)
			count++
		} else {
			out = append(out, w)
		}
	}
	if count > 0 {
		return strings.Join(out, " "), count
	}
	return s, 0
}

// corpusIsYear returns true for strings like "1905", "1860s", "2012".
func corpusIsYear(s string) bool {
	base := strings.TrimSuffix(s, "s")
	if len(base) != 4 {
		return false
	}
	for _, r := range base {
		if !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

// corpusReplacePlaces finds " in <Capitalized>" patterns not at sentence
// start and replaces the place with [PLACE].
func corpusReplacePlaces(s string) (string, int) {
	count := 0
	words := strings.Fields(s)
	var out []string
	for i, w := range words {
		if i > 1 && (strings.ToLower(words[i-1]) == "in" || strings.ToLower(words[i-1]) == "from" || strings.ToLower(words[i-1]) == "near") {
			clean := strings.TrimRight(w, ".,;:!?)")
			suffix := w[len(clean):]
			if len(clean) > 1 && unicode.IsUpper([]rune(clean)[0]) && !corpusAlreadySlotted(clean) {
				out = append(out, "[PLACE]"+suffix)
				count++
				continue
			}
		}
		out = append(out, w)
	}
	if count > 0 {
		return strings.Join(out, " "), count
	}
	return s, 0
}

// corpusReplaceCategories replaces noun phrases after "is a/an/the" with [CATEGORY].
func corpusReplaceCategories(s string) (string, int) {
	lower := strings.ToLower(s)
	triggers := []string{" is a ", " is an ", " is the ", " are the ", " are a "}

	for _, trigger := range triggers {
		idx := strings.Index(lower, trigger)
		if idx < 0 {
			continue
		}
		afterIdx := idx + len(trigger)
		rest := s[afterIdx:]
		phrase := corpusExtractCategoryPhrase(rest)
		if len(phrase) > 2 {
			return s[:afterIdx] + "[CATEGORY]" + s[afterIdx+len(phrase):], 1
		}
	}
	return s, 0
}

// corpusExtractCategoryPhrase extracts a noun phrase that terminates at
// a clause boundary (comma, "that", "which", "where", "who", period).
func corpusExtractCategoryPhrase(text string) string {
	clauses := []string{" that ", " which ", " where ", " who ", " whose "}
	end := len(text)

	for _, cl := range clauses {
		idx := strings.Index(strings.ToLower(text), cl)
		if idx >= 0 && idx < end {
			end = idx
		}
	}

	commaIdx := strings.Index(text, ",")
	if commaIdx >= 0 && commaIdx < end {
		end = commaIdx
	}

	periodIdx := strings.Index(text, ".")
	if periodIdx >= 0 && periodIdx < end {
		end = periodIdx
	}

	return strings.TrimSpace(text[:end])
}

// corpusReplaceSubject replaces the first capitalized phrase at the start
// of the sentence with [SUBJECT].
func corpusReplaceSubject(s string) (string, int) {
	if strings.HasPrefix(s, "[") {
		return s, 0 // already has a slot at the start
	}

	words := strings.Fields(s)
	if len(words) == 0 {
		return s, 0
	}

	var subjWords []string
	connectors := map[string]bool{"of": true, "the": true, "and": true, "de": true, "von": true, "van": true, "for": true}

	for i, w := range words {
		clean := strings.TrimRight(w, ".,;:!?)")
		if len(clean) == 0 {
			break
		}
		firstRune := []rune(clean)[0]
		if unicode.IsUpper(firstRune) || (i > 0 && connectors[strings.ToLower(clean)]) {
			subjWords = append(subjWords, w)
		} else {
			break
		}
	}

	if len(subjWords) == 0 {
		return s, 0
	}

	subject := strings.Join(subjWords, " ")
	// Only replace if it looks like a proper noun / entity (not just "The").
	if len(subjWords) == 1 {
		clean := strings.TrimRight(subjWords[0], ".,;:!?)")
		if corpusIsStopWord(strings.ToLower(clean)) {
			return s, 0
		}
	}

	rest := strings.TrimPrefix(s, subject)
	return "[SUBJECT]" + rest, 1
}

// corpusReplaceObjects replaces remaining capitalized multi-word phrases
// (not at sentence start and not already slotted) with [OBJECT].
func corpusReplaceObjects(s string) (string, int) {
	words := strings.Fields(s)
	count := 0
	maxReplacements := 1

	for i := 1; i < len(words) && count < maxReplacements; i++ {
		w := words[i]
		clean := strings.TrimRight(w, ".,;:!?)")
		if len(clean) < 2 || corpusAlreadySlotted(clean) {
			continue
		}

		firstRune := []rune(clean)[0]
		if !unicode.IsUpper(firstRune) {
			continue
		}

		// Skip if right after a period (sentence-start-like).
		if i > 0 && strings.HasSuffix(words[i-1], ".") {
			continue
		}

		if corpusIsStopWord(strings.ToLower(clean)) {
			continue
		}

		// Replace this word (and following capitalized words) with [OBJECT].
		phraseEnd := i + 1
		connectors := map[string]bool{"of": true, "the": true, "and": true}
		for phraseEnd < len(words) {
			pw := strings.TrimRight(words[phraseEnd], ".,;:!?)")
			pr := []rune(pw)
			if len(pr) == 0 {
				break
			}
			if unicode.IsUpper(pr[0]) || connectors[strings.ToLower(pw)] {
				phraseEnd++
			} else {
				break
			}
		}

		suffix := ""
		lastWord := words[phraseEnd-1]
		lastClean := strings.TrimRight(lastWord, ".,;:!?)")
		if len(lastClean) < len(lastWord) {
			suffix = lastWord[len(lastClean):]
		}

		replacement := "[OBJECT]" + suffix
		newWords := make([]string, 0, len(words)-phraseEnd+i+1)
		newWords = append(newWords, words[:i]...)
		newWords = append(newWords, replacement)
		newWords = append(newWords, words[phraseEnd:]...)
		words = newWords
		count++
	}

	if count > 0 {
		return strings.Join(words, " "), count
	}
	return s, 0
}

// corpusAlreadySlotted returns true if the word is a slot marker.
func corpusAlreadySlotted(w string) bool {
	return strings.HasPrefix(w, "[") && strings.HasSuffix(w, "]")
}

// corpusIsStopWord returns true for common words that are capitalized only
// because of sentence position, not because they're entities.
var corpusStopWords = map[string]bool{
	"the": true, "a": true, "an": true, "this": true, "that": true,
	"these": true, "those": true, "it": true, "its": true, "they": true,
	"their": true, "he": true, "she": true, "his": true, "her": true,
	"we": true, "our": true, "my": true, "your": true, "i": true,
	"each": true, "every": true, "some": true, "many": true, "most": true,
	"such": true, "other": true, "several": true, "both": true, "all": true,
	"more": true, "less": true, "much": true, "few": true,
	"however": true, "moreover": true, "furthermore": true, "therefore": true,
	"although": true, "because": true, "since": true, "while": true,
	"despite": true, "during": true, "between": true, "among": true,
	"through": true, "after": true, "before": true, "when": true,
	"where": true, "which": true, "what": true, "who": true, "how": true,
	"not": true, "no": true, "nor": true, "but": true, "or": true,
	"and": true, "as": true, "if": true, "so": true, "yet": true,
	"key": true, "common": true, "modern": true, "major": true,
	"early": true, "later": true, "first": true, "second": true,
}

func corpusIsStopWord(w string) bool {
	return corpusStopWords[w]
}

// corpusClassifyFunction determines the discourse function of a sentence.
func corpusClassifyFunction(sentence string) string {
	lower := strings.ToLower(sentence)
	for _, df := range corpusDiscourseFunctions {
		for _, signal := range df.Signals {
			if strings.Contains(lower, signal) {
				return df.Name
			}
		}
	}
	return "definition" // default
}

// -----------------------------------------------------------------------
// Generation — select patterns, fill slots, smooth transitions
// -----------------------------------------------------------------------

// GenerateFromFacts generates flowing prose from a subject and its facts.
// It groups facts by relation type, selects matching sentence patterns,
// fills them with fact data, and uses the bigram model to smooth transitions.
func (c *CorpusNLG) GenerateFromFacts(subject string, facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	c.mu.Lock()
	c.genCounter++
	c.mu.Unlock()

	groups := c.corpusGroupFactsByFunction(facts)

	c.mu.RLock()
	defer c.mu.RUnlock()

	var sentences []string
	prevWord := ""

	for _, group := range groups {
		function := group.function
		groupFacts := group.facts

		// Try to find a matching pattern.
		pattern, patIdx := c.selectPattern(function, groupFacts)

		var sentence string
		if pattern != nil {
			slots := c.buildSlotMap(subject, groupFacts)
			sentence = c.FillPattern(*pattern, slots)

			// Record pattern usage for anti-repetition.
			if patIdx >= 0 {
				c.mu.RUnlock()
				c.mu.Lock()
				c.lastUsed[patIdx] = c.genCounter
				c.mu.Unlock()
				c.mu.RLock()
			}
		} else {
			// Fallback: generate directly from facts.
			sentence = c.corpusFallbackGenerate(subject, groupFacts)
		}

		if sentence == "" {
			continue
		}

		// Add transition from previous sentence if needed.
		if len(sentences) > 0 && prevWord != "" {
			transition := c.GenerateTransition(prevWord, function)
			if transition != "" {
				sentence = transition + " " + corpusLowerFirst(sentence)
			}
		}

		sentences = append(sentences, sentence)

		// Track last word for transitions.
		words := strings.Fields(sentence)
		if len(words) > 0 {
			prevWord = strings.ToLower(strings.TrimRight(words[len(words)-1], ".,;:!?)"))
		}
	}

	if len(sentences) == 0 {
		return ""
	}

	result := strings.Join(sentences, " ")
	result = corpusPronominalize(result, subject)
	return result
}

// corpusFactFunctionGroup groups facts under a discourse function.
type corpusFactFunctionGroup struct {
	function string
	facts    []edgeFact
}

// corpusGroupFactsByFunction groups facts into discourse function categories.
func (c *CorpusNLG) corpusGroupFactsByFunction(facts []edgeFact) []corpusFactFunctionGroup {
	order := []string{"definition", "origin", "property", "usage", "relation", "evaluation"}
	grouped := make(map[string][]edgeFact)

	for _, f := range facts {
		fn := corpusRelToFunction(f.Relation)
		grouped[fn] = append(grouped[fn], f)
	}

	var groups []corpusFactFunctionGroup
	for _, fn := range order {
		if fs, ok := grouped[fn]; ok {
			groups = append(groups, corpusFactFunctionGroup{function: fn, facts: fs})
		}
	}
	return groups
}

// corpusRelToFunction maps a RelType to a discourse function.
func corpusRelToFunction(r RelType) string {
	switch r {
	case RelIsA, RelDescribedAs, RelKnownFor:
		return "definition"
	case RelCreatedBy, RelFoundedBy, RelFoundedIn:
		return "origin"
	case RelHas, RelOffers:
		return "property"
	case RelUsedFor:
		return "usage"
	case RelRelatedTo, RelPartOf, RelSimilarTo, RelLocatedIn:
		return "relation"
	default:
		return "definition"
	}
}

// selectPattern finds the best matching pattern for a function and facts.
// Returns nil if no suitable pattern is found.
func (c *CorpusNLG) selectPattern(function string, facts []edgeFact) (*MinedPattern, int) {
	indices, ok := c.byFunction[function]
	if !ok || len(indices) == 0 {
		return nil, -1
	}

	type scored struct {
		idx   int
		score float64
	}

	var candidates []scored
	for _, idx := range indices {
		pat := c.patterns[idx]
		score := c.ScorePattern(pat, facts)
		if score > 0 {
			candidates = append(candidates, scored{idx: idx, score: score})
		}
	}

	if len(candidates) == 0 {
		return nil, -1
	}

	// Sort by score descending, take top 10.
	for i := 0; i < len(candidates)-1; i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].score > candidates[i].score {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}
	if len(candidates) > 10 {
		candidates = candidates[:10]
	}

	// Weighted random selection.
	totalScore := 0.0
	for _, cand := range candidates {
		totalScore += cand.score
	}
	r := rand.Float64() * totalScore
	cumulative := 0.0
	for _, cand := range candidates {
		cumulative += cand.score
		if r <= cumulative {
			pat := c.patterns[cand.idx]
			return &pat, cand.idx
		}
	}

	pat := c.patterns[candidates[0].idx]
	return &pat, candidates[0].idx
}

// ScorePattern scores how well a pattern matches the current facts.
func (c *CorpusNLG) ScorePattern(pattern MinedPattern, facts []edgeFact) float64 {
	score := pattern.Quality

	// Prefer patterns whose slot count is close to the number of facts.
	slotDiff := corpusAbs(pattern.SlotCount - len(facts) - 1) // -1 for subject
	if slotDiff == 0 {
		score += 2.0
	} else {
		score += 1.0 / float64(slotDiff+1)
	}

	// Prefer medium-length patterns (more natural).
	if pattern.WordCount >= 8 && pattern.WordCount <= 25 {
		score += 1.0
	}

	// Anti-repetition: penalize recently used patterns.
	for idx, pat := range c.patterns {
		if pat.Structure == pattern.Structure {
			if lastGen, used := c.lastUsed[idx]; used {
				recency := c.genCounter - lastGen
				if recency < 3 {
					score -= 3.0 / float64(recency+1)
				}
			}
			break
		}
	}

	if score < 0 {
		score = 0.01
	}

	return score
}

// buildSlotMap creates a mapping from slot names to values based on facts.
func (c *CorpusNLG) buildSlotMap(subject string, facts []edgeFact) map[string]string {
	slots := map[string]string{
		"SUBJECT": subject,
	}

	for _, f := range facts {
		switch f.Relation {
		case RelIsA, RelDescribedAs, RelKnownFor:
			if _, ok := slots["CATEGORY"]; !ok {
				slots["CATEGORY"] = f.Object
			} else if _, ok := slots["OBJECT"]; !ok {
				slots["OBJECT"] = f.Object
			}
		case RelCreatedBy, RelFoundedBy:
			slots["PERSON"] = f.Object
		case RelFoundedIn:
			slots["DATE"] = f.Object
		case RelLocatedIn:
			slots["PLACE"] = f.Object
		case RelHas, RelOffers, RelUsedFor, RelRelatedTo, RelPartOf, RelSimilarTo:
			if _, ok := slots["OBJECT"]; !ok {
				slots["OBJECT"] = f.Object
			}
		default:
			if _, ok := slots["OBJECT"]; !ok {
				slots["OBJECT"] = f.Object
			}
		}
	}

	return slots
}

// FillPattern replaces [SLOT] markers in a pattern with actual values.
func (c *CorpusNLG) FillPattern(pattern MinedPattern, slots map[string]string) string {
	result := pattern.Structure

	slotTypes := []string{"SUBJECT", "CATEGORY", "OBJECT", "PERSON", "DATE", "PLACE"}
	for _, slot := range slotTypes {
		tag := "[" + slot + "]"
		if val, ok := slots[slot]; ok {
			result = strings.Replace(result, tag, val, 1)
		}
	}

	// Clean up any unfilled slots by removing them gracefully.
	for _, slot := range slotTypes {
		tag := "[" + slot + "]"
		result = strings.ReplaceAll(result, tag, "")
	}

	// Clean up double spaces and trailing punctuation issues.
	result = strings.Join(strings.Fields(result), " ")
	result = strings.TrimSpace(result)

	// Ensure it ends with a period.
	if len(result) > 0 && !strings.HasSuffix(result, ".") && !strings.HasSuffix(result, "!") && !strings.HasSuffix(result, "?") {
		result += "."
	}

	// Capitalize first letter.
	if len(result) > 0 {
		runes := []rune(result)
		runes[0] = unicode.ToUpper(runes[0])
		result = string(runes)
	}

	return result
}

// corpusFallbackGenerate creates a sentence directly from facts when no
// pattern matches.
func (c *CorpusNLG) corpusFallbackGenerate(subject string, facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	f := facts[0]
	switch f.Relation {
	case RelIsA:
		return subject + " is " + f.Object + "."
	case RelCreatedBy:
		return subject + " was created by " + f.Object + "."
	case RelFoundedBy:
		return subject + " was founded by " + f.Object + "."
	case RelFoundedIn:
		return subject + " was established in " + f.Object + "."
	case RelLocatedIn:
		return subject + " is located in " + f.Object + "."
	case RelHas:
		return subject + " features " + f.Object + "."
	case RelUsedFor:
		return subject + " is used for " + f.Object + "."
	case RelPartOf:
		return subject + " is part of " + f.Object + "."
	case RelRelatedTo:
		return subject + " is related to " + f.Object + "."
	default:
		return subject + " is associated with " + f.Object + "."
	}
}

// -----------------------------------------------------------------------
// Bigram model — transition and opener generation
// -----------------------------------------------------------------------

// BigramNext picks the next word probabilistically from bigram counts.
// Uses weighted random selection where probability = count / total.
// Falls back to Laplace smoothing for unknown words.
func (c *CorpusNLG) BigramNext(word string) string {
	w := strings.ToLower(word)
	nexts, ok := c.bigrams[w]
	if !ok || len(nexts) == 0 {
		// Laplace smoothing fallback: return a common word.
		common := []string{"the", "a", "and", "of", "in", "is", "to", "that", "for", "with"}
		return common[rand.Intn(len(common))]
	}

	total := c.totalBigrams[w]
	if total == 0 {
		return "the"
	}

	// Laplace smoothing: add 1 to all counts.
	vocabSize := len(nexts)
	smoothedTotal := total + vocabSize

	r := rand.Intn(smoothedTotal)
	cumulative := 0
	for next, count := range nexts {
		cumulative += count + 1 // +1 for Laplace smoothing
		if r < cumulative {
			return next
		}
	}

	// Fallback: return most frequent.
	bestWord := ""
	bestCount := 0
	for next, count := range nexts {
		if count > bestCount {
			bestWord = next
			bestCount = count
		}
	}
	return bestWord
}

// GenerateTransition uses the bigram model to generate a natural 2-5 word
// transition phrase. Starts from prevWord and walks the bigram chain,
// biased toward words found in the target function's sentences.
func (c *CorpusNLG) GenerateTransition(prevWord string, targetFunction string) string {
	// Transition connectors that bridge naturally between sentences.
	transitions := map[string][]string{
		"definition": {"notably,", "in particular,", "specifically,"},
		"origin":     {"originally", "historically,"},
		"property":   {"additionally,", "furthermore,"},
		"usage":      {"in practice,", "practically,"},
		"relation":   {"relatedly,", "similarly,"},
		"evaluation": {"significantly,", "importantly,"},
	}

	// First try bigram walk.
	word := strings.ToLower(prevWord)
	var chain []string

	length := 2 + rand.Intn(3) // 2-4 words
	for i := 0; i < length; i++ {
		next := c.BigramNext(word)
		if next == "" {
			break
		}
		chain = append(chain, next)

		// Stop at a natural boundary.
		if corpusIsTransitionEnd(next) && i >= 1 {
			break
		}
		word = next
	}

	if len(chain) >= 2 {
		result := strings.Join(chain, " ")
		runes := []rune(result)
		runes[0] = unicode.ToUpper(runes[0])
		return string(runes)
	}

	// Fallback to pre-selected transitions.
	if ts, ok := transitions[targetFunction]; ok {
		return ts[rand.Intn(len(ts))]
	}

	return ""
}

// corpusIsTransitionEnd returns true for words that naturally end a transition.
func corpusIsTransitionEnd(w string) bool {
	ends := map[string]bool{
		"the": true, "a": true, "an": true, "this": true, "its": true,
		"their": true, "these": true, "that": true,
	}
	return ends[w]
}

// GenerateOpener generates a natural sentence opener for a discourse
// function by walking bigrams from common function-specific seed words.
func (c *CorpusNLG) GenerateOpener(function string) string {
	seeds, ok := corpusOpenerSeeds[function]
	if !ok {
		seeds = corpusOpenerSeeds["definition"]
	}

	// Pick a random seed.
	seed := seeds[rand.Intn(len(seeds))]

	// Walk 2-4 words from the seed.
	var chain []string
	chain = append(chain, seed)
	word := seed
	length := 2 + rand.Intn(3) // walk 2-4 more words

	for i := 0; i < length; i++ {
		next := c.BigramNext(word)
		if next == "" {
			break
		}
		chain = append(chain, next)
		word = next
	}

	if len(chain) < 2 {
		return seed
	}

	result := strings.Join(chain, " ")
	runes := []rune(result)
	runes[0] = unicode.ToUpper(runes[0])
	return string(runes)
}

// RecordSuccess boosts a pattern's quality score when the response
// gets positive implicit feedback (user continues the conversation).
func (c *CorpusNLG) RecordSuccess(patternIdx int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if patternIdx >= 0 && patternIdx < len(c.patterns) {
		c.patterns[patternIdx].Quality += 0.1
	}
}

// -----------------------------------------------------------------------
// Pronominalization — vary subject references
// -----------------------------------------------------------------------

// corpusPronominalize replaces repeated subject mentions with pronouns and
// definite descriptions to create more natural prose.
func corpusPronominalize(text string, subject string) string {
	if subject == "" {
		return text
	}

	// Split into sentences.
	parts := strings.Split(text, ". ")
	if len(parts) <= 1 {
		return text
	}

	// References to alternate: pronoun, definite description.
	refs := []string{"it", "this", "the " + corpusGuessCategory(subject)}

	for i := 1; i < len(parts); i++ {
		if !strings.Contains(parts[i], subject) {
			continue
		}

		if strings.HasPrefix(parts[i], subject) {
			ref := refs[i%len(refs)]
			runes := []rune(ref)
			runes[0] = unicode.ToUpper(runes[0])
			ref = string(runes)
			parts[i] = strings.Replace(parts[i], subject, ref, 1)
		}
	}

	return strings.Join(parts, ". ")
}

// corpusGuessCategory returns a generic noun category for pronominalization.
func corpusGuessCategory(subject string) string {
	lower := strings.ToLower(subject)

	techTerms := []string{"algorithm", "system", "framework", "language", "protocol",
		"theory", "engine", "model", "architecture", "platform"}
	for _, t := range techTerms {
		if strings.Contains(lower, t) {
			return t
		}
	}

	words := strings.Fields(subject)
	if len(words) > 1 {
		return strings.ToLower(words[len(words)-1])
	}

	return "subject"
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

func corpusLowerFirst(s string) string {
	if s == "" {
		return s
	}
	runes := []rune(s)
	runes[0] = unicode.ToLower(runes[0])
	return string(runes)
}

func corpusAbs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// PatternCount returns the number of mined patterns.
func (c *CorpusNLG) PatternCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.patterns)
}

// BigramCount returns the number of unique words in the bigram model.
func (c *CorpusNLG) BigramCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.bigrams)
}

// VocabSize returns the total vocabulary size.
func (c *CorpusNLG) VocabSize() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.vocab)
}

// Patterns returns a copy of all mined patterns.
func (c *CorpusNLG) Patterns() []MinedPattern {
	c.mu.RLock()
	defer c.mu.RUnlock()
	cp := make([]MinedPattern, len(c.patterns))
	copy(cp, c.patterns)
	return cp
}

// FunctionCounts returns the number of patterns per discourse function.
func (c *CorpusNLG) FunctionCounts() map[string]int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	counts := make(map[string]int)
	for fn, indices := range c.byFunction {
		counts[fn] = len(indices)
	}
	return counts
}
