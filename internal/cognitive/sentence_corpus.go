package cognitive

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"sync"
	"time"
	"unicode"
)

// -----------------------------------------------------------------------
// Sentence Corpus — Layer 2 of the generative architecture.
//
// Instead of generating sentences from templates ("X is a Y"), we
// retrieve REAL human-written sentences from Wikipedia and adapt them
// by swapping entities. The corpus IS the model.
//
// Example:
//   Fact:      (Einstein, born_in, Ulm)
//   Corpus:    "Marie Curie was born in Warsaw, Poland in 1867."
//   Adapted:   "Albert Einstein was born in Ulm."
//
// Every word in the output was written by a real human. Nous just found
// the right words and swapped in the right entities.
// -----------------------------------------------------------------------

// SentenceExemplar is a real human-written sentence paired with the
// triple it expresses. Used for retrieval-based generation.
type SentenceExemplar struct {
	Sentence string  `json:"s"`
	Subject  string  `json:"sub"`
	Object   string  `json:"obj"`
	Relation RelType `json:"r"`
}

// SentenceCorpus holds exemplar sentences grouped by relation type.
type SentenceCorpus struct {
	mu        sync.RWMutex
	exemplars map[RelType][]SentenceExemplar
	totalSize int
}

// NewSentenceCorpus creates an empty corpus.
func NewSentenceCorpus() *SentenceCorpus {
	return &SentenceCorpus{
		exemplars: make(map[RelType][]SentenceExemplar),
	}
}

// Add inserts an exemplar into the corpus.
func (sc *SentenceCorpus) Add(ex SentenceExemplar) {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	sc.exemplars[ex.Relation] = append(sc.exemplars[ex.Relation], ex)
	sc.totalSize++
}

// Size returns the total number of exemplars.
func (sc *SentenceCorpus) Size() int {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	return sc.totalSize
}

// RelationCounts returns the number of exemplars per relation type.
func (sc *SentenceCorpus) RelationCounts() map[RelType]int {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	counts := make(map[RelType]int, len(sc.exemplars))
	for rel, exs := range sc.exemplars {
		counts[rel] = len(exs)
	}
	return counts
}

// -----------------------------------------------------------------------
// Persistence — JSON lines format for simplicity and streamability.
// -----------------------------------------------------------------------

// corpusEntry is the on-disk format.
type corpusEntry struct {
	S   string `json:"s"`   // sentence
	Sub string `json:"sub"` // subject
	Obj string `json:"obj"` // object
	R   string `json:"r"`   // relation string
}

// Save writes the corpus to a JSON file.
func (sc *SentenceCorpus) Save(path string) error {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	var entries []corpusEntry
	for _, exs := range sc.exemplars {
		for _, ex := range exs {
			entries = append(entries, corpusEntry{
				S:   ex.Sentence,
				Sub: ex.Subject,
				Obj: ex.Object,
				R:   relTypeToString(ex.Relation),
			})
		}
	}

	data, err := json.Marshal(entries)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// Load reads the corpus from a JSON file.
func (sc *SentenceCorpus) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var entries []corpusEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return err
	}

	sc.mu.Lock()
	defer sc.mu.Unlock()

	sc.exemplars = make(map[RelType][]SentenceExemplar)
	sc.totalSize = 0

	for _, e := range entries {
		rel := stringToRelType(e.R)
		ex := SentenceExemplar{
			Sentence: e.S,
			Subject:  e.Sub,
			Object:   e.Obj,
			Relation: rel,
		}
		sc.exemplars[rel] = append(sc.exemplars[rel], ex)
		sc.totalSize++
	}
	return nil
}

// -----------------------------------------------------------------------
// Retrieval — find the best exemplar for a target fact.
// -----------------------------------------------------------------------

// RetrieveSentence finds the best matching exemplar for a fact triple
// and returns the adapted sentence with entities swapped.
// Returns "" if no suitable exemplar is found.
func (sc *SentenceCorpus) RetrieveSentence(subject string, rel RelType, object string) string {
	sc.mu.RLock()
	candidates := sc.exemplars[rel]
	if len(candidates) == 0 {
		sc.mu.RUnlock()
		return ""
	}

	// Score all candidates and pick the best.
	bestScore := -100.0
	bestIdx := -1
	subjectLower := strings.ToLower(subject)

	for i, ex := range candidates {
		// Never retrieve an exemplar about the same entity.
		if strings.EqualFold(ex.Subject, subject) {
			continue
		}
		score := scoreExemplar(ex, subjectLower, object)
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	if bestIdx < 0 {
		sc.mu.RUnlock()
		return ""
	}

	best := candidates[bestIdx]
	sc.mu.RUnlock()

	return adaptSentence(best, subject, object)
}

// RetrieveVaried finds a good exemplar using time-based variation
// so repeated calls get different phrasings.
func (sc *SentenceCorpus) RetrieveVaried(subject string, rel RelType, object string) string {
	sc.mu.RLock()
	candidates := sc.exemplars[rel]
	if len(candidates) == 0 {
		sc.mu.RUnlock()
		return ""
	}

	// Score all candidates.
	type scored struct {
		idx   int
		score float64
	}
	var top []scored
	subjectLower := strings.ToLower(subject)

	for i, ex := range candidates {
		if strings.EqualFold(ex.Subject, subject) {
			continue
		}
		s := scoreExemplar(ex, subjectLower, object)
		if s > -10 {
			top = append(top, scored{i, s})
		}
	}

	if len(top) == 0 {
		sc.mu.RUnlock()
		return ""
	}

	// Sort by score descending, take top 5, pick using time-based index.
	// Simple selection sort for top-5 since we just need the best few.
	for i := 0; i < len(top) && i < 5; i++ {
		best := i
		for j := i + 1; j < len(top); j++ {
			if top[j].score > top[best].score {
				best = j
			}
		}
		top[i], top[best] = top[best], top[best]
		if best != i {
			top[i], top[best] = scored{top[best].idx, top[best].score}, scored{top[i].idx, top[i].score}
		}
	}

	// Cap at top 5.
	if len(top) > 5 {
		top = top[:5]
	}

	// Time-based variation.
	pick := int(time.Now().UnixNano()/1000) % len(top)
	chosen := candidates[top[pick].idx]
	sc.mu.RUnlock()

	return adaptSentence(chosen, subject, object)
}

// scoreExemplar rates how well an exemplar matches a target fact.
func scoreExemplar(ex SentenceExemplar, targetSubjLower, targetObj string) float64 {
	score := 0.0

	// Prefer similar object types (both years, both proper nouns, etc.).
	exObjIsYear := isYearLike(ex.Object)
	targetObjIsYear := isYearLike(targetObj)
	if exObjIsYear == targetObjIsYear {
		score += 2.0
	} else {
		score -= 2.0 // type mismatch penalty
	}

	exObjIsProper := startsUpper(ex.Object)
	targetObjIsProper := startsUpper(targetObj)
	if exObjIsProper == targetObjIsProper {
		score += 1.0
	}

	// Prefer similar sentence lengths (short sentences for simple facts).
	lenRatio := float64(len(ex.Subject)) / math.Max(float64(len(targetSubjLower)), 1)
	if lenRatio > 0.5 && lenRatio < 2.0 {
		score += 1.0
	}

	// Prefer shorter, cleaner sentences (less dangling context to truncate).
	if len(ex.Sentence) < 100 {
		score += 1.0
	} else if len(ex.Sentence) > 180 {
		score -= 1.0
	}

	// Prefer same plurality (singular↔singular, plural↔plural).
	if isLikelyPlural(ex.Subject) == isLikelyPlural(targetSubjLower) {
		score += 1.5
	}

	// Penalize sentences where the object is deeply embedded (hard to adapt).
	objIdx := strings.Index(ex.Sentence, ex.Object)
	if objIdx < 0 {
		score -= 5.0 // can't find object — bad exemplar
	}

	return score
}

// -----------------------------------------------------------------------
// Adaptation — swap entities in a retrieved sentence.
// -----------------------------------------------------------------------

// adaptSentence replaces the exemplar's subject and object with the
// target entities, producing a new sentence grounded in real human text.
func adaptSentence(ex SentenceExemplar, targetSubj, targetObj string) string {
	result := ex.Sentence

	// Replace subject (first occurrence only).
	result = strings.Replace(result, ex.Subject, capitalizeFirst(targetSubj), 1)

	// Replace object (first occurrence only).
	if ex.Object != "" && targetObj != "" {
		result = strings.Replace(result, ex.Object, targetObj, 1)
	}

	// Fix subject-verb agreement: if the original subject was plural
	// but the target is singular (or vice versa), fix "is/are", "was/were", "has/have".
	result = fixAdaptedAgreement(result, targetSubj, ex.Subject)

	// Truncate dangling context: if the original sentence had specific
	// details after the object (like a year or parenthetical), remove
	// them since they belong to the original entity, not ours.
	result = truncateDanglingContext(result, targetObj, ex.Object)

	// Ensure proper ending.
	result = strings.TrimSpace(result)
	if result != "" && !strings.HasSuffix(result, ".") && !strings.HasSuffix(result, "!") && !strings.HasSuffix(result, "?") {
		result += "."
	}

	return result
}

// truncateDanglingContext removes leftover specifics from the original
// entity that don't apply to the target. For example:
//   "X was born in Warsaw, Poland in 1867." adapted for Ulm
//   → "X was born in Ulm." (removes ", Poland in 1867")
func truncateDanglingContext(sentence, targetObj, origObj string) string {
	if targetObj == "" {
		return sentence
	}

	idx := strings.Index(sentence, targetObj)
	if idx < 0 {
		return sentence
	}

	afterObj := idx + len(targetObj)
	if afterObj >= len(sentence) {
		return sentence
	}

	rest := sentence[afterObj:]

	// If the rest starts with a comma followed by content that has
	// numbers or proper nouns (leftover specifics), truncate.
	trimRest := strings.TrimSpace(rest)
	if strings.HasPrefix(trimRest, ",") || strings.HasPrefix(trimRest, "(") {
		// Check if what follows contains year-like numbers or is short enough
		// to be leftover context.
		afterComma := strings.TrimLeft(trimRest, ", (")
		if containsYear(afterComma) || len(afterComma) < 30 {
			// Truncate to just after the target object.
			return strings.TrimSpace(sentence[:afterObj]) + "."
		}
	}

	return sentence
}

// fixAdaptedAgreement corrects is/are, was/were, has/have mismatches
// when swapping between singular and plural subjects.
// Only fixes plural→singular (are→is), since singular→plural is less
// reliable (many proper nouns end in 's' but are singular: Mars, Paris).
func fixAdaptedAgreement(sentence, targetSubj, origSubj string) string {
	origPlural := isLikelyPlural(origSubj)
	targetPlural := isLikelyPlural(targetSubj)

	if origPlural == targetPlural {
		return sentence // no change needed
	}

	// Only fix when the original was clearly plural (lowercase, ends in 's')
	// and the target is clearly singular, or vice versa.
	// Skip if the target is a proper noun (capitalized) — too risky.
	if origPlural && !targetPlural {
		// Plural exemplar → singular target: are→is, were→was, have→has
		sentence = strings.Replace(sentence, " are ", " is ", 1)
		sentence = strings.Replace(sentence, " were ", " was ", 1)
		sentence = strings.Replace(sentence, " have ", " has ", 1)
	}
	// Don't fix singular→plural — too many false positives with proper nouns.
	return sentence
}

// -----------------------------------------------------------------------
// Extraction — build the corpus from Wikipedia articles.
// -----------------------------------------------------------------------

// ArticleToExemplars extracts sentence-triple pairs from a Wikipedia
// article. Each exemplar is a real sentence paired with the triple it
// expresses, suitable for later retrieval and adaptation.
func ArticleToExemplars(title, plainText string) []SentenceExemplar {
	if plainText == "" {
		return nil
	}

	sentences := splitSentences(plainText)
	var exemplars []SentenceExemplar
	seen := make(map[string]bool)

	for _, sent := range sentences {
		sent = strings.TrimSpace(sent)

		// Quality filters.
		if len(sent) < 30 || len(sent) > 200 {
			continue
		}
		if isBoilerplate(sent) {
			continue
		}
		// Must start with a capital letter (proper sentence).
		if len(sent) > 0 && (sent[0] < 'A' || sent[0] > 'Z') {
			continue
		}
		// Add period if missing (splitSentences strips them).
		if !strings.HasSuffix(sent, ".") && !strings.HasSuffix(sent, "!") && !strings.HasSuffix(sent, "?") {
			sent += "."
		}
		// Clean wiki markup remnants.
		if strings.Contains(sent, "]]") || strings.Contains(sent, "[[") {
			continue
		}

		triples := ExtractTriples(sent)
		for _, t := range triples {
			if t.Relation == RelDescribedAs {
				continue // leads are handled by Layer 1
			}

			// Subject and object must appear literally in the sentence
			// for clean entity boundary detection during adaptation.
			if !strings.Contains(sent, t.Subject) {
				continue
			}
			if !strings.Contains(sent, t.Object) {
				continue
			}

			// Skip very short subjects/objects (likely fragments).
			if len(t.Subject) < 2 || len(t.Object) < 2 {
				continue
			}

			// Deduplicate by sentence text.
			key := sent
			if seen[key] {
				continue
			}
			seen[key] = true

			exemplars = append(exemplars, SentenceExemplar{
				Sentence: sent,
				Subject:  t.Subject,
				Object:   t.Object,
				Relation: t.Relation,
			})
		}
	}

	return exemplars
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

func isYearLike(s string) bool {
	s = strings.TrimSpace(s)
	if len(s) != 4 {
		return false
	}
	for _, ch := range s {
		if !unicode.IsDigit(ch) {
			return false
		}
	}
	return s[0] == '1' || s[0] == '2' // 1000-2999
}

func startsUpper(s string) bool {
	if s == "" {
		return false
	}
	return s[0] >= 'A' && s[0] <= 'Z'
}

func containsYear(s string) bool {
	// Simple check: contains a 4-digit number starting with 1 or 2.
	for i := 0; i+3 < len(s); i++ {
		if (s[i] == '1' || s[i] == '2') && s[i+1] >= '0' && s[i+1] <= '9' && s[i+2] >= '0' && s[i+2] <= '9' && s[i+3] >= '0' && s[i+3] <= '9' {
			return true
		}
	}
	return false
}

// stringToRelType converts a relation string back to RelType.
func stringToRelType(s string) RelType {
	switch s {
	case "is_a":
		return RelIsA
	case "located_in":
		return RelLocatedIn
	case "part_of":
		return RelPartOf
	case "created_by":
		return RelCreatedBy
	case "founded_by":
		return RelFoundedBy
	case "founded_in":
		return RelFoundedIn
	case "has":
		return RelHas
	case "offers":
		return RelOffers
	case "used_for":
		return RelUsedFor
	case "related_to":
		return RelRelatedTo
	case "similar_to":
		return RelSimilarTo
	case "causes":
		return RelCauses
	case "contradicts":
		return RelContradicts
	case "follows":
		return RelFollows
	case "prefers":
		return RelPrefers
	case "dislikes":
		return RelDislikes
	case "domain":
		return RelDomain
	case "described_as":
		return RelDescribedAs
	default:
		return RelRelatedTo
	}
}
