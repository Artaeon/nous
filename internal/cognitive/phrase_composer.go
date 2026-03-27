package cognitive

import (
	"encoding/json"
	"os"
	"strings"
	"sync"
	"time"
)

// -----------------------------------------------------------------------
// Phrase Composer — Layer 3 of the generative architecture.
//
// Layer 1: lead paragraphs (verbatim retrieval).
// Layer 2: sentence retrieval + entity swapping.
// Layer 3: phrase-level composition from structural templates.
//
// The insight: every Wikipedia sentence follows a syntactic pattern.
// By decomposing real sentences into reusable TEMPLATES with typed
// slots, we can compose NOVEL sentences by filling templates with new
// facts from the knowledge graph.
//
// Example:
//   Source:    "Marie Curie was a brilliant physicist who discovered radium."
//   Template: "[SUBJECT] was a [MODIFIER] [CATEGORY] who [VERB] [OBJECT]."
//   Slots:    SUBJECT=person, MODIFIER=evaluative, CATEGORY=profession,
//             VERB=past_action, OBJECT=thing
//
//   New fact: (Ada Lovelace, is_a, mathematician)
//   Output:   "Ada Lovelace was a brilliant mathematician who discovered radium."
//
// Every structural word came from a real human sentence. The novelty is
// in the COMBINATION of template + new entities, not in the words.
// -----------------------------------------------------------------------

// PhraseSlotKind describes what kind of content fills a phrase template slot.
// Distinct from SlotType (used by the template inducer for POS tags).
type PhraseSlotKind int

const (
	PhraseSlotSubject  PhraseSlotKind = iota // entity name (person, org, etc.)
	PhraseSlotObject                         // entity or concept
	PhraseSlotModifier                       // adjective, evaluative word
	PhraseSlotCategory                       // profession, type, class
	PhraseSlotVerb                           // action word or phrase
	PhraseSlotLocation                       // place name
	PhraseSlotYear                           // date or year
	PhraseSlotQuantity                       // number or amount
)

var phraseSlotKindNames = [...]string{
	"subject", "object", "modifier", "category",
	"verb", "location", "year", "quantity",
}

func (pk PhraseSlotKind) String() string {
	if int(pk) < len(phraseSlotKindNames) {
		return phraseSlotKindNames[pk]
	}
	return "unknown"
}

func parsePhraseSlotKind(s string) PhraseSlotKind {
	for i, name := range phraseSlotKindNames {
		if name == s {
			return PhraseSlotKind(i)
		}
	}
	return PhraseSlotObject
}

// PhraseSlot describes one fillable position in a phrase template.
type PhraseSlot struct {
	Name     string         `json:"name"`     // SUBJECT, OBJECT, MODIFIER, etc.
	Kind     PhraseSlotKind `json:"kind"`     // what kind of content fills it
	Original string         `json:"original"` // what was in this slot originally
}

// PhraseTemplate is a sentence structure extracted from real text,
// with named slots that can be filled with new content.
type PhraseTemplate struct {
	Pattern  string        `json:"pattern"`  // template with [SLOT] markers
	Slots    []PhraseSlot  `json:"slots"`    // the fillable positions
	Source   string        `json:"source"`   // original sentence
	Function DiscourseFunc `json:"function"` // discourse function
	Relation RelType       `json:"relation"` // what relation this template expresses
	Quality  int           `json:"quality"`  // 0-3, higher = better
}

// PhraseComposer generates novel sentences from templates + facts.
type PhraseComposer struct {
	mu        sync.RWMutex
	templates map[DiscourseFunc][]PhraseTemplate
	byRel     map[RelType][]int // relation -> indices into all
	all       []PhraseTemplate
}

// NewPhraseComposer creates an empty phrase composer.
func NewPhraseComposer() *PhraseComposer {
	return &PhraseComposer{
		templates: make(map[DiscourseFunc][]PhraseTemplate),
		byRel:     make(map[RelType][]int),
	}
}

// Size returns total template count.
func (pc *PhraseComposer) Size() int {
	pc.mu.RLock()
	defer pc.mu.RUnlock()
	return len(pc.all)
}

// AddTemplate adds a template to the composer.
func (pc *PhraseComposer) AddTemplate(t PhraseTemplate) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	idx := len(pc.all)
	pc.all = append(pc.all, t)
	pc.templates[t.Function] = append(pc.templates[t.Function], t)
	pc.byRel[t.Relation] = append(pc.byRel[t.Relation], idx)
}

// -----------------------------------------------------------------------
// Template Decomposition — extract structure from real sentences.
// -----------------------------------------------------------------------

// DecomposeToTemplate takes a real sentence with a known triple and
// extracts the structural template by replacing known entities with
// typed slots. Returns nil if the sentence is unsuitable.
func DecomposeToTemplate(sentence string, subject string, rel RelType, object string) *PhraseTemplate {
	sentence = strings.TrimSpace(sentence)
	if sentence == "" || subject == "" {
		return nil
	}

	// Subject and object must appear literally in the sentence.
	if !strings.Contains(sentence, subject) {
		return nil
	}
	if object != "" && !strings.Contains(sentence, object) {
		return nil
	}

	// Quality filters.
	if len(sentence) < 20 || len(sentence) > 250 {
		return nil
	}

	pattern := sentence
	var slots []PhraseSlot

	// Replace subject with typed slot.
	subjKind := guessSubjectKind(subject)
	pattern = strings.Replace(pattern, subject, "[SUBJECT]", 1)
	slots = append(slots, PhraseSlot{
		Name:     "SUBJECT",
		Kind:     subjKind,
		Original: subject,
	})

	// Replace object with typed slot.
	if object != "" {
		objKind := guessObjectKind(object, rel)
		pattern = strings.Replace(pattern, object, "[OBJECT]", 1)
		slots = append(slots, PhraseSlot{
			Name:     "OBJECT",
			Kind:     objKind,
			Original: object,
		})
	}

	// Extract additional slots from the structural middle.
	pattern, extraSlots := extractPhraseMiddleSlots(pattern, rel)
	slots = append(slots, extraSlots...)

	// Classify discourse function.
	lower := strings.ToLower(sentence)
	fn := classifyTemplateFunction(lower, rel)

	// Score quality.
	quality := scorePhraseTemplateQuality(sentence, pattern, slots)

	if quality < 1 {
		return nil
	}

	return &PhraseTemplate{
		Pattern:  pattern,
		Slots:    slots,
		Source:   sentence,
		Function: fn,
		Relation: rel,
		Quality:  quality,
	}
}

// guessSubjectKind infers what kind of entity the subject is.
func guessSubjectKind(subject string) PhraseSlotKind {
	return PhraseSlotSubject
}

// guessObjectKind infers slot kind from the object and relation.
func guessObjectKind(object string, rel RelType) PhraseSlotKind {
	switch rel {
	case RelLocatedIn:
		return PhraseSlotLocation
	case RelFoundedIn:
		if isYearLike(object) {
			return PhraseSlotYear
		}
		return PhraseSlotLocation
	case RelIsA:
		return PhraseSlotCategory
	default:
		if isYearLike(object) {
			return PhraseSlotYear
		}
		return PhraseSlotObject
	}
}

// extractPhraseMiddleSlots finds additional structured elements in the
// template between [SUBJECT] and [OBJECT] that can be further parameterized.
func extractPhraseMiddleSlots(pattern string, rel RelType) (string, []PhraseSlot) {
	var extra []PhraseSlot

	// Look for evaluative modifiers right before [OBJECT] in is_a patterns.
	// E.g., "[SUBJECT] was a brilliant [OBJECT]" -> extract "brilliant" as MODIFIER.
	if rel == RelIsA || rel == RelDescribedAs {
		modifiers := []string{
			"brilliant", "renowned", "famous", "notable", "prominent",
			"leading", "pioneering", "distinguished", "celebrated",
			"influential", "important", "significant", "major",
			"well-known", "popular", "innovative", "outstanding",
			"exceptional", "remarkable", "accomplished", "eminent",
			"legendary", "iconic", "visionary", "prolific",
		}
		lower := strings.ToLower(pattern)
		for _, mod := range modifiers {
			idx := strings.Index(lower, mod)
			if idx >= 0 {
				// Check it's a standalone word.
				before := idx - 1
				after := idx + len(mod)
				if (before < 0 || pattern[before] == ' ') &&
					(after >= len(pattern) || pattern[after] == ' ') {
					original := pattern[idx : idx+len(mod)]
					pattern = pattern[:idx] + "[MODIFIER]" + pattern[idx+len(mod):]
					extra = append(extra, PhraseSlot{
						Name:     "MODIFIER",
						Kind:     PhraseSlotModifier,
						Original: original,
					})
					break
				}
			}
		}
	}

	return pattern, extra
}

// classifyTemplateFunction determines the discourse function of a template.
func classifyTemplateFunction(lower string, rel RelType) DiscourseFunc {
	switch rel {
	case RelIsA:
		if containsEvalPattern(lower) {
			return DFEvaluates
		}
		return DFDefines
	case RelLocatedIn, RelFoundedIn:
		return DFContext
	case RelCauses:
		return DFConsequence
	case RelSimilarTo:
		return DFCompares
	default:
		if containsCausalPattern(lower) {
			return DFExplainsWhy
		}
		if containsExamplePattern(lower) {
			return DFGivesExample
		}
		if containsComparePattern(lower) {
			return DFCompares
		}
		if containsEvalPattern(lower) {
			return DFEvaluates
		}
		if containsContextPattern(lower) {
			return DFContext
		}
		return DFDescribes
	}
}

// scorePhraseTemplateQuality rates how good a template is for reuse.
func scorePhraseTemplateQuality(sentence, pattern string, slots []PhraseSlot) int {
	q := 0

	hasSubj, hasObj := false, false
	for _, s := range slots {
		if s.Name == "SUBJECT" {
			hasSubj = true
		}
		if s.Name == "OBJECT" {
			hasObj = true
		}
	}
	if !hasSubj {
		return 0
	}

	q++ // base: has subject

	if hasObj {
		q++ // has object — more reusable
	}

	// Ideal length.
	if len(sentence) >= 30 && len(sentence) <= 150 {
		q++
	}

	// Proper ending.
	if strings.HasSuffix(sentence, ".") {
		q++
	}

	// The pattern should have meaningful structural content beyond just slots.
	stripped := pattern
	stripped = strings.ReplaceAll(stripped, "[SUBJECT]", "")
	stripped = strings.ReplaceAll(stripped, "[OBJECT]", "")
	stripped = strings.ReplaceAll(stripped, "[MODIFIER]", "")
	words := strings.Fields(stripped)
	if len(words) >= 3 {
		q++
	}

	// Cap at 3.
	if q > 3 {
		q = 3
	}

	return q
}

// -----------------------------------------------------------------------
// Composition — fill templates with new facts.
// -----------------------------------------------------------------------

// Compose generates a novel sentence from a fact triple and desired
// discourse function. It finds the best matching template and fills it
// with the new entities.
func (pc *PhraseComposer) Compose(subject string, rel RelType, object string, fn DiscourseFunc) string {
	pc.mu.RLock()
	defer pc.mu.RUnlock()

	candidates := pc.findCandidates(subject, rel, object, fn)
	if len(candidates) == 0 {
		return ""
	}

	// Score and rank candidates.
	type scored struct {
		tmpl  PhraseTemplate
		score float64
	}
	var ranked []scored

	for _, t := range candidates {
		s := pc.scoreCandidate(t, subject, rel, object)
		ranked = append(ranked, scored{t, s})
	}

	// Sort by score descending.
	for i := 0; i < len(ranked) && i < 5; i++ {
		best := i
		for j := i + 1; j < len(ranked); j++ {
			if ranked[j].score > ranked[best].score {
				best = j
			}
		}
		if best != i {
			ranked[i], ranked[best] = ranked[best], ranked[i]
		}
	}

	if len(ranked) > 5 {
		ranked = ranked[:5]
	}

	// Time-based variation for variety.
	pick := int(time.Now().UnixNano()/1000) % len(ranked)
	chosen := ranked[pick].tmpl

	return fillPhraseTemplate(chosen, subject, object)
}

// ComposeAll generates multiple sentence options for a fact.
func (pc *PhraseComposer) ComposeAll(subject string, rel RelType, object string, fn DiscourseFunc, limit int) []string {
	pc.mu.RLock()
	defer pc.mu.RUnlock()

	candidates := pc.findCandidates(subject, rel, object, fn)
	if len(candidates) == 0 {
		return nil
	}

	type scored struct {
		tmpl  PhraseTemplate
		score float64
	}
	var ranked []scored
	for _, t := range candidates {
		s := pc.scoreCandidate(t, subject, rel, object)
		ranked = append(ranked, scored{t, s})
	}
	for i := 0; i < len(ranked) && i < limit; i++ {
		best := i
		for j := i + 1; j < len(ranked); j++ {
			if ranked[j].score > ranked[best].score {
				best = j
			}
		}
		if best != i {
			ranked[i], ranked[best] = ranked[best], ranked[i]
		}
	}
	if len(ranked) > limit {
		ranked = ranked[:limit]
	}

	seen := make(map[string]bool)
	var results []string
	for _, r := range ranked {
		s := fillPhraseTemplate(r.tmpl, subject, object)
		if s != "" && !seen[s] {
			seen[s] = true
			results = append(results, s)
		}
	}
	return results
}

// findCandidates retrieves templates that could work for the given fact.
func (pc *PhraseComposer) findCandidates(subject string, rel RelType, object string, fn DiscourseFunc) []PhraseTemplate {
	var candidates []PhraseTemplate

	// Primary: same relation type.
	if indices, ok := pc.byRel[rel]; ok {
		for _, idx := range indices {
			t := pc.all[idx]
			// Never use a template from the same entity.
			if len(t.Slots) > 0 && strings.EqualFold(t.Slots[0].Original, subject) {
				continue
			}
			candidates = append(candidates, t)
		}
	}

	// Secondary: same discourse function if we don't have enough.
	if len(candidates) < 3 {
		for _, t := range pc.templates[fn] {
			if len(t.Slots) > 0 && strings.EqualFold(t.Slots[0].Original, subject) {
				continue
			}
			dup := false
			for _, c := range candidates {
				if c.Source == t.Source {
					dup = true
					break
				}
			}
			if !dup {
				candidates = append(candidates, t)
			}
		}
	}

	return candidates
}

// scoreCandidate rates how well a template fits a target fact.
func (pc *PhraseComposer) scoreCandidate(t PhraseTemplate, subject string, rel RelType, object string) float64 {
	score := 0.0

	// Strong preference for matching relation type.
	if t.Relation == rel {
		score += 5.0
	}

	// Quality bonus.
	score += float64(t.Quality)

	// Slot kind matching.
	for _, slot := range t.Slots {
		switch slot.Name {
		case "SUBJECT":
			origIsProper := startsUpper(slot.Original)
			targetIsProper := startsUpper(subject)
			if origIsProper == targetIsProper {
				score += 1.5
			}
			origWords := len(strings.Fields(slot.Original))
			targetWords := len(strings.Fields(subject))
			if origWords == targetWords {
				score += 0.5
			}
		case "OBJECT":
			origIsYear := isYearLike(slot.Original)
			targetIsYear := isYearLike(object)
			if origIsYear == targetIsYear {
				score += 2.0
			} else {
				score -= 3.0
			}
			origIsProper := startsUpper(slot.Original)
			targetIsProper := startsUpper(object)
			if origIsProper == targetIsProper {
				score += 1.0
			}
		}
	}

	// Prefer shorter, cleaner templates.
	if len(t.Pattern) < 80 {
		score += 1.0
	} else if len(t.Pattern) > 150 {
		score -= 1.0
	}

	return score
}

// fillPhraseTemplate inserts new entities into a template's slots.
func fillPhraseTemplate(t PhraseTemplate, subject, object string) string {
	result := t.Pattern

	result = strings.Replace(result, "[SUBJECT]", subject, 1)

	if object != "" {
		result = strings.Replace(result, "[OBJECT]", object, 1)
	}

	// If there's an unfilled MODIFIER slot, use the original modifier.
	// This preserves natural phrasing — the modifier came from real text.
	for _, slot := range t.Slots {
		if slot.Name == "MODIFIER" && strings.Contains(result, "[MODIFIER]") {
			result = strings.Replace(result, "[MODIFIER]", slot.Original, 1)
		}
	}

	// Clean up any remaining unfilled slots by removing them.
	result = strings.ReplaceAll(result, "[MODIFIER] ", "")
	result = strings.ReplaceAll(result, "[MODIFIER]", "")

	result = collapseSpaces(result)
	result = strings.TrimSpace(result)

	if result != "" && !strings.HasSuffix(result, ".") && !strings.HasSuffix(result, "!") && !strings.HasSuffix(result, "?") {
		result += "."
	}

	return result
}

// collapseSpaces reduces runs of whitespace to a single space.
func collapseSpaces(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	prev := false
	for _, r := range s {
		if r == ' ' || r == '\t' || r == '\n' {
			if !prev {
				b.WriteRune(' ')
			}
			prev = true
		} else {
			b.WriteRune(r)
			prev = false
		}
	}
	return b.String()
}

// -----------------------------------------------------------------------
// Article-level extraction — build templates from Wikipedia articles.
// -----------------------------------------------------------------------

// ExtractTemplatesFromArticle processes a Wikipedia article and extracts
// phrase templates from each sentence that has a recognized triple.
func ExtractTemplatesFromArticle(title, text string) []PhraseTemplate {
	if text == "" {
		return nil
	}

	sentences := splitSentences(text)
	var templates []PhraseTemplate
	seen := make(map[string]bool)

	for _, sent := range sentences {
		sent = strings.TrimSpace(sent)

		if len(sent) < 30 || len(sent) > 200 {
			continue
		}
		if isBoilerplate(sent) {
			continue
		}
		if len(sent) > 0 && (sent[0] < 'A' || sent[0] > 'Z') {
			continue
		}
		if strings.Contains(sent, "]]") || strings.Contains(sent, "[[") {
			continue
		}
		if !strings.HasSuffix(sent, ".") && !strings.HasSuffix(sent, "!") && !strings.HasSuffix(sent, "?") {
			sent += "."
		}

		if seen[sent] {
			continue
		}

		triples := ExtractTriples(sent)
		for _, t := range triples {
			if t.Relation == RelDescribedAs {
				continue
			}
			if !strings.Contains(sent, t.Subject) || !strings.Contains(sent, t.Object) {
				continue
			}
			if len(t.Subject) < 2 || len(t.Object) < 2 {
				continue
			}

			tmpl := DecomposeToTemplate(sent, t.Subject, t.Relation, t.Object)
			if tmpl == nil {
				continue
			}

			seen[sent] = true
			templates = append(templates, *tmpl)
		}
	}

	return templates
}

// -----------------------------------------------------------------------
// Persistence — JSON format matching codebase conventions.
// -----------------------------------------------------------------------

type phraseEntry struct {
	Pattern  string      `json:"p"`
	Slots    []slotEntry `json:"sl"`
	Source   string      `json:"src"`
	Function string      `json:"f"`
	Relation string      `json:"r"`
	Quality  int         `json:"q"`
}

type slotEntry struct {
	Name     string `json:"n"`
	Type     string `json:"t"`
	Original string `json:"o"`
}

// Save writes templates to a JSON file.
func (pc *PhraseComposer) Save(path string) error {
	pc.mu.RLock()
	defer pc.mu.RUnlock()

	var entries []phraseEntry
	for _, t := range pc.all {
		var se []slotEntry
		for _, s := range t.Slots {
			se = append(se, slotEntry{
				Name:     s.Name,
				Type:     s.Kind.String(),
				Original: s.Original,
			})
		}
		entries = append(entries, phraseEntry{
			Pattern:  t.Pattern,
			Slots:    se,
			Source:   t.Source,
			Function: t.Function.String(),
			Relation: string(t.Relation),
			Quality:  t.Quality,
		})
	}

	data, err := json.MarshalIndent(entries, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// Load reads templates from a JSON file.
func (pc *PhraseComposer) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var entries []phraseEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return err
	}

	pc.mu.Lock()
	defer pc.mu.Unlock()

	pc.templates = make(map[DiscourseFunc][]PhraseTemplate)
	pc.byRel = make(map[RelType][]int)
	pc.all = nil

	for _, e := range entries {
		var slots []PhraseSlot
		for _, se := range e.Slots {
			slots = append(slots, PhraseSlot{
				Name:     se.Name,
				Kind:     parsePhraseSlotKind(se.Type),
				Original: se.Original,
			})
		}
		t := PhraseTemplate{
			Pattern:  e.Pattern,
			Slots:    slots,
			Source:   e.Source,
			Function: parseDFString(e.Function),
			Relation: RelType(e.Relation),
			Quality:  e.Quality,
		}
		idx := len(pc.all)
		pc.all = append(pc.all, t)
		pc.templates[t.Function] = append(pc.templates[t.Function], t)
		pc.byRel[t.Relation] = append(pc.byRel[t.Relation], idx)
	}

	return nil
}
