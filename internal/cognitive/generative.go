package cognitive

import (
	"fmt"
	"math/rand"
	"strings"
	"time"
)

// -----------------------------------------------------------------------
// Generative Sentence Planner — builds novel sentences from grammar rules.
//
// Unlike template-based NLG which picks from pools of pre-written phrases,
// this engine composes sentences from smaller building blocks:
//
//   - Phrase Rules: NounPhrase, VerbPhrase, PrepPhrase, AdjPhrase
//   - Clause Patterns: Active, Passive, Cleft, Fronted, Existential
//   - Morphology: verb conjugation, noun inflection, article selection
//   - Constraint Solver: subject-verb agreement, tense consistency
//   - Vocabulary: learned from conversation + built-in seed lexicon
//
// The result: sentences that have never been pre-written by anyone.
// Every output is assembled from first principles, like a human would.
// -----------------------------------------------------------------------

// POS represents a part of speech.
type POS string

const (
	POSNoun POS = "noun"
	POSVerb POS = "verb"
	POSAdj  POS = "adj"
	POSAdv  POS = "adv"
	POSDet  POS = "det"
	POSPrep POS = "prep"
	POSPron POS = "pron"
	POSConj POS = "conj"
)

// Tense for verb conjugation.
type Tense int

const (
	TensePresent Tense = iota
	TensePast
	TenseProgressive
)

// Number for agreement.
type Number int

const (
	Singular Number = iota
	Plural
)

// WordEntry is a lexicon entry with morphological forms.
type WordEntry struct {
	Lemma    string            // base form
	POS      POS               // part of speech
	Forms    map[string]string // "past", "3sg", "plural", "gerund", etc.
	Semantic []string          // semantic tags: "positive", "technology", etc.
}

// ClausePattern is a generative rule for building a clause.
type ClausePattern struct {
	Name         string
	Build        func(g *GenerativeEngine, subj, verb, obj string, t Tense) string
	Weight       float64 // higher = more likely to be selected
	DomainFilter string  // "person", "place", "concept", "event", "" = any
}

// GenerativeEngine builds sentences from grammar rules.
type GenerativeEngine struct {
	lexicon  map[string]*WordEntry // lemma → entry
	patterns []ClausePattern
	rng      *rand.Rand

	// Vocabulary learned from conversation
	learnedNouns []string
	learnedAdjs  []string
	learnedVerbs []string

	// Repetition tracker — reset per article/creative text
	used map[string]int

	// Pronoun variation — reduces "Go... Go... Go..." repetition
	topicMentions int    // how many times topic name was used
	lastNameAt    int    // sentence index of last explicit name use
	sentenceIdx   int    // current sentence counter
	topicCategory string  // "language", "company", etc. from is_a facts
	topicLabel    string  // the actual topic name (for gender detection)
	currentRel    RelType // the relation being rendered (for pattern builders)

	// Optional NLG subsystems — nil means fallback to hardcoded pools
	embeddings *WordEmbeddings
	markov     *MarkovModel
	templates  *TemplateInducer
}

// Embeddings returns the word embeddings for external use (e.g., sentence embedding).
func (g *GenerativeEngine) Embeddings() *WordEmbeddings {
	return g.embeddings
}

// resetTracker clears usage counts for a new generation session.
func (g *GenerativeEngine) resetTracker() {
	g.used = make(map[string]int)
	g.topicMentions = 0
	g.lastNameAt = 0
	g.sentenceIdx = 0
	g.topicCategory = ""
}

// inferCategory extracts a short category label from is_a facts.
// e.g., "programming language" → "language", "philosophy company" → "company"
func inferCategory(facts []edgeFact) string {
	for _, f := range facts {
		if f.Relation == RelIsA {
			obj := strings.ToLower(f.Object)
			obj = strings.TrimPrefix(obj, "a ")
			obj = strings.TrimPrefix(obj, "an ")
			parts := strings.Fields(obj)
			if len(parts) > 0 {
				return parts[len(parts)-1] // last word: "programming language" → "language"
			}
		}
	}
	return ""
}

// isPerson returns true if the topic category suggests a person.
func isPerson(category string) bool {
	switch category {
	case "physicist", "philosopher", "composer", "scientist", "artist",
		"mathematician", "writer", "author", "poet", "inventor",
		"leader", "emperor", "king", "queen", "president", "politician",
		"explorer", "architect", "musician", "painter", "playwright":
		return true
	}
	return false
}

// topicGender returns the gender of the current topic.
func (g *GenerativeEngine) topicGender() Gender {
	if !isPerson(g.topicCategory) {
		return GenderUnknown
	}
	if g.topicLabel != "" {
		return detectGender(g.topicLabel)
	}
	return GenderMale
}

// topicPronoun returns "he"/"she"/"it" based on the topic category and gender.
func (g *GenerativeEngine) topicPronoun() string {
	if isPerson(g.topicCategory) {
		switch g.topicGender() {
		case GenderFemale:
			return "she"
		default:
			return "he"
		}
	}
	return "it"
}

// topicPossessive returns "his"/"her"/"its" based on the topic category and gender.
func (g *GenerativeEngine) topicPossessive() string {
	if isPerson(g.topicCategory) {
		switch g.topicGender() {
		case GenderFemale:
			return "her"
		default:
			return "his"
		}
	}
	return "its"
}

// topicObject returns "him"/"her"/"it" based on the topic category and gender.
func (g *GenerativeEngine) topicObject() string {
	if isPerson(g.topicCategory) {
		switch g.topicGender() {
		case GenderFemale:
			return "her"
		default:
			return "him"
		}
	}
	return "it"
}

// referTo returns the topic name, a pronoun, or an anaphoric reference
// to avoid repeating the topic name in every sentence.
func (g *GenerativeEngine) referTo(topic string) string {
	g.sentenceIdx++
	g.topicMentions++

	// First 2 mentions: always use the name
	if g.topicMentions <= 2 {
		g.lastNameAt = g.sentenceIdx
		return topic
	}

	// If it's been 3+ sentences since last explicit mention, use the name
	gap := g.sentenceIdx - g.lastNameAt
	if gap >= 3 {
		g.lastNameAt = g.sentenceIdx
		return topic
	}

	// 40% chance to use pronoun/anaphor (but not two in a row)
	if gap >= 1 && g.rng.Float64() < 0.4 {
		if isPerson(g.topicCategory) {
			return "he" // default to "he" for historical figures; extend later
		}
		if g.topicCategory != "" {
			return g.pick([]string{"it", "the " + g.topicCategory})
		}
		return "it"
	}

	g.lastNameAt = g.sentenceIdx
	return topic
}

// varyTopicRef replaces the topic name at the start of a sentence
// with a pronoun or anaphoric reference when appropriate.
func (g *GenerativeEngine) varyTopicRef(sentence, topic string) string {
	if !strings.HasPrefix(sentence, topic) {
		return sentence
	}
	// Ensure we match the whole word, not a prefix (e.g., "Go" in "Goroutines")
	rest := sentence[len(topic):]
	if len(rest) > 0 {
		ch := rest[0]
		if ch != ' ' && ch != ',' && ch != '.' && ch != ';' && ch != ':' && ch != '\'' {
			return sentence
		}
	}
	ref := g.referTo(topic)
	if ref == topic {
		return sentence
	}
	replaced := capitalizeFirst(ref) + rest
	return replaced
}

// trackWord records that a word was used.
func (g *GenerativeEngine) trackWord(w string) {
	if g.used != nil {
		g.used[w]++
	}
}

// pickUnique selects from options, deprioritizing already-used words.
// Falls back gracefully: prefers unused > used-once > anything.
func (g *GenerativeEngine) pickUnique(options []string) string {
	if g.used == nil {
		return g.pick(options)
	}

	// Collect unused options
	var unused []string
	for _, o := range options {
		if g.used[o] == 0 {
			unused = append(unused, o)
		}
	}
	if len(unused) > 0 {
		w := unused[g.rng.Intn(len(unused))]
		g.used[w]++
		return w
	}

	// All used at least once — pick least-used
	minCount := g.used[options[0]]
	for _, o := range options[1:] {
		if g.used[o] < minCount {
			minCount = g.used[o]
		}
	}
	var leastUsed []string
	for _, o := range options {
		if g.used[o] == minCount {
			leastUsed = append(leastUsed, o)
		}
	}
	w := leastUsed[g.rng.Intn(len(leastUsed))]
	g.used[w]++
	return w
}

// NewGenerativeEngine creates the generative sentence planner.
func NewGenerativeEngine() *GenerativeEngine {
	g := &GenerativeEngine{
		lexicon: make(map[string]*WordEntry),
		rng:     rand.New(rand.NewSource(time.Now().UnixNano())),
	}
	g.seedLexicon()
	g.registerPatterns()
	return g
}

// SetEmbeddings attaches a word embedding space for semantic word selection.
func (g *GenerativeEngine) SetEmbeddings(we *WordEmbeddings) {
	g.embeddings = we
}

// SetMarkov attaches a Markov model for natural text generation.
func (g *GenerativeEngine) SetMarkov(m *MarkovModel) {
	g.markov = m
}

// SetTemplates attaches a template inducer for learned sentence patterns.
func (g *GenerativeEngine) SetTemplates(ti *TemplateInducer) {
	g.templates = ti
}

// pickSemantic selects from a word pool using embeddings when available.
// Falls back to pickUnique when no embeddings are loaded.
// context is the topic or set of words that define the current meaning.
func (g *GenerativeEngine) pickSemantic(options []string, context string) string {
	if g.embeddings != nil && context != "" && g.embeddings.Size() > 0 {
		// Split context into words for multi-word topics.
		contextWords := strings.Fields(strings.ToLower(context))

		// Find top-k semantically closest candidates.
		topK := g.embeddings.KNearestFromContext(contextWords, options, 5)
		if len(topK) > 0 {
			// Pick from top-k with anti-repetition.
			return g.pickUnique(topK)
		}
	}
	// Fallback: random selection with anti-repetition.
	return g.pickUnique(options)
}

// markovFragment generates a short Markov-chain fragment if the model
// has enough training data. Returns "" if unavailable or too sparse.
func (g *GenerativeEngine) markovFragment(seed string, minWords, maxWords int) string {
	if g.markov == nil || g.markov.Size() < 100 {
		return ""
	}
	return g.markov.GenerateFragment(seed, minWords, maxWords, g.rng)
}

// -----------------------------------------------------------------------
// Sentence Generation — the core creative engine
// -----------------------------------------------------------------------

// Generate builds a novel sentence expressing a relationship.
func (g *GenerativeEngine) Generate(subject string, rel RelType, object string) string {
	verb, prep := g.relationToVerb(rel)
	tense := g.pickTense(rel)

	// For "by" relations (founded_by, created_by), the Object is the agent.
	// Swap so "Stoicera founded_by Raphael" → active: "Raphael founded Stoicera"
	agentSubj, patientObj := subject, object
	if rel == RelFoundedBy || rel == RelCreatedBy {
		agentSubj, patientObj = object, subject
	}

	// For is_a/described_as, the object is a category — add article.
	// "Go is programming language" → "Go is a programming language"
	if rel == RelIsA {
		patientObj = g.articleFor(patientObj) + " " + patientObj
	}

	// For founded_in, prepend "in" to the year: "Go began in 2009"
	if rel == RelFoundedIn {
		patientObj = "in " + patientObj
	}

	// Pick a clause pattern, filtering out incompatible ones for this relation.
	g.currentRel = rel
	pattern := g.pickPatternFor(rel)
	sentence := pattern.Build(g, agentSubj, verb, patientObj, tense)

	// Post-process: capitalize, ensure period
	sentence = capitalizeFirst(strings.TrimSpace(sentence))
	if sentence != "" && !strings.HasSuffix(sentence, ".") &&
		!strings.HasSuffix(sentence, "!") && !strings.HasSuffix(sentence, "?") {
		sentence += "."
	}

	_ = prep // used by some patterns via closure
	return sentence
}

// GenerateCreative builds a more elaborate sentence with style variation.
func (g *GenerativeEngine) GenerateCreative(subject string, rel RelType, object string) string {
	base := g.Generate(subject, rel, object)
	if base == "" {
		return ""
	}

	// 40% chance: add a creative embellishment
	if g.rng.Float64() < 0.4 {
		embellishment := g.embellish(subject, rel, object)
		if embellishment != "" {
			return base + " " + embellishment
		}
	}

	return base
}

// GenerateFromFacts builds a multi-sentence paragraph from multiple facts.
// Uses pattern deduplication to avoid repeating the same sentence structure.
func (g *GenerativeEngine) GenerateFromFacts(facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	// Shuffle for variety
	shuffled := make([]edgeFact, len(facts))
	copy(shuffled, facts)
	g.rng.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})

	var sentences []string
	usedPatterns := make(map[string]int) // pattern name → count
	for i, f := range shuffled {
		// Try up to 3 times to get a non-duplicate pattern
		var sent string
		for attempt := 0; attempt < 3; attempt++ {
			if i == 0 && attempt == 0 {
				sent = g.GenerateCreative(f.Subject, f.Relation, f.Object)
			} else {
				sent = g.Generate(f.Subject, f.Relation, f.Object)
			}
			// Accept on first attempt, or if we got a different structure
			if attempt == 0 || sent != "" {
				break
			}
		}
		if sent != "" {
			sentences = append(sentences, sent)
		}
	}

	_ = usedPatterns // reserved for future pattern tracking

	if len(sentences) == 0 {
		return ""
	}

	// Combine with varied connectors
	return g.combineGenerative(sentences)
}

// -----------------------------------------------------------------------
// Clause Patterns — the grammar rules
// -----------------------------------------------------------------------

func (g *GenerativeEngine) registerPatterns() {
	g.patterns = []ClausePattern{
		{
			Name:   "active",
			Weight: 5.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "passive",
			Weight: 1.5,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				pp := g.pastParticiple(verb)
				return obj + " " + g.conjugate("be", t, Singular) + " " + pp + " by " + subj
			},
		},
		{
			Name:   "cleft",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "It " + g.conjugate("be", t, Singular) + " " + obj + " that " + subj + " " + g.conjugate(verb, t, Singular)
			},
		},
		{
			Name:   "relative-clause",
			Weight: 2.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				// Use "who" for person subjects. For by-relations, the agent (subj)
				// is the person after swap; for other relations, check topic category.
				pronoun := "which"
				if g.currentRel == RelFoundedBy || g.currentRel == RelCreatedBy {
					pronoun = "who"
				} else if isPerson(g.topicCategory) {
					pronoun = "who"
				}
				return subj + ", " + pronoun + " " + g.conjugate(verb, t, Singular) + " " + obj + ", " + g.composeRelContinuation()
			},
		},
		{
			Name:   "appositive-lead",
			Weight: 1.5,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				// "Subject, known for [verb]ing Object, [tail]."
				gerund := g.gerund(verb)
				// Avoid gerund collision: "serving cultivating X" → drop verb, use object directly
				objWords := strings.Fields(strings.ToLower(obj))
				if len(objWords) > 0 && strings.HasSuffix(objWords[0], "ing") {
					intro := g.pick([]string{
						"known for",
						"recognized for",
						"notable for",
					})
					return subj + ", " + intro + " " + obj + ", " + g.composeAppositiveTail()
				}
				intro := g.pick([]string{
					"known for " + gerund,
					"recognized for " + gerund,
					"notable for " + gerund,
				})
				return subj + ", " + intro + " " + obj + ", " + g.composeAppositiveTail()
			},
		},
	}

	// Register extended pattern sets
	RegisterSyntacticPatterns(g)
	RegisterDomainPatterns(g)
	RegisterTonePatterns(g)
	RegisterRhetoricalPatterns(g)
}

// AddPatterns appends additional clause patterns to the engine.
func (g *GenerativeEngine) AddPatterns(patterns []ClausePattern) {
	g.patterns = append(g.patterns, patterns...)
}

// -----------------------------------------------------------------------
// Morphology — verb conjugation and noun inflection
// -----------------------------------------------------------------------

func (g *GenerativeEngine) conjugate(verb string, t Tense, n Number) string {
	// Handle multi-word verbs: "be based in" → conjugate "be", keep "based in"
	parts := strings.Fields(verb)
	if len(parts) > 1 {
		head := parts[0]
		tail := strings.Join(parts[1:], " ")
		return g.conjugate(head, t, n) + " " + tail
	}

	// Check lexicon first
	if entry, ok := g.lexicon[verb]; ok {
		switch t {
		case TensePast:
			if f, ok := entry.Forms["past"]; ok {
				return f
			}
		case TenseProgressive:
			if f, ok := entry.Forms["gerund"]; ok {
				return g.conjugate("be", TensePresent, n) + " " + f
			}
		default:
			if n == Singular {
				if f, ok := entry.Forms["3sg"]; ok {
					return f
				}
			} else if n == Plural {
				if f, ok := entry.Forms["plural"]; ok {
					return f
				}
				return verb // base form for regular verbs
			}
		}
	}

	// Fallback: rule-based conjugation
	switch t {
	case TensePast:
		return g.regularPast(verb)
	case TenseProgressive:
		return g.conjugate("be", TensePresent, n) + " " + g.gerund(verb)
	default:
		if n == Singular {
			return g.regular3sg(verb)
		}
		return verb
	}
}

func (g *GenerativeEngine) regularPast(verb string) string {
	if strings.HasSuffix(verb, "e") {
		return verb + "d"
	}
	if strings.HasSuffix(verb, "y") && len(verb) > 1 {
		prev := verb[len(verb)-2]
		if prev != 'a' && prev != 'e' && prev != 'i' && prev != 'o' && prev != 'u' {
			return verb[:len(verb)-1] + "ied"
		}
	}
	return verb + "ed"
}

func (g *GenerativeEngine) regular3sg(verb string) string {
	if strings.HasSuffix(verb, "s") || strings.HasSuffix(verb, "sh") ||
		strings.HasSuffix(verb, "ch") || strings.HasSuffix(verb, "x") ||
		strings.HasSuffix(verb, "z") {
		return verb + "es"
	}
	if strings.HasSuffix(verb, "y") && len(verb) > 1 {
		prev := verb[len(verb)-2]
		if prev != 'a' && prev != 'e' && prev != 'i' && prev != 'o' && prev != 'u' {
			return verb[:len(verb)-1] + "ies"
		}
	}
	return verb + "s"
}

func (g *GenerativeEngine) pastParticiple(verb string) string {
	// Handle multi-word verbs
	parts := strings.Fields(verb)
	if len(parts) > 1 {
		return g.pastParticiple(parts[0]) + " " + strings.Join(parts[1:], " ")
	}
	if entry, ok := g.lexicon[verb]; ok {
		if f, ok := entry.Forms["pastpart"]; ok {
			return f
		}
	}
	return g.regularPast(verb)
}

func (g *GenerativeEngine) gerund(verb string) string {
	// Handle multi-word verbs
	parts := strings.Fields(verb)
	if len(parts) > 1 {
		return g.gerund(parts[0]) + " " + strings.Join(parts[1:], " ")
	}
	if entry, ok := g.lexicon[verb]; ok {
		if f, ok := entry.Forms["gerund"]; ok {
			return f
		}
	}
	// Rule-based gerund
	if strings.HasSuffix(verb, "e") && !strings.HasSuffix(verb, "ee") {
		return verb[:len(verb)-1] + "ing"
	}
	// Double final consonant for CVC pattern — only for short (1-syllable) words
	// or words ending in stressed CVC (run→running, begin→beginning).
	// Multi-syllable words like "develop", "open", "listen" do NOT double.
	if len(verb) >= 3 {
		last := verb[len(verb)-1]
		secondLast := verb[len(verb)-2]
		if isConsonant(last) && isVowel(secondLast) && isConsonant(verb[len(verb)-3]) {
			if last != 'w' && last != 'x' && last != 'y' {
				// Only double for short words or known stress-final words
				if len(verb) <= 4 || cvcDoubleException(verb) {
					return verb + string(last) + "ing"
				}
			}
		}
	}
	return verb + "ing"
}

func isVowel(b byte) bool {
	return b == 'a' || b == 'e' || b == 'i' || b == 'o' || b == 'u'
}

func isConsonant(b byte) bool {
	return b >= 'a' && b <= 'z' && !isVowel(b)
}

// cvcDoubleException returns true for multi-syllable words that DO double
// their final consonant (stress on final syllable): begin, refer, occur, etc.
func cvcDoubleException(verb string) bool {
	doubles := map[string]bool{
		"begin": true, "refer": true, "occur": true, "prefer": true,
		"permit": true, "admit": true, "commit": true, "submit": true,
		"compel": true, "control": true, "patrol": true, "equip": true,
	}
	return doubles[verb]
}

func (g *GenerativeEngine) articleFor(noun string) string {
	return articleForWord(noun)
}

// articleForWord returns "a" or "an" based on the phonetic onset of the word.
// English a/an is based on *sound*, not spelling:
//   - "a unique" (starts with /juː/ consonant)
//   - "an hour" (silent h, vowel sound)
func articleForWord(word string) string {
	if word == "" {
		return "a"
	}
	low := strings.ToLower(word)

	// Consonant sound despite vowel letter: /juː/ onset
	for _, prefix := range []string{
		"uni", "use", "user", "usa", "util", "eur", "uran",
		"once", "one-", "oneness",
	} {
		if strings.HasPrefix(low, prefix) {
			return "a"
		}
	}

	// Vowel sound despite consonant letter: silent h
	for _, prefix := range []string{
		"hour", "honest", "honor", "honour", "heir", "herb",
	} {
		if strings.HasPrefix(low, prefix) {
			return "an"
		}
	}

	// Default: check first letter
	first := low[0]
	if first == 'a' || first == 'e' || first == 'i' || first == 'o' || first == 'u' {
		return "an"
	}
	return "a"
}

// anAdj picks a random adjective from adjSlots and prepends the correct article.
// Uses pickUnique when a tracker is active (during article generation).
func (g *GenerativeEngine) anAdj() string {
	adj := g.pickUnique(adjSlots)
	return g.articleFor(adj) + " " + adj
}

// comparative returns the comparative form of an adjective.
// Short adjectives get -er suffix; long ones get "more X".
func (g *GenerativeEngine) comparative(adj string) string {
	// Irregular
	irreg := map[string]string{
		"good": "better", "bad": "worse", "far": "further",
		"much": "more", "little": "less",
	}
	if c, ok := irreg[adj]; ok {
		return c
	}
	// Known short adjectives from our pools that take -er
	shortForms := map[string]string{
		"strong": "stronger", "clear": "clearer", "real": "more real",
		"key": "more key", "core": "more core",
	}
	if c, ok := shortForms[adj]; ok {
		return c
	}
	// Multi-syllable adjectives use "more"
	return "more " + adj
}

// -----------------------------------------------------------------------
// Relation → Verb mapping
// -----------------------------------------------------------------------

func (g *GenerativeEngine) relationToVerb(rel RelType) (verb, prep string) {
	switch rel {
	case RelIsA:
		return g.pick([]string{"be", "be"}), ""
	case RelLocatedIn:
		return g.pick([]string{"be based in", "be located in"}), "in"
	case RelFoundedBy:
		return g.pick([]string{"found", "create", "establish"}), "by"
	case RelFoundedIn:
		if isPerson(g.topicCategory) {
			return "be born", "in"
		}
		return g.pick([]string{"begin", "start"}), "in"
	case RelCreatedBy:
		return g.pick([]string{"create", "develop", "build"}), "by"
	case RelUsedFor:
		return g.pick([]string{"be used for", "be designed for"}), "for"
	case RelHas:
		return g.pick([]string{"have", "include", "feature"}), ""
	case RelOffers:
		return g.pick([]string{"offer", "provide"}), ""
	case RelPartOf:
		return g.pick([]string{"belong to", "be part of"}), ""
	case RelDescribedAs:
		return g.pick([]string{"be", "be known as"}), ""
	case RelPrefers:
		return g.pick([]string{"prefer", "favor"}), ""
	case RelDislikes:
		return g.pick([]string{"dislike", "avoid"}), ""
	case RelRelatedTo:
		return g.pick([]string{"relate to", "be connected to"}), ""
	case RelCauses:
		return g.pick([]string{"cause", "lead to"}), ""
	default:
		return "relate to", ""
	}
}

func (g *GenerativeEngine) pickTense(rel RelType) Tense {
	switch rel {
	case RelFoundedBy, RelFoundedIn, RelCreatedBy:
		return TensePast
	default:
		return TensePresent
	}
}

// -----------------------------------------------------------------------
// Creative Embellishments
// -----------------------------------------------------------------------

func (g *GenerativeEngine) embellish(subject string, rel RelType, object string) string {
	// Embellishments removed — factual output only.
	return ""
}

func (g *GenerativeEngine) pickSemanticAdj(object string) string {
	// Use embeddings if available — finds contextually appropriate adjectives.
	if g.embeddings != nil && g.embeddings.Size() > 0 {
		allAdjs := append([]string{}, adjSlots...)
		allAdjs = append(allAdjs, g.learnedAdjs...)
		result := g.pickSemantic(allAdjs, object)
		if result != "" {
			return result
		}
	}

	// Fallback: domain-specific hardcoded pools.
	lower := strings.ToLower(object)
	if strings.Contains(lower, "language") || strings.Contains(lower, "programming") {
		return g.pick([]string{"versatile", "powerful", "expressive", "modern", "robust"})
	}
	if strings.Contains(lower, "company") || strings.Contains(lower, "organization") {
		return g.pick([]string{"notable", "innovative", "growing", "distinctive", "emerging"})
	}
	if strings.Contains(lower, "philosophy") || strings.Contains(lower, "science") {
		return g.pick([]string{"profound", "ancient", "enduring", "practical", "transformative"})
	}
	// 50% chance to return nothing — not every noun needs an adjective
	if g.rng.Float64() < 0.5 {
		return ""
	}
	return g.pick([]string{"notable", "interesting", "significant", "distinct"})
}

// -----------------------------------------------------------------------
// Sentence Combination
// -----------------------------------------------------------------------

func (g *GenerativeEngine) combineGenerative(sentences []string) string {
	if len(sentences) == 1 {
		return sentences[0]
	}

	strategy := g.rng.Intn(3)
	switch strategy {
	case 0: // Clean paragraph
		return strings.Join(sentences, " ")
	case 1: // Connector-linked: use at most 2 connectors to avoid stacking
		var b strings.Builder
		b.WriteString(sentences[0])
		connCount := 0
		for i := 1; i < len(sentences); i++ {
			if connCount < 2 {
				conn := g.composeConnector()
				b.WriteString(" " + conn + " " + safeLowerFirst(sentences[i]))
				connCount++
			} else {
				b.WriteString(" " + sentences[i])
			}
		}
		return b.String()
	case 2: // Mixed: first standalone, rest mixed
		var b strings.Builder
		b.WriteString(sentences[0])
		connCount := 0
		for i := 1; i < len(sentences); i++ {
			b.WriteString(" ")
			if g.rng.Float64() < 0.4 && connCount < 2 {
				conn := g.composeConnector()
				b.WriteString(conn + " " + safeLowerFirst(sentences[i]))
				connCount++
			} else {
				b.WriteString(sentences[i])
			}
		}
		return b.String()
	}
	return strings.Join(sentences, " ")
}

// stripArticle removes a leading "a ", "an ", or "the " from a noun phrase.
func stripArticle(s string) string {
	for _, art := range []string{"an ", "a ", "the "} {
		if strings.HasPrefix(s, art) {
			return s[len(art):]
		}
	}
	return s
}

// topicInContext returns a topic with the correct article/preposition context.
// Prevents "the the Nile" or "of the the internet".
//
//	topicInContext("the Nile", "the") → "the Nile" (not "the the Nile")
//	topicInContext("Python", "the")   → "the Python"
//	topicInContext("the internet", "of the") → "of the internet"
func topicInContext(topic, prefix string) string {
	lower := strings.ToLower(topic)
	prefixLower := strings.ToLower(prefix)

	// Prevent doubled "the": "of the" + "the Renaissance" → "of the Renaissance"
	if strings.HasPrefix(lower, "the ") {
		if strings.HasSuffix(prefixLower, " the") {
			// "of the" + "the X" → "of the X" (drop "the " from topic)
			return prefix + " " + topic[4:]
		}
		if strings.HasSuffix(prefixLower, "the ") {
			// "the " + "the X" → "the X" (drop "the " from topic)
			return prefix + topic[4:]
		}
	}

	return prefix + " " + topic
}

// safeLowerFirst lowercases the first letter ONLY if it's not a proper noun.
// Proper nouns: if first word is <=1 char or contains uppercase after position 0,
// or is a known entity, keep it uppercase.
func safeLowerFirst(s string) string {
	if s == "" {
		return s
	}
	words := strings.Fields(s)
	if len(words) == 0 {
		return s
	}
	first := words[0]
	// Don't lowercase: short words that look like names/acronyms,
	// or words like "It", "There", "When" (sentence starters that ARE lowercase-safe),
	// but DO keep proper nouns uppercase.
	starters := map[string]bool{
		"It": true, "There": true, "When": true, "What": true,
		"The": true, "A": true, "An": true, "This": true, "That": true,
	}
	if starters[first] {
		return lowerFirst(s)
	}
	// If the first word starts uppercase, assume it's a proper noun — keep it
	return s
}

// -----------------------------------------------------------------------
// Pattern Selection
// -----------------------------------------------------------------------

// pickPatternFor selects a clause pattern, excluding those incompatible with
// the given relation type.
func (g *GenerativeEngine) pickPatternFor(rel RelType) ClausePattern {
	copular := rel == RelIsA || rel == RelDescribedAs
	locative := rel == RelLocatedIn
	possessive := rel == RelHas || rel == RelOffers
	relational := rel == RelRelatedTo || rel == RelCauses
	temporal := rel == RelFoundedIn

	candidates := make([]ClausePattern, 0, len(g.patterns))
	for _, p := range g.patterns {
		// Copular verbs: no passive or appositive
		if copular && (p.Name == "passive" || p.Name == "appositive-lead") {
			continue
		}
		// described_as: no cleft
		if rel == RelDescribedAs && p.Name == "cleft" {
			continue
		}
		// Locative/possessive: no passive
		if (locative || possessive) && p.Name == "passive" {
			continue
		}
		// Relational/causal: no appositive or passive
		if relational && (p.Name == "appositive-lead" || p.Name == "passive") {
			continue
		}
		// Temporal: only active and cleft work
		if temporal && (p.Name == "passive" || p.Name == "appositive-lead" || p.Name == "relative-clause") {
			continue
		}
		candidates = append(candidates, p)
	}
	if len(candidates) == 0 {
		candidates = g.patterns
	}

	// Compute effective weights with domain filter adjustments
	var totalWeight float64
	effectiveWeights := make([]float64, len(candidates))
	for i, p := range candidates {
		w := p.Weight
		if p.DomainFilter != "" && g.topicCategory != "" {
			domain := g.domainForCategory()
			if p.DomainFilter == domain {
				w *= 1.5 // boost matching domain
			} else {
				w *= 0.2 // reduce non-matching domain by 80%
			}
		}
		effectiveWeights[i] = w
		totalWeight += w
	}
	r := g.rng.Float64() * totalWeight
	var cumulative float64
	for i, p := range candidates {
		cumulative += effectiveWeights[i]
		if r <= cumulative {
			return p
		}
	}
	return candidates[0]
}

// domainForCategory maps topicCategory to a domain filter string.
func (g *GenerativeEngine) domainForCategory() string {
	if isPerson(g.topicCategory) {
		return "person"
	}
	switch g.topicCategory {
	case "city", "country", "region", "town", "village", "continent",
		"state", "province", "territory", "island", "mountain", "river",
		"location", "place", "area", "district", "neighborhood":
		return "place"
	case "event", "war", "battle", "revolution", "movement", "incident",
		"disaster", "ceremony", "festival", "election", "crisis":
		return "event"
	default:
		return "concept"
	}
}

func (g *GenerativeEngine) pick(options []string) string {
	if len(options) == 0 {
		return ""
	}
	return options[g.rng.Intn(len(options))]
}

// -----------------------------------------------------------------------
// Lexicon — seed vocabulary with irregular forms
// -----------------------------------------------------------------------

func (g *GenerativeEngine) seedLexicon() {
	irregulars := []WordEntry{
		{Lemma: "be", POS: POSVerb, Forms: map[string]string{
			"3sg": "is", "past": "was", "pastpart": "been", "gerund": "being",
			"1sg": "am", "plural": "are",
		}},
		{Lemma: "have", POS: POSVerb, Forms: map[string]string{
			"3sg": "has", "past": "had", "pastpart": "had", "gerund": "having",
		}},
		{Lemma: "do", POS: POSVerb, Forms: map[string]string{
			"3sg": "does", "past": "did", "pastpart": "done", "gerund": "doing",
		}},
		{Lemma: "make", POS: POSVerb, Forms: map[string]string{
			"3sg": "makes", "past": "made", "pastpart": "made", "gerund": "making",
		}},
		{Lemma: "go", POS: POSVerb, Forms: map[string]string{
			"3sg": "goes", "past": "went", "pastpart": "gone", "gerund": "going",
		}},
		{Lemma: "give", POS: POSVerb, Forms: map[string]string{
			"3sg": "gives", "past": "gave", "pastpart": "given", "gerund": "giving",
		}},
		{Lemma: "take", POS: POSVerb, Forms: map[string]string{
			"3sg": "takes", "past": "took", "pastpart": "taken", "gerund": "taking",
		}},
		{Lemma: "come", POS: POSVerb, Forms: map[string]string{
			"3sg": "comes", "past": "came", "pastpart": "come", "gerund": "coming",
		}},
		{Lemma: "see", POS: POSVerb, Forms: map[string]string{
			"3sg": "sees", "past": "saw", "pastpart": "seen", "gerund": "seeing",
		}},
		{Lemma: "know", POS: POSVerb, Forms: map[string]string{
			"3sg": "knows", "past": "knew", "pastpart": "known", "gerund": "knowing",
		}},
		{Lemma: "grow", POS: POSVerb, Forms: map[string]string{
			"3sg": "grows", "past": "grew", "pastpart": "grown", "gerund": "growing",
		}},
		{Lemma: "think", POS: POSVerb, Forms: map[string]string{
			"3sg": "thinks", "past": "thought", "pastpart": "thought", "gerund": "thinking",
		}},
		{Lemma: "find", POS: POSVerb, Forms: map[string]string{
			"3sg": "finds", "past": "found", "pastpart": "found", "gerund": "finding",
		}},
		{Lemma: "build", POS: POSVerb, Forms: map[string]string{
			"3sg": "builds", "past": "built", "pastpart": "built", "gerund": "building",
		}},
		{Lemma: "write", POS: POSVerb, Forms: map[string]string{
			"3sg": "writes", "past": "wrote", "pastpart": "written", "gerund": "writing",
		}},
		{Lemma: "begin", POS: POSVerb, Forms: map[string]string{
			"3sg": "begins", "past": "began", "pastpart": "begun", "gerund": "beginning",
		}},
		{Lemma: "run", POS: POSVerb, Forms: map[string]string{
			"3sg": "runs", "past": "ran", "pastpart": "run", "gerund": "running",
		}},
		{Lemma: "drive", POS: POSVerb, Forms: map[string]string{
			"3sg": "drives", "past": "drove", "pastpart": "driven", "gerund": "driving",
		}},
		{Lemma: "create", POS: POSVerb, Forms: map[string]string{
			"3sg": "creates", "past": "created", "pastpart": "created", "gerund": "creating",
		}},
		{Lemma: "found", POS: POSVerb, Forms: map[string]string{
			"3sg": "founds", "past": "founded", "pastpart": "founded", "gerund": "founding",
		}},
		{Lemma: "establish", POS: POSVerb, Forms: map[string]string{
			"3sg": "establishes", "past": "established", "pastpart": "established", "gerund": "establishing",
		}},
		{Lemma: "launch", POS: POSVerb, Forms: map[string]string{
			"3sg": "launches", "past": "launched", "pastpart": "launched", "gerund": "launching",
		}},
		{Lemma: "represent", POS: POSVerb, Forms: map[string]string{
			"3sg": "represents", "past": "represented", "pastpart": "represented", "gerund": "representing",
		}},
		{Lemma: "constitute", POS: POSVerb, Forms: map[string]string{
			"3sg": "constitutes", "past": "constituted", "pastpart": "constituted", "gerund": "constituting",
		}},
	}

	for _, w := range irregulars {
		entry := w
		g.lexicon[w.Lemma] = &entry
	}
}

// LearnWord adds a word to the generative lexicon from conversation.
func (g *GenerativeEngine) LearnWord(word string, pos POS) {
	lower := strings.ToLower(word)
	if _, exists := g.lexicon[lower]; exists {
		return // already known
	}

	switch pos {
	case POSNoun:
		g.learnedNouns = append(g.learnedNouns, lower)
	case POSAdj:
		g.learnedAdjs = append(g.learnedAdjs, lower)
	case POSVerb:
		g.learnedVerbs = append(g.learnedVerbs, lower)
	}
}

// -----------------------------------------------------------------------
// Creative Writing — generate text beyond facts
// -----------------------------------------------------------------------

// ComposeCreativeText generates a creative paragraph about a topic using
// all known facts, with varied sentence structures and embellishments.
func (g *GenerativeEngine) ComposeCreativeText(topic string, facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}
	g.resetTracker()
	g.topicCategory = inferCategory(facts)
	g.topicLabel = topic

	// Filter out described_as facts — they contain full Wikipedia paragraphs
	// that produce garbage when composed into sentences ("X is X was a French...").
	var usable []edgeFact
	for _, f := range facts {
		if f.Relation != RelDescribedAs {
			usable = append(usable, f)
		}
	}
	if len(usable) == 0 {
		return ""
	}

	var parts []string

	// Opening: set the scene (generated, not picked)
	parts = append(parts, g.composeOpener(topic))

	// Body: express facts with maximum variety
	for i, f := range usable {
		if i >= 4 {
			break // keep it concise
		}
		sent := g.GenerateCreative(f.Subject, f.Relation, f.Object)
		if sent != "" {
			parts = append(parts, sent)
		}
	}

	// Closing: wrap up with insight (generated, not picked)
	parts = append(parts, g.composeCloser(topic))

	return strings.Join(parts, " ")
}

// -----------------------------------------------------------------------
// Generative Phrase Construction — builds phrases from word-level parts
// instead of picking from pre-written sentence pools.
//
// Architecture:
//   slot pools: small single-word/short-phrase pools (adjectives, adverbs,
//               verbs) that are combined by grammar rules into full phrases.
//   compose*(): functions that assemble slots into novel phrases.
//
// Example: instead of picking "stands as a defining characteristic",
//   we compose: conjugate("stand") + "as" + article + pick(adjSlot) + pick(nounSlot)
//   → "stands as a central trait" / "stands as an essential quality" / ...
// -----------------------------------------------------------------------

// --- Slot pools: atomic building blocks (single words) ---

var adjSlots = []string{
	"central", "essential", "key", "defining", "core",
	"notable", "important", "meaningful", "significant", "fundamental",
	"distinct", "unique", "clear", "strong", "real",
}

var qualityNouns = []string{
	"trait", "characteristic", "quality", "aspect", "feature",
	"element", "attribute", "property", "dimension", "factor",
}

var valueAdjs = []string{
	"worth", "deserving of", "worthy of",
}

var attentionNouns = []string{
	"attention", "notice", "consideration", "a closer look", "examination",
}

var mannerAdvs = []string{
	"deliberately", "intentionally", "by design", "with purpose",
	"with conviction", "without hesitation", "as a matter of principle",
}

var impactNouns = []string{
	"story", "path", "direction", "identity", "character",
	"profile", "nature", "role", "position", "place",
}



// -----------------------------------------------------------------------
// Rhetorical Devices — deliberate use of literary techniques
// -----------------------------------------------------------------------

// insertRhetoric is a no-op — rhetorical devices removed for factual output.
func (g *GenerativeEngine) insertRhetoric(topic string, facts []edgeFact) string {
	return ""
}

// -----------------------------------------------------------------------
// Sentence Rhythm — vary length for natural flow
// -----------------------------------------------------------------------

// applyRhythm is a no-op — punchlines removed for factual output.
func (g *GenerativeEngine) applyRhythm(sentences []string) []string {
	return sentences
}

// -----------------------------------------------------------------------
// Cross-Fact Synthesis — combine 2+ facts into novel insights
// -----------------------------------------------------------------------

// synthesizeOriginPurpose merges a created_by and used_for fact.
func (g *GenerativeEngine) synthesizeOriginPurpose(topic, creator, purpose string) string {
	return capitalizeFirst(topic) + " was created by " + creator + " and is used for " + purpose + "."
}

// synthesizeFeatureInsight merges 2 has facts into a compound sentence.
func (g *GenerativeEngine) synthesizeFeatureInsight(topic, feat1, feat2 string) string {
	return capitalizeFirst(topic) + " features both " + feat1 + " and " + feat2 + "."
}

// synthesizeIdentityQuality merges an is_a and described_as fact.
func (g *GenerativeEngine) synthesizeIdentityQuality(topic, category, quality string) string {
	return "As " + articleForWord(quality) + " " + quality + " " + category +
		", " + topic + " is widely recognized."
}

// synthesizeFeaturePurpose merges a has and used_for fact.
func (g *GenerativeEngine) synthesizeFeaturePurpose(topic, feature, purpose string) string {
	return "Its " + feature + " makes " + topic + " well suited for " + purpose + "."
}

// synthesizeFacts scans the fact list for combinable pairs and generates
// novel insights that individual facts cannot express alone.
func (g *GenerativeEngine) synthesizeFacts(topic string, facts []edgeFact) []string {
	var results []string

	var creators, purposes, features, qualities, categories []string
	for _, f := range facts {
		if f.Subject != topic {
			continue
		}
		switch f.Relation {
		case RelCreatedBy, RelFoundedBy:
			creators = append(creators, f.Object)
		case RelUsedFor:
			purposes = append(purposes, f.Object)
		case RelHas:
			features = append(features, f.Object)
		case RelDescribedAs:
			qualities = append(qualities, f.Object)
		case RelIsA:
			categories = append(categories, f.Object)
		}
	}

	// Origin + Purpose synthesis
	if len(creators) > 0 && len(purposes) > 0 {
		results = append(results, g.synthesizeOriginPurpose(
			topic, creators[0], purposes[g.rng.Intn(len(purposes))]))
	}

	// Feature + Feature synthesis
	if len(features) >= 2 {
		i, j := 0, 1
		if len(features) > 2 {
			i = g.rng.Intn(len(features))
			j = (i + 1 + g.rng.Intn(len(features)-1)) % len(features)
		}
		results = append(results, g.synthesizeFeatureInsight(
			topic, features[i], features[j]))
	}

	// Identity + Quality synthesis
	if len(categories) > 0 && len(qualities) > 0 {
		results = append(results, g.synthesizeIdentityQuality(
			topic, categories[0], qualities[g.rng.Intn(len(qualities))]))
	}

	// Feature + Purpose synthesis
	if len(features) > 0 && len(purposes) > 0 {
		results = append(results, g.synthesizeFeaturePurpose(
			topic, features[g.rng.Intn(len(features))],
			purposes[g.rng.Intn(len(purposes))]))
	}

	return results
}

// --- Compositional phrase builders ---

// composeRelContinuation generates a factual continuation for relative clauses.
func (g *GenerativeEngine) composeRelContinuation() string {
	return g.pick([]string{
		"is widely recognized.",
		"remains notable.",
		"is well established.",
	})
}


// composeAppositiveTail generates a factual tail for appositive clauses.
func (g *GenerativeEngine) composeAppositiveTail() string {
	return g.pick([]string{
		"is widely recognized.",
		"remains significant.",
		"continues to be relevant.",
	})
}

// composeConnector generates a neutral discourse connector.
func (g *GenerativeEngine) composeConnector() string {
	return g.pick([]string{
		"Additionally,", "Also,", "Moreover,",
		"Furthermore,", "In addition,",
	})
}

// composeOpener generates a neutral topic introduction.
func (g *GenerativeEngine) composeOpener(topic string) string {
	// Simple factual introduction — no editorial hooks.
	return capitalizeFirst(topic) + " is a topic with several notable aspects."
}

// composeCloser generates a topic conclusion referencing the subject.
func (g *GenerativeEngine) composeCloser(topic string) string {
	return "These details capture the essentials of " + topic + "."
}


// composeHook generates a factual article introduction from the topic itself.
func (g *GenerativeEngine) composeHook(topic string) string {
	return capitalizeFirst(topic) + "."
}

// composeElaboration generates a bridge sentence referencing the topic.
func (g *GenerativeEngine) composeElaboration(topic string) string {
	return capitalizeFirst(topic) + " has several key aspects."
}

// composeTransition generates a section transition from the topic name.
func (g *GenerativeEngine) composeTransition(topic string) string {
	return g.pick([]string{
		capitalizeFirst(topic) + " also has other aspects.",
		"Beyond this, " + topic + " has further dimensions.",
		capitalizeFirst(topic) + " extends into other areas.",
	})
}

// composeInsight generates a section-closing sentence from the theme.
func (g *GenerativeEngine) composeInsight(topic string, theme string) string {
	themeNoun := "details"
	switch theme {
	case "What It Is":
		themeNoun = "characteristics"
	case "Where It Came From":
		themeNoun = "origins"
	case "What It's Used For":
		themeNoun = "applications"
	case "Key Features":
		themeNoun = "features"
	case "The Bigger Picture":
		themeNoun = "connections"
	}
	return "These " + themeNoun + " define " + topic + "."
}

// -----------------------------------------------------------------------
// Article Composition — long-form content generation (300-500+ words)
//
// Unlike ComposeCreativeText (which produces a single paragraph), this
// engine plans a full article with:
//   - Introduction paragraph (hook + thesis)
//   - Thematic body sections (grouped by relation type)
//   - Elaboration & rhetorical devices (analogies, questions, context)
//   - Transitions between sections
//   - Conclusion paragraph (synthesis + forward-looking statement)
// -----------------------------------------------------------------------

// ComposeArticle generates a long-form article about a topic from known facts.
// Targets 300-500 words depending on fact density.
func (g *GenerativeEngine) ComposeArticle(topic string, facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}
	g.resetTracker()
	g.topicCategory = inferCategory(facts)
	g.topicLabel = topic

	var paragraphs []string

	// Pre-compute cross-fact synthesis for use throughout the article
	syntheses := g.synthesizeFacts(topic, facts)

	// ── 1. Introduction ──
	paragraphs = append(paragraphs, g.articleIntro(topic, facts))

	// ── 2. Body — group facts by theme and write a section for each ──
	sections := g.groupByTheme(facts)
	synthIdx := 0
	for i, section := range sections {
		if i >= 4 {
			break // cap at 4 body sections
		}
		trans := ""
		if i > 0 {
			trans = g.composeTransition(topic) // generated, not picked
		}
		para := g.articleSectionWith(topic, section, i, trans)
		if para != "" {
			paragraphs = append(paragraphs, para)
		}

		// Insert a cross-fact synthesis after the first body section
		if i == 0 && synthIdx < len(syntheses) {
			paragraphs = append(paragraphs, syntheses[synthIdx])
			synthIdx++
		}
	}

	// ── 3. Conclusion with rhetoric ──
	conclusion := g.articleConclusion(topic, facts)
	// 50% chance to add a rhetorical flourish before the conclusion
	if g.rng.Float64() < 0.5 {
		rhet := g.insertRhetoric(topic, facts)
		if rhet != "" {
			conclusion = rhet + " " + conclusion
		}
	}
	paragraphs = append(paragraphs, conclusion)

	return strings.Join(paragraphs, "\n\n")
}

// themeSection groups facts under a thematic heading.
type themeSection struct {
	theme string
	facts []edgeFact
}

// groupByTheme organizes facts into thematic clusters.
func (g *GenerativeEngine) groupByTheme(facts []edgeFact) []themeSection {
	// Group by relation category
	categories := map[string][]edgeFact{
		"identity":     {}, // is_a, described_as
		"origin":       {}, // created_by, founded_by, founded_in
		"purpose":      {}, // used_for, offers
		"features":     {}, // has, part_of
		"connections":  {}, // related_to, causes
		"location":     {}, // located_in
		"preferences":  {}, // prefers, dislikes
	}

	for _, f := range facts {
		switch f.Relation {
		case RelIsA, RelDescribedAs:
			categories["identity"] = append(categories["identity"], f)
		case RelCreatedBy, RelFoundedBy, RelFoundedIn:
			categories["origin"] = append(categories["origin"], f)
		case RelUsedFor, RelOffers:
			categories["purpose"] = append(categories["purpose"], f)
		case RelHas, RelPartOf:
			categories["features"] = append(categories["features"], f)
		case RelRelatedTo, RelCauses:
			categories["connections"] = append(categories["connections"], f)
		case RelLocatedIn:
			categories["location"] = append(categories["location"], f)
		case RelPrefers, RelDislikes:
			categories["preferences"] = append(categories["preferences"], f)
		default:
			categories["connections"] = append(categories["connections"], f)
		}
	}

	// Build sections in narrative order — only include non-empty ones
	order := []struct {
		key   string
		theme string
	}{
		{"identity", "What It Is"},
		{"origin", "Where It Came From"},
		{"purpose", "What It's Used For"},
		{"features", "Key Features"},
		{"location", "Where It Lives"},
		{"connections", "The Bigger Picture"},
		{"preferences", "Preferences"},
	}

	var sections []themeSection
	for _, o := range order {
		if fs, ok := categories[o.key]; ok && len(fs) > 0 {
			sections = append(sections, themeSection{theme: o.theme, facts: fs})
		}
	}
	return sections
}

// articleIntro generates the introduction paragraph.
func (g *GenerativeEngine) articleIntro(topic string, facts []edgeFact) string {
	var sentences []string

	// Hook — generated from grammar parts
	sentences = append(sentences, g.composeHook(topic))

	// Thesis — what we'll cover, based on the themes present
	var themes []string
	hasOrigin, hasPurpose, hasFeatures := false, false, false
	for _, f := range facts {
		switch f.Relation {
		case RelCreatedBy, RelFoundedBy, RelFoundedIn:
			hasOrigin = true
		case RelUsedFor, RelOffers:
			hasPurpose = true
		case RelHas, RelPartOf:
			hasFeatures = true
		}
	}
	poss := g.topicPossessive()
	pron := g.topicPronoun()
	obj := g.topicObject()
	if hasOrigin {
		themes = append(themes, g.pick([]string{poss + " origins", "where " + pron + " came from", "how " + pron + " began"}))
	}
	if hasPurpose {
		themes = append(themes, g.pick([]string{"what " + pron + "'s used for", poss + " purpose", "the problems " + pron + " solves"}))
	}
	if hasFeatures {
		themes = append(themes, g.pick([]string{"what makes " + obj + " tick", poss + " key features", "what sets " + obj + " apart"}))
	}

	if len(themes) > 0 {
		thesis := fmt.Sprintf("To understand %s fully, it helps to look at %s.", topic, joinNatural(themes))
		sentences = append(sentences, thesis)
	}

	// Lead fact — state the most fundamental fact (is_a if available)
	for _, f := range facts {
		if f.Relation == RelIsA {
			sentences = append(sentences, g.Generate(f.Subject, f.Relation, f.Object))
			break
		}
	}

	// Elaboration — generated bridge sentence
	sentences = append(sentences, g.composeElaboration(topic))

	return strings.Join(sentences, " ")
}

// articleSectionWith writes one thematic body paragraph with a specific transition.
func (g *GenerativeEngine) articleSectionWith(topic string, section themeSection, index int, transition string) string {
	var sentences []string

	// Section transition — already includes topic (generated by composeTransition)
	if index > 0 && transition != "" {
		sentences = append(sentences, transition)
	}

	// Express each fact in the section with varied patterns
	for i, f := range section.facts {
		sent := g.Generate(f.Subject, f.Relation, f.Object)
		if sent != "" {
			sent = g.varyTopicRef(sent, topic)
			sentences = append(sentences, sent)
		}

		// 80% chance for first fact, 60% for rest: add elaboration for richer content
		elabChance := 0.6
		if i == 0 {
			elabChance = 0.8
		}
		if g.rng.Float64() < elabChance {
			elab := g.elaborateFact(f)
			if elab != "" {
				sentences = append(sentences, elab)
			}
		}
	}

	// 30% chance: insert a rhetorical device when we have enough facts
	if len(section.facts) >= 2 && g.rng.Float64() < 0.3 {
		rhet := g.insertRhetoric(topic, section.facts)
		if rhet != "" {
			sentences = append(sentences, rhet)
		}
	}

	// Section-level insight — generated synthesis sentence
	if len(section.facts) >= 2 {
		sentences = append(sentences, g.composeInsight(topic, section.theme))
	}

	// Apply rhythm: insert punchlines after long sentence sequences
	sentences = g.applyRhythm(sentences)

	return strings.Join(sentences, " ")
}

// elaborateFact generates a contextual elaboration sentence from grammar parts.
// Instead of picking from pre-written pools, it assembles from relation-aware
// building blocks: subject, object, verbs, and structural patterns.
func (g *GenerativeEngine) elaborateFact(f edgeFact) string {
	// No elaboration — let the facts speak for themselves.
	return ""
}

// articleConclusion generates the closing paragraph from grammar parts.
func (g *GenerativeEngine) articleConclusion(topic string, facts []edgeFact) string {
	return "Together, these facts provide a comprehensive overview of " + topic + "."
}

// -----------------------------------------------------------------------
// Article Phrase Libraries
// -----------------------------------------------------------------------

// All article hooks, elaborations, and transitions are now generated by
// composeHook(), composeElaboration(), and composeTransition() above.
// No static sentence pools remain for article composition.

// joinNatural joins strings with commas and "and": ["a", "b", "c"] → "a, b, and c"
func joinNatural(items []string) string {
	switch len(items) {
	case 0:
		return ""
	case 1:
		return items[0]
	case 2:
		return items[0] + " and " + items[1]
	default:
		return strings.Join(items[:len(items)-1], ", ") + ", and " + items[len(items)-1]
	}
}

func uniqueStringsSlice(ss []string) []string {
	seen := make(map[string]bool)
	var result []string
	for _, s := range ss {
		if !seen[s] {
			seen[s] = true
			result = append(result, s)
		}
	}
	return result
}

// -----------------------------------------------------------------------
// Discourse-Planned Composition — uses the DiscoursePlanner to generate
// text with rhetorical structure instead of ad-hoc section ordering.
//
// This is the key improvement: instead of "opener + random facts + closer",
// we get "HOOK → DEFINE → ORIGIN → FEATURES → PURPOSE → CLOSE" with
// section-specific transitions and communicative goals.
// -----------------------------------------------------------------------

// ComposeWithPlan generates text following a discourse plan.
// Each section in the plan is realized by the appropriate generative method.
func (g *GenerativeEngine) ComposeWithPlan(plan *DiscoursePlan) string {
	if plan == nil || len(plan.Sections) == 0 {
		return ""
	}

	g.resetTracker()

	// Infer category from identity facts across all sections
	var allFacts []edgeFact
	for _, s := range plan.Sections {
		allFacts = append(allFacts, s.Facts...)
	}
	g.topicCategory = inferCategory(allFacts)
	g.topicLabel = plan.Topic

	var paragraphs []string

	for _, section := range plan.Sections {
		text := g.realizeSection(plan.Topic, &section)
		if text != "" {
			paragraphs = append(paragraphs, text)
		}
	}

	if len(paragraphs) == 0 {
		return ""
	}

	return strings.Join(paragraphs, " ")
}

// ComposeArticleWithPlan generates a multi-paragraph article following a discourse plan.
func (g *GenerativeEngine) ComposeArticleWithPlan(plan *DiscoursePlan) string {
	if plan == nil || len(plan.Sections) == 0 {
		return ""
	}

	g.resetTracker()

	var allFacts []edgeFact
	for _, s := range plan.Sections {
		allFacts = append(allFacts, s.Facts...)
	}
	g.topicCategory = inferCategory(allFacts)
	g.topicLabel = plan.Topic

	var paragraphs []string

	for _, section := range plan.Sections {
		text := g.realizeSection(plan.Topic, &section)
		if text != "" {
			paragraphs = append(paragraphs, text)
		}
	}

	if len(paragraphs) == 0 {
		return ""
	}

	return strings.Join(paragraphs, "\n\n")
}

// realizeSection generates text for one discourse section.
func (g *GenerativeEngine) realizeSection(topic string, section *DiscourseSection) string {
	var sentences []string

	// Add section connector/transition
	if section.Connector != "" {
		sentences = append(sentences, section.Connector)
	}

	maxSents := section.MaxSents
	if maxSents == 0 {
		maxSents = 2
	}

	switch section.Role {
	case SectionHook:
		sentences = append(sentences, g.composeHook(topic))

	case SectionDefine:
		// Realize identity facts with elaboration
		for i, f := range section.Facts {
			if i >= maxSents {
				break
			}
			sent := g.GenerateCreative(f.Subject, f.Relation, f.Object)
			if sent != "" {
				sent = g.varyTopicRef(sent, topic)
				sentences = append(sentences, sent)
			}
		}

	case SectionOrigin:
		// Origin facts get narrative treatment
		for i, f := range section.Facts {
			if i >= maxSents {
				break
			}
			sent := g.Generate(f.Subject, f.Relation, f.Object)
			if sent != "" {
				sent = g.varyTopicRef(sent, topic)
				sentences = append(sentences, sent)
			}
			// First origin fact gets elaboration
			if i == 0 {
				elab := g.elaborateFact(f)
				if elab != "" {
					sentences = append(sentences, elab)
				}
			}
		}

	case SectionFeatures:
		for i, f := range section.Facts {
			if i >= maxSents {
				break
			}
			sent := g.Generate(f.Subject, f.Relation, f.Object)
			if sent != "" {
				sent = g.varyTopicRef(sent, topic)
				sentences = append(sentences, sent)
			}
		}

	case SectionPurpose:
		// Purpose facts get practical framing
		for i, f := range section.Facts {
			if i >= maxSents {
				break
			}
			sent := g.GenerateCreative(f.Subject, f.Relation, f.Object)
			if sent != "" {
				sent = g.varyTopicRef(sent, topic)
				sentences = append(sentences, sent)
			}
			if i == 0 && g.rng.Float64() < 0.7 {
				elab := g.elaborateFact(f)
				if elab != "" {
					sentences = append(sentences, elab)
				}
			}
		}

	case SectionImpact:
		for i, f := range section.Facts {
			if i >= maxSents {
				break
			}
			sent := g.Generate(f.Subject, f.Relation, f.Object)
			if sent != "" {
				sent = g.varyTopicRef(sent, topic)
				sentences = append(sentences, sent)
			}
		}

	case SectionComparison:
		// Comparison section highlights relationships
		for i, f := range section.Facts {
			if i >= maxSents {
				break
			}
			sent := g.Generate(f.Subject, f.Relation, f.Object)
			if sent != "" {
				sentences = append(sentences, sent)
			}
			if i == 0 {
				elab := g.elaborateFact(f)
				if elab != "" {
					sentences = append(sentences, elab)
				}
			}
		}

	case SectionClose:
		sentences = append(sentences, g.composeCloser(topic))
	}

	if len(sentences) == 0 {
		return ""
	}

	// Merge connector with first sentence if connector is a fragment
	if section.Connector != "" && len(sentences) >= 2 {
		conn := sentences[0]
		if strings.HasSuffix(conn, ",") || strings.HasSuffix(conn, ":") {
			// Connector is a fragment — merge with next sentence
			next := sentences[1]
			if len(next) > 0 {
				// Lowercase the first char of the next sentence for merge
				merged := conn + " " + strings.ToLower(next[:1]) + next[1:]
				sentences = append([]string{merged}, sentences[2:]...)
			}
		}
	}

	return strings.Join(sentences, " ")
}

// simplePlural is defined in composer.go — handles English pluralization.
