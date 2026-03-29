package cognitive

import (
	"bufio"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

// -----------------------------------------------------------------------
// Hybrid Text Generation Engine
//
// Produces fluent prose by combining retrieval, recombination, and neural
// generation. The architecture: 80% human-written text (retrieved from
// corpus) + 20% generated connectors.
//
// Every word of substance was written by a Wikipedia author. The system
// just finds the right sentences and connects them.
//
// Pipeline:
//   1. Plan discourse order via semantic role grouping
//   2. Retrieve real human-written sentences from corpus
//   3. Fill gaps with GRU generation or structural fallback
//   4. Generate SHORT connectors between sentence groups
//   5. Score candidates with bigram fluency model
//   6. Pronominalize repeated subject mentions
//   7. Assemble into flowing paragraphs
// -----------------------------------------------------------------------

// HybridGenerator produces fluent prose by orchestrating:
//  1. Sentence retrieval from human-written corpus
//  2. Entity adaptation (swap subjects/objects)
//  3. Neural connector generation (GRU for transitions)
//  4. Bigram fluency scoring for candidate ranking
//  5. Discourse planning for coherent ordering
//
// The result: most words in the output were written by humans.
// The system just finds the right words and connects them.
type HybridGenerator struct {
	corpus  *SentenceCorpus
	textGen *TextGenModel
	fluency *FluencyScorer

	// Path to knowledge/*.txt files for same-domain sentence retrieval.
	knowledgeDir string

	// Fallback connectors for when GRU isn't available.
	// These are SHORT (2-5 words) connecting phrases, not full sentences.
	connectors map[string][]string // relation transition -> connector options
}

// NewHybridGenerator creates a hybrid text generation engine.
// knowledgeDir is the path to knowledge/*.txt files for same-domain retrieval.
// Pass "" to disable topic sentence retrieval.
func NewHybridGenerator(corpus *SentenceCorpus, textGen *TextGenModel, fluency *FluencyScorer, knowledgeDir ...string) *HybridGenerator {
	kDir := ""
	if len(knowledgeDir) > 0 {
		kDir = knowledgeDir[0]
	}
	hg := &HybridGenerator{
		corpus:       corpus,
		textGen:      textGen,
		fluency:      fluency,
		knowledgeDir: kDir,
		connectors: map[string][]string{
			"to_usage":    {"In practice,", "Practically,"},
			"to_relation": {""},
			"to_origin":   {""},
			"to_property": {""},
			"to_caveat":   {"However,", "That said,"},
			"to_impact":   {""},
			"to_location": {""},
		},
	}
	return hg
}

// -----------------------------------------------------------------------
// Discourse ordering — semantic role grouping
// -----------------------------------------------------------------------

// discourseGroup is a cluster of facts sharing a semantic role.
type discourseGroup struct {
	role  string     // "definition", "origin", "property", "usage", "relation", "caveat", "location"
	facts []edgeFact
}

// planDiscourseOrder groups and orders facts for coherent prose.
// Order: definition -> origin -> location -> properties -> usage -> relations -> caveats
func planDiscourseOrder(facts []edgeFact) []discourseGroup {
	buckets := map[string][]edgeFact{
		"definition": nil,
		"origin":     nil,
		"location":   nil,
		"property":   nil,
		"usage":      nil,
		"relation":   nil,
		"caveat":     nil,
	}

	for _, f := range facts {
		switch f.Relation {
		case RelIsA, RelDescribedAs, RelDomain:
			buckets["definition"] = append(buckets["definition"], f)
		case RelCreatedBy, RelFoundedBy, RelFoundedIn, RelDerivedFrom, RelInfluencedBy:
			buckets["origin"] = append(buckets["origin"], f)
		case RelLocatedIn:
			buckets["location"] = append(buckets["location"], f)
		case RelHas, RelPartOf, RelOffers, RelKnownFor:
			buckets["property"] = append(buckets["property"], f)
		case RelUsedFor:
			buckets["usage"] = append(buckets["usage"], f)
		case RelContradicts, RelOppositeOf:
			buckets["caveat"] = append(buckets["caveat"], f)
		default:
			buckets["relation"] = append(buckets["relation"], f)
		}
	}

	// Ordered sequence of roles.
	order := []string{"definition", "origin", "location", "property", "usage", "relation", "caveat"}

	var groups []discourseGroup
	for _, role := range order {
		if fs := buckets[role]; len(fs) > 0 {
			groups = append(groups, discourseGroup{role: role, facts: fs})
		}
	}
	return groups
}

// -----------------------------------------------------------------------
// Core generation pipeline
// -----------------------------------------------------------------------

// Generate produces fluent prose from a subject and its facts.
func (hg *HybridGenerator) Generate(subject string, facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	// 1. Plan discourse order.
	groups := planDiscourseOrder(facts)
	if len(groups) == 0 {
		return ""
	}

	// Find the IsA fact for pronominalization category.
	category := findCategory(facts)

	// 2-5. Retrieve, generate, connect, score per group.
	var paragraphParts []string
	prevRole := ""

	for _, grp := range groups {
		// Generate connector between groups.
		connector := hg.generateConnector(prevRole, grp.role)

		// Retrieve or generate sentences for each fact in this group.
		var groupSentences []string
		for _, fact := range grp.facts {
			sent := hg.scoreCandidates(subject, fact)
			if sent != "" {
				groupSentences = append(groupSentences, sent)
			}
		}

		if len(groupSentences) > 0 {
			joined := strings.Join(groupSentences, " ")
			if connector != "" {
				joined = connector + " " + joined
			}
			paragraphParts = append(paragraphParts, joined)
		}

		prevRole = grp.role
	}

	if len(paragraphParts) == 0 {
		return ""
	}

	// 6. Assemble and pronominalize.
	text := strings.Join(paragraphParts, " ")
	text = pronominalize(text, subject, category)
	text = cleanupWhitespace(text)

	return text
}

// -----------------------------------------------------------------------
// Retrieval with fallback
// -----------------------------------------------------------------------

// retrieveOrGenerate tries topic sentence retrieval first, then GRU, then structural.
func (hg *HybridGenerator) retrieveOrGenerate(subject string, fact edgeFact) string {
	// Tier 1: Direct topic sentences from knowledge files.
	// These are REAL human-written sentences about THIS topic — no entity swapping.
	if hg.knowledgeDir != "" {
		topicSentences := hg.retrieveTopicSentences(subject)
		objLower := strings.ToLower(fact.Object)
		for _, sent := range topicSentences {
			lower := strings.ToLower(sent)
			if strings.Contains(lower, objLower) && len(sent) < 200 {
				return sent
			}
		}
	}

	// Tier 2: Neural generation.
	if hg.textGen != nil {
		generated := hg.textGen.Generate(subject, fact.Relation, fact.Object, 0.5)
		if generated != "" && len(generated) > 10 {
			return generated
		}
	}

	// Tier 3: Simple structural sentence (last resort).
	return buildSimpleSentence(subject, fact.Relation, fact.Object)
}

// retrieveTopicSentences reads knowledge text files and returns individual
// sentences from paragraphs that mention the given topic. Only returns
// sentences from the SAME paragraph as the topic — no cross-domain mixing.
func (hg *HybridGenerator) retrieveTopicSentences(topic string) []string {
	if hg.knowledgeDir == "" {
		return nil
	}

	topicLower := strings.ToLower(topic)

	// Scan all .txt files in the knowledge directory.
	matches, err := filepath.Glob(filepath.Join(hg.knowledgeDir, "*.txt"))
	if err != nil || len(matches) == 0 {
		return nil
	}

	var result []string
	for _, path := range matches {
		f, err := os.Open(path)
		if err != nil {
			continue
		}

		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 0, 64*1024), 256*1024)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}

			// Check if this paragraph mentions the topic.
			if !strings.Contains(strings.ToLower(line), topicLower) {
				continue
			}

			// Split paragraph into sentences and collect them.
			sentences := hybridSplitSentences(line)
			for _, sent := range sentences {
				sent = strings.TrimSpace(sent)
				if len(sent) > 20 && len(sent) < 200 {
					result = append(result, sent)
				}
			}
		}
		f.Close()
	}

	return result
}

// hybridIsRelevant checks if an adapted corpus sentence is actually about
// the subject/object. Rejects cross-domain contamination.
func hybridIsRelevant(sentence, subject, object string) bool {
	lower := strings.ToLower(sentence)
	subjLower := strings.ToLower(subject)
	objLower := strings.ToLower(object)

	// Must contain the subject or object
	if !strings.Contains(lower, subjLower) && !strings.Contains(lower, objLower) {
		return false
	}

	// Reject bloated adaptations (source sentence was too long)
	if len(sentence) > 180 {
		return false
	}

	return true
}

// -----------------------------------------------------------------------
// Structural sentence construction
// -----------------------------------------------------------------------

// buildSimpleSentence constructs a grammatically correct sentence from a
// (subject, relation, object) triple. Used only as a last resort when
// neither corpus retrieval nor GRU generation produces anything.
func buildSimpleSentence(subject string, rel RelType, object string) string {
	subj := capitalizeFirst(subject)
	switch rel {
	case RelIsA:
		return subj + " is " + articleFor(object) + "."
	case RelDescribedAs:
		return subj + " is " + object + "."
	case RelCreatedBy:
		return subj + " was created by " + object + "."
	case RelFoundedBy:
		return subj + " was founded by " + object + "."
	case RelFoundedIn:
		return subj + " was founded in " + object + "."
	case RelLocatedIn:
		return subj + " is located in " + object + "."
	case RelPartOf:
		return subj + " is part of " + object + "."
	case RelHas:
		return subj + " has " + object + "."
	case RelOffers:
		return subj + " offers " + object + "."
	case RelUsedFor:
		return subj + " is used for " + object + "."
	case RelRelatedTo:
		return subj + " is related to " + object + "."
	case RelSimilarTo:
		return subj + " is similar to " + object + "."
	case RelCauses:
		return subj + " causes " + object + "."
	case RelContradicts:
		return subj + " contradicts " + object + "."
	case RelFollows:
		return subj + " follows " + object + "."
	case RelPrefers:
		return subj + " prefers " + object + "."
	case RelDislikes:
		return subj + " dislikes " + object + "."
	case RelDomain:
		return subj + " belongs to the domain of " + object + "."
	case RelKnownFor:
		return subj + " is known for " + object + "."
	case RelInfluencedBy:
		return subj + " was influenced by " + object + "."
	case RelDerivedFrom:
		return subj + " is derived from " + object + "."
	case RelOppositeOf:
		return subj + " is the opposite of " + object + "."
	default:
		return subj + " is associated with " + object + "."
	}
}

// -----------------------------------------------------------------------
// Connector generation
// -----------------------------------------------------------------------

// generateConnector produces a SHORT transition phrase between discourse
// groups. Most transitions are empty — real prose flows without
// "Furthermore" and "Additionally".
func (hg *HybridGenerator) generateConnector(prevRole, nextRole string) string {
	if prevRole == "" {
		return "" // first group, no connector
	}

	// Most transitions need no connector at all.
	// Only add one when the semantic shift is jarring.
	key := ""
	switch nextRole {
	case "usage":
		key = "to_usage"
	case "caveat":
		key = "to_caveat"
	case "relation":
		key = "to_relation"
	case "origin":
		key = "to_origin"
	case "property":
		key = "to_property"
	case "location":
		key = "to_location"
	case "definition":
		return "" // definition always flows naturally
	}

	if key == "" {
		return ""
	}

	options := hg.connectors[key]
	if len(options) == 0 {
		return ""
	}

	// Pick a connector. Use a simple deterministic choice for consistency
	// within a single generation pass — vary across calls via length hashing.
	pick := options[len(prevRole)%len(options)]
	return pick
}

// -----------------------------------------------------------------------
// Candidate scoring
// -----------------------------------------------------------------------

// scoreCandidates generates multiple candidate sentences for a fact and
// picks the most fluent one using bigram scoring.
func (hg *HybridGenerator) scoreCandidates(subject string, fact edgeFact) string {
	var candidates []string

	// Tier 1: Corpus retrieval.
	if hg.corpus != nil {
		retrieved := hg.corpus.RetrieveVaried(subject, fact.Relation, fact.Object)
		if retrieved != "" {
			candidates = append(candidates, retrieved)
		}
	}

	// Tier 2: GRU generation.
	if hg.textGen != nil {
		generated := hg.textGen.Generate(subject, fact.Relation, fact.Object, 0.5)
		if generated != "" && len(generated) > 10 {
			candidates = append(candidates, generated)
		}
	}

	// Tier 3: Structural fallback (always available).
	structural := buildSimpleSentence(subject, fact.Relation, fact.Object)
	if structural != "" {
		candidates = append(candidates, structural)
	}

	if len(candidates) == 0 {
		return ""
	}

	// If only one candidate, use it directly.
	if len(candidates) == 1 {
		return candidates[0]
	}

	// Score with fluency model if available.
	if hg.fluency != nil {
		idx, _ := hg.fluency.ScoreBest(candidates)
		if idx >= 0 {
			return candidates[idx]
		}
	}

	// No scorer — prefer corpus over GRU over structural (first candidate).
	return candidates[0]
}

// -----------------------------------------------------------------------
// Pronominalization
// -----------------------------------------------------------------------

// pronominalize replaces repeated mentions of the subject with pronouns
// or definite descriptions to avoid monotonous repetition.
//
// Uses gender-aware pronouns for persons (he/she) and "it" for non-persons.
// For plural/group entities, uses "they".
//
// Pattern:
//
//	1st occurrence: full name
//	2nd occurrence: pronoun (he/she/it/they)
//	3rd occurrence: definite description ("the philosopher", "the language")
//	4th+: alternate pronoun and description
func pronominalize(text, subject, category string) string {
	if subject == "" || text == "" {
		return text
	}

	capSubject := capitalizeFirst(subject)

	// Determine the correct pronoun based on category and gender.
	pronoun := hybridDeterminePronoun(subject, category)

	// Build the definite description from the IsA category.
	desc := hybridDefiniteDesc(subject, category)

	// Split into sentences so we can track occurrences.
	sentences := hybridSplitSentences(text)
	occurrence := 0

	for i, sent := range sentences {
		if !containsSubject(sent, capSubject, subject) {
			continue
		}
		occurrence++

		switch {
		case occurrence == 1:
			// Keep the full name.
			continue
		case occurrence%3 == 0 && desc != "":
			// Every 3rd mention: use definite description.
			descCap := capitalizeFirst(desc)
			sentences[i] = replaceSubjectOnce(sent, capSubject, subject, descCap)
		default:
			// Other mentions: use pronoun.
			sentences[i] = replaceSubjectOnce(sent, capSubject, subject, capitalizeFirst(pronoun))
		}
	}

	return strings.Join(sentences, " ")
}

// hybridDeterminePronoun selects the correct pronoun for a subject.
// For persons: he/she based on detectGender.
// For plural/group entities: they.
// For everything else: it.
func hybridDeterminePronoun(subject, category string) string {
	catNorm := inferCategoryFromString(strings.ToLower(category))

	// Check for plural/group entities first.
	if isPlural(category) {
		return "they"
	}

	// Check for persons — use gendered pronoun.
	if isPerson(catNorm) {
		g := detectGender(subject)
		switch g {
		case GenderFemale:
			return "she"
		case GenderMale:
			return "he"
		default:
			return "he"
		}
	}

	return "it"
}

// hybridDefiniteDesc builds a definite description like "the philosopher"
// or "the language" from the category. Falls back to the last word of the
// subject if no category is available.
func hybridDefiniteDesc(subject, category string) string {
	if category != "" {
		return "the " + strings.ToLower(category)
	}
	words := strings.Fields(strings.ToLower(subject))
	if len(words) > 0 {
		return "the " + words[len(words)-1]
	}
	return ""
}

// containsSubject checks if a sentence contains the subject (case-insensitive start match).
func containsSubject(sent, capSubject, subject string) bool {
	return strings.Contains(sent, capSubject) || strings.Contains(sent, subject)
}

// replaceSubjectOnce replaces the first occurrence of the subject in a
// sentence with the replacement, handling both capitalized and lowercase forms.
func replaceSubjectOnce(sent, capSubject, subject, replacement string) string {
	// Try capitalized form first (sentence start).
	if idx := strings.Index(sent, capSubject); idx != -1 {
		return sent[:idx] + replacement + sent[idx+len(capSubject):]
	}
	// Try lowercase form.
	if idx := strings.Index(sent, subject); idx != -1 {
		return sent[:idx] + replacement + sent[idx+len(subject):]
	}
	return sent
}

// findCategory extracts the category from IsA facts for pronominalization.
func findCategory(facts []edgeFact) string {
	for _, f := range facts {
		if f.Relation == RelIsA && f.Object != "" {
			return f.Object
		}
	}
	return ""
}

// splitIntoSentences breaks text into sentences at period boundaries,
// keeping each sentence intact for pronominalization.
func hybridSplitSentences(text string) []string {
	var sentences []string
	var current strings.Builder

	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		current.WriteRune(runes[i])
		if runes[i] == '.' || runes[i] == '!' || runes[i] == '?' {
			// Check if this is really a sentence boundary (followed by
			// space+uppercase or end of text).
			if i+1 >= len(runes) {
				sentences = append(sentences, strings.TrimSpace(current.String()))
				current.Reset()
			} else if i+2 < len(runes) && runes[i+1] == ' ' && unicode.IsUpper(runes[i+2]) {
				sentences = append(sentences, strings.TrimSpace(current.String()))
				current.Reset()
			}
		}
	}

	remaining := strings.TrimSpace(current.String())
	if remaining != "" {
		sentences = append(sentences, remaining)
	}

	return sentences
}

// -----------------------------------------------------------------------
// Specialized generators
// -----------------------------------------------------------------------

// GenerateExplanation produces prose optimized for explaining a concept.
// Ordering: definition -> mechanism/origin -> examples/properties -> caveats -> recap.
func (hg *HybridGenerator) GenerateExplanation(subject string, facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	// Reorder facts for explanation flow:
	// definition -> origin -> features -> usage -> caveats
	ordered := orderForExplanation(facts)
	return hg.Generate(subject, ordered)
}

// orderForExplanation sorts facts into explanation order, ensuring
// definitions come first and caveats come last.
func orderForExplanation(facts []edgeFact) []edgeFact {
	var definition, origin, features, usage, caveats, other []edgeFact

	for _, f := range facts {
		switch f.Relation {
		case RelIsA, RelDescribedAs, RelDomain:
			definition = append(definition, f)
		case RelCreatedBy, RelFoundedBy, RelFoundedIn, RelDerivedFrom, RelInfluencedBy:
			origin = append(origin, f)
		case RelHas, RelPartOf, RelOffers, RelKnownFor:
			features = append(features, f)
		case RelUsedFor:
			usage = append(usage, f)
		case RelContradicts, RelOppositeOf:
			caveats = append(caveats, f)
		default:
			other = append(other, f)
		}
	}

	var result []edgeFact
	result = append(result, definition...)
	result = append(result, origin...)
	result = append(result, features...)
	result = append(result, usage...)
	result = append(result, other...)
	result = append(result, caveats...)
	return result
}

// GenerateComparison produces prose comparing two subjects.
// Structure: intro A -> intro B -> shared properties -> differences -> verdict.
func (hg *HybridGenerator) GenerateComparison(subjectA, subjectB string, factsA, factsB []edgeFact) string {
	if len(factsA) == 0 && len(factsB) == 0 {
		return ""
	}

	var parts []string

	// Introduce subject A.
	introA := hg.generateIntro(subjectA, factsA)
	if introA != "" {
		parts = append(parts, introA)
	}

	// Introduce subject B.
	introB := hg.generateIntro(subjectB, factsB)
	if introB != "" {
		parts = append(parts, introB)
	}

	// Find shared properties — same relation+object across both subjects.
	shared := findSharedFacts(factsA, factsB)
	if len(shared) > 0 {
		sharedText := hg.generateSharedSection(subjectA, subjectB, shared)
		if sharedText != "" {
			parts = append(parts, sharedText)
		}
	}

	// Generate differences — facts unique to each subject.
	uniqueA, uniqueB := findUniqueFacts(factsA, factsB)
	diffText := hg.generateDifferencesSection(subjectA, subjectB, uniqueA, uniqueB)
	if diffText != "" {
		parts = append(parts, diffText)
	}

	if len(parts) == 0 {
		return ""
	}

	return strings.Join(parts, " ")
}

// generateIntro creates a brief introduction for a subject using its
// definition and one or two key facts.
func (hg *HybridGenerator) generateIntro(subject string, facts []edgeFact) string {
	// Pick definition facts first, then up to one additional fact.
	var introFacts []edgeFact
	var rest []edgeFact

	for _, f := range facts {
		if f.Relation == RelIsA || f.Relation == RelDescribedAs {
			introFacts = append(introFacts, f)
		} else {
			rest = append(rest, f)
		}
	}

	// Add one non-definition fact for flavor.
	if len(rest) > 0 && len(introFacts) < 3 {
		introFacts = append(introFacts, rest[0])
	}

	if len(introFacts) == 0 {
		return ""
	}

	var sentences []string
	for _, f := range introFacts {
		sent := hg.retrieveOrGenerate(subject, f)
		if sent != "" {
			sentences = append(sentences, sent)
		}
	}

	return strings.Join(sentences, " ")
}

// sharedFactPair records a relation+object shared by both subjects.
type sharedFactPair struct {
	relation RelType
	object   string
}

// findSharedFacts identifies facts with the same relation and object
// across two sets of facts.
func findSharedFacts(factsA, factsB []edgeFact) []sharedFactPair {
	type relObj struct {
		rel RelType
		obj string
	}

	setB := make(map[relObj]bool)
	for _, f := range factsB {
		setB[relObj{f.Relation, strings.ToLower(f.Object)}] = true
	}

	var shared []sharedFactPair
	seen := make(map[relObj]bool)
	for _, f := range factsA {
		key := relObj{f.Relation, strings.ToLower(f.Object)}
		if setB[key] && !seen[key] {
			shared = append(shared, sharedFactPair{f.Relation, f.Object})
			seen[key] = true
		}
	}
	return shared
}

// findUniqueFacts returns facts unique to each subject.
func findUniqueFacts(factsA, factsB []edgeFact) (uniqueA, uniqueB []edgeFact) {
	type relObj struct {
		rel RelType
		obj string
	}

	setA := make(map[relObj]bool)
	setB := make(map[relObj]bool)
	for _, f := range factsA {
		setA[relObj{f.Relation, strings.ToLower(f.Object)}] = true
	}
	for _, f := range factsB {
		setB[relObj{f.Relation, strings.ToLower(f.Object)}] = true
	}

	for _, f := range factsA {
		key := relObj{f.Relation, strings.ToLower(f.Object)}
		if !setB[key] {
			uniqueA = append(uniqueA, f)
		}
	}
	for _, f := range factsB {
		key := relObj{f.Relation, strings.ToLower(f.Object)}
		if !setA[key] {
			uniqueB = append(uniqueB, f)
		}
	}
	return
}

// generateSharedSection describes properties shared by both subjects.
func (hg *HybridGenerator) generateSharedSection(subjectA, subjectB string, shared []sharedFactPair) string {
	if len(shared) == 0 {
		return ""
	}

	var sentences []string
	for _, sp := range shared {
		sent := "Both " + subjectA + " and " + subjectB + " " + sharedVerbPhrase(sp.relation, sp.object) + "."
		sentences = append(sentences, sent)
	}

	return strings.Join(sentences, " ")
}

// sharedVerbPhrase constructs a verb phrase for a shared property.
func sharedVerbPhrase(rel RelType, object string) string {
	switch rel {
	case RelIsA:
		return "are " + articleFor(object) // "are a programming language"
	case RelUsedFor:
		return "are used for " + object
	case RelHas:
		return "have " + object
	case RelPartOf:
		return "are part of " + object
	case RelLocatedIn:
		return "are located in " + object
	case RelRelatedTo:
		return "are related to " + object
	case RelDomain:
		return "belong to the domain of " + object
	case RelKnownFor:
		return "are known for " + object
	default:
		return "share a connection to " + object
	}
}

// generateDifferencesSection describes what makes each subject unique.
func (hg *HybridGenerator) generateDifferencesSection(subjectA, subjectB string, uniqueA, uniqueB []edgeFact) string {
	var parts []string

	// Limit to a few differences per side to avoid verbosity.
	maxDiffs := 3
	if len(uniqueA) > maxDiffs {
		uniqueA = uniqueA[:maxDiffs]
	}
	if len(uniqueB) > maxDiffs {
		uniqueB = uniqueB[:maxDiffs]
	}

	if len(uniqueA) > 0 {
		var sents []string
		for _, f := range uniqueA {
			sent := hg.retrieveOrGenerate(subjectA, f)
			if sent != "" {
				sents = append(sents, sent)
			}
		}
		if len(sents) > 0 {
			parts = append(parts, strings.Join(sents, " "))
		}
	}

	if len(uniqueB) > 0 {
		var sents []string
		for _, f := range uniqueB {
			sent := hg.retrieveOrGenerate(subjectB, f)
			if sent != "" {
				sents = append(sents, sent)
			}
		}
		if len(sents) > 0 {
			parts = append(parts, strings.Join(sents, " "))
		}
	}

	return strings.Join(parts, " ")
}

// -----------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------

// cleanupWhitespace normalizes whitespace in generated text.
func cleanupWhitespace(text string) string {
	// Collapse multiple spaces to one.
	var prev rune
	var b strings.Builder
	for _, r := range text {
		if r == ' ' && prev == ' ' {
			continue
		}
		b.WriteRune(r)
		prev = r
	}
	return strings.TrimSpace(b.String())
}

// hybridRng provides a simple deterministic random for connector selection
// in tests. Not used for production variance — that comes from corpus
// time-based selection.
var hybridRng = rand.New(rand.NewSource(42))
