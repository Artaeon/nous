package cognitive

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// -----------------------------------------------------------------------
// Knowledge Packages — instant bulk loading of world knowledge, vocabulary,
// and facts into Nous without any LLM or neural network.
//
// A package is a JSON file containing:
//   - Facts: subject→relation→object triples for the CognitiveGraph
//   - Vocabulary: new words for the generative engine's slot pools
//   - LongTermMemory entries: persistent key-value facts
//
// Packages are composable: load as many as you want, they merge additively.
// -----------------------------------------------------------------------

// KnowledgePackage is the on-disk format for a knowledge package.
type KnowledgePackage struct {
	// Metadata
	Name        string `json:"name"`
	Version     string `json:"version"`
	Description string `json:"description"`
	Author      string `json:"author,omitempty"`
	Domain      string `json:"domain"` // e.g. "science", "geography", "language"

	// Knowledge: subject-relation-object triples
	Facts []PackageFact `json:"facts,omitempty"`

	// Vocabulary expansions for the generative engine
	Vocabulary *VocabExpansion `json:"vocabulary,omitempty"`

	// Long-term memory entries (key-value world knowledge)
	Memories []PackageMemory `json:"memories,omitempty"`
}

// PackageFact is a single knowledge triple.
type PackageFact struct {
	Subject  string `json:"s"`
	Relation string `json:"r"` // matches RelType strings: "is_a", "has", "used_for", etc.
	Object   string `json:"o"`
}

// VocabExpansion adds words to the generative engine's slot pools.
type VocabExpansion struct {
	Adjectives      []string   `json:"adjectives,omitempty"`       // → adjSlots
	QualityNouns    []string   `json:"quality_nouns,omitempty"`    // → qualityNouns
	ImpactNouns     []string   `json:"impact_nouns,omitempty"`     // → impactNouns
	AttentionNouns  []string   `json:"attention_nouns,omitempty"`  // → attentionNouns
	MannerAdverbs   []string   `json:"manner_adverbs,omitempty"`   // → mannerAdvs
	Connectors      []string   `json:"connectors,omitempty"`       // → connVerbs
	Metaphors       []string   `json:"metaphors,omitempty"`        // → metaphorVehicles
	Punchlines      []string   `json:"punchlines,omitempty"`       // → punchlines
	ContrastPairs   [][2]string `json:"contrast_pairs,omitempty"` // → contrastPairs
	ValueAdjs       []string   `json:"value_adjectives,omitempty"` // → valueAdjs

	// Learned words (added to the engine's dynamic vocabulary)
	Nouns []string `json:"nouns,omitempty"`
	Verbs []string `json:"verbs,omitempty"`
}

// PackageMemory is a key-value fact for long-term memory.
type PackageMemory struct {
	Key      string `json:"key"`
	Value    string `json:"value"`
	Category string `json:"category"` // e.g. "fact", "definition", "historical"
}

// PackageLoader handles loading and managing knowledge packages.
type PackageLoader struct {
	graph     *CognitiveGraph
	engine    *GenerativeEngine
	composer  *Composer
	packDir   string            // directory containing .json packages
	loaded    map[string]string // name → version of loaded packages

	// Wiki on-demand loading: maps lowercase topic → package file path
	wikiIndex    map[string]string // topic → file path
	wikiLoaded   map[string]bool   // file path → already loaded
	MaxStartupFacts int            // max facts to load at startup (0 = unlimited)
}

// NewPackageLoader creates a loader wired to the cognitive systems.
func NewPackageLoader(graph *CognitiveGraph, engine *GenerativeEngine, composer *Composer, packDir string) *PackageLoader {
	return &PackageLoader{
		graph:       graph,
		engine:      engine,
		composer:    composer,
		packDir:     packDir,
		loaded:      make(map[string]string),
		wikiIndex:   make(map[string]string),
		wikiLoaded:  make(map[string]bool),
		MaxStartupFacts: 100000, // default: load up to 100K facts at startup
	}
}

// LoadFile loads a single knowledge package from a JSON file.
func (pl *PackageLoader) LoadFile(path string) (*PackageLoadResult, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read package %s: %w", path, err)
	}

	var pkg KnowledgePackage
	if err := json.Unmarshal(data, &pkg); err != nil {
		return nil, fmt.Errorf("parse package %s: %w", path, err)
	}

	return pl.Install(&pkg), nil
}

// LoadAll loads all .json packages from the package directory.
// Wiki batch packages (wiki-batch-*.json) are indexed for on-demand loading
// rather than loaded at startup, to keep memory usage reasonable.
func (pl *PackageLoader) LoadAll() ([]*PackageLoadResult, error) {
	if pl.packDir == "" {
		return nil, fmt.Errorf("no package directory configured")
	}

	totalFacts := 0
	var results []*PackageLoadResult
	var wikiFiles []string

	err := filepath.WalkDir(pl.packDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() || !strings.HasSuffix(d.Name(), ".json") {
			return nil
		}
		// Defer wiki batch packages for on-demand loading
		if strings.HasPrefix(d.Name(), "wiki-batch-") {
			wikiFiles = append(wikiFiles, path)
			return nil
		}
		// Check startup fact limit
		if pl.MaxStartupFacts > 0 && totalFacts >= pl.MaxStartupFacts {
			return nil
		}
		result, loadErr := pl.LoadFile(path)
		if loadErr != nil {
			return nil
		}
		totalFacts += result.FactsLoaded
		results = append(results, result)
		return nil
	})
	if err != nil {
		return nil, err
	}

	// Build wiki index: scan each wiki package for article subjects
	for _, wf := range wikiFiles {
		pl.indexWikiPackage(wf)
	}

	if len(pl.wikiIndex) > 0 {
		// Add a summary result for the indexed wiki (FactsLoaded=0 since not loaded yet)
		results = append(results, &PackageLoadResult{
			Name:    fmt.Sprintf("wikipedia-index (%d topics on-demand)", len(pl.wikiIndex)),
			Version: "1.0",
			Domain:  "wikipedia",
		})
	}

	return results, nil
}

// isFragmentSubject returns true if a fact subject is a sentence fragment
// (pronoun, determiner phrase, or other non-entity text) that shouldn't be
// indexed or loaded as a standalone topic.
func isFragmentSubject(subj string) bool {
	if len(subj) < 2 {
		return true
	}
	// Reject subjects that start with a lowercase letter — real entities are capitalised
	if subj[0] >= 'a' && subj[0] <= 'z' {
		return true
	}
	// Reject pronoun and determiner-initial subjects
	lower := strings.ToLower(subj)
	fragmentPrefixes := []string{
		"it ", "he ", "she ", "they ", "we ", "its ",
		"the ", "this ", "that ", "these ", "those ",
		"his ", "her ", "their ", "our ",
		"a ", "an ",
		"some ", "many ", "most ", "several ", "each ", "every ",
	}
	for _, pfx := range fragmentPrefixes {
		if strings.HasPrefix(lower, pfx) {
			return true
		}
	}
	// Reject bare pronouns
	bareFragments := map[string]bool{
		"it": true, "he": true, "she": true, "they": true,
		"we": true, "its": true, "them": true, "him": true,
		"her": true, "this": true, "that": true, "there": true,
		"these": true, "those": true,
	}
	if bareFragments[lower] {
		return true
	}
	// Reject overly long subjects (likely sentence fragments)
	if strings.Count(subj, " ") >= 6 {
		return true
	}
	// Reject subjects containing verbs — likely sentence fragments, not entities.
	// Only check for common finite verb forms after the first word.
	words := strings.Fields(lower)
	if len(words) >= 3 {
		verbForms := map[string]bool{
			"is": true, "are": true, "was": true, "were": true,
			"has": true, "had": true, "have": true,
			"can": true, "could": true, "would": true, "will": true,
			"should": true, "may": true, "might": true, "must": true,
		}
		for _, w := range words[1:] {
			if verbForms[w] {
				return true
			}
		}
	}
	return false
}

// indexWikiPackage scans a wiki package file and indexes its topics without
// loading facts into memory. Only reads fact subjects to build the index.
// Skips fragment subjects (pronouns, determiners, sentence fragments).
func (pl *PackageLoader) indexWikiPackage(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}

	var pkg KnowledgePackage
	if err := json.Unmarshal(data, &pkg); err != nil {
		return
	}

	// Index each unique non-fragment subject to this file.
	// Prefer batches that contain described_as facts — these are the
	// primary article with the lead paragraph. Other batches may have
	// stale triples from different articles that mention the same subject.
	seen := make(map[string]bool)
	describedSubjects := make(map[string]bool)

	// First pass: find subjects that have described_as in this batch.
	for _, f := range pkg.Facts {
		if f.Relation == "described_as" && !isFragmentSubject(f.Subject) {
			describedSubjects[strings.ToLower(f.Subject)] = true
		}
	}

	// Second pass: index subjects. described_as batches always win.
	for _, f := range pkg.Facts {
		if isFragmentSubject(f.Subject) {
			continue
		}
		subj := strings.ToLower(f.Subject)
		if !seen[subj] {
			seen[subj] = true
			_, exists := pl.wikiIndex[subj]
			if !exists || describedSubjects[subj] {
				pl.wikiIndex[subj] = path
			}
		}
	}
}

// LookupWiki loads wiki knowledge on demand for a given topic.
// Returns the number of facts loaded (0 if topic not in index).
//
// Loads exactly ONE batch file — the single best match. Never mixes
// facts from different entities. Resolution order:
//
//  1. Strip leading articles ("a black hole" → "black hole")
//  2. Normalize numbers ("world war 2" → also try "world war ii")
//  3. Exact match
//  4. Plural/singular variants
//  5. Disambiguation page (loads alongside the best match)
//  6. Single best partial match (surname/word-boundary)
//  7. Broader fallback (query contains an indexed topic)
func (pl *PackageLoader) LookupWiki(topic string) int {
	topic = strings.ToLower(strings.TrimSpace(topic))
	if topic == "" {
		return 0
	}

	// 1. Strip leading articles
	for _, art := range []string{"a ", "an ", "the "} {
		if strings.HasPrefix(topic, art) {
			stripped := topic[len(art):]
			if len(stripped) > 0 {
				if _, exact := pl.wikiIndex[topic]; !exact {
					topic = stripped
				}
			}
			break
		}
	}

	// 2. Build candidate forms: exact, number variants, plural variants
	candidates := []string{topic}
	candidates = append(candidates, normalizeNumbers(topic)...)
	candidates = append(candidates, pluralVariants(topic)...)

	// 3. Try exact match on all candidate forms
	for _, c := range candidates {
		if path, ok := pl.wikiIndex[c]; ok {
			loaded := pl.loadWikiIfNeeded(path)
			// Also load the disambiguation page if one exists
			pl.loadDisambiguation(topic)
			return loaded
		}
	}

	// 4. Single best partial match — find the ONE indexed topic that
	//    best matches the query. Prefer disambiguation pages, then
	//    surname matches, then shortest name.
	var bestPath string
	bestScore := 0
	bestLen := 0
	for indexed, p := range pl.wikiIndex {
		if !containsPhrase(indexed, topic) {
			continue
		}
		score := wikiMatchScore(indexed, topic)
		// Prefer higher score, then shorter name (more focused article)
		if score > bestScore || (score == bestScore && (bestLen == 0 || len(indexed) < bestLen)) {
			bestPath = p
			bestScore = score
			bestLen = len(indexed)
		}
	}
	if bestPath != "" {
		loaded := pl.loadWikiIfNeeded(bestPath)
		pl.loadDisambiguation(topic)
		return loaded
	}

	// 5. Broader fallback — query contains an indexed topic
	if strings.Contains(topic, " ") {
		var fallbackPath string
		fallbackLen := 0
		for indexed, p := range pl.wikiIndex {
			if len(indexed) >= 3 && containsPhrase(topic, indexed) && len(indexed) > fallbackLen {
				fallbackPath = p
				fallbackLen = len(indexed)
			}
		}
		if fallbackPath != "" {
			return pl.loadWikiIfNeeded(fallbackPath)
		}
	}

	return 0
}

// loadDisambiguation loads the disambiguation page for a topic if one exists.
// This allows LookupDescription to use the disambiguation page's lead
// sentence to identify the most notable entity.
func (pl *PackageLoader) loadDisambiguation(topic string) {
	disambKey := topic + " (disambiguation)"
	if path, ok := pl.wikiIndex[disambKey]; ok {
		pl.loadWikiIfNeeded(path)
	}
}

// normalizeNumbers returns alternate forms with number↔Roman numeral conversion.
// "world war 2" → ["world war ii"], "henry viii" → ["henry 8"]
func normalizeNumbers(topic string) []string {
	arabicToRoman := map[string]string{
		"1": "i", "2": "ii", "3": "iii", "4": "iv", "5": "v",
		"6": "vi", "7": "vii", "8": "viii", "9": "ix", "10": "x",
	}
	romanToArabic := map[string]string{
		"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
		"vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10",
	}

	words := strings.Fields(topic)
	var variants []string

	for i, w := range words {
		if roman, ok := arabicToRoman[w]; ok {
			alt := make([]string, len(words))
			copy(alt, words)
			alt[i] = roman
			variants = append(variants, strings.Join(alt, " "))
		}
		if arabic, ok := romanToArabic[w]; ok {
			alt := make([]string, len(words))
			copy(alt, words)
			alt[i] = arabic
			variants = append(variants, strings.Join(alt, " "))
		}
	}
	return variants
}

// wikiMatchScore scores how well an indexed topic matches a query.
// Higher scores indicate more notable/relevant matches.
//
// Scoring:
//   - 10: disambiguation page ("gandhi (disambiguation)")
//   - 8: query is the last word — likely a surname match ("mahatma gandhi" for "gandhi")
//   - 6: exactly "FirstName Query" pattern (2 words, query last)
//   - 4: query is the first word ("gandhi smriti" for "gandhi")
//   - 2: query appears somewhere in the middle
//   - 0: default
func wikiMatchScore(indexed, query string) int {
	// Disambiguation pages are authoritative
	if strings.Contains(indexed, "(disambiguation)") {
		return 10
	}
	words := strings.Fields(indexed)
	if len(words) == 0 {
		return 0
	}
	// Query matches the last word → surname match (most notable for people)
	if words[len(words)-1] == query {
		if len(words) == 2 {
			return 8 // "Firstname Lastname" — strongest person match
		}
		return 6
	}
	// Query matches the first word
	if words[0] == query {
		return 4
	}
	return 2
}

// loadWikiIfNeeded loads a wiki package if not already loaded.
func (pl *PackageLoader) loadWikiIfNeeded(path string) int {
	if pl.wikiLoaded[path] {
		return 0
	}
	result, err := pl.loadWikiFile(path)
	if err != nil {
		return 0
	}
	pl.wikiLoaded[path] = true
	return result.FactsLoaded
}

// pluralVariants returns singular/plural variants of a topic.
func pluralVariants(topic string) []string {
	var variants []string
	if strings.HasSuffix(topic, "s") {
		// Try singular: "black holes" → "black hole"
		variants = append(variants, strings.TrimSuffix(topic, "s"))
		if strings.HasSuffix(topic, "es") {
			variants = append(variants, strings.TrimSuffix(topic, "es"))
		}
		if strings.HasSuffix(topic, "ies") {
			variants = append(variants, strings.TrimSuffix(topic, "ies")+"y")
		}
	} else {
		// Try plural: "black hole" → "black holes"
		variants = append(variants, topic+"s")
		if strings.HasSuffix(topic, "y") {
			variants = append(variants, strings.TrimSuffix(topic, "y")+"ies")
		}
	}
	return variants
}

// containsPhrase checks if text contains phrase as a whole-word-boundary match
// (bounded by spaces or string edges). Unlike containsWord in tracker.go,
// this handles multi-word phrases like "world war ii".
func containsPhrase(text, phrase string) bool {
	word := phrase
	idx := strings.Index(text, word)
	for idx >= 0 {
		// Check left boundary
		leftOk := idx == 0 || text[idx-1] == ' '
		// Check right boundary
		end := idx + len(word)
		rightOk := end == len(text) || text[end] == ' '
		if leftOk && rightOk {
			return true
		}
		// Search for next occurrence
		next := strings.Index(text[idx+1:], word)
		if next < 0 {
			break
		}
		idx = idx + 1 + next
	}
	return false
}

// loadWikiFile loads a wiki package, filtering out fragment subjects.
func (pl *PackageLoader) loadWikiFile(path string) (*PackageLoadResult, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read package %s: %w", path, err)
	}

	var pkg KnowledgePackage
	if err := json.Unmarshal(data, &pkg); err != nil {
		return nil, fmt.Errorf("parse package %s: %w", path, err)
	}

	// Filter out fragment subjects before installing
	clean := make([]PackageFact, 0, len(pkg.Facts))
	for _, f := range pkg.Facts {
		if !isFragmentSubject(f.Subject) {
			clean = append(clean, f)
		}
	}
	pkg.Facts = clean

	return pl.Install(&pkg), nil
}

// HasWikiEntry returns true if the wiki index contains an exact match for the topic.
func (pl *PackageLoader) HasWikiEntry(topic string) bool {
	topic = strings.ToLower(strings.TrimSpace(topic))
	_, ok := pl.wikiIndex[topic]
	return ok
}

// WikiIndexSize returns the number of indexed wiki topics.
func (pl *PackageLoader) WikiIndexSize() int {
	return len(pl.wikiIndex)
}

// PackageLoadResult tracks what was loaded.
type PackageLoadResult struct {
	Name         string
	Version      string
	Domain       string
	FactsLoaded  int
	VocabLoaded  int
	MemsLoaded   int
	Duration     time.Duration
}

func (r *PackageLoadResult) String() string {
	return fmt.Sprintf("%s v%s [%s]: %d facts, %d vocab, %d memories (%.1fms)",
		r.Name, r.Version, r.Domain, r.FactsLoaded, r.VocabLoaded, r.MemsLoaded,
		float64(r.Duration.Microseconds())/1000.0)
}

// Install loads a package into the cognitive systems.
func (pl *PackageLoader) Install(pkg *KnowledgePackage) *PackageLoadResult {
	start := time.Now()
	result := &PackageLoadResult{
		Name:    pkg.Name,
		Version: pkg.Version,
		Domain:  pkg.Domain,
	}

	// 1. Load facts into the cognitive graph
	if pl.graph != nil {
		for _, f := range pkg.Facts {
			rel := parseRelString(f.Relation)
			pl.graph.AddEdge(f.Subject, f.Object, rel, "pkg:"+pkg.Name)

			// Also teach the generative engine about these entities
			if pl.engine != nil {
				pl.engine.LearnWord(f.Subject, POSNoun)
				pl.engine.LearnWord(f.Object, POSNoun)
			}
			result.FactsLoaded++
		}
	}

	// 2. Expand vocabulary
	if pkg.Vocabulary != nil {
		result.VocabLoaded = pl.expandVocab(pkg.Vocabulary)
	}

	// 3. Load long-term memories
	for _, m := range pkg.Memories {
		if pl.graph != nil {
			// Store as graph edges for queryability
			pl.graph.AddEdge(m.Key, m.Value, RelDescribedAs, "pkg:"+pkg.Name)
		}
		result.MemsLoaded++
	}

	result.Duration = time.Since(start)
	pl.loaded[pkg.Name] = pkg.Version
	return result
}

// expandVocab merges new vocabulary into the generative engine's slot pools.
func (pl *PackageLoader) expandVocab(v *VocabExpansion) int {
	count := 0
	count += appendUnique(&adjSlots, v.Adjectives)
	count += appendUnique(&qualityNouns, v.QualityNouns)
	count += appendUnique(&impactNouns, v.ImpactNouns)
	count += appendUnique(&attentionNouns, v.AttentionNouns)
	count += appendUnique(&mannerAdvs, v.MannerAdverbs)
	count += appendUnique(&valueAdjs, v.ValueAdjs)
	// connVerbs, metaphorVehicles, punchlines, contrastPairs removed — skip those fields.

	// Learned vocabulary (dynamic)
	if pl.engine != nil {
		for _, n := range v.Nouns {
			pl.engine.LearnWord(n, POSNoun)
			count++
		}
		for _, vb := range v.Verbs {
			pl.engine.LearnWord(vb, POSVerb)
			count++
		}
	}

	return count
}

// appendUnique adds only new entries to a slice.
func appendUnique(target *[]string, additions []string) int {
	existing := make(map[string]bool, len(*target))
	for _, s := range *target {
		existing[strings.ToLower(s)] = true
	}
	added := 0
	for _, s := range additions {
		if !existing[strings.ToLower(s)] {
			*target = append(*target, s)
			existing[strings.ToLower(s)] = true
			added++
		}
	}
	return added
}

// parseRelString converts a string to RelType.
func parseRelString(s string) RelType {
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

// ListLoaded returns all currently loaded packages.
func (pl *PackageLoader) ListLoaded() map[string]string {
	result := make(map[string]string, len(pl.loaded))
	for k, v := range pl.loaded {
		result[k] = v
	}
	return result
}

// CreatePackage is a helper to generate a package JSON file from structured data.
func CreatePackage(name, version, description, domain string, facts []PackageFact, vocab *VocabExpansion) ([]byte, error) {
	pkg := KnowledgePackage{
		Name:        name,
		Version:     version,
		Description: description,
		Domain:      domain,
		Facts:       facts,
		Vocabulary:  vocab,
	}
	return json.MarshalIndent(pkg, "", "  ")
}
