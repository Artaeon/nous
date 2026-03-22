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

// indexWikiPackage scans a wiki package file and indexes its topics without
// loading facts into memory. Only reads fact subjects to build the index.
func (pl *PackageLoader) indexWikiPackage(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}

	var pkg KnowledgePackage
	if err := json.Unmarshal(data, &pkg); err != nil {
		return
	}

	// Index each unique subject to this file.
	// Don't overwrite existing entries — the first batch (alphabetically)
	// typically has the primary article with the best described_as fact.
	seen := make(map[string]bool)
	for _, f := range pkg.Facts {
		subj := strings.ToLower(f.Subject)
		if !seen[subj] {
			seen[subj] = true
			if _, exists := pl.wikiIndex[subj]; !exists {
				pl.wikiIndex[subj] = path
			}
		}
	}
}

// LookupWiki loads wiki knowledge on demand for a given topic.
// Returns the number of facts loaded (0 if topic not in index).
func (pl *PackageLoader) LookupWiki(topic string) int {
	topic = strings.ToLower(strings.TrimSpace(topic))
	if topic == "" {
		return 0
	}

	path, ok := pl.wikiIndex[topic]
	if !ok {
		// Try partial match — check if any indexed topic contains the query
		for indexed, p := range pl.wikiIndex {
			if strings.Contains(indexed, topic) || strings.Contains(topic, indexed) {
				path = p
				ok = true
				break
			}
		}
	}
	if !ok {
		return 0
	}

	// Already loaded this package?
	if pl.wikiLoaded[path] {
		return 0
	}

	result, err := pl.LoadFile(path)
	if err != nil {
		return 0
	}
	pl.wikiLoaded[path] = true
	return result.FactsLoaded
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
