package cognitive

import (
	"math/rand"
	"regexp"
	"strings"
	"time"
)

// -----------------------------------------------------------------------
// Common Sense Knowledge Layer — associative knowledge that bridges
// everyday language to useful responses.
//
// Wikipedia tells Nous what things ARE. Common sense tells Nous what
// things MEAN in everyday life. When someone says "I'm bored", Wikipedia
// has a definition of boredom. Common sense knows to suggest activities.
//
// This is a knowledge layer, not a response generator. It provides
// associations that the composer and action router can use to connect
// user intent to real knowledge.
// -----------------------------------------------------------------------

// CSRelation classifies how two concepts are related in everyday knowledge.
type CSRelation int

const (
	CSRelatedTo    CSRelation = iota // general association
	CSUsedFor                        // dinner → eating
	CSCapableOf                      // car → driving
	CSHasProperty                    // sky → blue
	CSCausesDesire                   // bored → entertainment
	CSCausedBy                       // rain → clouds
	CSPartOf                         // chapter → book
	CSIsA                            // novel → book
	CSHasContext                     // promoted → work, career
)

// csRelationNames maps relation types to human-readable labels for composing suggestions.
var csRelationNames = map[CSRelation]string{
	CSRelatedTo:    "related to",
	CSUsedFor:      "used for",
	CSCapableOf:    "can",
	CSHasProperty:  "is",
	CSCausesDesire: "makes you want",
	CSCausedBy:     "caused by",
	CSPartOf:       "part of",
	CSIsA:          "is a type of",
	CSHasContext:   "in the context of",
}

// Association links a source concept to a target concept with a typed
// relationship and strength weight.
type Association struct {
	Target   string     // related concept
	Relation CSRelation // how they're related
	Weight   float64    // strength 0-1 (1 = strongest)
}

// CommonSenseGraph provides everyday associative knowledge.
// Maps concepts to related concepts with typed relationships.
type CommonSenseGraph struct {
	associations map[string][]Association
	rng          *rand.Rand
}

// ResolvedQuery is the result of mapping an everyday query to its
// underlying topic and context, so the rest of the system can find
// relevant knowledge.
type ResolvedQuery struct {
	Topic   string // the real subject ("food", "activity", "literature")
	Context string // what kind of answer is needed ("suggestion", "explanation")
	Raw     string // original query
}

// NewCommonSenseGraph creates a graph seeded with everyday knowledge.
func NewCommonSenseGraph() *CommonSenseGraph {
	csg := &CommonSenseGraph{
		associations: make(map[string][]Association),
		rng:          rand.New(rand.NewSource(time.Now().UnixNano())),
	}
	csg.seed()
	return csg
}

// Lookup returns all associations for a concept. Returns nil if unknown.
func (csg *CommonSenseGraph) Lookup(concept string) []Association {
	return csg.associations[strings.ToLower(strings.TrimSpace(concept))]
}

// LookupByRelation returns associations for a concept filtered by relation type.
func (csg *CommonSenseGraph) LookupByRelation(concept string, rel CSRelation) []Association {
	all := csg.Lookup(concept)
	var filtered []Association
	for _, a := range all {
		if a.Relation == rel {
			filtered = append(filtered, a)
		}
	}
	return filtered
}

// Size returns the total number of associations in the graph.
func (csg *CommonSenseGraph) Size() int {
	total := 0
	for _, assocs := range csg.associations {
		total += len(assocs)
	}
	return total
}

// TopicCount returns the number of distinct source concepts.
func (csg *CommonSenseGraph) TopicCount() int {
	return len(csg.associations)
}

// Add inserts an association into the graph.
func (csg *CommonSenseGraph) Add(source string, target string, rel CSRelation, weight float64) {
	key := strings.ToLower(strings.TrimSpace(source))
	if key == "" || target == "" {
		return
	}
	if weight < 0 {
		weight = 0
	}
	if weight > 1 {
		weight = 1
	}
	// Deduplicate: if an identical source→target→relation exists, keep the higher weight.
	for i, existing := range csg.associations[key] {
		if strings.EqualFold(existing.Target, target) && existing.Relation == rel {
			if weight > existing.Weight {
				csg.associations[key][i].Weight = weight
			}
			return
		}
	}
	csg.associations[key] = append(csg.associations[key], Association{
		Target:   target,
		Relation: rel,
		Weight:   weight,
	})
}

// AddBatch adds multiple associations for a single source concept.
func (csg *CommonSenseGraph) AddBatch(source string, associations []Association) {
	for _, a := range associations {
		csg.Add(source, a.Target, a.Relation, a.Weight)
	}
}

// -----------------------------------------------------------------------
// Suggest — compose practical suggestions from the association graph.
// -----------------------------------------------------------------------

// suggestionTemplate defines how to turn an association into a natural suggestion.
type suggestionTemplate struct {
	relation CSRelation
	patterns []string // %s = target concept
}

var defaultTemplates = []suggestionTemplate{
	{CSRelatedTo, []string{
		"You might enjoy %s",
		"How about %s?",
		"Consider %s",
		"%s could be worth trying",
	}},
	{CSUsedFor, []string{
		"%s is a solid choice",
		"Try %s",
		"%s works well for this",
	}},
	{CSCapableOf, []string{
		"%s is great for this",
		"You could try %s",
	}},
	{CSHasProperty, []string{
		"It's known for being %s",
		"One thing to note: it's %s",
	}},
	{CSCausesDesire, []string{
		"You might want %s",
		"That often calls for %s",
		"%s could help",
	}},
	{CSIsA, []string{
		"%s is a good option in that category",
		"There's always %s",
	}},
}

// templateForRelation returns a random template string for the given relation.
func (csg *CommonSenseGraph) templateForRelation(rel CSRelation) string {
	for _, st := range defaultTemplates {
		if st.relation == rel {
			return st.patterns[csg.rng.Intn(len(st.patterns))]
		}
	}
	// Fallback
	fallback := []string{"How about %s?", "Consider %s", "You might try %s"}
	return fallback[csg.rng.Intn(len(fallback))]
}

// Suggest returns practical suggestions for a topic, composed from the
// association graph. The context parameter biases which associations are
// most relevant (e.g. "recommendation", "activity", "explanation").
func (csg *CommonSenseGraph) Suggest(topic string, context string) []string {
	key := strings.ToLower(strings.TrimSpace(topic))
	assocs := csg.associations[key]
	if len(assocs) == 0 {
		return nil
	}

	// Score associations by relevance to context
	type scored struct {
		assoc Association
		score float64
	}
	var candidates []scored
	for _, a := range assocs {
		s := a.Weight
		s += csg.contextBoost(a, context)
		candidates = append(candidates, scored{a, s})
	}

	// Sort by score descending
	for i := 0; i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].score > candidates[i].score {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	// Take top suggestions, compose from templates
	max := 5
	if len(candidates) < max {
		max = len(candidates)
	}

	seen := make(map[string]bool)
	var suggestions []string
	for _, c := range candidates[:max] {
		// Skip property associations for suggestion context — they describe,
		// they don't suggest.
		if c.assoc.Relation == CSHasProperty && context != "explanation" {
			continue
		}
		tmpl := csg.templateForRelation(c.assoc.Relation)
		text := strings.Replace(tmpl, "%s", c.assoc.Target, 1)
		lower := strings.ToLower(text)
		if !seen[lower] {
			seen[lower] = true
			suggestions = append(suggestions, text)
		}
	}

	return suggestions
}

// contextBoost gives a score bonus to associations that match the query context.
func (csg *CommonSenseGraph) contextBoost(a Association, context string) float64 {
	ctx := strings.ToLower(context)
	switch {
	case strings.Contains(ctx, "suggest") || strings.Contains(ctx, "recommend"):
		switch a.Relation {
		case CSRelatedTo, CSUsedFor, CSIsA:
			return 0.3
		case CSCausesDesire:
			return 0.2
		}
	case strings.Contains(ctx, "explain") || strings.Contains(ctx, "why"):
		switch a.Relation {
		case CSCausedBy, CSHasProperty:
			return 0.4
		}
	case strings.Contains(ctx, "activ") || strings.Contains(ctx, "do"):
		switch a.Relation {
		case CSCausesDesire, CSCapableOf, CSRelatedTo:
			return 0.3
		}
	case strings.Contains(ctx, "meal") || strings.Contains(ctx, "food"):
		switch a.Relation {
		case CSRelatedTo, CSUsedFor, CSIsA:
			return 0.3
		}
	}
	return 0
}

// -----------------------------------------------------------------------
// Resolve — map everyday language to underlying topics.
// -----------------------------------------------------------------------

// resolvePattern is a compiled pattern for query resolution.
type resolvePattern struct {
	re      *regexp.Regexp
	topic   string
	context string
}

// resolvePatterns are compiled once and reused.
var resolvePatterns = func() []resolvePattern {
	raw := []struct {
		pattern string
		topic   string
		context string
	}{
		// Food/meal queries
		{`(?i)\b(?:what|what's)\b.*\b(?:eat|have|cook|make)\b.*\b(?:dinner|lunch|breakfast|supper|meal)\b`, "food", "meal suggestion"},
		{`(?i)\b(?:what|what's)\b.*\b(?:for|to)\b.*\b(?:dinner|lunch|breakfast|supper)\b`, "food", "meal suggestion"},
		{`(?i)\b(?:dinner|lunch|breakfast|supper|meal)\b.*\b(?:idea|suggest|recommend)\b`, "food", "meal suggestion"},
		{`(?i)\b(?:suggest|recommend)\b.*\b(?:food|meal|dish|recipe|dinner|lunch|breakfast)\b`, "food", "meal suggestion"},
		{`(?i)\bhungry\b`, "food", "meal suggestion"},

		// Boredom / activity queries
		{`(?i)\b(?:i'?m|i am|feeling)\b.*\bbored\b`, "activity", "suggestion"},
		{`(?i)\b(?:what|something)\b.*\b(?:do|try)\b`, "activity", "suggestion"},
		{`(?i)\b(?:bored|boring|nothing to do)\b`, "activity", "suggestion"},

		// Recommendation queries
		{`(?i)\brecommend\b.*\bbook\b`, "literature", "recommendation"},
		{`(?i)\bbook\b.*\brecommend`, "literature", "recommendation"},
		{`(?i)\b(?:good|best|great)\b.*\bbook`, "literature", "recommendation"},
		{`(?i)\b(?:suggest|recommend)\b.*\b(?:movie|film)\b`, "film", "recommendation"},
		{`(?i)\b(?:movie|film)\b.*\b(?:suggest|recommend)\b`, "film", "recommendation"},
		{`(?i)\b(?:good|best|great)\b.*\b(?:movie|film)\b`, "film", "recommendation"},
		{`(?i)\b(?:suggest|recommend)\b.*\bmusic\b`, "music", "recommendation"},
		{`(?i)\b(?:suggest|recommend)\b.*\b(?:song|album|artist|band)\b`, "music", "recommendation"},
		{`(?i)\b(?:suggest|recommend)\b.*\b(?:game|video game)\b`, "games", "recommendation"},
		{`(?i)\b(?:suggest|recommend)\b.*\b(?:show|tv|series)\b`, "television", "recommendation"},

		// Emotional / social queries
		{`(?i)\b(?:i'?m|i am|feeling)\b.*\b(?:stressed|anxious|worried|overwhelmed)\b`, "stress relief", "suggestion"},
		{`(?i)\b(?:i'?m|i am|feeling)\b.*\b(?:sad|down|depressed|lonely|upset)\b`, "emotional support", "suggestion"},
		{`(?i)\b(?:i'?m|i am|feeling)\b.*\b(?:happy|excited|great|wonderful)\b`, "celebration", "response"},
		{`(?i)\b(?:i|we)\b.*\b(?:got promoted|promotion|raise)\b`, "career", "congratulation"},
		{`(?i)\b(?:i|we)\b.*\b(?:got fired|laid off|lost my job)\b`, "career", "empathy"},

		// Science / explanation queries
		{`(?i)\bwhy\b.*\bsky\b.*\bblue\b`, "Rayleigh scattering", "explanation"},
		{`(?i)\bwhy\b.*\b(?:rain|rains)\b`, "precipitation", "explanation"},
		{`(?i)\bwhy\b.*\b(?:hot|cold|warm|cool)\b`, "temperature", "explanation"},
		{`(?i)\bhow\b.*\b(?:rainbow|rainbows)\b`, "light refraction", "explanation"},

		// Health queries
		{`(?i)\b(?:i have|i'?ve got|got)\b.*\bheadache\b`, "headache", "remedy"},
		{`(?i)\b(?:i'?m|i am|feeling)\b.*\btired\b`, "fatigue", "remedy"},
		{`(?i)\bcan'?t\b.*\bsleep\b`, "insomnia", "remedy"},
		{`(?i)\b(?:i'?m|i am|feeling)\b.*\bsick\b`, "illness", "remedy"},
	}

	patterns := make([]resolvePattern, len(raw))
	for i, r := range raw {
		patterns[i] = resolvePattern{
			re:      regexp.MustCompile(r.pattern),
			topic:   r.topic,
			context: r.context,
		}
	}
	return patterns
}()

// Resolve maps an everyday query to its underlying topic and context.
// Returns nil if the query doesn't match any known everyday pattern.
func (csg *CommonSenseGraph) Resolve(query string) *ResolvedQuery {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil
	}

	for _, rp := range resolvePatterns {
		if rp.re.MatchString(query) {
			return &ResolvedQuery{
				Topic:   rp.topic,
				Context: rp.context,
				Raw:     query,
			}
		}
	}

	return nil
}

// -----------------------------------------------------------------------
// ExtractAssociations — mine common sense from Wikipedia articles.
// -----------------------------------------------------------------------

// csSeeAlsoRe matches "== See also ==" sections in wiki markup.
var csSeeAlsoRe = regexp.MustCompile(`(?i)==\s*see\s+also\s*==`)

// csLinkRe matches [[Target]] or [[Target|Display]] wiki links.
var csLinkRe = regexp.MustCompile(`\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]`)

// csCategoryRe matches [[Category:Name]] markup with capture group for the name.
var csCategoryRe = regexp.MustCompile(`(?i)\[\[Category:([^\]|]+?)(?:\|[^\]]+?)?\]\]`)

// ExtractAssociations mines common sense associations from a Wikipedia article.
// It extracts:
//   - "See also" links as CSRelatedTo associations
//   - Category memberships as CSIsA or CSHasContext associations
//   - Inline links in the first paragraph as CSRelatedTo associations
func ExtractAssociations(title, text string) []Association {
	if title == "" || text == "" {
		return nil
	}

	var assocs []Association
	seen := make(map[string]bool) // avoid duplicates

	addUnique := func(target string, rel CSRelation, weight float64) {
		target = strings.TrimSpace(target)
		if target == "" || strings.EqualFold(target, title) {
			return
		}
		key := strings.ToLower(target) + "|" + string(rune(rel))
		if seen[key] {
			return
		}
		seen[key] = true
		assocs = append(assocs, Association{
			Target:   target,
			Relation: rel,
			Weight:   weight,
		})
	}

	// 1. Extract "See also" links — strong related-to signals
	if loc := csSeeAlsoRe.FindStringIndex(text); loc != nil {
		seeAlsoSection := text[loc[1]:]
		// Section ends at next heading or end of text
		if nextHead := strings.Index(seeAlsoSection, "=="); nextHead > 0 {
			seeAlsoSection = seeAlsoSection[:nextHead]
		}
		for _, match := range csLinkRe.FindAllStringSubmatch(seeAlsoSection, -1) {
			if len(match) >= 2 {
				addUnique(match[1], CSRelatedTo, 0.8)
			}
		}
	}

	// 2. Extract categories — contextual associations
	for _, match := range csCategoryRe.FindAllStringSubmatch(text, -1) {
		if len(match) >= 2 {
			cat := match[1]
			// Skip maintenance categories
			if strings.Contains(strings.ToLower(cat), "stub") ||
				strings.Contains(strings.ToLower(cat), "articles") ||
				strings.Contains(strings.ToLower(cat), "pages") ||
				strings.Contains(strings.ToLower(cat), "wikipedia") ||
				strings.Contains(strings.ToLower(cat), "webarchive") ||
				strings.Contains(strings.ToLower(cat), "cs1") {
				continue
			}
			addUnique(cat, CSHasContext, 0.6)
		}
	}

	// 3. Extract links from first paragraph — topically related
	firstPara := text
	if nlIdx := strings.Index(text, "\n\n"); nlIdx > 0 {
		firstPara = text[:nlIdx]
	}
	// Limit to first 500 chars to avoid noise
	if len(firstPara) > 500 {
		firstPara = firstPara[:500]
	}
	for _, match := range csLinkRe.FindAllStringSubmatch(firstPara, 10) {
		if len(match) >= 2 {
			addUnique(match[1], CSRelatedTo, 0.5)
		}
	}

	return assocs
}

// -----------------------------------------------------------------------
// LoadIntoGraph installs common sense associations into the CognitiveGraph,
// bridging the common sense layer to the existing knowledge infrastructure.
// -----------------------------------------------------------------------

// LoadIntoGraph adds all common sense associations as edges in the
// cognitive graph, tagged with source "commonsense".
func (csg *CommonSenseGraph) LoadIntoGraph(graph *CognitiveGraph) int {
	if graph == nil {
		return 0
	}
	count := 0
	for source, assocs := range csg.associations {
		for _, a := range assocs {
			rel := csRelToGraphRel(a.Relation)
			graph.AddEdge(source, a.Target, rel, "commonsense")
			count++
		}
	}
	return count
}

// csRelToGraphRel maps CSRelation to the existing RelType system.
func csRelToGraphRel(r CSRelation) RelType {
	switch r {
	case CSIsA:
		return RelIsA
	case CSPartOf:
		return RelPartOf
	case CSUsedFor:
		return RelUsedFor
	case CSRelatedTo:
		return RelRelatedTo
	case CSCausesDesire:
		return RelCauses
	case CSCausedBy:
		return RelCauses
	case CSHasProperty:
		return RelHas
	case CSCapableOf:
		return RelUsedFor
	case CSHasContext:
		return RelDomain
	default:
		return RelRelatedTo
	}
}

// -----------------------------------------------------------------------
// Seed data — comprehensive everyday associations.
// -----------------------------------------------------------------------

func (csg *CommonSenseGraph) seed() {
	// -------------------------------------------------------------------
	// FOOD & MEALS
	// -------------------------------------------------------------------
	csg.seedFood()

	// -------------------------------------------------------------------
	// ACTIVITIES & HOBBIES
	// -------------------------------------------------------------------
	csg.seedActivities()

	// -------------------------------------------------------------------
	// EMOTIONS & SOCIAL
	// -------------------------------------------------------------------
	csg.seedEmotions()

	// -------------------------------------------------------------------
	// RECOMMENDATIONS (books, movies, music, games)
	// -------------------------------------------------------------------
	csg.seedRecommendations()

	// -------------------------------------------------------------------
	// EVERYDAY OBJECTS & TECHNOLOGY
	// -------------------------------------------------------------------
	csg.seedObjects()

	// -------------------------------------------------------------------
	// WEATHER & NATURE
	// -------------------------------------------------------------------
	csg.seedNature()

	// -------------------------------------------------------------------
	// HEALTH & WELLNESS
	// -------------------------------------------------------------------
	csg.seedHealth()

	// -------------------------------------------------------------------
	// SOCIAL & RELATIONSHIPS
	// -------------------------------------------------------------------
	csg.seedSocial()

	// -------------------------------------------------------------------
	// WORK & CAREER
	// -------------------------------------------------------------------
	csg.seedWork()

	// -------------------------------------------------------------------
	// HOME & DAILY LIFE
	// -------------------------------------------------------------------
	csg.seedHome()

	// -------------------------------------------------------------------
	// TRAVEL & TRANSPORTATION
	// -------------------------------------------------------------------
	csg.seedTravel()
}

func (csg *CommonSenseGraph) seedFood() {
	// Dinner
	csg.Add("dinner", "pasta", CSRelatedTo, 0.9)
	csg.Add("dinner", "salad", CSRelatedTo, 0.8)
	csg.Add("dinner", "soup", CSRelatedTo, 0.8)
	csg.Add("dinner", "stir-fry", CSRelatedTo, 0.8)
	csg.Add("dinner", "sandwich", CSRelatedTo, 0.7)
	csg.Add("dinner", "curry", CSRelatedTo, 0.8)
	csg.Add("dinner", "pizza", CSRelatedTo, 0.9)
	csg.Add("dinner", "rice and beans", CSRelatedTo, 0.7)
	csg.Add("dinner", "tacos", CSRelatedTo, 0.8)
	csg.Add("dinner", "roast chicken", CSRelatedTo, 0.8)
	csg.Add("dinner", "grilled fish", CSRelatedTo, 0.7)
	csg.Add("dinner", "stew", CSRelatedTo, 0.7)
	csg.Add("dinner", "evening meal", CSHasProperty, 0.9)
	csg.Add("dinner", "eating", CSUsedFor, 0.9)
	csg.Add("dinner", "meal", CSIsA, 0.9)
	csg.Add("dinner", "cooking", CSRelatedTo, 0.7)
	csg.Add("dinner", "restaurant", CSRelatedTo, 0.6)
	csg.Add("dinner", "family time", CSRelatedTo, 0.6)

	// Lunch
	csg.Add("lunch", "sandwich", CSRelatedTo, 0.9)
	csg.Add("lunch", "salad", CSRelatedTo, 0.8)
	csg.Add("lunch", "soup", CSRelatedTo, 0.8)
	csg.Add("lunch", "wrap", CSRelatedTo, 0.7)
	csg.Add("lunch", "burger", CSRelatedTo, 0.7)
	csg.Add("lunch", "leftovers", CSRelatedTo, 0.8)
	csg.Add("lunch", "sushi", CSRelatedTo, 0.7)
	csg.Add("lunch", "meal", CSIsA, 0.9)
	csg.Add("lunch", "midday meal", CSHasProperty, 0.9)
	csg.Add("lunch", "eating", CSUsedFor, 0.9)
	csg.Add("lunch", "lunch break", CSRelatedTo, 0.7)

	// Breakfast
	csg.Add("breakfast", "eggs", CSRelatedTo, 0.9)
	csg.Add("breakfast", "toast", CSRelatedTo, 0.9)
	csg.Add("breakfast", "cereal", CSRelatedTo, 0.8)
	csg.Add("breakfast", "oatmeal", CSRelatedTo, 0.8)
	csg.Add("breakfast", "pancakes", CSRelatedTo, 0.8)
	csg.Add("breakfast", "fruit", CSRelatedTo, 0.8)
	csg.Add("breakfast", "yoghurt", CSRelatedTo, 0.7)
	csg.Add("breakfast", "coffee", CSRelatedTo, 0.9)
	csg.Add("breakfast", "smoothie", CSRelatedTo, 0.7)
	csg.Add("breakfast", "meal", CSIsA, 0.9)
	csg.Add("breakfast", "morning meal", CSHasProperty, 0.9)
	csg.Add("breakfast", "eating", CSUsedFor, 0.9)

	// General food concepts
	csg.Add("food", "cooking", CSRelatedTo, 0.8)
	csg.Add("food", "nutrition", CSRelatedTo, 0.7)
	csg.Add("food", "recipe", CSRelatedTo, 0.8)
	csg.Add("food", "grocery", CSRelatedTo, 0.7)
	csg.Add("food", "eating", CSUsedFor, 0.9)
	csg.Add("food", "sustenance", CSUsedFor, 0.8)
	csg.Add("food", "enjoyment", CSUsedFor, 0.7)

	// Cuisines
	csg.Add("cuisine", "Italian", CSRelatedTo, 0.8)
	csg.Add("cuisine", "Chinese", CSRelatedTo, 0.8)
	csg.Add("cuisine", "Mexican", CSRelatedTo, 0.8)
	csg.Add("cuisine", "Japanese", CSRelatedTo, 0.8)
	csg.Add("cuisine", "Indian", CSRelatedTo, 0.8)
	csg.Add("cuisine", "Thai", CSRelatedTo, 0.7)
	csg.Add("cuisine", "French", CSRelatedTo, 0.7)
	csg.Add("cuisine", "Mediterranean", CSRelatedTo, 0.7)
	csg.Add("cuisine", "Korean", CSRelatedTo, 0.7)
	csg.Add("cuisine", "Vietnamese", CSRelatedTo, 0.7)

	// Cooking
	csg.Add("cooking", "recipe", CSRelatedTo, 0.9)
	csg.Add("cooking", "ingredients", CSRelatedTo, 0.8)
	csg.Add("cooking", "preparing food", CSUsedFor, 0.9)
	csg.Add("cooking", "kitchen", CSHasContext, 0.8)
	csg.Add("cooking", "baking", CSRelatedTo, 0.7)
	csg.Add("cooking", "grilling", CSRelatedTo, 0.7)
	csg.Add("cooking", "roasting", CSRelatedTo, 0.7)

	// Snack
	csg.Add("snack", "fruit", CSRelatedTo, 0.8)
	csg.Add("snack", "nuts", CSRelatedTo, 0.8)
	csg.Add("snack", "chips", CSRelatedTo, 0.7)
	csg.Add("snack", "chocolate", CSRelatedTo, 0.7)
	csg.Add("snack", "crackers", CSRelatedTo, 0.7)
	csg.Add("snack", "popcorn", CSRelatedTo, 0.7)
	csg.Add("snack", "small meal", CSIsA, 0.8)
	csg.Add("snack", "between meals", CSHasContext, 0.8)

	// Hungry
	csg.Add("hungry", "eating", CSCausesDesire, 0.9)
	csg.Add("hungry", "food", CSCausesDesire, 0.9)
	csg.Add("hungry", "snack", CSCausesDesire, 0.8)
	csg.Add("hungry", "cooking", CSCausesDesire, 0.7)
	csg.Add("hungry", "not eating", CSCausedBy, 0.8)
}

func (csg *CommonSenseGraph) seedActivities() {
	// Boredom → activities
	csg.Add("bored", "go for a walk", CSCausesDesire, 0.9)
	csg.Add("bored", "read a book", CSCausesDesire, 0.9)
	csg.Add("bored", "learn something new", CSCausesDesire, 0.8)
	csg.Add("bored", "play a game", CSCausesDesire, 0.8)
	csg.Add("bored", "watch a movie", CSCausesDesire, 0.8)
	csg.Add("bored", "exercise", CSCausesDesire, 0.8)
	csg.Add("bored", "call a friend", CSCausesDesire, 0.7)
	csg.Add("bored", "cook something new", CSCausesDesire, 0.7)
	csg.Add("bored", "start a project", CSCausesDesire, 0.7)
	csg.Add("bored", "go outside", CSCausesDesire, 0.8)
	csg.Add("bored", "listen to music", CSCausesDesire, 0.7)
	csg.Add("bored", "write or journal", CSCausesDesire, 0.6)
	csg.Add("bored", "explore a new hobby", CSCausesDesire, 0.7)
	csg.Add("bored", "draw or sketch", CSCausesDesire, 0.6)
	csg.Add("bored", "do a puzzle", CSCausesDesire, 0.7)
	csg.Add("bored", "lack of stimulation", CSCausedBy, 0.8)

	// Activities general
	csg.Add("activity", "walking", CSRelatedTo, 0.8)
	csg.Add("activity", "reading", CSRelatedTo, 0.8)
	csg.Add("activity", "exercise", CSRelatedTo, 0.8)
	csg.Add("activity", "gaming", CSRelatedTo, 0.7)
	csg.Add("activity", "cooking", CSRelatedTo, 0.7)
	csg.Add("activity", "sports", CSRelatedTo, 0.7)
	csg.Add("activity", "gardening", CSRelatedTo, 0.7)
	csg.Add("activity", "hiking", CSRelatedTo, 0.7)
	csg.Add("activity", "swimming", CSRelatedTo, 0.6)
	csg.Add("activity", "cycling", CSRelatedTo, 0.6)
	csg.Add("activity", "yoga", CSRelatedTo, 0.6)
	csg.Add("activity", "meditation", CSRelatedTo, 0.6)
	csg.Add("activity", "painting", CSRelatedTo, 0.6)
	csg.Add("activity", "photography", CSRelatedTo, 0.6)
	csg.Add("activity", "volunteering", CSRelatedTo, 0.5)

	// Hobbies
	csg.Add("hobby", "reading", CSRelatedTo, 0.8)
	csg.Add("hobby", "gardening", CSRelatedTo, 0.8)
	csg.Add("hobby", "painting", CSRelatedTo, 0.7)
	csg.Add("hobby", "photography", CSRelatedTo, 0.7)
	csg.Add("hobby", "woodworking", CSRelatedTo, 0.6)
	csg.Add("hobby", "knitting", CSRelatedTo, 0.6)
	csg.Add("hobby", "playing music", CSRelatedTo, 0.7)
	csg.Add("hobby", "collecting", CSRelatedTo, 0.6)
	csg.Add("hobby", "writing", CSRelatedTo, 0.7)
	csg.Add("hobby", "cooking", CSRelatedTo, 0.7)
	csg.Add("hobby", "coding", CSRelatedTo, 0.6)
	csg.Add("hobby", "chess", CSRelatedTo, 0.6)
	csg.Add("hobby", "fishing", CSRelatedTo, 0.6)
	csg.Add("hobby", "personal interest", CSIsA, 0.9)

	// Exercise
	csg.Add("exercise", "running", CSRelatedTo, 0.9)
	csg.Add("exercise", "walking", CSRelatedTo, 0.8)
	csg.Add("exercise", "swimming", CSRelatedTo, 0.7)
	csg.Add("exercise", "cycling", CSRelatedTo, 0.7)
	csg.Add("exercise", "yoga", CSRelatedTo, 0.7)
	csg.Add("exercise", "weight training", CSRelatedTo, 0.7)
	csg.Add("exercise", "stretching", CSRelatedTo, 0.6)
	csg.Add("exercise", "health", CSUsedFor, 0.9)
	csg.Add("exercise", "fitness", CSUsedFor, 0.9)
	csg.Add("exercise", "stress relief", CSUsedFor, 0.8)
	csg.Add("exercise", "physical activity", CSIsA, 0.9)
}

func (csg *CommonSenseGraph) seedEmotions() {
	// Stress
	csg.Add("stressed", "take a break", CSCausesDesire, 0.9)
	csg.Add("stressed", "deep breathing", CSCausesDesire, 0.8)
	csg.Add("stressed", "go for a walk", CSCausesDesire, 0.8)
	csg.Add("stressed", "talk to someone", CSCausesDesire, 0.8)
	csg.Add("stressed", "rest", CSCausesDesire, 0.8)
	csg.Add("stressed", "listen to music", CSCausesDesire, 0.7)
	csg.Add("stressed", "exercise", CSCausesDesire, 0.7)
	csg.Add("stressed", "take a bath", CSCausesDesire, 0.6)
	csg.Add("stressed", "meditate", CSCausesDesire, 0.7)
	csg.Add("stressed", "overwork", CSCausedBy, 0.8)
	csg.Add("stressed", "pressure", CSCausedBy, 0.8)
	csg.Add("stressed", "anxiety", CSRelatedTo, 0.7)

	csg.Add("stress relief", "exercise", CSRelatedTo, 0.8)
	csg.Add("stress relief", "meditation", CSRelatedTo, 0.8)
	csg.Add("stress relief", "deep breathing", CSRelatedTo, 0.9)
	csg.Add("stress relief", "walk in nature", CSRelatedTo, 0.8)
	csg.Add("stress relief", "talking to a friend", CSRelatedTo, 0.7)
	csg.Add("stress relief", "warm bath", CSRelatedTo, 0.7)
	csg.Add("stress relief", "music", CSRelatedTo, 0.7)
	csg.Add("stress relief", "reading", CSRelatedTo, 0.6)
	csg.Add("stress relief", "good sleep", CSRelatedTo, 0.8)

	// Sadness
	csg.Add("sad", "talk to someone", CSCausesDesire, 0.9)
	csg.Add("sad", "rest", CSCausesDesire, 0.7)
	csg.Add("sad", "comfort food", CSCausesDesire, 0.6)
	csg.Add("sad", "watch something uplifting", CSCausesDesire, 0.7)
	csg.Add("sad", "go outside", CSCausesDesire, 0.7)
	csg.Add("sad", "write about your feelings", CSCausesDesire, 0.6)
	csg.Add("sad", "listen to music", CSCausesDesire, 0.7)

	csg.Add("emotional support", "listening", CSRelatedTo, 0.9)
	csg.Add("emotional support", "empathy", CSRelatedTo, 0.9)
	csg.Add("emotional support", "talking to a friend", CSRelatedTo, 0.8)
	csg.Add("emotional support", "self-care", CSRelatedTo, 0.8)
	csg.Add("emotional support", "rest", CSRelatedTo, 0.7)
	csg.Add("emotional support", "professional help", CSRelatedTo, 0.7)

	// Happiness / celebration
	csg.Add("happy", "celebration", CSRelatedTo, 0.8)
	csg.Add("happy", "share the news", CSCausesDesire, 0.7)
	csg.Add("happy", "gratitude", CSRelatedTo, 0.7)
	csg.Add("happy", "positive event", CSCausedBy, 0.8)

	csg.Add("celebration", "party", CSRelatedTo, 0.8)
	csg.Add("celebration", "dinner out", CSRelatedTo, 0.7)
	csg.Add("celebration", "sharing with friends", CSRelatedTo, 0.8)
	csg.Add("celebration", "treat yourself", CSRelatedTo, 0.7)
	csg.Add("celebration", "cake", CSRelatedTo, 0.6)

	// Promoted / career success
	csg.Add("promoted", "congratulations", CSRelatedTo, 0.9)
	csg.Add("promoted", "celebration", CSCausesDesire, 0.9)
	csg.Add("promoted", "career advancement", CSIsA, 0.9)
	csg.Add("promoted", "hard work", CSCausedBy, 0.8)
	csg.Add("promoted", "work", CSHasContext, 0.9)
	csg.Add("promoted", "career", CSHasContext, 0.9)

	// Anxiety
	csg.Add("anxious", "deep breathing", CSCausesDesire, 0.9)
	csg.Add("anxious", "grounding exercises", CSCausesDesire, 0.8)
	csg.Add("anxious", "talk to someone", CSCausesDesire, 0.8)
	csg.Add("anxious", "take things one step at a time", CSCausesDesire, 0.7)
	csg.Add("anxious", "uncertainty", CSCausedBy, 0.8)

	// Lonely
	csg.Add("lonely", "call a friend", CSCausesDesire, 0.9)
	csg.Add("lonely", "join a group or club", CSCausesDesire, 0.7)
	csg.Add("lonely", "visit a public place", CSCausesDesire, 0.6)
	csg.Add("lonely", "reach out to family", CSCausesDesire, 0.8)
	csg.Add("lonely", "isolation", CSCausedBy, 0.8)
}

func (csg *CommonSenseGraph) seedRecommendations() {
	// Books / literature
	csg.Add("literature", "fiction", CSRelatedTo, 0.8)
	csg.Add("literature", "non-fiction", CSRelatedTo, 0.8)
	csg.Add("literature", "science fiction", CSRelatedTo, 0.7)
	csg.Add("literature", "fantasy", CSRelatedTo, 0.7)
	csg.Add("literature", "mystery", CSRelatedTo, 0.7)
	csg.Add("literature", "history", CSRelatedTo, 0.7)
	csg.Add("literature", "biography", CSRelatedTo, 0.6)
	csg.Add("literature", "philosophy", CSRelatedTo, 0.6)
	csg.Add("literature", "poetry", CSRelatedTo, 0.6)
	csg.Add("literature", "reading", CSUsedFor, 0.9)

	csg.Add("book", "novel", CSRelatedTo, 0.8)
	csg.Add("book", "reading", CSUsedFor, 0.9)
	csg.Add("book", "knowledge", CSUsedFor, 0.8)
	csg.Add("book", "entertainment", CSUsedFor, 0.8)
	csg.Add("book", "fiction", CSRelatedTo, 0.8)
	csg.Add("book", "non-fiction", CSRelatedTo, 0.8)
	csg.Add("book", "library", CSRelatedTo, 0.7)
	csg.Add("book", "written work", CSIsA, 0.9)

	// Famous books (as examples for recommendations)
	csg.Add("fiction", "To Kill a Mockingbird", CSRelatedTo, 0.8)
	csg.Add("fiction", "1984", CSRelatedTo, 0.8)
	csg.Add("fiction", "Pride and Prejudice", CSRelatedTo, 0.7)
	csg.Add("fiction", "The Great Gatsby", CSRelatedTo, 0.7)
	csg.Add("fiction", "One Hundred Years of Solitude", CSRelatedTo, 0.7)
	csg.Add("science fiction", "Dune", CSRelatedTo, 0.8)
	csg.Add("science fiction", "Foundation", CSRelatedTo, 0.8)
	csg.Add("science fiction", "Neuromancer", CSRelatedTo, 0.7)
	csg.Add("science fiction", "The Left Hand of Darkness", CSRelatedTo, 0.7)
	csg.Add("science fiction", "Brave New World", CSRelatedTo, 0.7)
	csg.Add("fantasy", "The Lord of the Rings", CSRelatedTo, 0.9)
	csg.Add("fantasy", "A Wizard of Earthsea", CSRelatedTo, 0.7)
	csg.Add("fantasy", "The Name of the Wind", CSRelatedTo, 0.7)
	csg.Add("mystery", "Sherlock Holmes", CSRelatedTo, 0.8)
	csg.Add("mystery", "And Then There Were None", CSRelatedTo, 0.7)
	csg.Add("mystery", "The Girl with the Dragon Tattoo", CSRelatedTo, 0.7)
	csg.Add("philosophy", "Meditations by Marcus Aurelius", CSRelatedTo, 0.8)
	csg.Add("philosophy", "The Republic by Plato", CSRelatedTo, 0.7)
	csg.Add("philosophy", "Sapiens by Yuval Noah Harari", CSRelatedTo, 0.7)

	// Movies / film
	csg.Add("film", "drama", CSRelatedTo, 0.8)
	csg.Add("film", "comedy", CSRelatedTo, 0.8)
	csg.Add("film", "action", CSRelatedTo, 0.7)
	csg.Add("film", "thriller", CSRelatedTo, 0.7)
	csg.Add("film", "documentary", CSRelatedTo, 0.7)
	csg.Add("film", "science fiction", CSRelatedTo, 0.7)
	csg.Add("film", "animation", CSRelatedTo, 0.6)
	csg.Add("film", "cinema", CSRelatedTo, 0.8)
	csg.Add("film", "entertainment", CSUsedFor, 0.9)

	csg.Add("movie", "watching", CSUsedFor, 0.9)
	csg.Add("movie", "entertainment", CSUsedFor, 0.9)
	csg.Add("movie", "cinema", CSRelatedTo, 0.8)
	csg.Add("movie", "streaming", CSRelatedTo, 0.7)
	csg.Add("movie", "film", CSIsA, 0.9)

	// Music
	csg.Add("music", "listening", CSUsedFor, 0.9)
	csg.Add("music", "relaxation", CSUsedFor, 0.8)
	csg.Add("music", "entertainment", CSUsedFor, 0.8)
	csg.Add("music", "rock", CSRelatedTo, 0.7)
	csg.Add("music", "classical", CSRelatedTo, 0.7)
	csg.Add("music", "jazz", CSRelatedTo, 0.7)
	csg.Add("music", "pop", CSRelatedTo, 0.7)
	csg.Add("music", "electronic", CSRelatedTo, 0.6)
	csg.Add("music", "folk", CSRelatedTo, 0.6)
	csg.Add("music", "concert", CSRelatedTo, 0.7)
	csg.Add("music", "instrument", CSRelatedTo, 0.7)

	// Games
	csg.Add("games", "board games", CSRelatedTo, 0.8)
	csg.Add("games", "video games", CSRelatedTo, 0.8)
	csg.Add("games", "card games", CSRelatedTo, 0.7)
	csg.Add("games", "puzzles", CSRelatedTo, 0.7)
	csg.Add("games", "chess", CSRelatedTo, 0.7)
	csg.Add("games", "entertainment", CSUsedFor, 0.9)
	csg.Add("games", "fun", CSUsedFor, 0.9)
	csg.Add("games", "socializing", CSUsedFor, 0.7)

	// Television
	csg.Add("television", "drama series", CSRelatedTo, 0.8)
	csg.Add("television", "comedy series", CSRelatedTo, 0.8)
	csg.Add("television", "documentary series", CSRelatedTo, 0.7)
	csg.Add("television", "streaming", CSRelatedTo, 0.8)
	csg.Add("television", "entertainment", CSUsedFor, 0.9)
}

func (csg *CommonSenseGraph) seedObjects() {
	// Car
	csg.Add("car", "driving", CSUsedFor, 0.9)
	csg.Add("car", "transportation", CSUsedFor, 0.9)
	csg.Add("car", "commuting", CSUsedFor, 0.8)
	csg.Add("car", "vehicle", CSIsA, 0.9)
	csg.Add("car", "fuel", CSRelatedTo, 0.8)
	csg.Add("car", "maintenance", CSRelatedTo, 0.7)
	csg.Add("car", "insurance", CSRelatedTo, 0.7)
	csg.Add("car", "parking", CSRelatedTo, 0.7)
	csg.Add("car", "traffic", CSRelatedTo, 0.7)
	csg.Add("car", "flat tyre", CSRelatedTo, 0.6)
	csg.Add("car", "oil change", CSRelatedTo, 0.6)

	// Phone
	csg.Add("phone", "calling", CSUsedFor, 0.9)
	csg.Add("phone", "texting", CSUsedFor, 0.9)
	csg.Add("phone", "browsing", CSUsedFor, 0.8)
	csg.Add("phone", "communication device", CSIsA, 0.9)
	csg.Add("phone", "apps", CSRelatedTo, 0.8)
	csg.Add("phone", "camera", CSRelatedTo, 0.7)
	csg.Add("phone", "battery", CSRelatedTo, 0.7)
	csg.Add("phone", "charging", CSRelatedTo, 0.7)
	csg.Add("phone", "screen", CSPartOf, 0.8)

	// Computer
	csg.Add("computer", "programming", CSUsedFor, 0.8)
	csg.Add("computer", "browsing the internet", CSUsedFor, 0.9)
	csg.Add("computer", "work", CSUsedFor, 0.9)
	csg.Add("computer", "gaming", CSUsedFor, 0.7)
	csg.Add("computer", "electronic device", CSIsA, 0.9)
	csg.Add("computer", "keyboard", CSPartOf, 0.8)
	csg.Add("computer", "mouse", CSPartOf, 0.7)
	csg.Add("computer", "screen", CSPartOf, 0.8)
	csg.Add("computer", "software", CSRelatedTo, 0.8)
	csg.Add("computer", "hardware", CSRelatedTo, 0.7)
	csg.Add("computer", "slow performance", CSRelatedTo, 0.5)
	csg.Add("computer", "restart", CSRelatedTo, 0.5)

	// Keys
	csg.Add("keys", "locking", CSUsedFor, 0.9)
	csg.Add("keys", "unlocking", CSUsedFor, 0.9)
	csg.Add("keys", "house", CSHasContext, 0.8)
	csg.Add("keys", "car", CSHasContext, 0.7)
	csg.Add("keys", "losing them", CSRelatedTo, 0.6)

	// Wallet
	csg.Add("wallet", "money", CSRelatedTo, 0.9)
	csg.Add("wallet", "cards", CSRelatedTo, 0.8)
	csg.Add("wallet", "identification", CSRelatedTo, 0.7)
	csg.Add("wallet", "carrying money", CSUsedFor, 0.9)

	// Glasses
	csg.Add("glasses", "seeing", CSUsedFor, 0.9)
	csg.Add("glasses", "reading", CSUsedFor, 0.8)
	csg.Add("glasses", "vision correction", CSIsA, 0.9)

	// Umbrella
	csg.Add("umbrella", "rain protection", CSUsedFor, 0.9)
	csg.Add("umbrella", "rain", CSRelatedTo, 0.9)
	csg.Add("umbrella", "staying dry", CSUsedFor, 0.9)
}

func (csg *CommonSenseGraph) seedNature() {
	// Sky
	csg.Add("sky", "blue", CSHasProperty, 0.9)
	csg.Add("sky", "Rayleigh scattering", CSCausedBy, 0.9)
	csg.Add("sky", "clouds", CSRelatedTo, 0.8)
	csg.Add("sky", "sun", CSRelatedTo, 0.8)
	csg.Add("sky", "stars", CSRelatedTo, 0.7)
	csg.Add("sky", "sunset", CSRelatedTo, 0.7)
	csg.Add("sky", "atmosphere", CSPartOf, 0.8)

	// Rain
	csg.Add("rain", "umbrella", CSCausesDesire, 0.9)
	csg.Add("rain", "wet", CSHasProperty, 0.9)
	csg.Add("rain", "weather", CSIsA, 0.9)
	csg.Add("rain", "clouds", CSCausedBy, 0.8)
	csg.Add("rain", "precipitation", CSIsA, 0.9)
	csg.Add("rain", "staying indoors", CSCausesDesire, 0.7)
	csg.Add("rain", "puddles", CSRelatedTo, 0.6)
	csg.Add("rain", "raincoat", CSCausesDesire, 0.7)

	// Sun
	csg.Add("sun", "warm", CSHasProperty, 0.9)
	csg.Add("sun", "bright", CSHasProperty, 0.9)
	csg.Add("sun", "sunscreen", CSCausesDesire, 0.8)
	csg.Add("sun", "daylight", CSRelatedTo, 0.9)
	csg.Add("sun", "solar energy", CSRelatedTo, 0.7)
	csg.Add("sun", "star", CSIsA, 0.9)

	// Snow
	csg.Add("snow", "cold", CSHasProperty, 0.9)
	csg.Add("snow", "white", CSHasProperty, 0.9)
	csg.Add("snow", "winter", CSRelatedTo, 0.9)
	csg.Add("snow", "warm clothing", CSCausesDesire, 0.8)
	csg.Add("snow", "skiing", CSRelatedTo, 0.7)
	csg.Add("snow", "snowman", CSRelatedTo, 0.6)
	csg.Add("snow", "frozen precipitation", CSIsA, 0.9)

	// Seasons
	csg.Add("spring", "flowers", CSRelatedTo, 0.8)
	csg.Add("spring", "warm", CSHasProperty, 0.7)
	csg.Add("spring", "new growth", CSRelatedTo, 0.8)
	csg.Add("spring", "season", CSIsA, 0.9)
	csg.Add("summer", "hot", CSHasProperty, 0.9)
	csg.Add("summer", "vacation", CSRelatedTo, 0.8)
	csg.Add("summer", "swimming", CSRelatedTo, 0.7)
	csg.Add("summer", "beach", CSRelatedTo, 0.7)
	csg.Add("summer", "season", CSIsA, 0.9)
	csg.Add("autumn", "leaves falling", CSRelatedTo, 0.8)
	csg.Add("autumn", "cool", CSHasProperty, 0.8)
	csg.Add("autumn", "harvest", CSRelatedTo, 0.7)
	csg.Add("autumn", "season", CSIsA, 0.9)
	csg.Add("winter", "cold", CSHasProperty, 0.9)
	csg.Add("winter", "snow", CSRelatedTo, 0.8)
	csg.Add("winter", "warm clothing", CSCausesDesire, 0.8)
	csg.Add("winter", "heating", CSRelatedTo, 0.7)
	csg.Add("winter", "season", CSIsA, 0.9)

	// Nature
	csg.Add("nature", "trees", CSRelatedTo, 0.8)
	csg.Add("nature", "animals", CSRelatedTo, 0.8)
	csg.Add("nature", "plants", CSRelatedTo, 0.8)
	csg.Add("nature", "outdoors", CSRelatedTo, 0.9)
	csg.Add("nature", "hiking", CSRelatedTo, 0.7)
	csg.Add("nature", "fresh air", CSHasProperty, 0.8)

	// Ocean / sea
	csg.Add("ocean", "water", CSHasProperty, 0.9)
	csg.Add("ocean", "waves", CSRelatedTo, 0.8)
	csg.Add("ocean", "beach", CSRelatedTo, 0.8)
	csg.Add("ocean", "marine life", CSRelatedTo, 0.8)
	csg.Add("ocean", "swimming", CSRelatedTo, 0.7)
	csg.Add("ocean", "sailing", CSRelatedTo, 0.7)
}

func (csg *CommonSenseGraph) seedHealth() {
	// Headache
	csg.Add("headache", "rest", CSCausesDesire, 0.9)
	csg.Add("headache", "water", CSCausesDesire, 0.8)
	csg.Add("headache", "pain relief medicine", CSCausesDesire, 0.9)
	csg.Add("headache", "quiet environment", CSCausesDesire, 0.7)
	csg.Add("headache", "dehydration", CSCausedBy, 0.7)
	csg.Add("headache", "stress", CSCausedBy, 0.7)
	csg.Add("headache", "lack of sleep", CSCausedBy, 0.7)
	csg.Add("headache", "pain", CSHasProperty, 0.9)

	// Tired / fatigue
	csg.Add("tired", "sleep", CSCausesDesire, 0.9)
	csg.Add("tired", "rest", CSCausesDesire, 0.9)
	csg.Add("tired", "caffeine", CSCausesDesire, 0.7)
	csg.Add("tired", "nap", CSCausesDesire, 0.8)
	csg.Add("tired", "early bedtime", CSCausesDesire, 0.7)
	csg.Add("tired", "lack of sleep", CSCausedBy, 0.9)
	csg.Add("tired", "overwork", CSCausedBy, 0.7)
	csg.Add("tired", "poor diet", CSCausedBy, 0.5)

	csg.Add("fatigue", "rest", CSRelatedTo, 0.9)
	csg.Add("fatigue", "sleep", CSRelatedTo, 0.9)
	csg.Add("fatigue", "nutrition", CSRelatedTo, 0.7)
	csg.Add("fatigue", "exercise", CSRelatedTo, 0.6)
	csg.Add("fatigue", "medical check-up", CSRelatedTo, 0.5)

	// Insomnia
	csg.Add("insomnia", "consistent sleep schedule", CSCausesDesire, 0.8)
	csg.Add("insomnia", "avoid screens before bed", CSCausesDesire, 0.8)
	csg.Add("insomnia", "relaxation techniques", CSCausesDesire, 0.7)
	csg.Add("insomnia", "herbal tea", CSCausesDesire, 0.6)
	csg.Add("insomnia", "dark quiet room", CSCausesDesire, 0.7)
	csg.Add("insomnia", "stress", CSCausedBy, 0.7)
	csg.Add("insomnia", "caffeine", CSCausedBy, 0.6)
	csg.Add("insomnia", "sleep disorder", CSIsA, 0.9)

	// Sick / illness
	csg.Add("illness", "rest", CSRelatedTo, 0.9)
	csg.Add("illness", "fluids", CSRelatedTo, 0.8)
	csg.Add("illness", "doctor", CSRelatedTo, 0.8)
	csg.Add("illness", "medicine", CSRelatedTo, 0.8)
	csg.Add("illness", "staying home", CSRelatedTo, 0.7)
	csg.Add("illness", "soup", CSRelatedTo, 0.6)

	// Cold (common cold)
	csg.Add("cold", "rest", CSCausesDesire, 0.9)
	csg.Add("cold", "warm drinks", CSCausesDesire, 0.8)
	csg.Add("cold", "vitamin C", CSCausesDesire, 0.6)
	csg.Add("cold", "tissues", CSCausesDesire, 0.7)
	csg.Add("cold", "virus", CSCausedBy, 0.9)
	csg.Add("cold", "illness", CSIsA, 0.8)

	// General health
	csg.Add("health", "exercise", CSRelatedTo, 0.9)
	csg.Add("health", "nutrition", CSRelatedTo, 0.9)
	csg.Add("health", "sleep", CSRelatedTo, 0.9)
	csg.Add("health", "hydration", CSRelatedTo, 0.8)
	csg.Add("health", "mental health", CSRelatedTo, 0.8)
	csg.Add("health", "doctor", CSRelatedTo, 0.7)
	csg.Add("health", "preventive care", CSRelatedTo, 0.7)

	// Sleep
	csg.Add("sleep", "rest", CSIsA, 0.9)
	csg.Add("sleep", "recovery", CSUsedFor, 0.9)
	csg.Add("sleep", "health", CSUsedFor, 0.9)
	csg.Add("sleep", "bedroom", CSHasContext, 0.8)
	csg.Add("sleep", "pillow", CSRelatedTo, 0.7)
	csg.Add("sleep", "blanket", CSRelatedTo, 0.7)
	csg.Add("sleep", "8 hours recommended", CSHasProperty, 0.7)
}

func (csg *CommonSenseGraph) seedSocial() {
	// Friend
	csg.Add("friend", "call", CSRelatedTo, 0.8)
	csg.Add("friend", "meet up", CSRelatedTo, 0.8)
	csg.Add("friend", "text", CSRelatedTo, 0.8)
	csg.Add("friend", "hang out", CSRelatedTo, 0.8)
	csg.Add("friend", "shared interests", CSHasProperty, 0.7)
	csg.Add("friend", "trust", CSHasProperty, 0.8)
	csg.Add("friend", "support", CSRelatedTo, 0.8)
	csg.Add("friend", "companionship", CSUsedFor, 0.9)
	csg.Add("friend", "person", CSIsA, 0.9)

	// Family
	csg.Add("family", "visit", CSRelatedTo, 0.8)
	csg.Add("family", "call", CSRelatedTo, 0.8)
	csg.Add("family", "dinner together", CSRelatedTo, 0.7)
	csg.Add("family", "holidays", CSRelatedTo, 0.7)
	csg.Add("family", "love", CSHasProperty, 0.9)
	csg.Add("family", "support", CSRelatedTo, 0.8)
	csg.Add("family", "parents", CSRelatedTo, 0.8)
	csg.Add("family", "siblings", CSRelatedTo, 0.7)
	csg.Add("family", "children", CSRelatedTo, 0.7)

	// Birthday
	csg.Add("birthday", "cake", CSRelatedTo, 0.9)
	csg.Add("birthday", "party", CSRelatedTo, 0.8)
	csg.Add("birthday", "gifts", CSRelatedTo, 0.8)
	csg.Add("birthday", "celebration", CSIsA, 0.9)
	csg.Add("birthday", "wishes", CSRelatedTo, 0.8)
	csg.Add("birthday", "candles", CSRelatedTo, 0.7)

	// Party
	csg.Add("party", "friends", CSRelatedTo, 0.8)
	csg.Add("party", "music", CSRelatedTo, 0.8)
	csg.Add("party", "food", CSRelatedTo, 0.8)
	csg.Add("party", "drinks", CSRelatedTo, 0.7)
	csg.Add("party", "fun", CSHasProperty, 0.8)
	csg.Add("party", "social event", CSIsA, 0.9)

	// Gift
	csg.Add("gift", "birthday", CSHasContext, 0.8)
	csg.Add("gift", "holiday", CSHasContext, 0.7)
	csg.Add("gift", "thoughtfulness", CSHasProperty, 0.7)
	csg.Add("gift", "wrapping", CSRelatedTo, 0.6)
	csg.Add("gift", "present", CSIsA, 0.9)
	csg.Add("gift", "giving", CSUsedFor, 0.9)
}

func (csg *CommonSenseGraph) seedWork() {
	// Work
	csg.Add("work", "office", CSHasContext, 0.8)
	csg.Add("work", "meeting", CSRelatedTo, 0.8)
	csg.Add("work", "deadline", CSRelatedTo, 0.8)
	csg.Add("work", "colleagues", CSRelatedTo, 0.7)
	csg.Add("work", "salary", CSRelatedTo, 0.8)
	csg.Add("work", "career", CSRelatedTo, 0.8)
	csg.Add("work", "commuting", CSRelatedTo, 0.7)
	csg.Add("work", "productivity", CSRelatedTo, 0.7)
	csg.Add("work", "earning income", CSUsedFor, 0.9)

	// Career
	csg.Add("career", "promotion", CSRelatedTo, 0.8)
	csg.Add("career", "resume", CSRelatedTo, 0.7)
	csg.Add("career", "skills", CSRelatedTo, 0.8)
	csg.Add("career", "networking", CSRelatedTo, 0.7)
	csg.Add("career", "job interview", CSRelatedTo, 0.7)
	csg.Add("career", "professional growth", CSUsedFor, 0.8)
	csg.Add("career", "work", CSHasContext, 0.9)

	// Meeting
	csg.Add("meeting", "agenda", CSRelatedTo, 0.8)
	csg.Add("meeting", "presentation", CSRelatedTo, 0.7)
	csg.Add("meeting", "notes", CSRelatedTo, 0.7)
	csg.Add("meeting", "discussion", CSUsedFor, 0.9)
	csg.Add("meeting", "collaboration", CSUsedFor, 0.8)
	csg.Add("meeting", "work", CSHasContext, 0.8)

	// Study / learning
	csg.Add("study", "reading", CSRelatedTo, 0.9)
	csg.Add("study", "notes", CSRelatedTo, 0.8)
	csg.Add("study", "practice", CSRelatedTo, 0.8)
	csg.Add("study", "focus", CSRelatedTo, 0.8)
	csg.Add("study", "exams", CSRelatedTo, 0.7)
	csg.Add("study", "learning", CSUsedFor, 0.9)
	csg.Add("study", "quiet place", CSCausesDesire, 0.7)
	csg.Add("study", "coffee", CSRelatedTo, 0.6)
}

func (csg *CommonSenseGraph) seedHome() {
	// Home
	csg.Add("home", "living room", CSPartOf, 0.8)
	csg.Add("home", "kitchen", CSPartOf, 0.8)
	csg.Add("home", "bedroom", CSPartOf, 0.8)
	csg.Add("home", "bathroom", CSPartOf, 0.7)
	csg.Add("home", "garden", CSPartOf, 0.6)
	csg.Add("home", "comfort", CSHasProperty, 0.8)
	csg.Add("home", "shelter", CSUsedFor, 0.9)
	csg.Add("home", "family", CSRelatedTo, 0.8)
	csg.Add("home", "cleaning", CSRelatedTo, 0.6)
	csg.Add("home", "rent", CSRelatedTo, 0.7)
	csg.Add("home", "mortgage", CSRelatedTo, 0.6)

	// Cleaning
	csg.Add("cleaning", "vacuuming", CSRelatedTo, 0.8)
	csg.Add("cleaning", "dusting", CSRelatedTo, 0.7)
	csg.Add("cleaning", "laundry", CSRelatedTo, 0.8)
	csg.Add("cleaning", "dishes", CSRelatedTo, 0.8)
	csg.Add("cleaning", "organising", CSRelatedTo, 0.7)
	csg.Add("cleaning", "hygiene", CSUsedFor, 0.8)
	csg.Add("cleaning", "home", CSHasContext, 0.9)

	// Morning routine
	csg.Add("morning", "wake up", CSRelatedTo, 0.9)
	csg.Add("morning", "coffee", CSRelatedTo, 0.9)
	csg.Add("morning", "breakfast", CSRelatedTo, 0.9)
	csg.Add("morning", "shower", CSRelatedTo, 0.8)
	csg.Add("morning", "getting ready", CSRelatedTo, 0.8)
	csg.Add("morning", "fresh", CSHasProperty, 0.7)

	// Evening
	csg.Add("evening", "dinner", CSRelatedTo, 0.9)
	csg.Add("evening", "relaxation", CSRelatedTo, 0.8)
	csg.Add("evening", "television", CSRelatedTo, 0.7)
	csg.Add("evening", "reading", CSRelatedTo, 0.7)
	csg.Add("evening", "winding down", CSRelatedTo, 0.8)

	// Shopping
	csg.Add("shopping", "grocery", CSRelatedTo, 0.8)
	csg.Add("shopping", "clothes", CSRelatedTo, 0.7)
	csg.Add("shopping", "budget", CSRelatedTo, 0.7)
	csg.Add("shopping", "list", CSRelatedTo, 0.7)
	csg.Add("shopping", "buying things", CSUsedFor, 0.9)

	// Money
	csg.Add("money", "saving", CSRelatedTo, 0.8)
	csg.Add("money", "spending", CSRelatedTo, 0.8)
	csg.Add("money", "budget", CSRelatedTo, 0.8)
	csg.Add("money", "investing", CSRelatedTo, 0.7)
	csg.Add("money", "earning", CSRelatedTo, 0.8)
	csg.Add("money", "bills", CSRelatedTo, 0.7)
	csg.Add("money", "purchasing", CSUsedFor, 0.9)

	// Pet
	csg.Add("pet", "dog", CSRelatedTo, 0.9)
	csg.Add("pet", "cat", CSRelatedTo, 0.9)
	csg.Add("pet", "feeding", CSRelatedTo, 0.8)
	csg.Add("pet", "walking", CSRelatedTo, 0.7)
	csg.Add("pet", "vet", CSRelatedTo, 0.7)
	csg.Add("pet", "companionship", CSUsedFor, 0.9)
	csg.Add("pet", "animal", CSIsA, 0.9)

	// Dog
	csg.Add("dog", "walking", CSRelatedTo, 0.9)
	csg.Add("dog", "fetch", CSCapableOf, 0.8)
	csg.Add("dog", "loyal", CSHasProperty, 0.9)
	csg.Add("dog", "bark", CSCapableOf, 0.8)
	csg.Add("dog", "pet", CSIsA, 0.9)
	csg.Add("dog", "training", CSRelatedTo, 0.7)

	// Cat
	csg.Add("cat", "independent", CSHasProperty, 0.8)
	csg.Add("cat", "purring", CSCapableOf, 0.8)
	csg.Add("cat", "pet", CSIsA, 0.9)
	csg.Add("cat", "litter box", CSRelatedTo, 0.7)
	csg.Add("cat", "scratching post", CSRelatedTo, 0.7)
}

func (csg *CommonSenseGraph) seedTravel() {
	// Travel
	csg.Add("travel", "flight", CSRelatedTo, 0.8)
	csg.Add("travel", "hotel", CSRelatedTo, 0.8)
	csg.Add("travel", "passport", CSRelatedTo, 0.8)
	csg.Add("travel", "luggage", CSRelatedTo, 0.8)
	csg.Add("travel", "sightseeing", CSRelatedTo, 0.7)
	csg.Add("travel", "vacation", CSRelatedTo, 0.8)
	csg.Add("travel", "exploring", CSUsedFor, 0.8)
	csg.Add("travel", "adventure", CSRelatedTo, 0.7)
	csg.Add("travel", "planning", CSRelatedTo, 0.7)
	csg.Add("travel", "booking", CSRelatedTo, 0.7)

	// Vacation
	csg.Add("vacation", "beach", CSRelatedTo, 0.8)
	csg.Add("vacation", "mountains", CSRelatedTo, 0.7)
	csg.Add("vacation", "city trip", CSRelatedTo, 0.7)
	csg.Add("vacation", "relaxation", CSUsedFor, 0.9)
	csg.Add("vacation", "travel", CSIsA, 0.8)
	csg.Add("vacation", "time off", CSIsA, 0.9)
	csg.Add("vacation", "sightseeing", CSRelatedTo, 0.7)
	csg.Add("vacation", "souvenirs", CSRelatedTo, 0.5)

	// Commute
	csg.Add("commute", "car", CSRelatedTo, 0.8)
	csg.Add("commute", "bus", CSRelatedTo, 0.8)
	csg.Add("commute", "train", CSRelatedTo, 0.8)
	csg.Add("commute", "bicycle", CSRelatedTo, 0.7)
	csg.Add("commute", "walking", CSRelatedTo, 0.6)
	csg.Add("commute", "getting to work", CSUsedFor, 0.9)
	csg.Add("commute", "traffic", CSRelatedTo, 0.7)
	csg.Add("commute", "work", CSHasContext, 0.9)

	// Transport
	csg.Add("transport", "bus", CSRelatedTo, 0.8)
	csg.Add("transport", "train", CSRelatedTo, 0.8)
	csg.Add("transport", "taxi", CSRelatedTo, 0.7)
	csg.Add("transport", "bicycle", CSRelatedTo, 0.7)
	csg.Add("transport", "walking", CSRelatedTo, 0.6)
	csg.Add("transport", "getting around", CSUsedFor, 0.9)

	// Time
	csg.Add("time", "clock", CSRelatedTo, 0.8)
	csg.Add("time", "schedule", CSRelatedTo, 0.8)
	csg.Add("time", "deadline", CSRelatedTo, 0.7)
	csg.Add("time", "calendar", CSRelatedTo, 0.7)
	csg.Add("time", "planning", CSRelatedTo, 0.7)
	csg.Add("time", "time management", CSRelatedTo, 0.7)

	// Weekend
	csg.Add("weekend", "relaxation", CSRelatedTo, 0.8)
	csg.Add("weekend", "activities", CSRelatedTo, 0.8)
	csg.Add("weekend", "friends", CSRelatedTo, 0.7)
	csg.Add("weekend", "family", CSRelatedTo, 0.7)
	csg.Add("weekend", "hobbies", CSRelatedTo, 0.7)
	csg.Add("weekend", "errands", CSRelatedTo, 0.6)
	csg.Add("weekend", "time off", CSIsA, 0.9)
}
