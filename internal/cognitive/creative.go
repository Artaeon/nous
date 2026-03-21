package cognitive

import (
	"fmt"
	"math/rand"
	"strings"
	"time"
)

// -----------------------------------------------------------------------
// Creative Response Engine — generates poems, stories, jokes, and
// reflections WITHOUT any LLM. Every output is assembled from structural
// templates, combinatorial generation, and knowledge graph integration.
//
// Techniques:
//   - Template-based poetry: free verse, haiku, quatrain with rhyme maps
//   - Three-act micro-stories: setup, conflict, resolution
//   - Joke generation: question/answer, observational, pun-based
//   - Multi-perspective reflection: pros/cons using graph knowledge
//   - Metaphor generation: nature/emotion domain crossings
//   - Topic-aware: weaves in knowledge graph facts when available
// -----------------------------------------------------------------------

// PoemForm selects the structural form of a poem.
type PoemForm int

const (
	PoemFreeVerse PoemForm = iota
	PoemHaiku
	PoemQuatrain
)

// CreativeType identifies the kind of creative output requested.
type CreativeType string

const (
	CreativePoem    CreativeType = "poem"
	CreativeStory   CreativeType = "story"
	CreativeJoke    CreativeType = "joke"
	CreativeReflect CreativeType = "reflect"
)

// CreativeRequest bundles a creative generation request.
type CreativeRequest struct {
	Type     CreativeType
	Topic    string
	PoemForm PoemForm // only for poems
	Length   int      // story paragraph count (0 = default 3)
}

// CreativeEngine generates creative text from templates and combinatorics.
type CreativeEngine struct {
	Graph    *CognitiveGraph
	Composer *Composer
	rng      *rand.Rand

	// Vocabulary pools
	rhymes     map[string][]string
	adjectives map[string][]string // domain → adjectives
	nouns      map[string][]string // domain → nouns
	verbs      map[string][]string // domain → verbs
	metaphors  []metaphorTemplate
	imagery    []string

	// Story building blocks
	archetypes []characterArchetype
	settings   []string
	conflicts  []string
	endings    []string

	// Joke templates
	jokeTemplates []jokeTemplate
}

type metaphorTemplate struct {
	pattern string // "X is like {Y}" or "X, {ADJ} as {Y}"
	domain  string // source domain for Y: "nature", "weather", "emotion"
}

type characterArchetype struct {
	name        string
	description string
	traits      []string
}

type jokeTemplate struct {
	setup    string
	punchFn  func(topic string, rng *rand.Rand) string
	category string // "question", "observational", "pun"
}

// NewCreativeEngine creates a creative engine with rich vocabulary.
func NewCreativeEngine(g *CognitiveGraph, c *Composer) *CreativeEngine {
	ce := &CreativeEngine{
		Graph:    g,
		Composer: c,
		rng:      rand.New(rand.NewSource(time.Now().UnixNano())),
	}
	ce.initRhymes()
	ce.initVocabulary()
	ce.initMetaphors()
	ce.initStoryElements()
	ce.initJokeTemplates()
	return ce
}

// Generate routes a creative request to the appropriate sub-method.
func (ce *CreativeEngine) Generate(request CreativeRequest) string {
	topic := strings.TrimSpace(request.Topic)
	if topic == "" {
		topic = ce.randomTopic()
	}

	switch request.Type {
	case CreativePoem:
		return ce.WritePoem(topic, request.PoemForm)
	case CreativeStory:
		length := request.Length
		if length <= 0 {
			length = 3
		}
		return ce.WriteStory(topic, length)
	case CreativeJoke:
		return ce.TellJoke(topic)
	case CreativeReflect:
		return ce.Reflect(topic, "")
	default:
		return ce.WritePoem(topic, PoemFreeVerse)
	}
}

// -----------------------------------------------------------------------
// Poem Generation
// -----------------------------------------------------------------------

// WritePoem generates a poem in the specified form about a topic.
func (ce *CreativeEngine) WritePoem(topic string, form PoemForm) string {
	switch form {
	case PoemHaiku:
		return ce.writeHaiku(topic)
	case PoemQuatrain:
		return ce.writeQuatrain(topic)
	default:
		return ce.writeFreeVerse(topic)
	}
}

func (ce *CreativeEngine) writeFreeVerse(topic string) string {
	lines := make([]string, 0, 8)
	facts := ce.getTopicFacts(topic)

	// Opening line — metaphor or imagery
	opening := ce.generateMetaphor(topic)
	lines = append(lines, opening)

	// Body lines — 3-6 lines mixing imagery, emotion, and facts
	bodyCount := 3 + ce.rng.Intn(4)
	for i := 0; i < bodyCount; i++ {
		switch ce.rng.Intn(4) {
		case 0:
			lines = append(lines, ce.imageryLine(topic))
		case 1:
			lines = append(lines, ce.emotionLine(topic))
		case 2:
			if len(facts) > 0 {
				fact := facts[ce.rng.Intn(len(facts))]
				lines = append(lines, ce.poeticizeFact(fact))
			} else {
				lines = append(lines, ce.imageryLine(topic))
			}
		case 3:
			lines = append(lines, ce.sensoryLine(topic))
		}
	}

	// Closing line — circular or reflective
	closers := []string{
		fmt.Sprintf("and still, %s remains", topic),
		fmt.Sprintf("this is what %s teaches us", topic),
		fmt.Sprintf("in the end, only %s endures", topic),
		fmt.Sprintf("and %s whispers on", topic),
		fmt.Sprintf("so goes the way of %s", topic),
		fmt.Sprintf("forever echoing — %s", topic),
	}
	lines = append(lines, closers[ce.rng.Intn(len(closers))])

	return strings.Join(lines, "\n")
}

func (ce *CreativeEngine) writeHaiku(topic string) string {
	// Haiku: 3 lines, nature-themed, contemplative
	line1Pool := []string{
		fmt.Sprintf("%s drifts softly", topic),
		fmt.Sprintf("beneath the %s sky", ce.pick(ce.adjectives["weather"])),
		fmt.Sprintf("morning light on %s", topic),
		fmt.Sprintf("still water holds %s", topic),
		fmt.Sprintf("%s in the wind", topic),
		fmt.Sprintf("a leaf falls toward %s", topic),
		fmt.Sprintf("the %s moon rises", ce.pick(ce.adjectives["nature"])),
		fmt.Sprintf("old stones know of %s", topic),
	}

	line2Pool := []string{
		fmt.Sprintf("the %s %s stirs gently", ce.pick(ce.adjectives["nature"]), ce.pick(ce.nouns["nature"])),
		fmt.Sprintf("a %s passes through the %s", ce.pick(ce.nouns["nature"]), ce.pick(ce.nouns["weather"])),
		fmt.Sprintf("silence speaks of %s", ce.pick(ce.nouns["emotion"])),
		fmt.Sprintf("%s %s drifting slowly", ce.pick(ce.adjectives["weather"]), ce.pick(ce.nouns["nature"])),
		fmt.Sprintf("the world turns toward %s", ce.pick(ce.nouns["emotion"])),
		fmt.Sprintf("between the %s and the %s", ce.pick(ce.nouns["nature"]), ce.pick(ce.nouns["nature"])),
	}

	line3Pool := []string{
		fmt.Sprintf("%s endures", topic),
		fmt.Sprintf("all becomes %s", ce.pick(ce.adjectives["emotion"])),
		fmt.Sprintf("peace at last — %s", topic),
		fmt.Sprintf("nothing but %s", topic),
		fmt.Sprintf("the %s knows why", ce.pick(ce.nouns["nature"])),
		fmt.Sprintf("and %s remains", ce.pick(ce.nouns["emotion"])),
		fmt.Sprintf("so it has always been"),
	}

	return fmt.Sprintf("%s\n%s\n%s",
		ce.pick(line1Pool),
		ce.pick(line2Pool),
		ce.pick(line3Pool),
	)
}

func (ce *CreativeEngine) writeQuatrain(topic string) string {
	// Pick a rhyme pair for lines 1+3 and another for lines 2+4 (ABAB)
	// or lines 1+2 and 3+4 (AABB) — coin flip
	scheme := ce.rng.Intn(2) // 0=ABAB, 1=AABB

	endWord1 := ce.pickRhymableWord(topic)
	rhymes1 := ce.getRhymes(endWord1)
	endWord2 := ce.pickRhymableWord(topic)
	for endWord2 == endWord1 {
		endWord2 = ce.pickRhymableWord(topic)
	}
	rhymes2 := ce.getRhymes(endWord2)

	line1 := ce.buildRhymeLine(topic, endWord1)
	line2 := ce.buildRhymeLine(topic, endWord2)
	var line3, line4 string

	if scheme == 0 { // ABAB
		line3 = ce.buildRhymeLine(topic, ce.pick(rhymes1))
		line4 = ce.buildRhymeLine(topic, ce.pick(rhymes2))
	} else { // AABB
		line1 = ce.buildRhymeLine(topic, endWord1)
		line2 = ce.buildRhymeLine(topic, ce.pick(rhymes1))
		line3 = ce.buildRhymeLine(topic, endWord2)
		line4 = ce.buildRhymeLine(topic, ce.pick(rhymes2))
	}

	return fmt.Sprintf("%s\n%s\n%s\n%s", line1, line2, line3, line4)
}

func (ce *CreativeEngine) buildRhymeLine(topic, endWord string) string {
	templates := []string{
		fmt.Sprintf("the %s of %s brings %s", ce.pick(ce.nouns["emotion"]), topic, endWord),
		fmt.Sprintf("beneath a sky of %s and %s", ce.pick(ce.adjectives["nature"]), endWord),
		fmt.Sprintf("we search for %s amid the %s", topic, endWord),
		fmt.Sprintf("like %s that dances in the %s", topic, endWord),
		fmt.Sprintf("a gentle %s reveals the %s", ce.pick(ce.nouns["weather"]), endWord),
		fmt.Sprintf("and in the %s we find the %s", ce.pick(ce.nouns["nature"]), endWord),
		fmt.Sprintf("through %s and %s toward the %s", ce.pick(ce.nouns["emotion"]), ce.pick(ce.nouns["weather"]), endWord),
		fmt.Sprintf("the %s %s carries us to %s", ce.pick(ce.adjectives["emotion"]), ce.pick(ce.nouns["nature"]), endWord),
	}
	return ce.pick(templates)
}

func (ce *CreativeEngine) pickRhymableWord(topic string) string {
	// Pick a word that has known rhymes, related to the topic's emotional domain
	rhymable := make([]string, 0, len(ce.rhymes))
	for word := range ce.rhymes {
		rhymable = append(rhymable, word)
	}
	return rhymable[ce.rng.Intn(len(rhymable))]
}

func (ce *CreativeEngine) getRhymes(word string) []string {
	if r, ok := ce.rhymes[word]; ok && len(r) > 0 {
		return r
	}
	// Fallback: return some generic rhymes
	return []string{"light", "day", "way", "time", "mind"}
}

// -----------------------------------------------------------------------
// Story Generation
// -----------------------------------------------------------------------

// WriteStory generates a three-act micro-story about the topic.
func (ce *CreativeEngine) WriteStory(topic string, length int) string {
	if length < 2 {
		length = 3
	}
	if length > 7 {
		length = 7
	}

	facts := ce.getTopicFacts(topic)
	arch := ce.archetypes[ce.rng.Intn(len(ce.archetypes))]
	setting := ce.settings[ce.rng.Intn(len(ce.settings))]
	charName := ce.generateCharName(arch)

	paragraphs := make([]string, 0, length)

	// Act 1: Setup — introduce character + setting
	setup := ce.buildSetup(charName, arch, setting, topic, facts)
	paragraphs = append(paragraphs, setup)

	// Act 2: Conflict — problem/challenge (can be multiple paragraphs)
	conflictParagraphs := length - 2
	if conflictParagraphs < 1 {
		conflictParagraphs = 1
	}
	for i := 0; i < conflictParagraphs; i++ {
		conflict := ce.buildConflict(charName, arch, topic, facts, i)
		paragraphs = append(paragraphs, conflict)
	}

	// Act 3: Resolution
	resolution := ce.buildResolution(charName, arch, topic, facts)
	paragraphs = append(paragraphs, resolution)

	return strings.Join(paragraphs, "\n\n")
}

func (ce *CreativeEngine) buildSetup(name string, arch characterArchetype, setting, topic string, facts []string) string {
	trait := ce.pick(arch.traits)
	intros := []string{
		fmt.Sprintf("In %s, there lived %s named %s — %s and full of %s.", setting, arch.description, name, trait, ce.pick(ce.nouns["emotion"])),
		fmt.Sprintf("%s had always been %s. From the first days in %s, everyone knew %s was %s.", name, trait, setting, name, arch.description),
		fmt.Sprintf("Long ago, in %s, %s %s began a journey. %s was %s, though the world did not yet know it.", setting, arch.description, name, name, trait),
		fmt.Sprintf("They called %s %s, and in %s, that meant something. %s was %s by nature.", name, arch.description, setting, name, trait),
	}
	result := ce.pick(intros)

	// Weave in a fact if available
	if len(facts) > 0 {
		result += fmt.Sprintf(" %s had long been fascinated by %s — after all, %s.", name, topic, facts[0])
	} else {
		result += fmt.Sprintf(" %s had long been drawn to %s, sensing something profound within it.", name, topic)
	}
	return result
}

func (ce *CreativeEngine) buildConflict(name string, arch characterArchetype, topic string, facts []string, stage int) string {
	conflict := ce.conflicts[ce.rng.Intn(len(ce.conflicts))]
	obstacles := []string{
		fmt.Sprintf("But %s. %s struggled with the weight of it, turning to %s for answers that would not come.", conflict, name, topic),
		fmt.Sprintf("Then came the challenge: %s. %s searched for meaning in %s, but found only more questions.", conflict, name, topic),
		fmt.Sprintf("Everything changed when %s. %s stood at a crossroads, with %s pulling in one direction and doubt in another.", conflict, name, topic),
		fmt.Sprintf("The path grew difficult. %s, and %s began to wonder whether understanding %s was even possible.", conflict, name, topic),
	}
	result := ce.pick(obstacles)

	if stage > 0 && len(facts) > 1 {
		factIdx := (stage) % len(facts)
		result += fmt.Sprintf(" A clue emerged: %s. Perhaps this was the key.", facts[factIdx])
	}
	return result
}

func (ce *CreativeEngine) buildResolution(name string, arch characterArchetype, topic string, facts []string) string {
	endings := ce.endings
	ending := endings[ce.rng.Intn(len(endings))]

	resolutions := []string{
		fmt.Sprintf("In the end, %s. %s had not conquered %s — but had learned to live alongside it. And that was enough.", ending, name, topic),
		fmt.Sprintf("And then, %s. %s smiled, understanding at last that %s was never the destination, but the journey itself.", ending, name, topic),
		fmt.Sprintf("Finally, %s. %s carried the lesson of %s forward, changed forever by what had been discovered.", ending, name, topic),
		fmt.Sprintf("When it was over, %s. %s looked back at the path through %s and saw not struggle, but growth.", ending, name, topic),
	}
	result := ce.pick(resolutions)

	if len(facts) > 0 {
		result += fmt.Sprintf(" After all, %s.", facts[len(facts)-1])
	}
	return result
}

func (ce *CreativeEngine) generateCharName(arch characterArchetype) string {
	names := []string{
		"Elara", "Kael", "Mira", "Orion", "Lyra", "Thane",
		"Sera", "Dorian", "Astra", "Felix", "Nola", "Rowan",
		"Cleo", "Jasper", "Iris", "Atlas", "Luna", "Sage",
	}
	return names[ce.rng.Intn(len(names))]
}

// -----------------------------------------------------------------------
// Joke Generation
// -----------------------------------------------------------------------

// TellJoke generates a joke about the given topic.
func (ce *CreativeEngine) TellJoke(topic string) string {
	facts := ce.getTopicFacts(topic)

	// If we have knowledge graph facts, prefer topical humor
	if len(facts) > 0 {
		topical := ce.topicalJoke(topic, facts)
		if topical != "" {
			return topical
		}
	}

	// Fall back to template-based jokes
	tmpl := ce.jokeTemplates[ce.rng.Intn(len(ce.jokeTemplates))]
	return tmpl.punchFn(topic, ce.rng)
}

func (ce *CreativeEngine) topicalJoke(topic string, facts []string) string {
	fact := facts[ce.rng.Intn(len(facts))]
	templates := []string{
		fmt.Sprintf("Why did %s cross the road? Because %s — and that changes everything.", topic, fact),
		fmt.Sprintf("You know what's funny about %s? Well, %s. No wonder it keeps coming up!", topic, fact),
		fmt.Sprintf("I tried to explain %s to my friend. I said, \"%s.\" They still don't get it.", topic, fact),
		fmt.Sprintf("Have you ever noticed that %s? I mean, %s. Makes you think, right?", fact, topic),
	}
	return templates[ce.rng.Intn(len(templates))]
}

// -----------------------------------------------------------------------
// Reflection / Opinion Generation
// -----------------------------------------------------------------------

// Reflect generates a multi-perspective analysis on a topic.
func (ce *CreativeEngine) Reflect(topic string, question string) string {
	facts := ce.getTopicFacts(topic)
	sections := make([]string, 0, 5)

	// Opening frame
	openers := []string{
		fmt.Sprintf("%s is one of those topics that invites many perspectives.", strings.Title(topic)),
		fmt.Sprintf("When we consider %s, several dimensions come to mind.", topic),
		fmt.Sprintf("The question of %s is more nuanced than it might first appear.", topic),
		fmt.Sprintf("There are many ways to think about %s. Here are some perspectives worth considering.", topic),
	}
	sections = append(sections, ce.pick(openers))

	// If we have facts, weave them into the analysis
	if len(facts) > 0 {
		factSection := fmt.Sprintf("What we know: %s.", strings.Join(ce.limitSlice(facts, 3), ". Additionally, "))
		sections = append(sections, factSection)
	}

	// Perspective 1: The practical view
	practical := []string{
		fmt.Sprintf("From a practical standpoint, %s has real implications for how we approach everyday challenges. It shapes decisions, influences priorities, and affects outcomes in ways both subtle and profound.", topic),
		fmt.Sprintf("Practically speaking, %s matters because it touches the way we organize our lives, our work, and our interactions with others.", topic),
	}
	sections = append(sections, ce.pick(practical))

	// Perspective 2: The philosophical view
	philosophical := []string{
		fmt.Sprintf("On a deeper level, %s raises questions about values, purpose, and what we consider meaningful. Thinkers throughout history have grappled with similar themes, arriving at different but equally thoughtful conclusions.", topic),
		fmt.Sprintf("Philosophically, %s connects to fundamental questions about existence, knowledge, and the nature of understanding itself.", topic),
	}
	sections = append(sections, ce.pick(philosophical))

	// Perspective 3: The forward-looking view
	forward := []string{
		fmt.Sprintf("Looking ahead, our relationship with %s will likely continue to evolve. New understanding, new contexts, and new challenges will reshape how we think about it.", topic),
		fmt.Sprintf("The future of %s is open. What matters is that we approach it with curiosity, humility, and a willingness to reconsider what we think we know.", topic),
	}
	sections = append(sections, ce.pick(forward))

	return strings.Join(sections, "\n\n")
}

// -----------------------------------------------------------------------
// Knowledge Graph Integration
// -----------------------------------------------------------------------

func (ce *CreativeEngine) getTopicFacts(topic string) []string {
	if ce.Graph == nil {
		return nil
	}
	var facts []string

	// Get edges from and to this topic
	edges := ce.Graph.EdgesFrom(topic)
	edges = append(edges, ce.Graph.EdgesTo(topic)...)

	for _, edge := range edges {
		fact := ce.Graph.edgeToFact(edge)
		if fact != "" {
			facts = append(facts, fact)
		}
	}
	return facts
}

// -----------------------------------------------------------------------
// Poetic line generators
// -----------------------------------------------------------------------

func (ce *CreativeEngine) generateMetaphor(topic string) string {
	m := ce.metaphors[ce.rng.Intn(len(ce.metaphors))]
	var y string
	switch m.domain {
	case "nature":
		y = ce.pick(ce.nouns["nature"])
	case "weather":
		y = ce.pick(ce.nouns["weather"])
	case "emotion":
		y = ce.pick(ce.nouns["emotion"])
	default:
		y = ce.pick(ce.nouns["nature"])
	}
	result := strings.ReplaceAll(m.pattern, "{TOPIC}", topic)
	result = strings.ReplaceAll(result, "{Y}", y)
	result = strings.ReplaceAll(result, "{ADJ}", ce.pick(ce.adjectives[m.domain]))
	return result
}

func (ce *CreativeEngine) imageryLine(topic string) string {
	templates := []string{
		fmt.Sprintf("like %s scattered across %s", ce.pick(ce.nouns["nature"]), ce.pick(ce.nouns["texture"])),
		fmt.Sprintf("the color of %s when %s arrives", ce.pick(ce.nouns["emotion"]), ce.pick(ce.nouns["season"])),
		fmt.Sprintf("a %s %s where %s once stood", ce.pick(ce.adjectives["texture"]), ce.pick(ce.nouns["nature"]), topic),
		fmt.Sprintf("traces of %s on %s %s", topic, ce.pick(ce.adjectives["nature"]), ce.pick(ce.nouns["nature"])),
		fmt.Sprintf("the way %s bends the %s", topic, ce.pick(ce.nouns["nature"])),
		fmt.Sprintf("shadows of %s falling on %s", topic, ce.pick(ce.nouns["nature"])),
		fmt.Sprintf("in the %s of %s, something stirs", ce.pick(ce.nouns["nature"]), topic),
	}
	return ce.pick(templates)
}

func (ce *CreativeEngine) emotionLine(topic string) string {
	templates := []string{
		fmt.Sprintf("carrying the weight of %s like %s", ce.pick(ce.nouns["emotion"]), ce.pick(ce.nouns["nature"])),
		fmt.Sprintf("the %s ache of knowing %s", ce.pick(ce.adjectives["emotion"]), topic),
		fmt.Sprintf("somewhere between %s and %s", ce.pick(ce.nouns["emotion"]), ce.pick(ce.nouns["emotion"])),
		fmt.Sprintf("a %s tenderness that %s awakens", ce.pick(ce.adjectives["emotion"]), topic),
		fmt.Sprintf("what is %s but a form of %s", topic, ce.pick(ce.nouns["emotion"])),
		fmt.Sprintf("the heart knows %s before the mind does", topic),
	}
	return ce.pick(templates)
}

func (ce *CreativeEngine) sensoryLine(topic string) string {
	templates := []string{
		fmt.Sprintf("the taste of %s in %s air", topic, ce.pick(ce.adjectives["weather"])),
		fmt.Sprintf("a %s sound, like %s touching %s", ce.pick(ce.adjectives["texture"]), topic, ce.pick(ce.nouns["nature"])),
		fmt.Sprintf("%s feels like %s %s on skin", topic, ce.pick(ce.adjectives["texture"]), ce.pick(ce.nouns["weather"])),
		fmt.Sprintf("the scent of %s after %s", ce.pick(ce.nouns["nature"]), ce.pick(ce.nouns["weather"])),
		fmt.Sprintf("warm as %s, cold as %s", ce.pick(ce.nouns["nature"]), ce.pick(ce.nouns["emotion"])),
	}
	return ce.pick(templates)
}

func (ce *CreativeEngine) poeticizeFact(fact string) string {
	templates := []string{
		fmt.Sprintf("they say that %s", fact),
		fmt.Sprintf("consider this: %s", fact),
		fmt.Sprintf("it is known — %s", fact),
		fmt.Sprintf("as the world reminds us, %s", fact),
	}
	return ce.pick(templates)
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

func (ce *CreativeEngine) pick(pool []string) string {
	if len(pool) == 0 {
		return "something"
	}
	return pool[ce.rng.Intn(len(pool))]
}

func (ce *CreativeEngine) randomTopic() string {
	topics := []string{
		"time", "silence", "the sea", "memory", "light",
		"change", "solitude", "hope", "the stars", "music",
		"rain", "friendship", "courage", "the wind", "dreams",
	}
	return topics[ce.rng.Intn(len(topics))]
}

func (ce *CreativeEngine) limitSlice(s []string, max int) []string {
	if len(s) <= max {
		return s
	}
	return s[:max]
}

// -----------------------------------------------------------------------
// Vocabulary Initialization
// -----------------------------------------------------------------------

func (ce *CreativeEngine) initVocabulary() {
	ce.adjectives = map[string][]string{
		"nature": {
			"ancient", "wild", "gentle", "vast", "silent", "golden",
			"emerald", "deep", "tangled", "mossy", "windswept", "verdant",
			"towering", "fragile", "enduring", "radiant", "barren", "lush",
			"crystalline", "sprawling", "hushed", "primeval", "blooming",
			"withered", "luminous",
		},
		"emotion": {
			"tender", "fierce", "quiet", "burning", "wistful", "serene",
			"aching", "joyful", "bittersweet", "restless", "peaceful",
			"melancholy", "exuberant", "somber", "hopeful", "yearning",
			"resolute", "fragile", "defiant", "compassionate", "longing",
			"grateful", "haunted", "radiant", "contemplative",
		},
		"weather": {
			"misty", "stormy", "clear", "foggy", "crisp", "sultry",
			"overcast", "breezy", "thunderous", "balmy", "frosty",
			"damp", "scorching", "drizzling", "sweltering", "brisk",
			"temperate", "howling", "tranquil", "humid", "icy",
		},
		"texture": {
			"rough", "smooth", "velvet", "grainy", "silken", "coarse",
			"polished", "weathered", "soft", "jagged", "rippled",
			"worn", "burnished", "delicate", "heavy", "feathered",
			"gossamer", "brittle", "supple", "woven",
		},
		"color": {
			"crimson", "amber", "ivory", "cobalt", "scarlet", "silver",
			"indigo", "copper", "pearl", "obsidian", "sapphire", "rust",
			"jade", "violet", "ash", "gold", "charcoal", "turquoise",
			"burgundy", "slate",
		},
	}

	ce.nouns = map[string][]string{
		"nature": {
			"river", "mountain", "forest", "meadow", "ocean", "stone",
			"tree", "flower", "sky", "shore", "valley", "horizon",
			"leaf", "root", "seed", "tide", "current", "peak",
			"canyon", "reef", "marsh", "prairie", "grove", "cliff",
			"stream",
		},
		"weather": {
			"rain", "thunder", "snow", "mist", "fog", "storm",
			"lightning", "frost", "dew", "hail", "breeze", "wind",
			"sunlight", "moonlight", "dawn", "dusk", "twilight",
			"drizzle", "downpour", "gale",
		},
		"emotion": {
			"sorrow", "joy", "wonder", "longing", "hope", "grief",
			"courage", "fear", "love", "doubt", "peace", "desire",
			"gratitude", "awe", "regret", "compassion", "resilience",
			"nostalgia", "tenderness", "contentment", "yearning",
			"determination", "serenity", "melancholy", "bliss",
		},
		"season": {
			"spring", "summer", "autumn", "winter", "harvest",
			"solstice", "equinox",
		},
		"texture": {
			"velvet", "silk", "sand", "glass", "marble", "clay",
			"wool", "linen", "parchment", "stone", "ice", "iron",
			"copper", "crystal", "amber",
		},
	}

	ce.verbs = map[string][]string{
		"nature": {
			"flows", "grows", "blooms", "withers", "rises", "falls",
			"drifts", "echoes", "rustles", "swells", "recedes", "unfolds",
			"settles", "stirs", "awakens", "fades",
		},
		"emotion": {
			"aches", "soars", "trembles", "yearns", "embraces", "surrenders",
			"endures", "persists", "whispers", "cries", "laughs", "mourns",
			"hopes", "forgives", "remembers", "forgets",
		},
	}

	ce.imagery = []string{
		"the color of dusk on still water",
		"a single bird against an empty sky",
		"roots reaching deep beneath the surface",
		"light breaking through heavy clouds",
		"the last ember of a dying fire",
		"sand flowing through open fingers",
		"a bridge across an impossible distance",
		"the reflection that the mirror does not show",
		"footprints filling slowly with rain",
		"a door that opens both ways",
	}
}

func (ce *CreativeEngine) initMetaphors() {
	ce.metaphors = []metaphorTemplate{
		{pattern: "{TOPIC} is like a {Y} — {ADJ} and unyielding", domain: "nature"},
		{pattern: "{TOPIC}, {ADJ} as the {Y}", domain: "nature"},
		{pattern: "what is {TOPIC} but a {ADJ} {Y}", domain: "emotion"},
		{pattern: "{TOPIC} moves like {Y} across the land", domain: "weather"},
		{pattern: "in {TOPIC} we find the {Y} of all things", domain: "emotion"},
		{pattern: "{TOPIC} — the {Y} that never stills", domain: "nature"},
		{pattern: "they call it {TOPIC}, but it is really a kind of {Y}", domain: "emotion"},
		{pattern: "like {Y} before the {ADJ} dawn, so is {TOPIC}", domain: "weather"},
		{pattern: "{TOPIC} rises, {ADJ}, like a {Y} from the deep", domain: "nature"},
		{pattern: "there is a {Y} in {TOPIC} that the {ADJ} heart recognizes", domain: "emotion"},
	}
}

func (ce *CreativeEngine) initStoryElements() {
	ce.archetypes = []characterArchetype{
		{
			name:        "seeker",
			description: "a restless seeker",
			traits:      []string{"curious", "relentless", "open-hearted", "questioning", "observant"},
		},
		{
			name:        "builder",
			description: "a determined builder",
			traits:      []string{"patient", "methodical", "visionary", "stubborn", "creative"},
		},
		{
			name:        "wanderer",
			description: "a quiet wanderer",
			traits:      []string{"solitary", "perceptive", "adaptable", "contemplative", "free-spirited"},
		},
		{
			name:        "sage",
			description: "a thoughtful sage",
			traits:      []string{"wise", "measured", "compassionate", "experienced", "humble"},
		},
	}

	ce.settings = []string{
		"a quiet village at the edge of the world",
		"a vast ocean stretching beyond all horizons",
		"an ancient library where books whispered their secrets",
		"a distant planet where the stars sang at night",
		"a mountain town where the air was thin and the silence deep",
		"a coastal city where the tides governed all rhythm",
		"a forest so old that the trees remembered the first rains",
		"a desert where the sand held the memory of water",
	}

	ce.conflicts = []string{
		"the old certainties began to crumble",
		"a question arose that had no easy answer",
		"what once seemed simple revealed its hidden complexity",
		"the path forward split into many, and none were marked",
		"an unexpected loss shook the foundations of everything",
		"a stranger arrived with a truth no one wanted to hear",
		"the tools that had always worked suddenly failed",
		"the world demanded a choice between two impossible options",
	}

	ce.endings = []string{
		"clarity came not as a flash but as a slow dawn",
		"the answer had been there all along, hidden in plain sight",
		"it was not victory but understanding that brought peace",
		"the pieces fell into place, not perfectly, but honestly",
		"what had seemed like an ending turned out to be a beginning",
		"the question transformed, and with it, the questioner",
		"acceptance arrived quietly, without fanfare",
		"the struggle itself had been the teacher",
	}
}

func (ce *CreativeEngine) initJokeTemplates() {
	ce.jokeTemplates = []jokeTemplate{
		{
			category: "question",
			punchFn: func(topic string, rng *rand.Rand) string {
				answers := []string{
					"Because it couldn't find the manual!",
					"Because even experts need a day off!",
					"Because practice makes... well, more practice!",
					"Because it heard there was free coffee on the other side!",
					"Because nobody told it there was a shortcut!",
				}
				return fmt.Sprintf("Why did %s cross the road? %s", topic, answers[rng.Intn(len(answers))])
			},
		},
		{
			category: "observation",
			punchFn: func(topic string, rng *rand.Rand) string {
				observations := []string{
					fmt.Sprintf("Have you ever noticed that whenever you finally understand %s, it changes? It's like it knows you're watching.", topic),
					fmt.Sprintf("The funny thing about %s is that everyone's an expert until you ask them to explain it. Then suddenly it's 'complicated.'", topic),
					fmt.Sprintf("I asked someone to explain %s to me in simple terms. Three hours later, we were both confused.", topic),
					fmt.Sprintf("You know what %s and my alarm clock have in common? I keep trying to ignore both of them, and neither one gives up.", topic),
				}
				return observations[rng.Intn(len(observations))]
			},
		},
		{
			category: "pun",
			punchFn: func(topic string, rng *rand.Rand) string {
				puns := []string{
					fmt.Sprintf("What do you call a group of people who really love %s? A %s-ination!", topic, topic),
					fmt.Sprintf("I told my friend I was getting into %s. They said, 'That's a %s move.' I said, 'Exactly.'", topic, topic),
					fmt.Sprintf("My relationship with %s is complicated. It's a real %s-coaster.", topic, topic),
					fmt.Sprintf("I tried to make a belt out of %s. It was a waist of %s.", topic, topic),
				}
				return puns[rng.Intn(len(puns))]
			},
		},
		{
			category: "self-referential",
			punchFn: func(topic string, rng *rand.Rand) string {
				meta := []string{
					fmt.Sprintf("I'd tell you a joke about %s, but I'm still working on the punchline. Much like %s itself.", topic, topic),
					fmt.Sprintf("A computer, a philosopher, and %s walk into a bar. The bartender says, 'What is this, some kind of thought experiment?'", topic),
					fmt.Sprintf("Knock knock. Who's there? %s. %s who? That's the question everyone's been asking!", topic, topic),
					fmt.Sprintf("I was going to write a joke about %s, but then I realized the real joke is how much time we spend thinking about it.", topic),
				}
				return meta[rng.Intn(len(meta))]
			},
		},
	}
}

// -----------------------------------------------------------------------
// Rhyme Map — ~200 common words with rhyme partners
// -----------------------------------------------------------------------

func (ce *CreativeEngine) initRhymes() {
	ce.rhymes = map[string][]string{
		// Nature
		"sky":       {"fly", "high", "by", "why", "try", "eye", "sigh", "cry", "dry", "lie"},
		"sea":       {"free", "be", "tree", "key", "me", "we", "three", "plea", "flee", "agree"},
		"tree":      {"free", "sea", "be", "key", "me", "we", "three", "plea", "flee", "agree"},
		"rain":      {"pain", "gain", "plain", "train", "brain", "chain", "main", "vain", "remain", "contain"},
		"night":     {"light", "sight", "right", "bright", "might", "flight", "white", "fight", "tight", "height"},
		"light":     {"night", "sight", "right", "bright", "might", "flight", "white", "fight", "tight", "height"},
		"sun":       {"run", "one", "done", "fun", "won", "begun", "son", "none", "gun", "spun"},
		"moon":      {"soon", "tune", "june", "noon", "bloom", "room", "spoon", "boon"},
		"star":      {"far", "are", "bar", "car", "jar", "scar", "guitar", "afar"},
		"wind":      {"find", "mind", "kind", "blind", "behind", "remind", "defined", "aligned"},
		"stone":     {"alone", "bone", "known", "grown", "own", "tone", "zone", "phone", "throne"},
		"fire":      {"higher", "desire", "inspire", "wire", "tire", "entire", "acquire", "admire"},
		"snow":      {"flow", "grow", "know", "show", "glow", "below", "slow", "throw", "shadow"},
		"earth":     {"birth", "worth", "mirth", "hearth", "berth"},
		"river":     {"giver", "deliver", "shiver", "quiver", "forever"},
		"mountain":  {"fountain", "certain", "curtain"},
		"flower":    {"power", "hour", "tower", "shower", "devour"},
		"storm":     {"form", "warm", "norm", "transform", "reform", "perform", "inform"},
		"wave":      {"brave", "save", "gave", "cave", "grave", "crave", "pave"},
		"shore":     {"more", "door", "floor", "core", "before", "explore", "adore", "restore"},
		"cloud":     {"loud", "proud", "crowd", "shroud", "allowed", "bowed"},
		"field":     {"yield", "sealed", "revealed", "healed", "shield"},
		"leaf":      {"belief", "grief", "relief", "brief", "chief"},
		"seed":      {"need", "feed", "lead", "read", "speed", "freed", "deed", "agreed"},
		"root":      {"truth", "fruit", "pursuit", "route", "shoot", "suit"},

		// Emotion
		"love":      {"above", "dove", "of", "shove", "glove"},
		"fear":      {"near", "clear", "dear", "hear", "year", "appear", "tear", "here", "sincere"},
		"hope":      {"scope", "rope", "slope", "cope"},
		"dream":     {"stream", "seem", "team", "gleam", "scheme", "theme", "beam", "extreme"},
		"heart":     {"start", "art", "part", "smart", "apart", "chart", "dart"},
		"soul":      {"whole", "goal", "role", "control", "toll", "scroll", "bowl", "pole"},
		"mind":      {"find", "kind", "blind", "behind", "remind", "wind", "defined", "aligned"},
		"peace":     {"release", "increase", "cease", "fleece", "lease", "piece"},
		"joy":       {"boy", "toy", "employ", "enjoy", "destroy", "annoy", "deploy"},
		"sorrow":    {"tomorrow", "borrow", "follow", "hollow"},
		"pain":      {"rain", "gain", "plain", "train", "brain", "chain", "main", "vain", "remain"},
		"grace":     {"place", "face", "space", "race", "trace", "embrace", "pace"},
		"pride":     {"side", "guide", "wide", "ride", "hide", "tide", "inside", "decide"},
		"trust":     {"must", "just", "dust", "rust", "gust", "adjust", "robust"},
		"truth":     {"youth", "root", "pursuit", "proof"},
		"grief":     {"leaf", "belief", "relief", "brief", "chief"},

		// Time
		"time":      {"rhyme", "climb", "prime", "sublime", "chime", "mime", "paradigm"},
		"day":       {"way", "say", "play", "stay", "away", "may", "gray", "pray", "display", "delay"},
		"way":       {"day", "say", "play", "stay", "away", "may", "gray", "pray", "display", "delay"},
		"year":      {"fear", "near", "clear", "dear", "hear", "appear", "tear", "here", "sincere"},
		"dawn":      {"gone", "drawn", "born", "worn", "torn", "forlorn", "mourn"},
		"dusk":      {"trust", "must", "dust", "husk", "rush"},
		"spring":    {"ring", "sing", "bring", "thing", "king", "wing", "string", "swing", "cling"},
		"fall":      {"call", "all", "wall", "tall", "small", "hall", "ball"},
		"hour":      {"power", "flower", "tower", "shower", "devour"},
		"moment":    {"component", "atonement", "opponent"},

		// Life
		"life":      {"wife", "strife", "knife", "rife"},
		"death":     {"breath", "beneath", "wreath"},
		"birth":     {"earth", "worth", "mirth", "hearth"},
		"home":      {"roam", "foam", "dome", "poem", "chrome", "alone"},
		"road":      {"code", "load", "mode", "showed", "flowed", "bestowed"},
		"door":      {"more", "floor", "shore", "core", "before", "explore", "adore"},
		"song":      {"long", "strong", "wrong", "along", "belong", "among"},
		"voice":     {"choice", "rejoice", "noise"},
		"hand":      {"land", "stand", "band", "sand", "grand", "understand", "demand", "expand"},
		"name":      {"flame", "game", "same", "fame", "came", "claim", "aim", "frame"},
		"word":      {"heard", "bird", "third", "absurd", "occurred", "stirred"},
		"eye":       {"sky", "fly", "high", "by", "why", "try", "sigh", "cry", "dry", "lie"},
		"face":      {"place", "grace", "space", "race", "trace", "embrace", "pace"},

		// Abstract
		"thought":   {"brought", "caught", "fought", "taught", "sought", "wrought"},
		"change":    {"range", "strange", "arrange", "exchange"},
		"rest":      {"best", "test", "quest", "west", "chest", "nest", "blessed", "guest", "expressed"},
		"will":      {"still", "hill", "fill", "skill", "thrill", "until", "chill", "fulfill"},
		"call":      {"fall", "all", "wall", "tall", "small", "hall", "ball"},
		"end":       {"friend", "send", "bend", "blend", "tend", "mend", "defend", "pretend", "extend"},
		"start":     {"heart", "art", "part", "smart", "apart", "chart", "dart"},
		"hold":      {"gold", "old", "cold", "bold", "told", "fold", "unfold", "behold"},
		"stand":     {"hand", "land", "band", "sand", "grand", "understand", "demand", "expand"},
		"sleep":     {"deep", "keep", "leap", "weep", "sweep", "steep", "creep"},
		"wake":      {"lake", "make", "take", "break", "shake", "sake", "mistake", "cake"},
		"run":       {"sun", "one", "done", "fun", "won", "begun", "son", "none"},
		"fly":       {"sky", "high", "by", "why", "try", "eye", "sigh", "cry", "dry", "lie"},
		"sing":      {"ring", "spring", "bring", "thing", "king", "wing", "string", "swing", "cling"},
		"dance":     {"chance", "glance", "advance", "romance", "trance", "enhance", "stance"},
		"walk":      {"talk", "chalk", "stalk", "balk"},
		"speak":     {"seek", "week", "peak", "cheek", "meek", "unique", "creek", "sleek"},
		"grow":      {"know", "show", "flow", "snow", "glow", "below", "slow", "throw"},
		"rise":      {"eyes", "skies", "wise", "surprise", "size", "prize", "ties", "lies", "realize"},
		"fade":      {"made", "shade", "blade", "trade", "aid", "afraid", "parade", "cascade"},
		"burn":      {"turn", "learn", "return", "earn", "concern", "yearn", "discern"},
		"break":     {"wake", "lake", "make", "take", "shake", "sake", "mistake"},
		"lost":      {"cost", "frost", "crossed", "tossed", "exhaust"},
		"found":     {"ground", "sound", "round", "bound", "around", "profound", "surround"},
		"silence":   {"violence", "defiance", "compliance", "guidance", "alliance"},
		"dark":      {"spark", "mark", "park", "stark", "embark", "remark", "arc"},
		"blue":      {"true", "new", "through", "knew", "view", "few", "grew", "flew", "drew"},
		"gold":      {"hold", "old", "cold", "bold", "told", "fold", "unfold", "behold"},
		"deep":      {"sleep", "keep", "leap", "weep", "sweep", "steep", "creep"},
		"free":      {"sea", "tree", "be", "key", "me", "we", "three", "plea", "flee", "agree"},
		"wild":      {"child", "mild", "styled", "compiled", "beguiled"},
		"still":     {"will", "hill", "fill", "skill", "thrill", "until", "chill", "fulfill"},
	}
}
