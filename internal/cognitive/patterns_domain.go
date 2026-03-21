package cognitive

// RegisterDomainPatterns adds ~50 domain-specific patterns with DomainFilter set.
func RegisterDomainPatterns(g *GenerativeEngine) {
	g.AddPatterns([]ClausePattern{
		// --- People (15, DomainFilter: "person") ---
		{
			Name:         "person-born-in",
			Weight:       1.0,
			DomainFilter: "person",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Born into a world shaped by " + obj + ", " + subj + " went on to leave a lasting mark"
			},
		},
		{
			Name:         "person-throughout-career",
			Weight:       1.0,
			DomainFilter: "person",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Throughout " + g.topicPossessive() + " career, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:         "person-whose-work",
			Weight:       0.9,
			DomainFilter: "person",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + ", whose work in " + obj + " reshaped the field, remains influential"
			},
		},
		{
			Name:         "person-as-a",
			Weight:       1.0,
			DomainFilter: "person",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "As a thinker, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " with conviction"
			},
		},
		{
			Name:         "person-legacy",
			Weight:       0.9,
			DomainFilter: "person",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "The legacy of " + subj + " " + g.conjugate("be", t, Singular) + " inseparable from " + obj
			},
		},
		{
			Name:         "person-contributions",
			Weight:       1.0,
			DomainFilter: "person",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + "'s contributions to " + obj + " remain deeply significant"
			},
		},
		{
			Name:         "person-driven-to",
			Weight:       0.8,
			DomainFilter: "person",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Driven to " + verb + " " + obj + ", " + subj + " pursued this path with determination"
			},
		},
		{
			Name:         "person-remembered-for",
			Weight:       0.9,
			DomainFilter: "person",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate("be", t, Singular) + " remembered for " + g.gerund(verb) + " " + obj
			},
		},
		{
			Name:         "person-few-have",
			Weight:       0.7,
			DomainFilter: "person",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Few have " + g.pastParticiple(verb) + " " + obj + " the way " + subj + " did"
			},
		},
		{
			Name:         "person-in-the-eyes",
			Weight:       0.8,
			DomainFilter: "person",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "In the eyes of many, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " like no other"
			},
		},
		{
			Name:         "person-dedicated",
			Weight:       0.9,
			DomainFilter: "person",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Dedicated to " + obj + ", " + subj + " " + g.conjugate(verb, t, Singular) + " with unwavering focus"
			},
		},
		{
			Name:         "person-it-was",
			Weight:       0.8,
			DomainFilter: "person",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "It was " + subj + " who " + g.conjugate(verb, TensePast, Singular) + " " + obj + " in a way that endures"
			},
		},
		{
			Name:         "person-among-those",
			Weight:       0.7,
			DomainFilter: "person",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Among those who " + g.conjugate(verb, TensePast, Singular) + " " + obj + ", " + subj + " stands apart"
			},
		},
		{
			Name:         "person-through-work",
			Weight:       0.9,
			DomainFilter: "person",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Through " + g.topicPossessive() + " work, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " in a lasting way"
			},
		},
		{
			Name:         "person-shaped-by-vision",
			Weight:       0.8,
			DomainFilter: "person",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Shaped by a clear vision, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " with purpose"
			},
		},

		// --- Places (10, DomainFilter: "place") ---
		{
			Name:         "place-nestled",
			Weight:       1.0,
			DomainFilter: "place",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Nestled within " + obj + ", " + subj + " " + g.conjugate(verb, t, Singular) + " with quiet distinction"
			},
		},
		{
			Name:         "place-home-to",
			Weight:       1.0,
			DomainFilter: "place",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Home to " + obj + ", " + subj + " " + g.conjugate(verb, t, Singular) + " as a place of significance"
			},
		},
		{
			Name:         "place-located-in",
			Weight:       1.0,
			DomainFilter: "place",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + ", located near " + obj + ", stands as a place of enduring character"
			},
		},
		{
			Name:         "place-known-across",
			Weight:       0.9,
			DomainFilter: "place",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Known across the region, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:         "place-at-the-heart",
			Weight:       0.9,
			DomainFilter: "place",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "At the heart of " + subj + " lies " + obj
			},
		},
		{
			Name:         "place-rich-in",
			Weight:       0.8,
			DomainFilter: "place",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Rich in " + obj + ", " + subj + " " + g.conjugate(verb, t, Singular) + " with a unique identity"
			},
		},
		{
			Name:         "place-visitors-to",
			Weight:       0.7,
			DomainFilter: "place",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Visitors to " + subj + " discover " + obj + " at every turn"
			},
		},
		{
			Name:         "place-shaped-by-geography",
			Weight:       0.8,
			DomainFilter: "place",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Shaped by its geography, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:         "place-steeped-in",
			Weight:       0.8,
			DomainFilter: "place",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Steeped in " + obj + ", " + subj + " " + g.conjugate(verb, t, Singular) + " with deep roots"
			},
		},
		{
			Name:         "place-the-landscape",
			Weight:       0.7,
			DomainFilter: "place",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "The landscape of " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " in every direction"
			},
		},

		// --- Events (10, DomainFilter: "event") ---
		{
			Name:         "event-when-it",
			Weight:       1.0,
			DomainFilter: "event",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "When " + subj + " " + g.conjugate(verb, TensePast, Singular) + " " + obj + ", it marked a turning point"
			},
		},
		{
			Name:         "event-the-moment",
			Weight:       0.9,
			DomainFilter: "event",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "The pivotal moment when " + subj + " " + g.conjugate(verb, TensePast, Singular) + " " + obj + " changed the course of events"
			},
		},
		{
			Name:         "event-in-the-aftermath",
			Weight:       0.9,
			DomainFilter: "event",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "In the aftermath, " + subj + " " + g.conjugate(verb, TensePast, Singular) + " " + obj + " with lasting consequences"
			},
		},
		{
			Name:         "event-looking-back",
			Weight:       0.8,
			DomainFilter: "event",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Looking back, " + subj + " " + g.conjugate(verb, TensePast, Singular) + " " + obj + " in ways few anticipated"
			},
		},
		{
			Name:         "event-triggered-by",
			Weight:       0.8,
			DomainFilter: "event",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Triggered by " + obj + ", " + subj + " " + g.conjugate(verb, TensePast, Singular) + " with sudden force"
			},
		},
		{
			Name:         "event-historians-note",
			Weight:       0.7,
			DomainFilter: "event",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Historians note that " + subj + " " + g.conjugate(verb, TensePast, Singular) + " " + obj + " at a critical juncture"
			},
		},
		{
			Name:         "event-the-repercussions",
			Weight:       0.8,
			DomainFilter: "event",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "The repercussions of " + subj + " " + g.gerund(verb) + " " + obj + " were felt for decades"
			},
		},
		{
			Name:         "event-at-the-time",
			Weight:       0.9,
			DomainFilter: "event",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "At the time, " + subj + " " + g.conjugate(verb, TensePast, Singular) + " " + obj + " against all expectations"
			},
		},
		{
			Name:         "event-few-foresaw",
			Weight:       0.7,
			DomainFilter: "event",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Few foresaw that " + subj + " would " + verb + " " + obj + " so decisively"
			},
		},
		{
			Name:         "event-as-unfolded",
			Weight:       0.8,
			DomainFilter: "event",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "As events unfolded, " + subj + " " + g.conjugate(verb, TensePast, Singular) + " " + obj + " with finality"
			},
		},

		// --- Concepts (15, DomainFilter: "concept") ---
		{
			Name:         "concept-at-its-core",
			Weight:       1.0,
			DomainFilter: "concept",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "At its core, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:         "concept-the-notion",
			Weight:       0.9,
			DomainFilter: "concept",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "The notion that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " invites scrutiny"
			},
		},
		{
			Name:         "concept-broadly-speaking",
			Weight:       1.0,
			DomainFilter: "concept",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Broadly speaking, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:         "concept-in-the-world-of",
			Weight:       0.9,
			DomainFilter: "concept",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "In the broader landscape, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:         "concept-the-idea-that",
			Weight:       1.0,
			DomainFilter: "concept",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "The idea that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " has gained traction over time"
			},
		},
		{
			Name:         "concept-fundamentally",
			Weight:       0.9,
			DomainFilter: "concept",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Fundamentally, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:         "concept-as-a-framework",
			Weight:       0.8,
			DomainFilter: "concept",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "As a framework, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " with clarity"
			},
		},
		{
			Name:         "concept-when-examined",
			Weight:       0.8,
			DomainFilter: "concept",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "When examined closely, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " in a coherent way"
			},
		},
		{
			Name:         "concept-the-principle",
			Weight:       0.9,
			DomainFilter: "concept",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "The principle that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " is well supported"
			},
		},
		{
			Name:         "concept-in-theory",
			Weight:       0.8,
			DomainFilter: "concept",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "In theory and in practice, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:         "concept-underlying",
			Weight:       0.8,
			DomainFilter: "concept",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Underlying this " + g.conjugate("be", t, Singular) + " the fact that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:         "concept-put-differently",
			Weight:       0.7,
			DomainFilter: "concept",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Put differently, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:         "concept-at-a-higher-level",
			Weight:       0.7,
			DomainFilter: "concept",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "At a higher level, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " as part of a larger whole"
			},
		},
		{
			Name:         "concept-the-significance",
			Weight:       0.9,
			DomainFilter: "concept",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "The significance of " + subj + " " + g.gerund(verb) + " " + obj + " should not be understated"
			},
		},
		{
			Name:         "concept-to-understand",
			Weight:       0.8,
			DomainFilter: "concept",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "To understand " + subj + " " + g.conjugate("be", t, Singular) + " to understand " + obj
			},
		},
	})
}
