package cognitive

// RegisterDomainPatterns adds domain-specific patterns with DomainFilter set.
// These produce factual, encyclopedic sentences appropriate to each domain
// without editorial commentary or value judgments.
func RegisterDomainPatterns(g *GenerativeEngine) {
	g.AddPatterns([]ClausePattern{
		// --- People (DomainFilter: "person") ---
		{
			Name:         "person-throughout-career",
			Weight:       1.0,
			DomainFilter: "person",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Throughout " + g.topicPossessive() + " career, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:         "person-contributions",
			Weight:       1.0,
			DomainFilter: "person",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + "'s contributions to " + obj + " remain significant"
			},
		},

		// --- Places (DomainFilter: "place") ---
		{
			Name:         "place-home-to",
			Weight:       1.0,
			DomainFilter: "place",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " is home to " + obj
			},
		},
		{
			Name:         "place-known-for",
			Weight:       1.0,
			DomainFilter: "place",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " is known for " + obj
			},
		},

		// --- Events (DomainFilter: "event") ---
		{
			Name:         "event-occurred-when",
			Weight:       1.0,
			DomainFilter: "event",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, TensePast, Singular) + " " + obj
			},
		},

		// --- Concepts (DomainFilter: "concept") ---
		{
			Name:         "concept-at-its-core",
			Weight:       1.0,
			DomainFilter: "concept",
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "At its core, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
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
	})
}
