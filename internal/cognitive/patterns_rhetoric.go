package cognitive

// RegisterRhetoricalPatterns adds a small set of clean informational patterns.
// These use factual framing rather than editorial devices.
func RegisterRhetoricalPatterns(g *GenerativeEngine) {
	g.AddPatterns([]ClausePattern{
		{
			Name:   "definition-specifically",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Specifically, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "definition-in-particular",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "In particular, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "definition-notably",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Notably, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
	})
}
