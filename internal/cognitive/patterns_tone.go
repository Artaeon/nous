package cognitive

// RegisterTonePatterns adds clean informational sentence patterns.
// These produce encyclopedic prose — no editorial commentary, casual speech,
// narrative hooks, or reflective asides.
func RegisterTonePatterns(g *GenerativeEngine) {
	g.AddPatterns([]ClausePattern{
		{
			Name:   "informational-additionally",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Additionally, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "informational-in-addition",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "In addition, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "informational-more-specifically",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "More specifically, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
	})
}
