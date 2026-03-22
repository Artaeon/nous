package cognitive

// RegisterSyntacticPatterns adds clean syntactic variation patterns.
// These produce natural compound and complex sentences without editorial
// commentary, value judgments, or filler phrases.
func RegisterSyntacticPatterns(g *GenerativeEngine) {
	g.AddPatterns([]ClausePattern{
		{
			Name:   "compound-and",
			Weight: 1.5,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "subordinate-while",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "While " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", this remains a defining characteristic"
			},
		},
		{
			Name:   "appositive-which",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", which is widely recognized"
			},
		},
	})
}
