package cognitive

// RegisterSyntacticPatterns adds ~60 syntactic variation patterns.
func RegisterSyntacticPatterns(g *GenerativeEngine) {
	g.AddPatterns([]ClausePattern{
		// --- Compound sentences (15) ---
		{
			Name:   "compound-moreover",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", and moreover, this matters"
			},
		},
		{
			Name:   "compound-not-only",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Not only does " + subj + " " + verb + " " + obj + ", but " + obj + " stands on its own"
			},
		},
		{
			Name:   "compound-in-fact",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj + "; in fact, the significance is hard to overstate"
			},
		},
		{
			Name:   "compound-in-turn",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", which in turn shapes how we understand " + subj
			},
		},
		{
			Name:   "compound-distinction",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " — and that distinction matters"
			},
		},
		{
			Name:   "compound-deeper",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "While " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", the significance runs deeper"
			},
		},
		{
			Name:   "compound-furthermore",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj + "; furthermore, this has lasting implications"
			},
		},
		{
			Name:   "compound-equally",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", and equally important, it shapes the broader landscape"
			},
		},
		{
			Name:   "compound-yet",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", yet the full picture is richer still"
			},
		},
		{
			Name:   "compound-as-a-result",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj + "; as a result, the impact extends widely"
			},
		},
		{
			Name:   "compound-indeed",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " — indeed, this is a defining trait"
			},
		},
		{
			Name:   "compound-simultaneously",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", and simultaneously, new possibilities emerge"
			},
		},
		{
			Name:   "compound-that-said",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " — that said, the context matters"
			},
		},
		{
			Name:   "compound-consequently",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj + "; consequently, the implications are wide-reaching"
			},
		},
		{
			Name:   "compound-above-all",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Above all, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},

		// --- Subordinate clauses (15) ---
		{
			Name:   "subordinate-because",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Because " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", the implications are far-reaching"
			},
		},
		{
			Name:   "subordinate-although",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Although the details vary, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "subordinate-given",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Given that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", one can draw meaningful conclusions"
			},
		},
		{
			Name:   "subordinate-once",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Once you understand that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", the rest follows naturally"
			},
		},
		{
			Name:   "subordinate-whether",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Whether or not one agrees, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "subordinate-since",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Since " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", the point is well established"
			},
		},
		{
			Name:   "subordinate-unless",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Unless one overlooks the evidence, " + subj + " clearly " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "subordinate-wherever",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Wherever one looks, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "subordinate-as-long-as",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "As long as " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", the foundation holds"
			},
		},
		{
			Name:   "subordinate-even-though",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Even though opinions differ, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "subordinate-insofar-as",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Insofar as " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", the argument stands"
			},
		},
		{
			Name:   "subordinate-provided",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Provided that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", the conclusion is sound"
			},
		},
		{
			Name:   "subordinate-after",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "After examining the evidence, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "subordinate-before",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Before anything else, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "subordinate-while",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "While it may seem simple, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},

		// --- Participial phrases (10) ---
		{
			Name:   "participial-having",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				pp := g.pastParticiple(verb)
				return "Having " + pp + " " + obj + ", " + subj + " stands in a unique position"
			},
		},
		{
			Name:   "participial-gerund-with",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				gerund := g.gerund(verb)
				return capitalizeFirst(gerund) + " " + obj + " with precision, " + subj + " has earned its place"
			},
		},
		{
			Name:   "participial-known-for",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				gerund := g.gerund(verb)
				return "Known for " + gerund + " " + obj + ", " + subj + " continues to make an impact"
			},
		},
		{
			Name:   "participial-recognized",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Recognized for " + obj + ", " + subj + " holds a distinctive place"
			},
		},
		{
			Name:   "participial-built-on",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Built on the idea that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", the foundation is solid"
			},
		},
		{
			Name:   "participial-driven-by",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Driven by " + obj + ", " + subj + " " + g.conjugate(verb, t, Singular) + " with purpose"
			},
		},
		{
			Name:   "participial-shaped-by",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Shaped by " + obj + ", " + subj + " has evolved significantly"
			},
		},
		{
			Name:   "participial-drawing-on",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Drawing on " + obj + ", " + subj + " " + g.conjugate(verb, t, Singular) + " with depth"
			},
		},
		{
			Name:   "participial-rooted-in",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Rooted in " + obj + ", " + subj + " " + g.conjugate(verb, t, Singular) + " from a strong foundation"
			},
		},
		{
			Name:   "participial-standing-as",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Standing as " + obj + ", " + subj + " " + g.conjugate(verb, t, Singular) + " with authority"
			},
		},

		// --- Appositive expansions (10) ---
		{
			Name:   "appositive-category",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + ", a notable presence, " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "appositive-often",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " — often regarded highly — " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "appositive-sometimes-called",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + ", sometimes described as " + obj + ", " + g.conjugate(verb, t, Singular) + " in a meaningful way"
			},
		},
		{
			Name:   "appositive-widely-known",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + ", widely known for " + obj + ", " + g.conjugate(verb, t, Singular) + " with clarity"
			},
		},
		{
			Name:   "appositive-by-all-accounts",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + ", by all accounts, " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "appositive-in-essence",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " — in essence, " + obj + " — " + g.conjugate(verb, t, Singular) + " at every level"
			},
		},
		{
			Name:   "appositive-that-is",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + ", that is to say " + obj + ", " + g.conjugate(verb, t, Singular) + " consistently"
			},
		},
		{
			Name:   "appositive-a-force",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + ", a force in its own right, " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "appositive-long-established",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + ", long established as " + obj + ", continues to shape the field"
			},
		},
		{
			Name:   "appositive-no-stranger",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + ", no stranger to " + obj + ", " + g.conjugate(verb, t, Singular) + " with confidence"
			},
		},

		// --- Inverted structures (10) ---
		{
			Name:   "inverted-rare",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Rare is the case that " + g.conjugate(verb, t, Singular) + " " + obj + " like " + subj
			},
		},
		{
			Name:   "inverted-what-makes",
			Weight: 1.0,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "What makes " + subj + " distinct is " + obj
			},
		},
		{
			Name:   "inverted-it-is-through",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "It is through " + obj + " that " + subj + " " + g.conjugate(verb, t, Singular)
			},
		},
		{
			Name:   "inverted-only-by",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				gerund := g.gerund(verb)
				return "Only by understanding " + obj + " can one grasp what " + subj + " " + g.conjugate("achieve", t, Singular) + " through " + gerund
			},
		},
		{
			Name:   "inverted-seldom",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Seldom does one encounter something like " + subj + ", which " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "inverted-central-to",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Central to " + subj + " " + g.conjugate("be", t, Singular) + " " + obj
			},
		},
		{
			Name:   "inverted-nowhere-else",
			Weight: 0.6,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Nowhere else does one find " + obj + " quite as " + subj + " " + g.conjugate(verb, t, Singular) + " it"
			},
		},
		{
			Name:   "inverted-essential",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Essential to understanding " + subj + " " + g.conjugate("be", t, Singular) + " " + obj
			},
		},
		{
			Name:   "inverted-equally-clear",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Equally clear " + g.conjugate("be", t, Singular) + " the fact that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "inverted-undeniable",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Undeniable " + g.conjugate("be", t, Singular) + " the role of " + subj + " in " + g.gerund(verb) + " " + obj
			},
		},
	})
}
