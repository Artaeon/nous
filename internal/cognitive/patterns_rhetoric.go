package cognitive

// RegisterRhetoricalPatterns adds ~40 rhetorical device patterns.
func RegisterRhetoricalPatterns(g *GenerativeEngine) {
	g.AddPatterns([]ClausePattern{
		// --- Analogy (10) ---
		{
			Name:   "analogy-much-like",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Much like a cornerstone, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "analogy-in-the-same-way",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", in the same way that a foundation supports a structure"
			},
		},
		{
			Name:   "analogy-if-then",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "If a compass points north, then " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "analogy-just-as",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Just as a river shapes the land, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "analogy-think-of",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Think of " + subj + " as " + g.gerund(verb) + " " + obj + " the way light fills a room"
			},
		},
		{
			Name:   "analogy-like-a-thread",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Like a thread running through fabric, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "analogy-parallel",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "The parallel is clear: " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "analogy-not-unlike",
			Weight: 0.6,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Not unlike a catalyst, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "analogy-mirror",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", mirroring patterns seen throughout history"
			},
		},
		{
			Name:   "analogy-as-a-lens",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "As a lens focuses light, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},

		// --- Enumeration (8) ---
		{
			Name:   "enum-first-and-foremost",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "First and foremost, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "enum-among-these",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Among these, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " and stands apart"
			},
		},
		{
			Name:   "enum-for-one-thing",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "For one thing, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "enum-most-notably",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Most notably, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "enum-in-addition",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "In addition, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "enum-on-top-of-that",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "On top of that, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "enum-what-is-more",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "What is more, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "enum-equally-important",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Equally important, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},

		// --- Example/illustration (8) ---
		{
			Name:   "example-consider",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Consider " + subj + ": it " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "example-take-for-instance",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Take " + subj + ", for instance — it " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "example-case-in-point",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "A case in point: " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "example-to-illustrate",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "To illustrate, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "example-as-seen-in",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "As seen in " + subj + ", " + g.gerund(verb) + " " + obj + " proves the point"
			},
		},
		{
			Name:   "example-this-shows",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " — this shows the principle in action"
			},
		},
		{
			Name:   "example-witness",
			Weight: 0.6,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Witness how " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "example-observe",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Observe that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},

		// --- Definition (8) ---
		{
			Name:   "definition-by-we-mean",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "By " + subj + ", we mean " + obj
			},
		},
		{
			Name:   "definition-that-is",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " — that is, " + obj + " — shapes the broader context"
			},
		},
		{
			Name:   "definition-to-put-simply",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "To put it simply, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "definition-in-other-words",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "In other words, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "definition-what-this-means",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "What this means is that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "definition-stated-plainly",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Stated plainly, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "definition-essentially",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Essentially, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "definition-to-clarify",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "To clarify, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},

		// --- Concession (6) ---
		{
			Name:   "concession-while-some",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "While some may question " + subj + ", the fact remains that it " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "concession-granted",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Granted, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", yet the impact endures"
			},
		},
		{
			Name:   "concession-despite",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Despite the challenges, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "concession-admittedly",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Admittedly, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", but the value is clear"
			},
		},
		{
			Name:   "concession-nonetheless",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj + "; nonetheless, questions persist"
			},
		},
		{
			Name:   "concession-even-so",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Even so, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " with remarkable consistency"
			},
		},
	})
}
