package cognitive

// RegisterTonePatterns adds ~50 tonal variation patterns.
func RegisterTonePatterns(g *GenerativeEngine) {
	g.AddPatterns([]ClausePattern{
		// --- Formal/academic (15) ---
		{
			Name:   "formal-widely-recognized",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "It is widely recognized that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "formal-scholarly-consensus",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Scholarly consensus holds that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "formal-one-may-observe",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "One may observe that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "formal-evidence-suggests",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "The evidence suggests that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "formal-it-follows",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "It follows that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "formal-as-demonstrated",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "As has been demonstrated, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "formal-it-is-evident",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "It is evident that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "formal-upon-examination",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Upon examination, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "formal-analysis-reveals",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Analysis reveals that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "formal-in-this-regard",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "In this regard, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "formal-accordingly",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Accordingly, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "formal-by-this-measure",
			Weight: 0.6,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "By this measure, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "formal-the-data-indicate",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "The data indicate that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "formal-it-bears-noting",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "It bears noting that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "formal-to-this-end",
			Weight: 0.6,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "To this end, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},

		// --- Casual/conversational (15) ---
		{
			Name:   "casual-heres-the-thing",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Here's the thing about " + subj + " — it " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "casual-interesting",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "You know what's interesting? " + capitalizeFirst(subj) + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "casual-basically",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "So basically, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "casual-think-about-it",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Think about it: " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "casual-turns-out",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Turns out, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "casual-the-cool-part",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "The cool part? " + capitalizeFirst(subj) + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "casual-believe-it-or-not",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Believe it or not, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "casual-look",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Look, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " — it's that straightforward"
			},
		},
		{
			Name:   "casual-long-story-short",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Long story short, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "casual-bottom-line",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Bottom line: " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "casual-real-talk",
			Weight: 0.6,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Real talk — " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "casual-honestly",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Honestly, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "casual-no-surprise",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "No surprise here — " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "casual-plain-and-simple",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return subj + " " + g.conjugate(verb, t, Singular) + " " + obj + ", plain and simple"
			},
		},
		{
			Name:   "casual-at-the-end-of-the-day",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "At the end of the day, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},

		// --- Narrative/storytelling (10) ---
		{
			Name:   "narrative-picture-this",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Picture this: " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "narrative-story-begins",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "The story of " + subj + " begins with " + obj
			},
		},
		{
			Name:   "narrative-imagine",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Imagine a world where " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "narrative-once-upon",
			Weight: 0.6,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Once, " + subj + " " + g.conjugate(verb, TensePast, Singular) + " " + obj + ", and the rest is history"
			},
		},
		{
			Name:   "narrative-there-came",
			Weight: 0.6,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "There came a time when " + subj + " " + g.conjugate(verb, TensePast, Singular) + " " + obj
			},
		},
		{
			Name:   "narrative-as-the-story-goes",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "As the story goes, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "narrative-chapter",
			Weight: 0.6,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "In this chapter, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "narrative-the-tale",
			Weight: 0.6,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "The tale of " + subj + " " + g.gerund(verb) + " " + obj + " is one worth telling"
			},
		},
		{
			Name:   "narrative-it-all-started",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "It all started when " + subj + " " + g.conjugate(verb, TensePast, Singular) + " " + obj
			},
		},
		{
			Name:   "narrative-the-journey",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "The journey of " + subj + " " + g.gerund(verb) + " " + obj + " unfolds in fascinating ways"
			},
		},

		// --- Reflective (10) ---
		{
			Name:   "reflective-worth-noting",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "It's worth noting that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "reflective-looking-closely",
			Weight: 0.9,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Looking at it closely, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "reflective-the-more-you-examine",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "The more you examine " + subj + ", the more " + obj + " reveals itself"
			},
		},
		{
			Name:   "reflective-on-reflection",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "On reflection, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " in ways that reward careful thought"
			},
		},
		{
			Name:   "reflective-stepping-back",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Stepping back, one sees that " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "reflective-with-hindsight",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "With hindsight, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj + " more clearly than ever"
			},
		},
		{
			Name:   "reflective-in-quiet-moments",
			Weight: 0.6,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "In quiet moments, one appreciates how " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "reflective-upon-closer-look",
			Weight: 0.8,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "Upon closer look, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "reflective-there-is-something",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "There is something meaningful about how " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
		{
			Name:   "reflective-when-you-pause",
			Weight: 0.7,
			Build: func(g *GenerativeEngine, subj, verb, obj string, t Tense) string {
				return "When you pause to consider it, " + subj + " " + g.conjugate(verb, t, Singular) + " " + obj
			},
		},
	})
}
