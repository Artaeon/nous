package cognitive

import (
	"math/rand"
	"strings"
)

// -----------------------------------------------------------------------
// Frame System — structural templates for different output types.
//
// A Frame defines the STRUCTURE of a response, not its content.
// "Email" has greeting → context → body → closing → signoff.
// "Brainstorm" has intro → categories → ideas → synthesis.
//
// The Thinking Engine fills each frame section with generated content.
// This is how Nous can write emails, brainstorm, explain, compare —
// all without an LLM. The structure comes from frames, the content
// from graph + Markov + templates + entity extraction.
// -----------------------------------------------------------------------

// OutputFormat specifies the structural format of the output.
type OutputFormat int

const (
	FormatProse      OutputFormat = iota // flowing text
	FormatEmail                         // email structure
	FormatList                          // bullet/numbered list
	FormatComparison                    // side-by-side comparison
	FormatTutorial                      // step-by-step instructions
	FormatStory                         // narrative arc
)

// Additional tone constants — base type Tone is defined in composer.go.
// ToneNeutral (0), ToneCasual (1), ToneWarm (2), ToneDirect (3) already exist.
const (
	ToneFormal     Tone = 10 // professional, polished
	ToneAcademic   Tone = 11 // precise, scholarly
	ToneEmpathetic Tone = 12 // warm, supportive
)

// SectionType classifies what a frame section does.
type SectionType int

const (
	SecGreeting  SectionType = iota // salutation
	SecOpening                      // context-setting opener
	SecBody                         // main content
	SecList                         // enumerated items
	SecClosing                      // wrap-up
	SecSignoff                      // farewell/signature
	SecHeading                      // section header
	SecSynthesis                    // summarizing insight
)

// FrameSection is one structural unit in an output frame.
type FrameSection struct {
	Type     SectionType
	Role     string // descriptive name: "greeting", "main_points", etc.
	Goal     string // communicative goal: "set context", "list ideas"
	Required bool   // mandatory or skippable
}

// Frame is a structural template for a type of output.
type Frame struct {
	Name     string
	Sections []FrameSection
}

// -----------------------------------------------------------------------
// Frame Library — templates for common output types
// -----------------------------------------------------------------------

var emailFrame = Frame{
	Name: "email",
	Sections: []FrameSection{
		{Type: SecGreeting, Role: "greeting", Goal: "address the recipient", Required: true},
		{Type: SecOpening, Role: "opening", Goal: "set context and purpose", Required: true},
		{Type: SecBody, Role: "body", Goal: "deliver the main content", Required: true},
		{Type: SecClosing, Role: "closing", Goal: "state next steps or request", Required: true},
		{Type: SecSignoff, Role: "signoff", Goal: "professional sign-off", Required: true},
	},
}

var brainstormFrame = Frame{
	Name: "brainstorm",
	Sections: []FrameSection{
		{Type: SecOpening, Role: "context", Goal: "frame the brainstorm topic", Required: true},
		{Type: SecList, Role: "ideas", Goal: "list generated ideas", Required: true},
		{Type: SecSynthesis, Role: "synthesis", Goal: "highlight most promising ideas", Required: false},
	},
}

var explanationFrame = Frame{
	Name: "explanation",
	Sections: []FrameSection{
		{Type: SecOpening, Role: "hook", Goal: "grab attention and frame topic", Required: true},
		{Type: SecBody, Role: "definition", Goal: "define the concept clearly", Required: true},
		{Type: SecBody, Role: "mechanism", Goal: "explain how it works", Required: false},
		{Type: SecBody, Role: "example", Goal: "give a concrete example", Required: false},
		{Type: SecClosing, Role: "significance", Goal: "explain why it matters", Required: true},
	},
}

var comparisonFrame = Frame{
	Name: "comparison",
	Sections: []FrameSection{
		{Type: SecOpening, Role: "intro", Goal: "frame what we're comparing", Required: true},
		{Type: SecBody, Role: "item_a", Goal: "describe first item", Required: true},
		{Type: SecBody, Role: "item_b", Goal: "describe second item", Required: true},
		{Type: SecBody, Role: "differences", Goal: "highlight key differences", Required: true},
		{Type: SecSynthesis, Role: "verdict", Goal: "summarize or recommend", Required: false},
	},
}

var tutorialFrame = Frame{
	Name: "tutorial",
	Sections: []FrameSection{
		{Type: SecOpening, Role: "goal", Goal: "state what we'll learn", Required: true},
		{Type: SecBody, Role: "prerequisites", Goal: "what you need to know first", Required: false},
		{Type: SecList, Role: "steps", Goal: "step-by-step instructions", Required: true},
		{Type: SecBody, Role: "tips", Goal: "helpful tips and gotchas", Required: false},
		{Type: SecClosing, Role: "next_steps", Goal: "where to go from here", Required: false},
	},
}

var adviceFrame = Frame{
	Name: "advice",
	Sections: []FrameSection{
		{Type: SecOpening, Role: "empathy", Goal: "acknowledge the situation", Required: true},
		{Type: SecBody, Role: "analysis", Goal: "break down the problem", Required: true},
		{Type: SecList, Role: "suggestions", Goal: "actionable recommendations", Required: true},
		{Type: SecClosing, Role: "encouragement", Goal: "supportive closing thought", Required: true},
	},
}

var argumentFrame = Frame{
	Name: "argument",
	Sections: []FrameSection{
		{Type: SecOpening, Role: "thesis", Goal: "state the position", Required: true},
		{Type: SecBody, Role: "evidence", Goal: "supporting evidence and reasoning", Required: true},
		{Type: SecBody, Role: "counterpoint", Goal: "acknowledge opposing views", Required: false},
		{Type: SecClosing, Role: "conclusion", Goal: "reinforce the thesis", Required: true},
	},
}

var summaryFrame = Frame{
	Name: "summary",
	Sections: []FrameSection{
		{Type: SecOpening, Role: "overview", Goal: "state what we're summarizing", Required: true},
		{Type: SecList, Role: "key_points", Goal: "the main takeaways", Required: true},
		{Type: SecClosing, Role: "conclusion", Goal: "overall significance", Required: false},
	},
}

var storyFrame = Frame{
	Name: "story",
	Sections: []FrameSection{
		{Type: SecBody, Role: "setup", Goal: "set the scene and introduce elements", Required: true},
		{Type: SecBody, Role: "development", Goal: "build tension or explore the theme", Required: true},
		{Type: SecBody, Role: "resolution", Goal: "resolve or deliver insight", Required: true},
	},
}

var planFrame = Frame{
	Name: "plan",
	Sections: []FrameSection{
		{Type: SecOpening, Role: "objective", Goal: "state what we're planning", Required: true},
		{Type: SecList, Role: "phases", Goal: "ordered phases or milestones", Required: true},
		{Type: SecBody, Role: "considerations", Goal: "things to keep in mind", Required: false},
		{Type: SecClosing, Role: "timeline", Goal: "suggested timeline", Required: false},
	},
}

var poemFrame = Frame{
	Name: "poem",
	Sections: []FrameSection{
		{Type: SecBody, Role: "verse", Goal: "express the theme through imagery", Required: true},
	},
}

var converseFrame = Frame{
	Name: "converse",
	Sections: []FrameSection{
		{Type: SecBody, Role: "response", Goal: "engage with the user's message", Required: true},
	},
}

// -----------------------------------------------------------------------
// Frame Selection
// -----------------------------------------------------------------------

// SelectFrame chooses the right frame for a task.
func SelectFrame(task ThinkTask, format OutputFormat) *Frame {
	// Explicit format overrides
	if format == FormatEmail {
		return &emailFrame
	}
	if format == FormatList {
		return &brainstormFrame
	}
	if format == FormatComparison {
		return &comparisonFrame
	}
	if format == FormatTutorial {
		return &tutorialFrame
	}

	// Task-based selection
	switch task {
	case TaskCompose:
		return &emailFrame
	case TaskBrainstorm:
		return &brainstormFrame
	case TaskAnalyze:
		return &explanationFrame
	case TaskTeach:
		return &tutorialFrame
	case TaskAdvise:
		return &adviceFrame
	case TaskCompare:
		return &comparisonFrame
	case TaskSummarize:
		return &summaryFrame
	case TaskCreate:
		return &storyFrame
	case TaskPlan:
		return &planFrame
	case TaskDebate:
		return &argumentFrame
	case TaskReflect:
		return &adviceFrame
	default:
		return &converseFrame
	}
}

// -----------------------------------------------------------------------
// Tone-Aware Phrase Pools
// -----------------------------------------------------------------------

// Greetings by tone
var greetingsByTone = map[Tone][]string{
	ToneFormal: {
		"Dear %s,",
		"Good day, %s.",
	},
	ToneCasual: {
		"Hey %s!",
		"Hi %s!",
		"Hi there, %s!",
	},
	ToneNeutral: {
		"Hello %s,",
		"Hi %s,",
	},
}

// Email openers by tone
var emailOpenersByTone = map[Tone][]string{
	ToneFormal: {
		"I hope this message finds you well.",
		"Thank you for your time.",
		"I am writing to bring something to your attention.",
	},
	ToneCasual: {
		"Hope you're doing well!",
		"Quick note for you.",
		"Just wanted to reach out.",
	},
	ToneNeutral: {
		"I wanted to reach out regarding something.",
		"I'm writing about a matter I'd like to discuss.",
	},
}

// Email closers by tone
var emailClosersByTone = map[Tone][]string{
	ToneFormal: {
		"Thank you for your time and consideration.",
		"I appreciate your attention to this matter.",
		"Please do not hesitate to reach out if you have any questions.",
		"I look forward to hearing from you.",
	},
	ToneCasual: {
		"Let me know what you think!",
		"Thanks in advance!",
		"Looking forward to your thoughts.",
		"Cheers!",
	},
	ToneNeutral: {
		"Thank you for considering this.",
		"Looking forward to your response.",
		"Please let me know your thoughts.",
	},
}

// Sign-offs by tone
var signoffsByTone = map[Tone][]string{
	ToneFormal: {
		"Best regards,",
		"Sincerely,",
		"Kind regards,",
		"Respectfully,",
	},
	ToneCasual: {
		"Cheers,",
		"Thanks,",
		"Talk soon,",
		"Best,",
	},
	ToneNeutral: {
		"Best,",
		"Thanks,",
		"Regards,",
	},
}

// pickFromTone selects a phrase from the tone-appropriate pool.
func pickFromTone(pools map[Tone][]string, tone Tone, rng *rand.Rand) string {
	pool, ok := pools[tone]
	if !ok || len(pool) == 0 {
		pool = pools[ToneNeutral]
	}
	if len(pool) == 0 {
		return ""
	}
	return pool[rng.Intn(len(pool))]
}

// -----------------------------------------------------------------------
// Brainstorm Category Templates
// -----------------------------------------------------------------------

// 5W1H brainstorm categories — universal for any topic
var brainstormCategories = []string{
	"Approaches", "Variations", "Tools", "Creative angles",
	"Practical steps", "Unconventional ideas",
}

// SCAMPER prompts — creative thinking framework
var scamperPrompts = map[string]string{
	"Substitute":      "What could you replace or swap out?",
	"Combine":         "What could you merge or bring together?",
	"Adapt":           "What could you borrow from something else?",
	"Modify":          "What could you change, enlarge, or reduce?",
	"Put to other use": "What else could this be used for?",
	"Eliminate":        "What could you remove or simplify?",
	"Reverse":          "What if you did the opposite?",
}

// -----------------------------------------------------------------------
// Content Templates — for generating text about unknown topics
// -----------------------------------------------------------------------

var purposeTemplates = []string{
	"I am writing regarding %s.",
	"I wanted to reach out about %s.",
	"This is regarding %s.",
	"The purpose of this message is to discuss %s.",
}

var bodyTemplates = []string{
	"When it comes to %s, there are several aspects worth considering.",
	"Looking at %s from different angles reveals interesting possibilities.",
	"%s involves multiple dimensions that are worth exploring.",
	"There are a few key things to understand about %s.",
}

var adviceOpeners = []string{
	"I understand this can feel challenging.",
	"That's a real concern, and it's worth thinking through carefully.",
	"This is something many people face, and there are good approaches.",
	"Let's break this down and look at it step by step.",
}

var encouragements = []string{
	"You've got this — take it one step at a time.",
	"The fact that you're thinking about this shows you're on the right track.",
	"Remember, progress matters more than perfection.",
	"Trust the process and be patient with yourself.",
}

// -----------------------------------------------------------------------
// Helper: detect output format from query
// -----------------------------------------------------------------------

// InferFormat guesses the desired output format from the query.
func InferFormat(query string) OutputFormat {
	lower := strings.ToLower(query)
	if strings.Contains(lower, "email") || strings.Contains(lower, "letter") ||
		strings.Contains(lower, "message to") || strings.Contains(lower, "write to") {
		return FormatEmail
	}
	if strings.Contains(lower, "list") || strings.Contains(lower, "bullet") ||
		strings.Contains(lower, "steps") || strings.Contains(lower, "checklist") {
		return FormatList
	}
	if strings.Contains(lower, "compare") || strings.Contains(lower, "versus") ||
		strings.Contains(lower, " vs ") || strings.Contains(lower, "difference") {
		return FormatComparison
	}
	if strings.Contains(lower, "how to") || strings.Contains(lower, "tutorial") ||
		strings.Contains(lower, "guide") || strings.Contains(lower, "step by step") {
		return FormatTutorial
	}
	return FormatProse
}

// InferTone guesses the appropriate tone from context.
func InferTone(query string, recipient string) Tone {
	lower := strings.ToLower(query)
	recipientLower := strings.ToLower(recipient)

	// Explicit tone hints
	if strings.Contains(lower, "formal") || strings.Contains(lower, "professional") {
		return ToneFormal
	}
	if strings.Contains(lower, "casual") || strings.Contains(lower, "informal") ||
		strings.Contains(lower, "friendly") {
		return ToneCasual
	}

	// Recipient-based inference
	formalRecipients := []string{"boss", "professor", "manager", "director", "client",
		"hr", "ceo", "supervisor", "dean", "principal", "recruiter"}
	for _, f := range formalRecipients {
		if strings.Contains(recipientLower, f) {
			return ToneFormal
		}
	}
	casualRecipients := []string{"friend", "buddy", "mate", "bro", "sis", "dude"}
	for _, c := range casualRecipients {
		if strings.Contains(recipientLower, c) {
			return ToneCasual
		}
	}

	return ToneNeutral
}
