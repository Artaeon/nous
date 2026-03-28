package cognitive

import (
	"regexp"
	"strings"
)

// -----------------------------------------------------------------------
// Follow-Up Resolver — detects and resolves conversational follow-ups.
//
// When a user says "tell me more", "what about X?", "compare with Y",
// the system needs to:
//   1. Detect that it IS a follow-up (not a standalone query)
//   2. Classify the TYPE of follow-up
//   3. Resolve it by carrying over prior context
//   4. Produce a fully resolved query the rest of the pipeline can handle
//
// This is the bridge between terse human dialogue and the context-free
// query processing that the knowledge/thinking engines expect.
// -----------------------------------------------------------------------

// FollowUpType classifies the type of follow-up.
type FollowUpType int

const (
	FollowUpNone      FollowUpType = iota // not a follow-up
	FollowUpDeeper                        // "go deeper", "tell me more", "elaborate"
	FollowUpPivot                         // "what about X?", "how about Y?"
	FollowUpCompare                       // "compare with Z", "how does it differ from..."
	FollowUpClarify                       // "what do you mean by...", "can you explain..."
	FollowUpNarrow                        // "specifically about...", "focus on..."
	FollowUpBroaden                       // "more generally...", "in a broader sense..."
	FollowUpChallenge                     // "but what about...", "isn't that wrong because..."
	FollowUpApply                         // "how would that work for...", "what if I..."
)

// String returns the name of a FollowUpType.
func (ft FollowUpType) String() string {
	switch ft {
	case FollowUpNone:
		return "none"
	case FollowUpDeeper:
		return "deeper"
	case FollowUpPivot:
		return "pivot"
	case FollowUpCompare:
		return "compare"
	case FollowUpClarify:
		return "clarify"
	case FollowUpNarrow:
		return "narrow"
	case FollowUpBroaden:
		return "broaden"
	case FollowUpChallenge:
		return "challenge"
	case FollowUpApply:
		return "apply"
	default:
		return "unknown"
	}
}

// FollowUpResult captures the resolved follow-up.
type FollowUpResult struct {
	Type          FollowUpType
	IsFollowUp    bool
	OriginalQuery string
	ResolvedQuery string   // the fully resolved query with context carried over
	CarryOver     []string // what context was carried from previous turns
	NewEntity     string   // newly introduced entity (for pivot/compare)
	PriorTopic    string   // the topic being followed up on
}

// FollowUpResolver detects and resolves follow-up queries.
type FollowUpResolver struct {
	deeperPatterns    []string
	pivotPatterns     []*regexp.Regexp
	comparePatterns   []*regexp.Regexp
	clarifyPatterns   []string
	narrowPatterns    []string
	broadenPatterns   []string
	challengePatterns []string
	applyPatterns     []*regexp.Regexp
}

// NewFollowUpResolver creates a resolver with all detection patterns.
func NewFollowUpResolver() *FollowUpResolver {
	return &FollowUpResolver{
		deeperPatterns: []string{
			"tell me more",
			"go deeper",
			"elaborate",
			"more detail",
			"more details",
			"expand on that",
			"keep going",
			"continue",
			"go on",
			"more about that",
			"more about this",
			"can you elaborate",
			"explain further",
			"dig deeper",
			"in more depth",
			"tell me everything",
		},
		pivotPatterns: []*regexp.Regexp{
			regexp.MustCompile(`(?i)^what about (.+?)[\?\.]?$`),
			regexp.MustCompile(`(?i)^how about (.+?)[\?\.]?$`),
			regexp.MustCompile(`(?i)^and (.+?)[\?\.]?$`),
			regexp.MustCompile(`(?i)^what of (.+?)[\?\.]?$`),
			regexp.MustCompile(`(?i)^now (?:tell me )?about (.+?)[\?\.]?$`),
			regexp.MustCompile(`(?i)^(?:ok|okay|alright),?\s*(?:now\s+)?(?:what about|how about) (.+?)[\?\.]?$`),
			regexp.MustCompile(`(?i)^what about (.+)$`),
			regexp.MustCompile(`(?i)^how about (.+)$`),
		},
		comparePatterns: []*regexp.Regexp{
			regexp.MustCompile(`(?i)compare (?:it |that |this )?with (.+?)[\?\.]?$`),
			regexp.MustCompile(`(?i)(?:how does (?:it|that|this) )?(?:differ|compare) (?:from|to|with) (.+?)[\?\.]?$`),
			regexp.MustCompile(`(?i)versus (.+?)[\?\.]?$`),
			regexp.MustCompile(`(?i)^(.+?) vs\.? (?:that|it|this)[\?\.]?$`),
			regexp.MustCompile(`(?i)^(?:and |what about )(.+?) (?:vs\.?|versus|compared to) (?:that|it|this)[\?\.]?$`),
			regexp.MustCompile(`(?i)^(?:how is |how does )(.+?) different[\?\.]?$`),
			regexp.MustCompile(`(?i)(?:difference between (?:that|it) and )(.+?)[\?\.]?$`),
		},
		clarifyPatterns: []string{
			"what do you mean",
			"can you clarify",
			"i don't understand",
			"i don't get it",
			"what does that mean",
			"can you explain that",
			"what are you saying",
			"i'm confused",
			"clarify that",
			"not sure i follow",
			"explain what you mean",
			"be more specific",
		},
		narrowPatterns: []string{
			"specifically",
			"focus on",
			"just the",
			"only about",
			"narrow it down",
			"zero in on",
			"in particular",
			"concentrate on",
			"let's focus",
		},
		broadenPatterns: []string{
			"more generally",
			"in general",
			"broadly speaking",
			"big picture",
			"stepping back",
			"zoom out",
			"in a broader sense",
			"at a higher level",
			"overall",
			"the bigger picture",
		},
		challengePatterns: []string{
			"but what about",
			"isn't it true that",
			"that's not right",
			"that's not correct",
			"but doesn't",
			"what about the case where",
			"that ignores",
			"you're forgetting",
			"but surely",
			"however",
			"i disagree",
			"that can't be right",
			"counterpoint",
			"on the other hand",
		},
		applyPatterns: []*regexp.Regexp{
			regexp.MustCompile(`(?i)how would that work for (.+?)[\?\.]?$`),
			regexp.MustCompile(`(?i)what if [iI] (.+?)[\?\.]?$`),
			regexp.MustCompile(`(?i)in my case[,.]?\s*(.+?)[\?\.]?$`),
			regexp.MustCompile(`(?i)for example[,.]?\s*(?:if|when) (.+?)[\?\.]?$`),
			regexp.MustCompile(`(?i)(?:how )?(?:would|could|can) (?:I|we|someone) (?:use|apply) (?:that|this|it) (?:for|to|with|in) (.+?)[\?\.]?$`),
			regexp.MustCompile(`(?i)(?:how )?(?:does|would) (?:that|this|it) apply to (.+?)[\?\.]?$`),
			regexp.MustCompile(`(?i)^(?:apply|use) (?:that|this|it) (?:to|for|with) (.+?)[\?\.]?$`),
		},
	}
}

// Resolve detects if input is a follow-up and resolves it using conversation state.
func (fr *FollowUpResolver) Resolve(input string, state *ConversationState) *FollowUpResult {
	lower := strings.ToLower(strings.TrimSpace(input))

	result := &FollowUpResult{
		OriginalQuery: input,
		PriorTopic:    state.ActiveTopic,
	}

	// No prior context means nothing to follow up on
	if state.TurnCount == 0 && state.ActiveTopic == "" {
		result.Type = FollowUpNone
		result.ResolvedQuery = input
		return result
	}

	// Try each follow-up type in priority order.
	// Compare before pivot because "compare with X" is more specific than "what about X".
	if fr.tryCompare(lower, state, result) {
		return result
	}
	if fr.tryApply(lower, input, state, result) {
		return result
	}
	if fr.tryDeeper(lower, state, result) {
		return result
	}
	if fr.tryClarify(lower, state, result) {
		return result
	}
	if fr.tryChallenge(lower, state, result) {
		return result
	}
	if fr.tryNarrow(lower, input, state, result) {
		return result
	}
	if fr.tryBroaden(lower, state, result) {
		return result
	}
	if fr.tryPivot(lower, state, result) {
		return result
	}

	// Not a follow-up
	result.Type = FollowUpNone
	result.ResolvedQuery = input
	return result
}

// -----------------------------------------------------------------------
// Detection methods — one per follow-up type
// -----------------------------------------------------------------------

func (fr *FollowUpResolver) tryDeeper(lower string, state *ConversationState, result *FollowUpResult) bool {
	for _, pattern := range fr.deeperPatterns {
		if strings.Contains(lower, pattern) {
			result.Type = FollowUpDeeper
			result.IsFollowUp = true

			// Build resolved query: "explain [topic] in more detail"
			topic := state.ActiveTopic
			if topic == "" {
				topic = "the previous topic"
			}
			result.ResolvedQuery = "explain " + topic + " in more detail"

			// Carry over the active topic and any entities
			result.CarryOver = []string{topic}
			for _, t := range state.TopicStack {
				if t != topic {
					result.CarryOver = append(result.CarryOver, t)
					break // just one additional context topic
				}
			}
			return true
		}
	}
	return false
}

func (fr *FollowUpResolver) tryPivot(lower string, state *ConversationState, result *FollowUpResult) bool {
	for _, re := range fr.pivotPatterns {
		if matches := re.FindStringSubmatch(lower); len(matches) > 1 {
			newEntity := strings.TrimSpace(matches[1])
			newEntity = trimTrailingPunctuation(newEntity)

			result.Type = FollowUpPivot
			result.IsFollowUp = true
			result.NewEntity = newEntity

			// Build resolved query incorporating prior context
			if state.ActiveTopic != "" {
				result.ResolvedQuery = "tell me about " + newEntity +
					" in the context of " + state.ActiveTopic
				result.CarryOver = []string{state.ActiveTopic}
			} else {
				result.ResolvedQuery = "tell me about " + newEntity
			}
			return true
		}
	}
	return false
}

func (fr *FollowUpResolver) tryCompare(lower string, state *ConversationState, result *FollowUpResult) bool {
	for _, re := range fr.comparePatterns {
		if matches := re.FindStringSubmatch(lower); len(matches) > 1 {
			newEntity := strings.TrimSpace(matches[1])
			newEntity = trimTrailingPunctuation(newEntity)

			result.Type = FollowUpCompare
			result.IsFollowUp = true
			result.NewEntity = newEntity

			// Build comparison query between prior topic and new entity
			priorTopic := state.ActiveTopic
			if priorTopic == "" {
				priorTopic = "the previous topic"
			}
			result.ResolvedQuery = "compare " + priorTopic + " with " + newEntity
			result.CarryOver = []string{priorTopic}
			return true
		}
	}
	return false
}

func (fr *FollowUpResolver) tryClarify(lower string, state *ConversationState, result *FollowUpResult) bool {
	for _, pattern := range fr.clarifyPatterns {
		if strings.Contains(lower, pattern) {
			result.Type = FollowUpClarify
			result.IsFollowUp = true

			// Try to extract the unclear term
			unclearTerm := ""

			// "what do you mean by X" pattern
			byIdx := strings.Index(lower, " by ")
			if byIdx != -1 {
				unclearTerm = strings.TrimSpace(lower[byIdx+4:])
				unclearTerm = trimTrailingPunctuation(unclearTerm)
			}

			topic := state.ActiveTopic
			if topic == "" {
				topic = "the previous point"
			}

			if unclearTerm != "" {
				result.ResolvedQuery = "define and explain " + unclearTerm +
					" in the context of " + topic
				result.NewEntity = unclearTerm
			} else {
				result.ResolvedQuery = "clarify the explanation of " + topic +
					" in simpler terms"
			}
			result.CarryOver = []string{topic}
			return true
		}
	}
	return false
}

func (fr *FollowUpResolver) tryNarrow(lower string, original string, state *ConversationState, result *FollowUpResult) bool {
	for _, pattern := range fr.narrowPatterns {
		if strings.Contains(lower, pattern) {
			result.Type = FollowUpNarrow
			result.IsFollowUp = true

			// Extract the focus area — what comes after the pattern keyword
			idx := strings.Index(lower, pattern)
			rest := strings.TrimSpace(lower[idx+len(pattern):])
			rest = trimTrailingPunctuation(rest)

			topic := state.ActiveTopic
			if topic == "" {
				topic = "the previous topic"
			}

			if rest != "" {
				result.ResolvedQuery = "explain " + rest + " aspect of " + topic
				result.NewEntity = rest
			} else {
				// "specifically" without a target — just narrow the topic
				result.ResolvedQuery = "explain the key specifics of " + topic
			}
			result.CarryOver = []string{topic}
			return true
		}
	}
	return false
}

func (fr *FollowUpResolver) tryBroaden(lower string, state *ConversationState, result *FollowUpResult) bool {
	for _, pattern := range fr.broadenPatterns {
		if strings.Contains(lower, pattern) {
			result.Type = FollowUpBroaden
			result.IsFollowUp = true

			topic := state.ActiveTopic
			if topic == "" {
				topic = "the previous topic"
			}

			result.ResolvedQuery = "explain " + topic + " at a broader level, covering the big picture"
			result.CarryOver = []string{topic}

			// Also carry over sub-topics from the stack
			for i, t := range state.TopicStack {
				if i > 0 && t != topic && len(result.CarryOver) < 4 {
					result.CarryOver = append(result.CarryOver, t)
				}
			}
			return true
		}
	}
	return false
}

func (fr *FollowUpResolver) tryChallenge(lower string, state *ConversationState, result *FollowUpResult) bool {
	for _, pattern := range fr.challengePatterns {
		if strings.Contains(lower, pattern) {
			result.Type = FollowUpChallenge
			result.IsFollowUp = true

			topic := state.ActiveTopic
			if topic == "" {
				topic = "the previous claim"
			}

			// Extract the challenge content — everything after the challenge marker
			idx := strings.Index(lower, pattern)
			challenge := strings.TrimSpace(lower[idx+len(pattern):])
			challenge = trimTrailingPunctuation(challenge)

			if challenge != "" {
				result.ResolvedQuery = "address the counterpoint about " + topic +
					": " + challenge
				result.NewEntity = challenge
			} else {
				result.ResolvedQuery = "address counterarguments about " + topic
			}
			result.CarryOver = []string{topic}

			// Include any assumptions for the challenge to address
			for _, a := range state.Assumptions {
				result.CarryOver = append(result.CarryOver, "assumed: "+a)
			}
			return true
		}
	}
	return false
}

func (fr *FollowUpResolver) tryApply(lower string, original string, state *ConversationState, result *FollowUpResult) bool {
	for _, re := range fr.applyPatterns {
		if matches := re.FindStringSubmatch(lower); len(matches) > 1 {
			application := strings.TrimSpace(matches[1])
			application = trimTrailingPunctuation(application)

			result.Type = FollowUpApply
			result.IsFollowUp = true
			result.NewEntity = application

			topic := state.ActiveTopic
			if topic == "" {
				topic = "the previous concept"
			}

			result.ResolvedQuery = "apply the concept of " + topic +
				" to the scenario: " + application
			result.CarryOver = []string{topic}
			return true
		}
	}
	return false
}
