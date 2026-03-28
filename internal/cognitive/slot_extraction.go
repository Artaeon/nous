package cognitive

import (
	"regexp"
	"strings"
)

// -----------------------------------------------------------------------
// Slot extraction with type validation.
//
// Extracts structured parameters (slots) from user queries using regex
// patterns and heuristic rules. Each slot is typed and validated against
// expectations for that type.
//
// Slot types map to TaskParams fields and inform downstream generation:
//   - Topic    -> what the query is about
//   - Goal     -> what the user wants to achieve
//   - Constraint -> budget, time, dietary, etc.
//   - ComparisonAxis -> "in terms of performance", "regarding price"
//   - TimeHorizon -> "this week", "by Friday", "within 30 days"
// -----------------------------------------------------------------------

// IntentSlotType defines the expected type of an intent slot value.
// (Named IntentSlotType to avoid conflict with SlotType in templates.go.)
type IntentSlotType int

const (
	IntentSlotTopic          IntentSlotType = iota
	IntentSlotGoal
	IntentSlotConstraint
	IntentSlotComparisonAxis
	IntentSlotTimeHorizon
	IntentSlotEntity
	IntentSlotQuantity
	IntentSlotFormat
	IntentSlotTone
	IntentSlotAudience
	IntentSlotLocation
	IntentSlotPerson
)

// ExtractedSlot is one validated slot.
type ExtractedSlot struct {
	Name       string
	Value      string
	Type       IntentSlotType
	Confidence float64
	Validated  bool // passed type validation
}

// ExtractedSlots holds all slots extracted from a query.
type ExtractedSlots struct {
	Topic          string
	Goal           string
	Constraints    []string
	ComparisonAxes []string
	TimeHorizon    string
	Entities       []ExtractedSlot
	AllSlots       map[string]*ExtractedSlot
	FilledCount    int
	ExpectedCount  int
}

// Compiled regexes for slot extraction (built once).
var (
	goalRe = regexp.MustCompile(`(?i)(?:i (?:want|need|would like|'d like) to |help me |i'm trying to |let me |i wish to )(.+?)(?:\.|$)`)

	constraintRes = []*regexp.Regexp{
		regexp.MustCompile(`(?i)(?:under|below|less than|cheaper than|no more than)\s+\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(?:dollars|usd|euros|gbp)?`),
		regexp.MustCompile(`(?i)(?:within|in less than|under|no more than)\s+(\d+\s*(?:minutes?|mins?|hours?|hrs?|days?|weeks?|months?))`),
		regexp.MustCompile(`(?i)(?:at least|minimum|no less than|more than)\s+\$?\s*(\d[\d,]*(?:\.\d+)?)`),
		regexp.MustCompile(`(?i)(?:between)\s+\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(?:and|to|-)\s*\$?\s*(\d[\d,]*(?:\.\d+)?)`),
		regexp.MustCompile(`(?i)(?:without|no|excluding|not including|except)\s+([\w\s]+?)(?:\s*[,.]|\s+and\s+|\s+or\s+|$)`),
		regexp.MustCompile(`(?i)(?:with|including|must have|requires?|needs?)\s+([\w\s]+?)(?:\s*[,.]|\s+and\s+|\s+or\s+|$)`),
		regexp.MustCompile(`(?i)(?:for|suitable for|appropriate for)\s+(beginners?|advanced|intermediate|kids?|children|adults?|professionals?|seniors?)`),
	}

	compAxisRes = []*regexp.Regexp{
		regexp.MustCompile(`(?i)(?:in terms of|regarding|with respect to|when it comes to|as far as)\s+(.+?)(?:\s*[,.]|$)`),
		regexp.MustCompile(`(?i)(?:for|based on|considering)\s+(price|cost|performance|speed|quality|reliability|durability|ease of use|features?|security|safety|comfort|design|battery|weight|size|portability|value)(?:\s*[,.]|\s+and\s+|$)`),
	}

	timeHorizonRes = []*regexp.Regexp{
		regexp.MustCompile(`(?i)(?:by|before|until|due)\s+(tomorrow|next (?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?)`),
		regexp.MustCompile(`(?i)(?:in|within|over)\s+(?:the\s+)?(?:next\s+)?(\d+\s*(?:minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?))`),
		regexp.MustCompile(`(?i)(today|tonight|this (?:week|month|year|weekend|morning|afternoon|evening))`),
		regexp.MustCompile(`(?i)(short[ -]term|medium[ -]term|long[ -]term)`),
	}

	personRe   = regexp.MustCompile(`(?i)(?:for|to|from|with|by)\s+(?:my\s+)?(mom|dad|mother|father|brother|sister|boss|friend|colleague|partner|wife|husband|teacher|professor|doctor|manager|team)`)
	locationRe = regexp.MustCompile(`(?i)(?:in|at|near|around|from)\s+([\w\s]+?(?:city|town|country|state|village|airport|station|park|beach|mountain|lake|river|street|avenue|boulevard))`)
	quantityRe = regexp.MustCompile(`(?i)(\d+(?:\.\d+)?)\s*(kg|lbs?|pounds?|miles?|km|kilometers?|gallons?|liters?|litres?|cups?|oz|ounces?|grams?|meters?|metres?|feet|inches?|cm|mm|percent|%|dollars?|euros?|\$|items?|pieces?|servings?|people|persons?)`)
	slotAudienceRe = regexp.MustCompile(`(?i)(?:for|aimed at|targeted at|suitable for|appropriate for|written for|designed for)\s+(?:a\s+)?([\w-]+(?:\s+[\w-]+){0,3}?)(?:\s*[,.]|\s+(?:who|that|and)|$)`)
	formatRe   = regexp.MustCompile(`(?i)(?:as|in|formatted as|format(?:ted)? (?:as|in)|in the form of)\s+(?:a\s+|an\s+)?(list|bullet points?|table|paragraph|essay|email|letter|report|summary|outline|json|csv|markdown|code|steps?|instructions?)`)
	toneRe     = regexp.MustCompile(`(?i)(?:in a |use a |with a |make it )?(formal|casual|professional|friendly|humorous|serious|academic|conversational|technical|simple|playful|sarcastic|empathetic|neutral)\s+(?:tone|style|manner|voice|way)`)
)

// ExtractSlots extracts typed, validated slots from the input.
// The coarse intent and sub-intent guide which slots to expect.
func ExtractSlots(input string, coarse CoarseIntent, subIntent string) *ExtractedSlots {
	slots := &ExtractedSlots{
		AllSlots: make(map[string]*ExtractedSlot),
	}

	lower := strings.ToLower(input)

	// Determine expected slot count based on intent
	slots.ExpectedCount = expectedSlotsForIntent(coarse, subIntent)

	// 1. Extract topic from main noun phrase
	topic := extractTopicSlot(lower, coarse, subIntent)
	if topic != "" {
		slots.Topic = topic
		slot := &ExtractedSlot{
			Name:       "topic",
			Value:      topic,
			Type:       IntentSlotTopic,
			Confidence: 0.80,
			Validated:  true,
		}
		ValidateSlot(slot)
		slots.AllSlots["topic"] = slot
		slots.Entities = append(slots.Entities, *slot)
	}

	// 2. Extract goal from verb phrases
	if m := goalRe.FindStringSubmatch(lower); len(m) > 1 {
		goal := strings.TrimSpace(m[1])
		if goal != "" {
			slots.Goal = goal
			slot := &ExtractedSlot{
				Name:       "goal",
				Value:      goal,
				Type:       IntentSlotGoal,
				Confidence: 0.75,
				Validated:  true,
			}
			ValidateSlot(slot)
			slots.AllSlots["goal"] = slot
			slots.Entities = append(slots.Entities, *slot)
		}
	}

	// 3. Extract constraints
	for _, re := range constraintRes {
		if matches := re.FindAllStringSubmatch(lower, -1); len(matches) > 0 {
			for _, m := range matches {
				constraint := strings.TrimSpace(m[0])
				slots.Constraints = append(slots.Constraints, constraint)
				slot := &ExtractedSlot{
					Name:       "constraint",
					Value:      constraint,
					Type:       IntentSlotConstraint,
					Confidence: 0.70,
				}
				ValidateSlot(slot)
				key := "constraint_" + itoa(len(slots.Constraints))
				slots.AllSlots[key] = slot
				slots.Entities = append(slots.Entities, *slot)
			}
		}
	}

	// 4. Extract comparison axes
	for _, re := range compAxisRes {
		if matches := re.FindAllStringSubmatch(lower, -1); len(matches) > 0 {
			for _, m := range matches {
				if len(m) > 1 {
					axis := strings.TrimSpace(m[1])
					if axis != "" {
						slots.ComparisonAxes = append(slots.ComparisonAxes, axis)
						slot := &ExtractedSlot{
							Name:       "comparison_axis",
							Value:      axis,
							Type:       IntentSlotComparisonAxis,
							Confidence: 0.75,
						}
						ValidateSlot(slot)
						key := "comparison_axis_" + itoa(len(slots.ComparisonAxes))
						slots.AllSlots[key] = slot
						slots.Entities = append(slots.Entities, *slot)
					}
				}
			}
		}
	}

	// 5. Extract time horizon
	for _, re := range timeHorizonRes {
		if m := re.FindStringSubmatch(lower); len(m) > 1 {
			horizon := strings.TrimSpace(m[1])
			if horizon != "" {
				slots.TimeHorizon = horizon
				slot := &ExtractedSlot{
					Name:       "time_horizon",
					Value:      horizon,
					Type:       IntentSlotTimeHorizon,
					Confidence: 0.80,
				}
				ValidateSlot(slot)
				slots.AllSlots["time_horizon"] = slot
				slots.Entities = append(slots.Entities, *slot)
				break
			}
		}
	}

	// 6. Extract person references
	if m := personRe.FindStringSubmatch(lower); len(m) > 1 {
		person := strings.TrimSpace(m[1])
		slot := &ExtractedSlot{
			Name:       "person",
			Value:      person,
			Type:       IntentSlotPerson,
			Confidence: 0.80,
		}
		ValidateSlot(slot)
		slots.AllSlots["person"] = slot
		slots.Entities = append(slots.Entities, *slot)
	}

	// 7. Extract location
	if m := locationRe.FindStringSubmatch(lower); len(m) > 1 {
		location := strings.TrimSpace(m[1])
		slot := &ExtractedSlot{
			Name:       "location",
			Value:      location,
			Type:       IntentSlotLocation,
			Confidence: 0.70,
		}
		ValidateSlot(slot)
		slots.AllSlots["location"] = slot
		slots.Entities = append(slots.Entities, *slot)
	}

	// 8. Extract quantities
	if m := quantityRe.FindStringSubmatch(lower); len(m) > 2 {
		quantity := strings.TrimSpace(m[0])
		slot := &ExtractedSlot{
			Name:       "quantity",
			Value:      quantity,
			Type:       IntentSlotQuantity,
			Confidence: 0.85,
		}
		ValidateSlot(slot)
		slots.AllSlots["quantity"] = slot
		slots.Entities = append(slots.Entities, *slot)
	}

	// 9. Extract audience
	if m := slotAudienceRe.FindStringSubmatch(lower); len(m) > 1 {
		audience := strings.TrimSpace(m[1])
		// Filter out common false positives
		if !slotIsStopWord(audience) && len(audience) > 2 {
			slot := &ExtractedSlot{
				Name:       "audience",
				Value:      audience,
				Type:       IntentSlotAudience,
				Confidence: 0.70,
			}
			ValidateSlot(slot)
			slots.AllSlots["audience"] = slot
			slots.Entities = append(slots.Entities, *slot)
		}
	}

	// 10. Extract format requests
	if m := formatRe.FindStringSubmatch(lower); len(m) > 1 {
		format := strings.TrimSpace(m[1])
		slot := &ExtractedSlot{
			Name:       "format",
			Value:      format,
			Type:       IntentSlotFormat,
			Confidence: 0.85,
		}
		ValidateSlot(slot)
		slots.AllSlots["format"] = slot
		slots.Entities = append(slots.Entities, *slot)
	}

	// 11. Extract tone
	if m := toneRe.FindStringSubmatch(lower); len(m) > 1 {
		tone := strings.TrimSpace(m[1])
		slot := &ExtractedSlot{
			Name:       "tone",
			Value:      tone,
			Type:       IntentSlotTone,
			Confidence: 0.85,
		}
		ValidateSlot(slot)
		slots.AllSlots["tone"] = slot
		slots.Entities = append(slots.Entities, *slot)
	}

	// Count filled slots
	slots.FilledCount = len(slots.AllSlots)

	return slots
}

// ValidateSlot checks whether a slot value is valid for its declared type.
// Returns true if valid, false otherwise. Also sets the Validated field.
func ValidateSlot(slot *ExtractedSlot) bool {
	if slot.Value == "" {
		slot.Validated = false
		slot.Confidence *= 0.5
		return false
	}

	switch slot.Type {
	case IntentSlotTopic:
		// Topic should be non-empty and not just stop words
		slot.Validated = len(slot.Value) > 0 && !slotIsAllStopWords(slot.Value)

	case IntentSlotGoal:
		// Goal should contain a verb-like phrase
		slot.Validated = len(slot.Value) > 2

	case IntentSlotConstraint:
		// Constraint should have some concrete qualifier
		slot.Validated = len(slot.Value) > 3

	case IntentSlotComparisonAxis:
		// Axis should be a recognizable dimension
		slot.Validated = len(slot.Value) > 1

	case IntentSlotTimeHorizon:
		// Time horizon should look like a temporal expression
		slot.Validated = isTemporalExpression(slot.Value)

	case IntentSlotEntity:
		// Entity should be non-empty
		slot.Validated = len(slot.Value) > 0

	case IntentSlotQuantity:
		// Quantity should contain digits
		slot.Validated = slotContainsDigit(slot.Value)

	case IntentSlotFormat:
		// Format should be a recognized format name
		validFormats := map[string]bool{
			"list": true, "bullet points": true, "bullet point": true,
			"table": true, "paragraph": true, "essay": true,
			"email": true, "letter": true, "report": true,
			"summary": true, "outline": true, "json": true,
			"csv": true, "markdown": true, "code": true,
			"steps": true, "step": true, "instructions": true,
		}
		slot.Validated = validFormats[strings.ToLower(slot.Value)]

	case IntentSlotTone:
		validTones := map[string]bool{
			"formal": true, "casual": true, "professional": true,
			"friendly": true, "humorous": true, "serious": true,
			"academic": true, "conversational": true, "technical": true,
			"simple": true, "playful": true, "sarcastic": true,
			"empathetic": true, "neutral": true,
		}
		slot.Validated = validTones[strings.ToLower(slot.Value)]

	case IntentSlotAudience:
		slot.Validated = len(slot.Value) > 2 && !slotIsStopWord(slot.Value)

	case IntentSlotLocation:
		slot.Validated = len(slot.Value) > 1

	case IntentSlotPerson:
		slot.Validated = len(slot.Value) > 1

	default:
		slot.Validated = len(slot.Value) > 0
	}

	// Adjust confidence based on validation
	if !slot.Validated {
		slot.Confidence *= 0.5
	}

	return slot.Validated
}

// extractTopicSlot extracts the main topic from user input.
// Uses different strategies depending on the coarse intent.
func extractTopicSlot(lower string, coarse CoarseIntent, subIntent string) string {
	// Strip common prefixes to expose the noun phrase
	prefixes := []string{
		// Question prefixes
		"what is ", "what's ", "what are ", "what was ", "what were ",
		"who is ", "who are ", "who was ", "who were ",
		"where is ", "where are ",
		"when is ", "when was ", "when did ",
		"why is ", "why are ", "why does ", "why do ",
		"how does ", "how do ", "how is ", "how are ", "how did ",
		// Explain prefixes
		"explain ", "describe ", "tell me about ", "teach me about ",
		"tell me everything about ", "tell me all about ",
		"give me an overview of ", "give me a full overview of ",
		"walk me through ", "deep dive into ",
		"define ", "definition of ",
		// Compare prefixes
		"compare ", "difference between ", "differences between ",
		// Task prefixes
		"write me ", "write a ", "write an ", "write ",
		"create a ", "create an ", "create ",
		"make a ", "make an ", "make me ",
		"compose a ", "compose an ", "compose ",
		"help me ", "help me with ",
		"plan a ", "plan an ", "plan ",
		"build a ", "build an ", "build ",
		// Command prefixes
		"please ", "can you ", "could you ", "would you ",
		"i want to ", "i need to ", "i'd like to ",
	}

	topic := lower
	for _, p := range prefixes {
		if strings.HasPrefix(topic, p) {
			topic = topic[len(p):]
			break
		}
	}

	// Strip trailing noise
	topic = strings.TrimRight(topic, "?!.,;:")
	topic = strings.TrimSpace(topic)

	// Strip trailing qualifiers that are extracted as separate slots
	trailingRes := []*regexp.Regexp{
		regexp.MustCompile(`(?i)\s+(?:in terms of|regarding|with respect to|when it comes to)\s+.+$`),
		regexp.MustCompile(`(?i)\s+(?:for|aimed at|suitable for)\s+(?:beginners?|advanced|intermediate|kids?|children|adults?|professionals?)\s*$`),
		regexp.MustCompile(`(?i)\s+(?:by|before|until|within|in the next)\s+.+$`),
	}
	for _, re := range trailingRes {
		topic = re.ReplaceAllString(topic, "")
	}

	topic = strings.TrimSpace(topic)

	// For comparison sub-intents, try to extract just the compared items
	if subIntent == "compare_tradeoff" || subIntent == "compare" {
		vsRe := regexp.MustCompile(`(?i)(.+?)\s+(?:vs\.?|versus|or|compared to|against)\s+(.+)`)
		if m := vsRe.FindStringSubmatch(topic); len(m) > 2 {
			return strings.TrimSpace(m[1]) + " vs " + strings.TrimSpace(m[2])
		}
	}

	return topic
}

// expectedSlotsForIntent returns the number of slots typically expected
// for a given intent combination.
func expectedSlotsForIntent(coarse CoarseIntent, subIntent string) int {
	switch coarse {
	case CoarseQuery:
		switch subIntent {
		case "factual_qa":
			return 1 // topic
		case "deep_explain":
			return 2 // topic + audience/format
		case "compare_tradeoff":
			return 3 // topic + comparison axes
		default:
			return 1
		}
	case CoarseTask:
		switch subIntent {
		case "compose":
			return 3 // topic + tone + audience
		case "planning":
			return 3 // topic + time horizon + constraints
		case "transform":
			return 2 // topic + format
		default:
			return 2
		}
	case CoarseConversation:
		return 1 // topic
	case CoarseNavigation:
		return 1 // tool-specific entity
	case CoarseMeta:
		return 0
	}
	return 1
}

// isTemporalExpression checks if a string looks like a time/date expression.
func isTemporalExpression(s string) bool {
	lower := strings.ToLower(s)

	temporalWords := []string{
		"today", "tonight", "tomorrow", "yesterday",
		"this week", "this month", "this year", "this weekend",
		"next week", "next month", "next year",
		"last week", "last month", "last year",
		"short-term", "short term", "medium-term", "medium term",
		"long-term", "long term",
		"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
		"january", "february", "march", "april", "may", "june",
		"july", "august", "september", "october", "november", "december",
	}
	for _, tw := range temporalWords {
		if strings.Contains(lower, tw) {
			return true
		}
	}

	// Check for durations: "5 minutes", "2 hours", "30 days"
	durationRe := regexp.MustCompile(`\d+\s*(?:seconds?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)`)
	return durationRe.MatchString(lower)
}

// slotContainsDigit returns true if the string contains at least one digit.
func slotContainsDigit(s string) bool {
	for _, c := range s {
		if c >= '0' && c <= '9' {
			return true
		}
	}
	return false
}

// slotIsStopWord returns true if the word is a common stop word.
func slotIsStopWord(s string) bool {
	stops := map[string]bool{
		"a": true, "an": true, "the": true, "is": true, "are": true,
		"was": true, "were": true, "be": true, "been": true, "being": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
		"may": true, "might": true, "shall": true, "can": true,
		"of": true, "in": true, "to": true, "for": true, "with": true,
		"on": true, "at": true, "from": true, "by": true, "about": true,
		"it": true, "its": true, "this": true, "that": true, "these": true,
		"those": true, "i": true, "me": true, "my": true, "we": true,
		"you": true, "your": true, "he": true, "she": true, "they": true,
	}
	return stops[strings.ToLower(strings.TrimSpace(s))]
}

// slotIsAllStopWords returns true if every word in the string is a stop word.
func slotIsAllStopWords(s string) bool {
	words := strings.Fields(s)
	if len(words) == 0 {
		return true
	}
	for _, w := range words {
		if !slotIsStopWord(w) {
			return false
		}
	}
	return true
}

// itoa converts an int to a string without importing strconv.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := false
	if n < 0 {
		neg = true
		n = -n
	}
	digits := make([]byte, 0, 10)
	for n > 0 {
		digits = append(digits, byte('0'+n%10))
		n /= 10
	}
	if neg {
		digits = append(digits, '-')
	}
	// reverse
	for i, j := 0, len(digits)-1; i < j; i, j = i+1, j-1 {
		digits[i], digits[j] = digits[j], digits[i]
	}
	return string(digits)
}
