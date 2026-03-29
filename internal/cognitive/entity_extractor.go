package cognitive

import (
	"regexp"
	"strings"
)

// -----------------------------------------------------------------------
// Smart Entity Extraction
//
// Extracts typed entities from user input AFTER intent classification.
// This bridges the gap between "I know what you want" (intent) and
// "I know the details" (entities).
//
// The neural NLU correctly classifies intents (calculate, translate, etc.)
// but does not always extract the specific entities that tool handlers
// need (expression, text, language, duration, units). This extractor
// fills that gap using intent-specific regex patterns and heuristics.
// -----------------------------------------------------------------------

// SmartEntityExtractor extracts typed entities from user input AFTER
// intent classification.
type SmartEntityExtractor struct{}

// NewSmartEntityExtractor creates a new SmartEntityExtractor.
func NewSmartEntityExtractor() *SmartEntityExtractor {
	return &SmartEntityExtractor{}
}

// ExtractForIntent extracts entities relevant to the classified intent.
// It mutates the provided entities map, adding any keys the downstream
// tool handler expects.
func (se *SmartEntityExtractor) ExtractForIntent(input string, intent string, entities map[string]string) {
	if entities == nil {
		return
	}

	switch intent {
	case "calculate", "compute":
		se.extractMath(input, entities)
	case "translate":
		se.extractTranslate(input, entities)
	case "weather":
		se.extractWeather(input, entities)
	case "timer":
		se.extractTimer(input, entities)
	case "convert":
		se.extractConvert(input, entities)
	}
}

// -----------------------------------------------------------------------
// Math / Calculate
// -----------------------------------------------------------------------

// extractMath parses natural-language math expressions into evaluable form.
//
//   "15% of 340"         → expression="0.15*340"
//   "whats 5 + 3"        → expression="5+3"
//   "square root of 144" → expression="sqrt(144)"
//   "20% tip on 85"      → expression="0.20*85"
//   "15 times 7"         → expression="15*7"
//   "100 divided by 4"   → expression="100/4"
func (se *SmartEntityExtractor) extractMath(input string, entities map[string]string) {
	if entities["expression"] != "" {
		return // already extracted
	}

	lower := strings.ToLower(input)

	// Strip common question prefixes
	for _, prefix := range []string{
		"whats ", "what's ", "what is ", "how much is ",
		"calculate ", "compute ", "evaluate ", "solve ",
	} {
		if strings.HasPrefix(lower, prefix) {
			lower = lower[len(prefix):]
			break
		}
	}
	lower = strings.TrimRight(lower, "?!. ")

	expr := lower

	// "square root of N" → "sqrt(N)"
	expr = reSquareRoot.ReplaceAllString(expr, "sqrt($1)")

	// "cube root of N" → "cbrt(N)"
	expr = reCubeRoot.ReplaceAllString(expr, "cbrt($1)")

	// "N% of M" → "0.0N*M"  (e.g. "15% of 340" → "0.15*340")
	expr = rePercentOf.ReplaceAllStringFunc(expr, func(m string) string {
		parts := rePercentOf.FindStringSubmatch(m)
		if len(parts) == 3 {
			return percentToDecimal(parts[1]) + "*" + parts[2]
		}
		return m
	})

	// "N% tip on M" → "0.0N*M"
	expr = rePercentTipOn.ReplaceAllStringFunc(expr, func(m string) string {
		parts := rePercentTipOn.FindStringSubmatch(m)
		if len(parts) == 3 {
			return percentToDecimal(parts[1]) + "*" + parts[2]
		}
		return m
	})

	// Word operators: "times" → "*", "divided by" → "/", "plus" → "+", "minus" → "-"
	expr = strings.ReplaceAll(expr, " divided by ", "/")
	expr = strings.ReplaceAll(expr, " times ", "*")
	expr = strings.ReplaceAll(expr, " plus ", "+")
	expr = strings.ReplaceAll(expr, " minus ", "-")
	expr = strings.ReplaceAll(expr, " multiplied by ", "*")
	expr = strings.ReplaceAll(expr, " over ", "/")

	// "N to the power of M" → "N^M"
	expr = rePowerOf.ReplaceAllString(expr, "$1^$2")

	// Clean up whitespace around operators
	expr = reSpacesAroundOps.ReplaceAllString(expr, "$1")

	entities["expression"] = strings.TrimSpace(expr)
}

// percentToDecimal converts "15" → "0.15", "5" → "0.05", "100" → "1.00"
func percentToDecimal(pct string) string {
	// Simple string-based conversion to avoid float formatting issues
	pct = strings.TrimSpace(pct)
	switch len(pct) {
	case 1:
		return "0.0" + pct
	case 2:
		return "0." + pct
	case 3:
		// "100" → "1.00"
		return pct[:1] + "." + pct[1:]
	default:
		return "0." + pct
	}
}

var (
	reSquareRoot     = regexp.MustCompile(`square root of\s+(\d+\.?\d*)`)
	reCubeRoot       = regexp.MustCompile(`cube root of\s+(\d+\.?\d*)`)
	rePercentOf      = regexp.MustCompile(`(\d+(?:\.\d+)?)%\s+of\s+(\d+\.?\d*)`)
	rePercentTipOn   = regexp.MustCompile(`(\d+(?:\.\d+)?)%\s+tip\s+on\s+(\d+\.?\d*)`)
	rePowerOf        = regexp.MustCompile(`(\d+\.?\d*)\s+to the power of\s+(\d+\.?\d*)`)
	reSpacesAroundOps = regexp.MustCompile(`\s*([+\-*/^])\s*`)
)

// -----------------------------------------------------------------------
// Translate
// -----------------------------------------------------------------------

// extractTranslate parses translation requests into text + target language.
//
//   "translate hello to japanese"          → text="hello", to="japanese"
//   "how do you say goodbye in french"     → text="goodbye", to="french"
//   "whats hello in spanish"               → text="hello", to="spanish"
func (se *SmartEntityExtractor) extractTranslate(input string, entities map[string]string) {
	if entities["text"] != "" && entities["to"] != "" {
		return // already extracted
	}

	lower := strings.ToLower(input)

	// Pattern: "translate X to/into Y"
	for _, sep := range []string{" to ", " into "} {
		idx := strings.LastIndex(lower, sep)
		if idx > 0 {
			textPart := input[:idx]
			langPart := strings.TrimSpace(input[idx+len(sep):])

			// Strip "translate" prefix from text
			textLower := strings.ToLower(textPart)
			for _, prefix := range []string{
				"translate ", "can you translate ", "please translate ",
			} {
				if strings.HasPrefix(textLower, prefix) {
					textPart = textPart[len(prefix):]
					break
				}
			}
			textPart = strings.TrimSpace(textPart)
			if textPart != "" && langPart != "" {
				entities["text"] = textPart
				entities["to"] = strings.TrimRight(langPart, "?!. ")
				return
			}
		}
	}

	// Pattern: "how do you say X in Y" / "whats X in Y"
	idx := strings.LastIndex(lower, " in ")
	if idx > 0 {
		langPart := strings.TrimSpace(input[idx+4:])
		textPart := input[:idx]

		textLower := strings.ToLower(textPart)
		for _, prefix := range []string{
			"how do you say ", "how to say ", "say ",
			"whats ", "what's ", "what is ",
		} {
			if strings.HasPrefix(textLower, prefix) {
				textPart = textPart[len(prefix):]
				break
			}
		}
		textPart = strings.TrimSpace(textPart)
		if textPart != "" && langPart != "" {
			entities["text"] = textPart
			entities["to"] = strings.TrimRight(langPart, "?!. ")
			return
		}
	}
}

// -----------------------------------------------------------------------
// Weather
// -----------------------------------------------------------------------

// extractWeather parses weather queries into location and optional date.
//
//   "whats the weather in paris"           → location="paris"
//   "is it going to rain tomorrow"         → location="", date="tomorrow"
func (se *SmartEntityExtractor) extractWeather(input string, entities map[string]string) {
	if entities["location"] != "" {
		return // already extracted
	}

	lower := strings.ToLower(input)

	// Pattern: "weather in/for/at LOCATION"
	for _, sep := range []string{" in ", " for ", " at "} {
		idx := strings.LastIndex(lower, sep)
		if idx > 0 {
			loc := strings.TrimSpace(input[idx+len(sep):])
			loc = strings.TrimRight(loc, "?!. ")
			if loc != "" {
				entities["location"] = loc
			}
		}
	}

	// Date extraction: tomorrow, today, etc.
	if entities["date"] == "" {
		for _, dateWord := range []string{"tomorrow", "today", "tonight", "this weekend"} {
			if strings.Contains(lower, dateWord) {
				entities["date"] = dateWord
				break
			}
		}
	}
}

// -----------------------------------------------------------------------
// Timer
// -----------------------------------------------------------------------

var reTimerDuration = regexp.MustCompile(`(\d+)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b`)

// extractTimer parses timer/reminder requests into a duration string.
//
//   "set a timer for 5 minutes"  → duration="5m"
//   "remind me in 10 min"        → duration="10m"
//   "timer 30 seconds"           → duration="30s"
func (se *SmartEntityExtractor) extractTimer(input string, entities map[string]string) {
	if entities["duration"] != "" {
		return // already extracted
	}

	lower := strings.ToLower(input)
	matches := reTimerDuration.FindStringSubmatch(lower)
	if len(matches) >= 3 {
		value := matches[1]
		unit := matches[2]

		var suffix string
		switch {
		case strings.HasPrefix(unit, "h"):
			suffix = "h"
		case strings.HasPrefix(unit, "m"):
			suffix = "m"
		case strings.HasPrefix(unit, "s"):
			suffix = "s"
		default:
			suffix = "m"
		}
		entities["duration"] = value + suffix
	}
}

// -----------------------------------------------------------------------
// Convert
// -----------------------------------------------------------------------

var reConvertPattern = regexp.MustCompile(`(?i)(\d+\.?\d*)\s+(\w+)\s+(?:to|in|into)\s+(\w+)`)
var reHowManyPattern = regexp.MustCompile(`(?i)how many\s+(\w+)\s+in\s+(?:a\s+)?(\w+)`)

// extractConvert parses unit conversion requests.
//
//   "convert 5 miles to km"       → value="5", from="miles", to_unit="km"
//   "how many feet in a mile"     → value="1", from="mile", to_unit="feet"
func (se *SmartEntityExtractor) extractConvert(input string, entities map[string]string) {
	if entities["value"] != "" && entities["from"] != "" && entities["to_unit"] != "" {
		return // already extracted
	}

	// Pattern: "N UNIT to UNIT"
	matches := reConvertPattern.FindStringSubmatch(input)
	if len(matches) >= 4 {
		entities["value"] = matches[1]
		entities["from"] = strings.ToLower(matches[2])
		entities["to_unit"] = strings.ToLower(matches[3])
		return
	}

	// Pattern: "how many UNIT in a UNIT"
	matches = reHowManyPattern.FindStringSubmatch(input)
	if len(matches) >= 3 {
		entities["value"] = "1"
		entities["from"] = strings.ToLower(matches[2])
		entities["to_unit"] = strings.ToLower(matches[1])
		return
	}
}
