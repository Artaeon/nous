package cognitive

import (
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"unicode"
)

// TemplateMiner extracts sentence patterns from Wikipedia descriptions
// and generalizes them into reusable templates for text generation.
type TemplateMiner struct {
	templates map[RelType]map[string]int // rel → template → frequency
}

func NewTemplateMiner() *TemplateMiner {
	return &TemplateMiner{
		templates: make(map[RelType]map[string]int),
	}
}

// MinedTemplate represents a single mined template with its frequency.
type MinedTemplate struct {
	Pattern string `json:"pattern"`
	Freq    int    `json:"freq"`
}

// MinedTemplateFile is the JSON output format.
type MinedTemplateFile struct {
	Version   string                      `json:"version"`
	Source    string                       `json:"source"`
	Templates map[string][]MinedTemplate  `json:"templates"`
}

// verbPattern maps a regex to the relation type it signals and the verb phrase to use in templates.
type verbPattern struct {
	re       *regexp.Regexp
	rel      RelType
	template string // the generalized verb phrase, e.g. "is a", "was founded by"
}

// Ordered by specificity — more specific patterns first.
var verbPatterns = []verbPattern{
	// Founded/created by
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+founded\s+by\s+(.+)`), RelFoundedBy, "was founded by"},
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+started\s+by\s+(.+)`), RelFoundedBy, "was started by"},
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+established\s+by\s+(.+)`), RelFoundedBy, "was established by"},
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+created\s+by\s+(.+)`), RelCreatedBy, "was created by"},
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+built\s+by\s+(.+)`), RelCreatedBy, "was built by"},
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+developed\s+by\s+(.+)`), RelCreatedBy, "was developed by"},
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+designed\s+by\s+(.+)`), RelCreatedBy, "was designed by"},
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+invented\s+by\s+(.+)`), RelCreatedBy, "was invented by"},
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+written\s+by\s+(.+)`), RelCreatedBy, "was written by"},
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+composed\s+by\s+(.+)`), RelCreatedBy, "was composed by"},
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+made\s+by\s+(.+)`), RelCreatedBy, "was made by"},
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+directed\s+by\s+(.+)`), RelCreatedBy, "was directed by"},
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+produced\s+by\s+(.+)`), RelCreatedBy, "was produced by"},

	// Founded in (year/date)
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+founded\s+in\s+(.+)`), RelFoundedIn, "was founded in"},
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+established\s+in\s+(.+)`), RelFoundedIn, "was established in"},
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+started\s+in\s+(.+)`), RelFoundedIn, "was started in"},
	{regexp.MustCompile(`(?i)^(.+?)\s+was\s+formed\s+in\s+(.+)`), RelFoundedIn, "was formed in"},

	// Located in
	{regexp.MustCompile(`(?i)^(.+?)\s+is\s+(?:a\s+\w+\s+)?(?:located|situated)\s+in\s+(.+)`), RelLocatedIn, "is located in"},
	{regexp.MustCompile(`(?i)^(.+?)\s+(?:lies|sits|stands)\s+in\s+(.+)`), RelLocatedIn, "lies in"},

	// Part of
	{regexp.MustCompile(`(?i)^(.+?)\s+is\s+(?:a\s+)?part\s+of\s+(.+)`), RelPartOf, "is part of"},
	{regexp.MustCompile(`(?i)^(.+?)\s+belongs?\s+to\s+(.+)`), RelPartOf, "belongs to"},

	// Used for
	{regexp.MustCompile(`(?i)^(.+?)\s+is\s+used\s+(?:for|in|to)\s+(.+)`), RelUsedFor, "is used for"},
	{regexp.MustCompile(`(?i)^(.+?)\s+is\s+designed\s+(?:for|to)\s+(.+)`), RelUsedFor, "is designed for"},

	// Has
	{regexp.MustCompile(`(?i)^(.+?)\s+(?:has|have)\s+(.+)`), RelHas, "has"},

	// Causes
	{regexp.MustCompile(`(?i)^(.+?)\s+causes?\s+(.+)`), RelCauses, "causes"},
}

// isAPattern matches "X is/are/was/were a/an/the Y" — the most common Wikipedia pattern.
var isAPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^(.+?)\s+is\s+an?\s+(.+)`),
	regexp.MustCompile(`(?i)^(.+?)\s+is\s+the\s+(.+)`),
	regexp.MustCompile(`(?i)^(.+?)\s+is\s+one\s+of\s+the\s+(.+)`),
	regexp.MustCompile(`(?i)^(.+?)\s+was\s+an?\s+(.+)`),
	regexp.MustCompile(`(?i)^(.+?)\s+was\s+the\s+(.+)`),
	regexp.MustCompile(`(?i)^(.+?)\s+are\s+(.+)`),
	regexp.MustCompile(`(?i)^(.+?)\s+were\s+(.+)`),
	regexp.MustCompile(`(?i)^(.+?)\s+is\s+(.+)`), // bare "is" last — least specific
}

// Regex to strip parenthetical asides from subjects: "Name (born 1990)" → "Name"
var parentheticalRe = regexp.MustCompile(`\s*\([^)]*\)\s*`)

// ProcessSentence extracts a template from a Wikipedia first sentence.
// Returns the relation type, the generalized template, and whether extraction succeeded.
func (tm *TemplateMiner) ProcessSentence(sentence, subject string) (RelType, string, bool) {
	// Clean up
	sentence = strings.TrimSpace(sentence)
	if len(sentence) < 15 || len(sentence) > 500 {
		return "", "", false
	}

	// Skip sentences that don't start with the subject
	if !subjectStartsSentence(sentence, subject) {
		return "", "", false
	}

	// Skip sentences with wiki markup
	if strings.Contains(sentence, "]]") || strings.Contains(sentence, "[[") ||
		strings.Contains(sentence, "{{") || strings.Contains(sentence, "}}") {
		return "", "", false
	}

	// Take only the first sentence (up to first period followed by space or end)
	sentence = firstSentence(sentence)
	if len(sentence) < 15 {
		return "", "", false
	}

	// Try specific verb patterns first (founded by, created by, etc.)
	for _, vp := range verbPatterns {
		if m := vp.re.FindStringSubmatch(sentence); m != nil {
			tmpl := "%s " + vp.template + " %s."
			if validateTemplate(tmpl) {
				return vp.rel, tmpl, true
			}
		}
	}

	// Try is_a patterns — extract the exact verb phrase used
	for _, pat := range isAPatterns {
		if m := pat.FindStringSubmatch(sentence); m != nil {
			// Extract the verb phrase between subject and object
			verbPhrase := extractVerbPhrase(sentence, m[1], m[2])
			if verbPhrase == "" {
				continue
			}
			tmpl := "%s " + verbPhrase + " %s."
			if validateTemplate(tmpl) {
				return RelIsA, tmpl, true
			}
		}
	}

	return "", "", false
}

// extractVerbPhrase pulls out the connector between subject and object.
// E.g., "Go is a programming language" → "is a"
// E.g., "The Beatles were an English rock band" → "were an"
func extractVerbPhrase(sentence, subjectMatch, objectMatch string) string {
	subEnd := strings.Index(sentence, subjectMatch) + len(subjectMatch)
	objStart := strings.LastIndex(sentence, objectMatch)
	if subEnd >= objStart || subEnd < 0 || objStart < 0 {
		return ""
	}
	vp := strings.TrimSpace(sentence[subEnd:objStart])
	// Clean up: should be a short verb phrase
	if len(vp) > 40 || len(vp) < 2 {
		return ""
	}
	// Must start with a verb
	vpLower := strings.ToLower(vp)
	validStarts := []string{"is ", "are ", "was ", "were ", "is a", "is an", "is the",
		"was a", "was an", "was the", "are a", "are the", "were a", "were the",
		"is one", "was one"}
	valid := false
	for _, vs := range validStarts {
		if strings.HasPrefix(vpLower, vs) || vpLower == strings.TrimSpace(vs) {
			valid = true
			break
		}
	}
	if !valid {
		return ""
	}
	return vp
}

// AddTemplate records a mined template.
func (tm *TemplateMiner) AddTemplate(rel RelType, tmpl string) {
	if tm.templates[rel] == nil {
		tm.templates[rel] = make(map[string]int)
	}
	tm.templates[rel][tmpl]++
}

// Export returns the templates filtered by minimum frequency.
func (tm *TemplateMiner) Export(minFreq int) *MinedTemplateFile {
	result := &MinedTemplateFile{
		Version:   "1.0",
		Source:    "simplewiki",
		Templates: make(map[string][]MinedTemplate),
	}

	for rel, tmpls := range tm.templates {
		relStr := relToString(rel)
		var filtered []MinedTemplate
		for tmpl, freq := range tmpls {
			if freq >= minFreq {
				filtered = append(filtered, MinedTemplate{Pattern: tmpl, Freq: freq})
			}
		}
		if len(filtered) > 0 {
			result.Templates[relStr] = filtered
		}
	}

	return result
}

// subjectStartsSentence checks if the sentence begins with the subject.
func subjectStartsSentence(sentence, subject string) bool {
	sentLower := strings.ToLower(sentence)
	subjLower := strings.ToLower(strings.TrimSpace(subject))
	if len(subjLower) < 2 {
		return false
	}
	// Reject generic subjects
	switch subjLower {
	case "it", "the", "they", "this", "that", "he", "she", "we":
		return false
	}
	return strings.HasPrefix(sentLower, subjLower)
}

// firstSentence extracts the first sentence from text.
func firstSentence(text string) string {
	// Look for ". " followed by an uppercase letter (new sentence)
	for i := 0; i < len(text)-2; i++ {
		if text[i] == '.' && text[i+1] == ' ' && i+2 < len(text) && unicode.IsUpper(rune(text[i+2])) {
			return text[:i+1]
		}
	}
	// No clear sentence boundary — use the whole text if it ends with a period
	if strings.HasSuffix(strings.TrimSpace(text), ".") {
		return strings.TrimSpace(text)
	}
	// Add period
	return strings.TrimSpace(text) + "."
}

// validateTemplate checks if a template is usable.
func validateTemplate(tmpl string) bool {
	// Must have exactly 2 %s placeholders
	if strings.Count(tmpl, "%s") != 2 {
		return false
	}
	// Must start with %s (subject slot)
	if !strings.HasPrefix(tmpl, "%s ") {
		return false
	}
	// Must end with %s. (object slot + period)
	if !strings.HasSuffix(tmpl, "%s.") {
		return false
	}
	// Reasonable length
	if len(tmpl) < 10 || len(tmpl) > 80 {
		return false
	}
	// No leaked proper nouns (capitalized words that aren't sentence-initial)
	words := strings.Fields(tmpl)
	for i, w := range words {
		if i == 0 {
			continue
		}
		if w == "%s" || w == "%s." {
			continue
		}
		if len(w) > 0 && unicode.IsUpper(rune(w[0])) {
			return false // leaked proper noun
		}
	}
	return true
}

func relToString(rel RelType) string {
	switch rel {
	case RelIsA:
		return "is_a"
	case RelLocatedIn:
		return "located_in"
	case RelPartOf:
		return "part_of"
	case RelCreatedBy:
		return "created_by"
	case RelFoundedBy:
		return "founded_by"
	case RelFoundedIn:
		return "founded_in"
	case RelHas:
		return "has"
	case RelOffers:
		return "offers"
	case RelUsedFor:
		return "used_for"
	case RelRelatedTo:
		return "related_to"
	case RelCauses:
		return "causes"
	case RelDescribedAs:
		return "described_as"
	case RelDomain:
		return "domain"
	default:
		return string(rel)
	}
}

// LoadMinedTemplates reads a mined template JSON file and appends the
// templates to the global relationTemplates map.
func LoadMinedTemplates(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var file MinedTemplateFile
	if err := json.Unmarshal(data, &file); err != nil {
		return err
	}

	for relStr, tmpls := range file.Templates {
		rel := parseRelString(relStr)
		if rel == "" {
			continue
		}
		existing := make(map[string]bool)
		for _, t := range relationTemplates[rel] {
			existing[t] = true
		}
		for _, mt := range tmpls {
			pattern := mt.Pattern
			// Skip is_a templates that assume plural/past tense — these require
			// context (historical vs current, singular vs plural) that the template
			// system can't determine at selection time.
			if rel == RelIsA {
				lower := strings.ToLower(pattern)
				if strings.Contains(lower, " are ") || strings.Contains(lower, " were ") {
					continue // plural-only templates
				}
				if strings.Contains(lower, " was ") {
					continue // past-tense-only templates
				}
			}
			if !existing[pattern] {
				relationTemplates[rel] = append(relationTemplates[rel], pattern)
			}
		}
	}

	return nil
}
