package cognitive

import (
	"encoding/xml"
	"fmt"
	"io"
	"regexp"
	"strings"
	"time"
)

// -----------------------------------------------------------------------
// Wikipedia Dump Parser — streams bzip2'd XML dumps into KnowledgePackages.
//
// Converts MediaWiki markup to plain text, extracts triples using the
// existing ExtractTriples patterns, and batches articles into packages
// that can be loaded by the PackageLoader.
//
// Designed for Simple English Wikipedia but works with any MediaWiki dump.
// -----------------------------------------------------------------------

// WikiArticle is a single parsed Wikipedia article.
type WikiArticle struct {
	Title string
	Text  string // plain text after markup stripping
	ID    int
}

// xmlPage mirrors the <page> element in a MediaWiki XML dump.
type xmlPage struct {
	Title    string      `xml:"title"`
	NS       int         `xml:"ns"`
	ID       int         `xml:"id"`
	Revision xmlRevision `xml:"revision"`
}

// xmlRevision mirrors the <revision> element.
type xmlRevision struct {
	Text string `xml:"text"`
}

// ParseWikiDump streams articles from a MediaWiki XML dump reader.
// The reader should already be decompressed (e.g. passed through compress/bzip2).
// Calls handler for each article in namespace 0, skipping redirects.
func ParseWikiDump(reader io.Reader, handler func(WikiArticle) error) error {
	decoder := xml.NewDecoder(reader)

	for {
		tok, err := decoder.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("xml decode: %w", err)
		}

		se, ok := tok.(xml.StartElement)
		if !ok || se.Name.Local != "page" {
			continue
		}

		var page xmlPage
		if err := decoder.DecodeElement(&page, &se); err != nil {
			continue // skip malformed pages
		}

		// Only main namespace
		if page.NS != 0 {
			continue
		}

		// Skip redirects
		text := strings.TrimSpace(page.Revision.Text)
		if strings.HasPrefix(strings.ToUpper(text), "#REDIRECT") {
			continue
		}

		plainText := StripWikitext(text)
		plainText = strings.TrimSpace(plainText)
		if plainText == "" {
			continue
		}

		article := WikiArticle{
			Title: page.Title,
			Text:  plainText,
			ID:    page.ID,
		}

		if err := handler(article); err != nil {
			return fmt.Errorf("handler error on %q: %w", page.Title, err)
		}
	}

	return nil
}

// -----------------------------------------------------------------------
// Wikitext stripping — convert MediaWiki markup to clean plain text.
// -----------------------------------------------------------------------

// Precompiled regexps for wikitext stripping.
var (
	// Comments: <!-- ... -->
	wikiCommentRe = regexp.MustCompile(`<!--[\s\S]*?-->`)

	// <ref> tags: both paired <ref>...</ref> and self-closing <ref ... />
	wikiRefPairedRe = regexp.MustCompile(`(?i)<ref[^>]*>[\s\S]*?</ref>`)
	wikiRefSelfRe   = regexp.MustCompile(`(?i)<ref[^/]*/\s*>`)

	// <nowiki>content</nowiki> — remove tags, keep content
	wikiNowikiRe = regexp.MustCompile(`(?i)<nowiki>([\s\S]*?)</nowiki>`)

	// All remaining HTML tags — remove tags, keep content
	wikiHTMLTagRe = regexp.MustCompile(`<[^>]+>`)

	// Tables: {| ... |}
	wikiTableRe = regexp.MustCompile(`\{\|[\s\S]*?\|\}`)

	// Category links: [[Category:...]]
	wikiCategoryStripRe = regexp.MustCompile(`(?i)\[\[Category:[^\]]*\]\]`)

	// File/Image links: [[File:...]] or [[Image:...]]
	wikiFileRe = regexp.MustCompile(`(?i)\[\[(?:File|Image):[^\]]*\]\]`)

	// Section headers: == Title == → ". Title"
	wikiSectionRe = regexp.MustCompile(`(?m)^=+\s*(.+?)\s*=+\s*$`)

	// Bold and italic
	wikiBoldItalicRe = regexp.MustCompile(`'{2,5}`)

	// Bullet and numbered lists: * item, # item, : item, ; item
	wikiListRe = regexp.MustCompile(`(?m)^[*#:;]+\s*`)

	// External links: [http://... display text] → display text
	wikiExtLinkWithTextRe = regexp.MustCompile(`\[https?://\S+\s+([^\]]+)\]`)
	// External links without display text: [http://...] → remove
	wikiExtLinkBareRe = regexp.MustCompile(`\[https?://\S+\]`)

	// Bare URLs
	wikiBareURLRe = regexp.MustCompile(`https?://\S+`)

	// Multiple spaces / blank lines
	wikiMultiSpaceRe   = regexp.MustCompile(`[ \t]{2,}`)
	wikiMultiNewlineRe = regexp.MustCompile(`\n{3,}`)

	// Magic words / behavior switches
	wikiMagicWordRe = regexp.MustCompile(`(?i)__[A-Z]+__`)
)

// StripWikitext converts MediaWiki markup to plain text.
func StripWikitext(wikitext string) string {
	s := wikitext

	// 1. Comments
	s = wikiCommentRe.ReplaceAllString(s, "")

	// 2. <nowiki> — protect content from further processing using placeholders
	var nowikiBlocks []string
	s = wikiNowikiRe.ReplaceAllStringFunc(s, func(match string) string {
		sub := wikiNowikiRe.FindStringSubmatch(match)
		if len(sub) < 2 {
			return match
		}
		idx := len(nowikiBlocks)
		nowikiBlocks = append(nowikiBlocks, sub[1])
		return fmt.Sprintf("\x00NOWIKI%d\x00", idx)
	})

	// 3. <ref> tags (must be before general HTML removal)
	s = wikiRefPairedRe.ReplaceAllString(s, "")
	s = wikiRefSelfRe.ReplaceAllString(s, "")

	// 4. Templates: {{...}} — handle nesting by counting braces
	s = stripNestedBraces(s, '{', '}')

	// 5. Tables: {| ... |} (in case nested stripping missed them)
	s = wikiTableRe.ReplaceAllString(s, "")

	// 6. Category and File/Image links
	s = wikiCategoryStripRe.ReplaceAllString(s, "")
	s = wikiFileRe.ReplaceAllString(s, "")

	// 7. Wikilinks: [[target|display]] → display, [[target]] → target
	s = stripWikilinks(s)

	// 8. Section headers → sentence boundary + text
	s = wikiSectionRe.ReplaceAllString(s, ". $1. ")

	// 9. Bold/italic markup
	s = wikiBoldItalicRe.ReplaceAllString(s, "")

	// 10. List items → sentence-like text
	s = wikiListRe.ReplaceAllString(s, "")

	// 11. External links
	s = wikiExtLinkWithTextRe.ReplaceAllString(s, "$1")
	s = wikiExtLinkBareRe.ReplaceAllString(s, "")
	s = wikiBareURLRe.ReplaceAllString(s, "")

	// 12. HTML tags (keep content)
	s = wikiHTMLTagRe.ReplaceAllString(s, "")

	// 13. Magic words
	s = wikiMagicWordRe.ReplaceAllString(s, "")

	// 14. Restore <nowiki> content
	for i, block := range nowikiBlocks {
		s = strings.ReplaceAll(s, fmt.Sprintf("\x00NOWIKI%d\x00", i), block)
	}

	// 15. Cleanup
	s = wikiMultiSpaceRe.ReplaceAllString(s, " ")
	s = wikiMultiNewlineRe.ReplaceAllString(s, "\n\n")

	// Clean up artifacts: ". ." patterns, leading/trailing dots
	s = strings.ReplaceAll(s, ". .", ".")
	s = strings.ReplaceAll(s, "..", ".")

	lines := strings.Split(s, "\n")
	var cleaned []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			cleaned = append(cleaned, line)
		}
	}

	return strings.Join(cleaned, "\n")
}

// stripNestedBraces removes content between matching brace pairs,
// handling arbitrary nesting depth. Used for {{ templates }}.
func stripNestedBraces(s string, open, close byte) string {
	var result []byte
	depth := 0
	i := 0
	for i < len(s) {
		if i+1 < len(s) && s[i] == open && s[i+1] == open {
			depth++
			i += 2
			continue
		}
		if i+1 < len(s) && s[i] == close && s[i+1] == close {
			if depth > 0 {
				depth--
			}
			i += 2
			continue
		}
		if depth == 0 {
			result = append(result, s[i])
		}
		i++
	}
	return string(result)
}

// stripWikilinks handles [[target|display]] → display and [[target]] → target.
// Also handles nested links by tracking bracket depth.
func stripWikilinks(s string) string {
	var result strings.Builder
	result.Grow(len(s))

	i := 0
	for i < len(s) {
		if i+1 < len(s) && s[i] == '[' && s[i+1] == '[' {
			// Find the matching ]]
			j := i + 2
			depth := 1
			for j < len(s)-1 && depth > 0 {
				if s[j] == '[' && s[j+1] == '[' {
					depth++
					j += 2
					continue
				}
				if s[j] == ']' && s[j+1] == ']' {
					depth--
					if depth == 0 {
						break
					}
					j += 2
					continue
				}
				j++
			}

			if depth == 0 {
				// Extract content between [[ and ]]
				content := s[i+2 : j]
				// Use display text (after |) if present
				if pipeIdx := strings.LastIndex(content, "|"); pipeIdx >= 0 {
					result.WriteString(content[pipeIdx+1:])
				} else {
					result.WriteString(content)
				}
				i = j + 2
			} else {
				// Unmatched — keep as-is
				result.WriteByte(s[i])
				i++
			}
		} else {
			result.WriteByte(s[i])
			i++
		}
	}

	return result.String()
}

// -----------------------------------------------------------------------
// Fact extraction from articles
// -----------------------------------------------------------------------

// ArticleToFacts extracts PackageFacts from a plain-text article.
// Uses ExtractTriples patterns. Creates a "described_as" fact from the
// lead paragraph (first 2-3 sentences) — preserving the original
// human-written Wikipedia text for high-quality responses.
func ArticleToFacts(title, plainText string) []PackageFact {
	if plainText == "" {
		return nil
	}

	var facts []PackageFact
	seen := make(map[string]bool)

	addFact := func(s, r, o string) {
		key := s + "|" + r + "|" + o
		if !seen[key] {
			seen[key] = true
			facts = append(facts, PackageFact{Subject: s, Relation: r, Object: o})
		}
	}

	sentences := splitSentences(plainText)

	// Lead paragraph: preserve the first 2-3 sentences as a described_as fact.
	// These are human-written, encyclopedia-quality text that the composer
	// can surface directly — no lossy triple round-trip needed.
	lead := buildLeadParagraph(sentences, title)
	if lead != "" {
		addFact(title, "described_as", lead)
	}

	// Extract triples from all sentences (including the lead ones —
	// triples provide structured facts for the knowledge graph).
	for _, sent := range sentences {
		sent = strings.TrimSpace(sent)
		if sent == "" || len(sent) < 10 {
			continue
		}
		if isBoilerplate(sent) {
			continue
		}

		triples := ExtractTriples(sent)
		for _, t := range triples {
			rel := relTypeToString(t.Relation)
			addFact(t.Subject, rel, t.Object)
		}
	}

	return facts
}

// buildLeadParagraph assembles the first 2-3 sentences of a Wikipedia
// article into a single lead paragraph. Skips sentences that are
// boilerplate, fragment-like, or don't relate to the article title.
// Returns "" if no usable lead text is found.
func buildLeadParagraph(sentences []string, title string) string {
	const maxSentences = 3
	const maxLen = 500 // keep lead compact
	const scanLimit = 15 // scan deeper for articles with image captions

	titleLower := strings.ToLower(title)
	var parts []string
	totalLen := 0

	for i, sent := range sentences {
		if i >= scanLimit {
			break
		}
		sent = strings.TrimSpace(sent)
		// Clean wiki markup remnants (image captions, stray brackets).
		sent = strings.ReplaceAll(sent, "]]", "")
		sent = strings.ReplaceAll(sent, "[[", "")
		// If a sentence contains a newline, take only the part after the
		// last newline — the earlier parts are usually image captions.
		if nlIdx := strings.LastIndex(sent, "\n"); nlIdx >= 0 {
			sent = strings.TrimSpace(sent[nlIdx+1:])
		}
		if sent == "" || len(sent) < 15 {
			continue
		}
		if isBoilerplate(sent) {
			continue
		}

		// First sentence must start with or prominently feature the title.
		// Wikipedia convention: title appears early in the first sentence.
		if len(parts) == 0 {
			if !leadSentenceMatchesTitle(sent, titleLower) {
				continue
			}
			// Skip image captions that happen to contain the title.
			// "This Renaissance painting shows..." is about a painting, not the Renaissance.
			sentLower := strings.ToLower(sent)
			if strings.HasPrefix(sentLower, "this ") {
				// Only allow if the sentence IS "This <title> is/was/are..."
				// (defining the topic), not "This <title> painting/photo/image shows..."
				captionWords := []string{"painting", "photo", "image", "picture",
					"map", "diagram", "chart", "statue", "drawing", "flag", "logo",
					"portrait", "view", "scene", "stamp", "coin", "monument"}
				isCaption := false
				for _, cw := range captionWords {
					if strings.Contains(sentLower, cw) {
						isCaption = true
						break
					}
				}
				if isCaption {
					continue
				}
			}
		}

		// Continuation sentences: must start with a capital letter or pronoun.
		// Skip fragments like "is the Ancient Greek word" (starts lowercase)
		// and image captions like "This painting shows..." that aren't about the topic.
		if len(parts) > 0 {
			if len(sent) > 0 && (sent[0] < 'A' || sent[0] > 'Z') {
				continue // lowercase start — likely a fragment
			}
			// Skip obvious image captions
			sentLower := strings.ToLower(sent)
			if strings.HasPrefix(sentLower, "this ") && !strings.Contains(sentLower, titleLower) {
				continue
			}
		}

		// Ensure sentence ends with punctuation.
		if !strings.HasSuffix(sent, ".") && !strings.HasSuffix(sent, "!") && !strings.HasSuffix(sent, "?") {
			sent += "."
		}

		parts = append(parts, sent)
		totalLen += len(sent)

		if len(parts) >= maxSentences || totalLen >= maxLen {
			break
		}
	}

	// Fallback: if no sentence matched the title, use the first good sentence.
	// This handles articles where the text starts with a variant of the title
	// (e.g. "Einstein" instead of "Albert Einstein") or image captions precede it.
	if len(parts) == 0 {
		for i, sent := range sentences {
			if i >= scanLimit {
				break
			}
			sent = strings.TrimSpace(sent)
			sent = strings.ReplaceAll(sent, "]]", "")
			sent = strings.ReplaceAll(sent, "[[", "")
			if nlIdx := strings.LastIndex(sent, "\n"); nlIdx >= 0 {
				sent = strings.TrimSpace(sent[nlIdx+1:])
			}
			if sent == "" || len(sent) < 20 {
				continue
			}
			if isBoilerplate(sent) {
				continue
			}
			// Must start with a capital letter — ensures it's a proper sentence,
			// not a fragment like "is the Ancient Greek word" or "from the atmosphere".
			if len(sent) > 0 && (sent[0] < 'A' || sent[0] > 'Z') {
				continue
			}
			if !strings.HasSuffix(sent, ".") && !strings.HasSuffix(sent, "!") && !strings.HasSuffix(sent, "?") {
				sent += "."
			}
			return sent
		}
		return ""
	}
	result := strings.Join(parts, " ")
	// Fix missing spaces after periods: "food.Photosynthesis" → "food. Photosynthesis"
	var fixed strings.Builder
	fixed.Grow(len(result) + 10)
	for i := 0; i < len(result); i++ {
		fixed.WriteByte(result[i])
		if result[i] == '.' && i+1 < len(result) && result[i+1] >= 'A' && result[i+1] <= 'Z' {
			fixed.WriteByte(' ')
		}
	}
	return fixed.String()
}

// leadSentenceMatchesTitle checks whether a sentence is a valid Wikipedia
// lead sentence for the given title. The title (or last word of multi-word
// titles) must appear near the start of the sentence.
func leadSentenceMatchesTitle(sent, titleLower string) bool {
	sentLower := strings.ToLower(sent)

	// Direct prefix match is always good.
	if strings.HasPrefix(sentLower, titleLower) {
		return true
	}

	// For multi-word titles, check if any significant word appears at the start.
	// "Mahatma Gandhi" article starts with "Mohandas Karmchand Gandhi".
	titleWords := strings.Fields(titleLower)
	if len(titleWords) > 1 {
		// Check last word (surname) — most distinctive.
		lastName := titleWords[len(titleWords)-1]
		if len(lastName) >= 4 && wordBoundaryContains(sentLower, lastName) {
			// Must be in the first half of the sentence.
			idx := strings.Index(sentLower, lastName)
			if idx >= 0 && idx < len(sentLower)/2 {
				return true
			}
		}
		// Check first word prefix.
		if strings.HasPrefix(sentLower, titleWords[0]) {
			return true
		}
	}

	// Full title must appear near the start — within first 50% of the sentence.
	idx := strings.Index(sentLower, titleLower)
	if idx < 0 {
		return false
	}
	maxPos := len(sentLower) / 2
	if maxPos > 80 {
		maxPos = 80
	}
	if idx > maxPos {
		return false
	}

	// Word-boundary check to avoid substring matches.
	endIdx := idx + len(titleLower)
	atWordStart := idx == 0 || sentLower[idx-1] == ' ' || sentLower[idx-1] == '('
	atWordEnd := endIdx == len(sentLower) || sentLower[endIdx] == ' ' || sentLower[endIdx] == ',' || sentLower[endIdx] == '.' || sentLower[endIdx] == ')'
	return atWordStart && atWordEnd
}

// wordBoundaryContains checks if text contains word at a word boundary.
func wordBoundaryContains(text, word string) bool {
	idx := strings.Index(text, word)
	if idx < 0 {
		return false
	}
	endIdx := idx + len(word)
	atStart := idx == 0 || text[idx-1] == ' ' || text[idx-1] == '('
	atEnd := endIdx == len(text) || text[endIdx] == ' ' || text[endIdx] == ',' || text[endIdx] == '.' || text[endIdx] == ')'
	return atStart && atEnd
}

// relTypeToString converts a RelType back to its string representation.
func relTypeToString(r RelType) string {
	switch r {
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
	case RelSimilarTo:
		return "similar_to"
	case RelCauses:
		return "causes"
	case RelContradicts:
		return "contradicts"
	case RelFollows:
		return "follows"
	case RelPrefers:
		return "prefers"
	case RelDislikes:
		return "dislikes"
	case RelDomain:
		return "domain"
	case RelDescribedAs:
		return "described_as"
	default:
		return "related_to"
	}
}

// BatchToPackage creates a KnowledgePackage from a batch of extracted facts.
func BatchToPackage(batchNum int, domain string, facts []PackageFact) *KnowledgePackage {
	return &KnowledgePackage{
		Name:        fmt.Sprintf("wiki-%s-batch-%04d", domain, batchNum),
		Version:     "1.0.0",
		Description: fmt.Sprintf("Wikipedia knowledge batch %d — auto-extracted from %s dump", batchNum, domain),
		Author:      "wikiimport",
		Domain:      domain,
		Facts:       facts,
	}
}

// PackageToJSON serializes a KnowledgePackage to pretty-printed JSON.
func PackageToJSON(pkg *KnowledgePackage) ([]byte, error) {
	return CreatePackage(pkg.Name, pkg.Version, pkg.Description, pkg.Domain, pkg.Facts, pkg.Vocabulary)
}

// WikiImportStats tracks progress during a wiki import.
type WikiImportStats struct {
	ArticlesProcessed int
	ArticlesSkipped   int
	FactsExtracted    int
	PackagesWritten   int
	StartTime         time.Time
}

func (s *WikiImportStats) String() string {
	elapsed := time.Since(s.StartTime)
	rate := float64(s.ArticlesProcessed) / elapsed.Seconds()
	return fmt.Sprintf("articles=%d skipped=%d facts=%d packages=%d elapsed=%s rate=%.0f/s",
		s.ArticlesProcessed, s.ArticlesSkipped, s.FactsExtracted,
		s.PackagesWritten, elapsed.Round(time.Millisecond), rate)
}
