package tools

import (
	"fmt"
	"net/url"
	"regexp"
	"strings"
)

// Link represents an extracted hyperlink from HTML.
type Link struct {
	URL      string
	Text     string
	Internal bool
}

// --- HTML tag/attribute regexes (compiled once) ---

var (
	reScript    = regexp.MustCompile(`(?is)<script[^>]*>.*?</script>`)
	reStyle     = regexp.MustCompile(`(?is)<style[^>]*>.*?</style>`)
	reNav       = regexp.MustCompile(`(?is)<nav[^>]*>.*?</nav>`)
	reFooter    = regexp.MustCompile(`(?is)<footer[^>]*>.*?</footer>`)
	reHeader    = regexp.MustCompile(`(?is)<header[^>]*>.*?</header>`)
	reSVG       = regexp.MustCompile(`(?is)<svg[^>]*>.*?</svg>`)
	reComment   = regexp.MustCompile(`(?s)<!--.*?-->`)
	reAnchor    = regexp.MustCompile(`(?is)<a\s[^>]*href\s*=\s*["']([^"']*?)["'][^>]*>(.*?)</a>`)
	reHrefAttr  = regexp.MustCompile(`(?i)href\s*=\s*["']([^"']*?)["']`)
	reTagOpen   = regexp.MustCompile(`<([a-zA-Z][a-zA-Z0-9]*)[^>]*>`)
	reTagClose  = regexp.MustCompile(`</[a-zA-Z][a-zA-Z0-9]*>`)
	reTagAny    = regexp.MustCompile(`<[^>]+>`)
	reTable     = regexp.MustCompile(`(?is)<table[^>]*>(.*?)</table>`)
	reTR        = regexp.MustCompile(`(?is)<tr[^>]*>(.*?)</tr>`)
	reTDTH      = regexp.MustCompile(`(?is)<(?:td|th)[^>]*>(.*?)</(?:td|th)>`)
	reOL        = regexp.MustCompile(`(?is)<ol[^>]*>(.*?)</ol>`)
	reUL        = regexp.MustCompile(`(?is)<ul[^>]*>(.*?)</ul>`)
	reLI        = regexp.MustCompile(`(?is)<li[^>]*>(.*?)</li>`)
	reMeta      = regexp.MustCompile(`(?is)<meta\s[^>]*>`)
	reMetaAttrs = regexp.MustCompile(`(?i)(name|property|content)\s*=\s*["']([^"']*?)["']`)
	reID        = regexp.MustCompile(`(?i)\bid\s*=\s*["']([^"']*?)["']`)
	reClass     = regexp.MustCompile(`(?i)\bclass\s*=\s*["']([^"']*?)["']`)
	reBlankLine = regexp.MustCompile(`\n{3,}`)
	reSpaces    = regexp.MustCompile(`[ \t]{2,}`)
	reTitle     = regexp.MustCompile(`(?is)<title[^>]*>(.*?)</title>`)

	reEntityAmp  = strings.NewReplacer("&amp;", "&", "&lt;", "<", "&gt;", ">", "&quot;", "\"", "&#39;", "'", "&apos;", "'", "&nbsp;", " ")
	reEntityNum  = regexp.MustCompile(`&#(\d+);`)
	reEntityHex  = regexp.MustCompile(`(?i)&#x([0-9a-f]+);`)

	// Regexes used in HTMLToMarkdown (compiled once at package level)
	reHead       = regexp.MustCompile(`(?is)<head[^>]*>.*?</head>`)
	reH          [6]*regexp.Regexp
	reStrong     = regexp.MustCompile(`(?is)<strong[^>]*>(.*?)</strong>`)
	reBold       = regexp.MustCompile(`(?is)<b\b[^>]*>(.*?)</b>`)
	reEmTag      = regexp.MustCompile(`(?is)<em\b[^>]*>(.*?)</em>`)
	reITag       = regexp.MustCompile(`(?is)<i\b[^>]*>(.*?)</i>`)
	reCode       = regexp.MustCompile(`(?is)<code[^>]*>(.*?)</code>`)
	rePre        = regexp.MustCompile(`(?is)<pre[^>]*>(.*?)</pre>`)
	reBQ         = regexp.MustCompile(`(?is)<blockquote[^>]*>(.*?)</blockquote>`)
	reBR         = regexp.MustCompile(`(?i)<br\s*/?\s*>`)
	reHR         = regexp.MustCompile(`(?i)<hr\s*/?\s*>`)
	rePOpen      = regexp.MustCompile(`(?i)<p[^>]*>`)
	rePClose     = regexp.MustCompile(`(?i)</p>`)
	reDiv        = regexp.MustCompile(`(?i)</?div[^>]*>`)
)

func init() {
	for i := 0; i < 6; i++ {
		tag := fmt.Sprintf("h%d", i+1)
		reH[i] = regexp.MustCompile(fmt.Sprintf(`(?is)<%s[^>]*>(.*?)</%s>`, tag, tag))
	}
}

// decodeEntities decodes common HTML entities.
func decodeEntities(s string) string {
	s = reEntityAmp.Replace(s)
	s = reEntityNum.ReplaceAllStringFunc(s, func(m string) string {
		sub := reEntityNum.FindStringSubmatch(m)
		if len(sub) == 2 {
			var n int
			fmt.Sscanf(sub[1], "%d", &n)
			if n > 0 && n < 0x110000 {
				return string(rune(n))
			}
		}
		return m
	})
	s = reEntityHex.ReplaceAllStringFunc(s, func(m string) string {
		sub := reEntityHex.FindStringSubmatch(m)
		if len(sub) == 2 {
			var n int
			fmt.Sscanf(sub[1], "%x", &n)
			if n > 0 && n < 0x110000 {
				return string(rune(n))
			}
		}
		return m
	})
	return s
}

// StripTags removes all HTML tags, returning plain text.
func StripTags(html string) string {
	text := reScript.ReplaceAllString(html, "")
	text = reStyle.ReplaceAllString(text, "")
	text = reComment.ReplaceAllString(text, "")
	text = reSVG.ReplaceAllString(text, "")
	text = reTagAny.ReplaceAllString(text, " ")
	text = decodeEntities(text)
	text = reSpaces.ReplaceAllString(text, " ")
	text = reBlankLine.ReplaceAllString(text, "\n\n")
	return strings.TrimSpace(text)
}

// HTMLToMarkdown converts HTML to readable markdown for LLM consumption.
func HTMLToMarkdown(html string) string {
	// Remove noise
	text := reScript.ReplaceAllString(html, "")
	text = reStyle.ReplaceAllString(text, "")
	text = reComment.ReplaceAllString(text, "")
	text = reSVG.ReplaceAllString(text, "")
	text = reNav.ReplaceAllString(text, "")
	text = reFooter.ReplaceAllString(text, "")
	text = reHeader.ReplaceAllString(text, "")

	// Strip <head> entirely (title, meta, etc are not content)
	text = reHead.ReplaceAllString(text, "")

	// Convert headings
	for i := 5; i >= 0; i-- {
		prefix := strings.Repeat("#", i+1) + " "
		re := reH[i]
		text = re.ReplaceAllStringFunc(text, func(m string) string {
			inner := re.FindStringSubmatch(m)
			if len(inner) < 2 {
				return m
			}
			content := StripTags(inner[1])
			content = strings.TrimSpace(content)
			if content == "" {
				return ""
			}
			return "\n\n" + prefix + content + "\n\n"
		})
	}

	// Convert links: <a href="url">text</a> -> [text](url)
	text = reAnchor.ReplaceAllStringFunc(text, func(m string) string {
		sub := reAnchor.FindStringSubmatch(m)
		if len(sub) < 3 {
			return m
		}
		href := strings.TrimSpace(sub[1])
		linkText := strings.TrimSpace(StripTags(sub[2]))
		if linkText == "" {
			linkText = href
		}
		return fmt.Sprintf("[%s](%s)", linkText, href)
	})

	// Convert bold/strong (match exact tags, not <body> etc.)
	text = reStrong.ReplaceAllString(text, "**$1**")
	text = reBold.ReplaceAllString(text, "**$1**")

	// Convert italic/em
	text = reEmTag.ReplaceAllString(text, "*$1*")
	text = reITag.ReplaceAllString(text, "*$1*")

	// Convert code
	text = reCode.ReplaceAllString(text, "`$1`")

	// Convert pre blocks
	text = rePre.ReplaceAllStringFunc(text, func(m string) string {
		inner := rePre.FindStringSubmatch(m)
		if len(inner) < 2 {
			return m
		}
		content := StripTags(inner[1])
		return "\n```\n" + strings.TrimSpace(content) + "\n```\n"
	})

	// Convert blockquote
	text = reBQ.ReplaceAllStringFunc(text, func(m string) string {
		inner := reBQ.FindStringSubmatch(m)
		if len(inner) < 2 {
			return m
		}
		content := StripTags(inner[1])
		lines := strings.Split(strings.TrimSpace(content), "\n")
		for i, l := range lines {
			lines[i] = "> " + strings.TrimSpace(l)
		}
		return "\n" + strings.Join(lines, "\n") + "\n"
	})

	// Convert unordered lists
	text = reUL.ReplaceAllStringFunc(text, func(m string) string {
		inner := reUL.FindStringSubmatch(m)
		if len(inner) < 2 {
			return m
		}
		items := reLI.FindAllStringSubmatch(inner[1], -1)
		var out strings.Builder
		out.WriteString("\n")
		for _, item := range items {
			if len(item) < 2 {
				continue
			}
			content := strings.TrimSpace(StripTags(item[1]))
			if content != "" {
				out.WriteString("- " + content + "\n")
			}
		}
		return out.String()
	})

	// Convert ordered lists
	text = reOL.ReplaceAllStringFunc(text, func(m string) string {
		inner := reOL.FindStringSubmatch(m)
		if len(inner) < 2 {
			return m
		}
		items := reLI.FindAllStringSubmatch(inner[1], -1)
		var out strings.Builder
		out.WriteString("\n")
		for i, item := range items {
			if len(item) < 2 {
				continue
			}
			content := strings.TrimSpace(StripTags(item[1]))
			if content != "" {
				out.WriteString(fmt.Sprintf("%d. %s\n", i+1, content))
			}
		}
		return out.String()
	})

	// Convert <br> and <hr>
	text = reBR.ReplaceAllString(text, "\n")
	text = reHR.ReplaceAllString(text, "\n---\n")

	// Convert <p> tags to double newlines
	text = rePOpen.ReplaceAllString(text, "\n\n")
	text = rePClose.ReplaceAllString(text, "\n\n")

	// Convert <div> to newlines
	text = reDiv.ReplaceAllString(text, "\n")

	// Strip remaining tags
	text = reTagAny.ReplaceAllString(text, "")

	// Decode entities
	text = decodeEntities(text)

	// Clean up whitespace
	text = reSpaces.ReplaceAllString(text, " ")
	text = reBlankLine.ReplaceAllString(text, "\n\n")

	// Trim each line
	lines := strings.Split(text, "\n")
	for i, l := range lines {
		lines[i] = strings.TrimRight(l, " \t")
	}
	text = strings.Join(lines, "\n")

	return strings.TrimSpace(text)
}

// ExtractBySelector extracts content matching a basic CSS selector.
// Supports: tag, .class, #id, tag.class, tag#id
func ExtractBySelector(html, selector string) string {
	selector = strings.TrimSpace(selector)
	if selector == "" {
		return html
	}

	tag := ""
	id := ""
	class := ""

	// Parse selector
	if strings.HasPrefix(selector, "#") {
		id = selector[1:]
	} else if strings.HasPrefix(selector, ".") {
		class = selector[1:]
	} else if strings.Contains(selector, "#") {
		parts := strings.SplitN(selector, "#", 2)
		tag = parts[0]
		id = parts[1]
	} else if strings.Contains(selector, ".") {
		parts := strings.SplitN(selector, ".", 2)
		tag = parts[0]
		class = parts[1]
	} else {
		tag = selector
	}

	// Build a regex to find the opening tag
	var pattern string
	if id != "" && tag != "" {
		pattern = fmt.Sprintf(`(?is)<%s\s[^>]*\bid\s*=\s*["']%s["'][^>]*>`, regexp.QuoteMeta(tag), regexp.QuoteMeta(id))
	} else if id != "" {
		pattern = fmt.Sprintf(`(?is)<([a-zA-Z][a-zA-Z0-9]*)\s[^>]*\bid\s*=\s*["']%s["'][^>]*>`, regexp.QuoteMeta(id))
	} else if class != "" && tag != "" {
		pattern = fmt.Sprintf(`(?is)<%s\s[^>]*\bclass\s*=\s*["'][^"']*\b%s\b[^"']*["'][^>]*>`, regexp.QuoteMeta(tag), regexp.QuoteMeta(class))
	} else if class != "" {
		pattern = fmt.Sprintf(`(?is)<([a-zA-Z][a-zA-Z0-9]*)\s[^>]*\bclass\s*=\s*["'][^"']*\b%s\b[^"']*["'][^>]*>`, regexp.QuoteMeta(class))
	} else if tag != "" {
		pattern = fmt.Sprintf(`(?is)<%s[^>]*>`, regexp.QuoteMeta(tag))
	} else {
		return ""
	}

	re, err := regexp.Compile(pattern)
	if err != nil {
		return ""
	}

	loc := re.FindStringIndex(html)
	if loc == nil {
		return ""
	}

	// Determine the tag name from the match
	matchTag := tag
	if matchTag == "" {
		// Extract tag name from the match
		m := re.FindStringSubmatch(html)
		if len(m) > 1 {
			matchTag = m[1]
		}
	}

	if matchTag == "" {
		return ""
	}

	// Find the matching closing tag by counting nesting
	rest := html[loc[0]:]
	depth := 0
	openRe := regexp.MustCompile(fmt.Sprintf(`(?i)<%s[\s>]`, regexp.QuoteMeta(matchTag)))
	closeRe := regexp.MustCompile(fmt.Sprintf(`(?i)</%s>`, regexp.QuoteMeta(matchTag)))

	pos := 0
	for pos < len(rest) {
		openLoc := openRe.FindStringIndex(rest[pos:])
		closeLoc := closeRe.FindStringIndex(rest[pos:])

		if closeLoc == nil {
			// No closing tag found, return what we have
			return rest
		}

		if openLoc != nil && openLoc[0] < closeLoc[0] {
			depth++
			pos += openLoc[1]
		} else {
			if depth <= 1 {
				endPos := pos + closeLoc[1]
				return rest[:endPos]
			}
			depth--
			pos += closeLoc[1]
		}
	}

	return rest
}

// ExtractLinks extracts all links from HTML, resolving relative URLs against baseURL.
func ExtractLinks(html, baseURL string) []Link {
	base, _ := url.Parse(baseURL)

	matches := reAnchor.FindAllStringSubmatch(html, -1)
	seen := make(map[string]bool)
	var links []Link

	for _, m := range matches {
		if len(m) < 3 {
			continue
		}
		href := strings.TrimSpace(m[1])
		text := strings.TrimSpace(StripTags(m[2]))

		if href == "" || href == "#" || strings.HasPrefix(href, "javascript:") || strings.HasPrefix(href, "mailto:") || strings.HasPrefix(href, "tel:") || strings.HasPrefix(href, "data:") || strings.HasPrefix(href, "blob:") {
			continue
		}

		// Resolve relative URLs
		if base != nil {
			if parsed, err := url.Parse(href); err == nil {
				resolved := base.ResolveReference(parsed)
				href = resolved.String()
			}
		}

		if seen[href] {
			continue
		}
		seen[href] = true

		internal := false
		if base != nil {
			if parsed, err := url.Parse(href); err == nil {
				internal = parsed.Host == "" || parsed.Host == base.Host
			}
		}

		links = append(links, Link{
			URL:      href,
			Text:     text,
			Internal: internal,
		})
	}

	return links
}

// ExtractTables extracts HTML tables as slices of rows (each row is a slice of cell strings).
func ExtractTables(html string) [][][]string {
	tableMatches := reTable.FindAllStringSubmatch(html, -1)
	var tables [][][]string

	for _, tm := range tableMatches {
		if len(tm) < 2 {
			continue
		}
		tableHTML := tm[1]
		rowMatches := reTR.FindAllStringSubmatch(tableHTML, -1)

		var rows [][]string
		for _, rm := range rowMatches {
			if len(rm) < 2 {
				continue
			}
			cellMatches := reTDTH.FindAllStringSubmatch(rm[1], -1)
			var cells []string
			for _, cm := range cellMatches {
				if len(cm) < 2 {
					continue
				}
				cells = append(cells, strings.TrimSpace(StripTags(cm[1])))
			}
			if len(cells) > 0 {
				rows = append(rows, cells)
			}
		}
		if len(rows) > 0 {
			tables = append(tables, rows)
		}
	}

	return tables
}

// ExtractMeta extracts meta tag name/property and content pairs.
func ExtractMeta(html string) map[string]string {
	result := make(map[string]string)

	metaTags := reMeta.FindAllString(html, -1)
	for _, tag := range metaTags {
		attrs := reMetaAttrs.FindAllStringSubmatch(tag, -1)
		name := ""
		content := ""
		for _, a := range attrs {
			if len(a) < 3 {
				continue
			}
			key := strings.ToLower(a[1])
			val := a[2]
			switch key {
			case "name", "property":
				name = val
			case "content":
				content = val
			}
		}
		if name != "" {
			result[name] = content
		}
	}

	// Also extract title
	if tm := reTitle.FindStringSubmatch(html); len(tm) > 1 {
		result["title"] = strings.TrimSpace(StripTags(tm[1]))
	}

	return result
}

// ExtractOpenGraph extracts OpenGraph (og:) meta tags.
func ExtractOpenGraph(html string) map[string]string {
	all := ExtractMeta(html)
	og := make(map[string]string)
	for k, v := range all {
		if strings.HasPrefix(k, "og:") {
			og[k] = v
		}
	}
	return og
}

// ExtractLists extracts <ul> and <ol> lists as slices of items.
func ExtractLists(html string) [][]string {
	var lists [][]string

	for _, re := range []*regexp.Regexp{reUL, reOL} {
		matches := re.FindAllStringSubmatch(html, -1)
		for _, m := range matches {
			if len(m) < 2 {
				continue
			}
			items := reLI.FindAllStringSubmatch(m[1], -1)
			var list []string
			for _, item := range items {
				if len(item) < 2 {
					continue
				}
				text := strings.TrimSpace(StripTags(item[1]))
				if text != "" {
					list = append(list, text)
				}
			}
			if len(list) > 0 {
				lists = append(lists, list)
			}
		}
	}

	return lists
}
