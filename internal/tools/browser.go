package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

// rateLimiter enforces per-domain request rate limiting (1 req/sec).
var rateLimiter = struct {
	mu   sync.Mutex
	last map[string]time.Time
}{last: make(map[string]time.Time)}

func waitForRateLimit(host string) {
	rateLimiter.mu.Lock()
	defer rateLimiter.mu.Unlock()
	if last, ok := rateLimiter.last[host]; ok {
		elapsed := time.Since(last)
		if elapsed < time.Second {
			time.Sleep(time.Second - elapsed)
		}
	}
	rateLimiter.last[host] = time.Now()
}

// isPrivateIP checks if an IP address is in a private/reserved range.
func isPrivateIP(ip net.IP) bool {
	privateRanges := []struct {
		network string
	}{
		{"127.0.0.0/8"},
		{"10.0.0.0/8"},
		{"172.16.0.0/12"},
		{"192.168.0.0/16"},
		{"169.254.0.0/16"},
		{"::1/128"},
		{"fe80::/10"},
		{"fc00::/7"},
	}
	for _, r := range privateRanges {
		_, cidr, err := net.ParseCIDR(r.network)
		if err != nil {
			continue
		}
		if cidr.Contains(ip) {
			return true
		}
	}
	return false
}

// validateHost resolves a hostname and rejects private/internal IPs to prevent SSRF.
func validateHost(hostname string) error {
	// Strip port if present
	host := hostname
	if h, _, err := net.SplitHostPort(hostname); err == nil {
		host = h
	}

	addrs, err := net.LookupHost(host)
	if err != nil {
		return fmt.Errorf("cannot resolve host %q: %w", host, err)
	}

	for _, addr := range addrs {
		ip := net.ParseIP(addr)
		if ip == nil {
			continue
		}
		if isPrivateIP(ip) {
			return fmt.Errorf("access to private/internal IP %s is not allowed", addr)
		}
	}
	return nil
}

// truncateUTF8 truncates a string to at most maxBytes bytes without splitting
// a multi-byte UTF-8 character.
func truncateUTF8(s string, maxBytes int) string {
	if len(s) <= maxBytes {
		return s
	}
	// Walk backward from maxBytes to find a valid rune boundary
	for maxBytes > 0 && !utf8.RuneStart(s[maxBytes]) {
		maxBytes--
	}
	return s[:maxBytes]
}

// fetchPage fetches a URL with standard protections (timeout, size limit, rate limiting).
func fetchPage(rawURL string) (string, string, error) {
	if rawURL == "" {
		return "", "", fmt.Errorf("url is required")
	}

	// Ensure scheme
	if !strings.HasPrefix(rawURL, "http://") && !strings.HasPrefix(rawURL, "https://") {
		rawURL = "https://" + rawURL
	}

	parsed, err := url.Parse(rawURL)
	if err != nil {
		return "", "", fmt.Errorf("invalid URL: %w", err)
	}

	// SSRF protection: reject private/internal IPs
	if err := validateHost(parsed.Host); err != nil {
		return "", "", err
	}

	// Rate limit by host
	waitForRateLimit(parsed.Host)

	// Basic robots.txt check
	if blocked := checkRobots(parsed); blocked {
		return "", "", fmt.Errorf("blocked by robots.txt for %s", rawURL)
	}

	client := &http.Client{
		Timeout: 30 * time.Second,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 5 {
				return fmt.Errorf("too many redirects")
			}
			return nil
		},
	}

	req, err := http.NewRequest("GET", rawURL, nil)
	if err != nil {
		return "", "", fmt.Errorf("invalid URL: %w", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; Nous/1.0; +https://github.com/artaeon/nous)")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
	req.Header.Set("Accept-Language", "en-US,en;q=0.5")

	resp, err := client.Do(req)
	if err != nil {
		return "", "", fmt.Errorf("fetch %s: %w", rawURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return "", "", fmt.Errorf("HTTP %d for %s", resp.StatusCode, rawURL)
	}

	// Limit to 5MB
	body, err := io.ReadAll(io.LimitReader(resp.Body, 5<<20))
	if err != nil {
		return "", "", fmt.Errorf("reading response: %w", err)
	}

	return string(body), resp.Request.URL.String(), nil
}

// checkRobots does a basic robots.txt check for the User-Agent "Nous".
// Returns true if the path appears to be disallowed.
func checkRobots(u *url.URL) bool {
	robotsURL := fmt.Sprintf("%s://%s/robots.txt", u.Scheme, u.Host)

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(robotsURL)
	if err != nil || resp.StatusCode != 200 {
		// Can't fetch robots.txt — assume allowed
		return false
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	if err != nil {
		return false
	}

	// Parse robots.txt: collect all Allow/Disallow rules for matching user-agents,
	// then apply the most specific (longest) matching rule.
	lines := strings.Split(string(body), "\n")
	inWildcard := false
	path := u.Path
	if path == "" {
		path = "/"
	}

	type rule struct {
		pattern string
		allow   bool
	}
	var rules []rule

	for _, line := range lines {
		line = strings.TrimSpace(line)
		lower := strings.ToLower(line)

		if strings.HasPrefix(lower, "user-agent:") {
			agent := strings.TrimSpace(line[len("user-agent:"):])
			inWildcard = agent == "*" || strings.EqualFold(agent, "nous")
			continue
		}

		if !inWildcard {
			continue
		}

		if strings.HasPrefix(lower, "disallow:") {
			pattern := strings.TrimSpace(line[len("disallow:"):])
			if pattern != "" {
				rules = append(rules, rule{pattern: pattern, allow: false})
			}
		} else if strings.HasPrefix(lower, "allow:") {
			pattern := strings.TrimSpace(line[len("allow:"):])
			if pattern != "" {
				rules = append(rules, rule{pattern: pattern, allow: true})
			}
		}
	}

	// Find the most specific matching rule (longest pattern wins; ties go to Allow)
	bestLen := -1
	bestAllow := true
	for _, r := range rules {
		if strings.HasPrefix(path, r.pattern) {
			pLen := len(r.pattern)
			if pLen > bestLen || (pLen == bestLen && r.allow) {
				bestLen = pLen
				bestAllow = r.allow
			}
		}
	}

	if bestLen >= 0 {
		return !bestAllow
	}
	return false
}

// RegisterBrowserTools adds browser automation tools to the registry.
func RegisterBrowserTools(r *Registry) {
	r.Register(Tool{
		Name:        "browse",
		Description: "Fetch and render a web page as readable text. Args: url (required), selector (optional CSS selector like '#main', '.content', 'article'), format (optional: 'text', 'links', 'raw', default 'text').",
		Execute: func(args map[string]string) (string, error) {
			return toolBrowse(args)
		},
	})

	r.Register(Tool{
		Name:        "screenshot",
		Description: "Take a screenshot of a web page (requires chromium/chrome). Args: url (required), width (optional, default 1280), height (optional, default 720). Saves to ~/.nous/screenshots/.",
		Execute: func(args map[string]string) (string, error) {
			return toolScreenshot(args)
		},
	})

	r.Register(Tool{
		Name:        "links",
		Description: "Extract all links from a web page. Args: url (required), filter (optional regex to filter URLs).",
		Execute: func(args map[string]string) (string, error) {
			return toolLinks(args)
		},
	})

	r.Register(Tool{
		Name:        "scrape",
		Description: "Extract structured data from a web page. Args: url (required), type (required: 'table', 'list', 'meta', 'og').",
		Execute: func(args map[string]string) (string, error) {
			return toolScrape(args)
		},
	})
}

func toolBrowse(args map[string]string) (string, error) {
	rawURL := args["url"]
	if rawURL == "" {
		return "", fmt.Errorf("browse requires 'url' argument")
	}

	html, finalURL, err := fetchPage(rawURL)
	if err != nil {
		return "", err
	}

	// Apply selector if provided
	selector := args["selector"]
	if selector != "" {
		extracted := ExtractBySelector(html, selector)
		if extracted != "" {
			html = extracted
		}
	}

	format := args["format"]
	if format == "" {
		format = "text"
	}

	var result string

	switch format {
	case "raw":
		result = html
	case "links":
		links := ExtractLinks(html, finalURL)
		var sb strings.Builder
		for _, l := range links {
			label := ""
			if l.Internal {
				label = " [internal]"
			}
			text := l.Text
			if text != "" {
				sb.WriteString(fmt.Sprintf("- [%s](%s)%s\n", text, l.URL, label))
			} else {
				sb.WriteString(fmt.Sprintf("- %s%s\n", l.URL, label))
			}
		}
		result = sb.String()
		if result == "" {
			result = "no links found"
		}
	default: // "text"
		result = HTMLToMarkdown(html)
	}

	// Truncate to reasonable size for LLM context
	if len(result) > 16384 {
		result = truncateUTF8(result, 16384) + "\n\n... (truncated — page content exceeds 16KB limit)"
	}

	if result == "" {
		return fmt.Sprintf("page at %s returned no extractable content", finalURL), nil
	}

	return result, nil
}

func toolScreenshot(args map[string]string) (string, error) {
	rawURL := args["url"]
	if rawURL == "" {
		return "", fmt.Errorf("screenshot requires 'url' argument")
	}

	if !strings.HasPrefix(rawURL, "http://") && !strings.HasPrefix(rawURL, "https://") {
		rawURL = "https://" + rawURL
	}

	// Validate URL scheme — only http/https allowed
	parsedScreenshot, err := url.Parse(rawURL)
	if err != nil {
		return "", fmt.Errorf("invalid URL: %w", err)
	}
	scheme := strings.ToLower(parsedScreenshot.Scheme)
	if scheme != "http" && scheme != "https" {
		return "", fmt.Errorf("unsupported URL scheme %q — only http and https are allowed", scheme)
	}

	// SSRF protection: reject private/internal IPs
	if err := validateHost(parsedScreenshot.Host); err != nil {
		return "", err
	}

	// Find a browser binary
	browserPath := ""
	candidates := []string{
		"chromium-browser", "chromium", "google-chrome-stable",
		"google-chrome", "chrome",
	}
	for _, name := range candidates {
		if path, err := exec.LookPath(name); err == nil {
			browserPath = path
			break
		}
	}

	if browserPath == "" {
		return "", fmt.Errorf("no chromium/chrome browser found — use the 'browse' tool instead for text content")
	}

	width := 1280
	height := 720
	if v, ok := args["width"]; ok {
		if n, err := strconv.Atoi(v); err == nil && n > 0 && n <= 3840 {
			width = n
		}
	}
	if v, ok := args["height"]; ok {
		if n, err := strconv.Atoi(v); err == nil && n > 0 && n <= 2160 {
			height = n
		}
	}

	// Create screenshots directory
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("cannot determine home directory: %w", err)
	}

	screenshotDir := filepath.Join(home, ".nous", "screenshots")
	if err := os.MkdirAll(screenshotDir, 0755); err != nil {
		return "", fmt.Errorf("cannot create screenshot directory: %w", err)
	}

	// Generate filename from URL
	parsed, _ := url.Parse(rawURL)
	safeName := regexp.MustCompile(`[^a-zA-Z0-9.-]`).ReplaceAllString(parsed.Host+parsed.Path, "_")
	if len(safeName) > 100 {
		safeName = safeName[:100]
	}
	timestamp := time.Now().Format("20060102-150405")
	filename := fmt.Sprintf("%s_%s.png", safeName, timestamp)
	outputPath := filepath.Join(screenshotDir, filename)

	// Launch headless chrome
	chromeArgs := []string{
		"--headless",
		"--disable-gpu",
		"--no-sandbox",
		"--disable-dev-shm-usage",
		"--disable-extensions",
		"--disable-background-networking",
		fmt.Sprintf("--window-size=%d,%d", width, height),
		fmt.Sprintf("--screenshot=%s", outputPath),
		"--hide-scrollbars",
		rawURL,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, browserPath, chromeArgs...)
	cmd.Env = append(os.Environ(), "DISPLAY=:0")

	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("screenshot failed: %v\n%s", err, string(output))
	}

	// Verify the screenshot was created
	info, err := os.Stat(outputPath)
	if err != nil {
		return "", fmt.Errorf("screenshot file not created: %w", err)
	}

	return fmt.Sprintf("screenshot saved to %s (%d bytes, %dx%d)", outputPath, info.Size(), width, height), nil
}

func toolLinks(args map[string]string) (string, error) {
	rawURL := args["url"]
	if rawURL == "" {
		return "", fmt.Errorf("links requires 'url' argument")
	}

	html, finalURL, err := fetchPage(rawURL)
	if err != nil {
		return "", err
	}

	links := ExtractLinks(html, finalURL)
	if len(links) == 0 {
		return "no links found", nil
	}

	// Apply filter if provided
	filter := args["filter"]
	var filterRe *regexp.Regexp
	if filter != "" {
		filterRe, err = regexp.Compile(filter)
		if err != nil {
			return "", fmt.Errorf("invalid filter regex: %w", err)
		}
	}

	var internal, external []Link
	for _, l := range links {
		if filterRe != nil && !filterRe.MatchString(l.URL) && !filterRe.MatchString(l.Text) {
			continue
		}
		if l.Internal {
			internal = append(internal, l)
		} else {
			external = append(external, l)
		}
	}

	var sb strings.Builder

	if len(internal) > 0 {
		sb.WriteString(fmt.Sprintf("Internal links (%d):\n", len(internal)))
		for _, l := range internal {
			if l.Text != "" {
				sb.WriteString(fmt.Sprintf("  - [%s](%s)\n", l.Text, l.URL))
			} else {
				sb.WriteString(fmt.Sprintf("  - %s\n", l.URL))
			}
		}
	}

	if len(external) > 0 {
		if sb.Len() > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString(fmt.Sprintf("External links (%d):\n", len(external)))
		for _, l := range external {
			if l.Text != "" {
				sb.WriteString(fmt.Sprintf("  - [%s](%s)\n", l.Text, l.URL))
			} else {
				sb.WriteString(fmt.Sprintf("  - %s\n", l.URL))
			}
		}
	}

	result := sb.String()
	if result == "" {
		return "no links matched the filter", nil
	}

	if len(result) > 8192 {
		result = truncateUTF8(result, 8192) + "\n... (truncated)"
	}

	return result, nil
}

func toolScrape(args map[string]string) (string, error) {
	rawURL := args["url"]
	if rawURL == "" {
		return "", fmt.Errorf("scrape requires 'url' argument")
	}

	scrapeType := args["type"]
	if scrapeType == "" {
		return "", fmt.Errorf("scrape requires 'type' argument (table, list, meta, og)")
	}

	html, _, err := fetchPage(rawURL)
	if err != nil {
		return "", err
	}

	switch scrapeType {
	case "table":
		tables := ExtractTables(html)
		if len(tables) == 0 {
			return "no tables found", nil
		}
		data, err := json.MarshalIndent(tables, "", "  ")
		if err != nil {
			return "", fmt.Errorf("marshal tables: %w", err)
		}
		result := string(data)
		if len(result) > 8192 {
			result = result[:8192] + "\n... (truncated)"
		}
		return result, nil

	case "list":
		lists := ExtractLists(html)
		if len(lists) == 0 {
			return "no lists found", nil
		}
		var sb strings.Builder
		for i, list := range lists {
			sb.WriteString(fmt.Sprintf("List %d:\n", i+1))
			for _, item := range list {
				sb.WriteString(fmt.Sprintf("  - %s\n", item))
			}
			sb.WriteString("\n")
		}
		result := sb.String()
		if len(result) > 8192 {
			result = result[:8192] + "\n... (truncated)"
		}
		return result, nil

	case "meta":
		meta := ExtractMeta(html)
		if len(meta) == 0 {
			return "no meta tags found", nil
		}
		data, err := json.MarshalIndent(meta, "", "  ")
		if err != nil {
			return "", fmt.Errorf("marshal meta: %w", err)
		}
		return string(data), nil

	case "og":
		og := ExtractOpenGraph(html)
		if len(og) == 0 {
			return "no OpenGraph tags found", nil
		}
		data, err := json.MarshalIndent(og, "", "  ")
		if err != nil {
			return "", fmt.Errorf("marshal og: %w", err)
		}
		return string(data), nil

	default:
		return "", fmt.Errorf("unknown scrape type %q — use 'table', 'list', 'meta', or 'og'", scrapeType)
	}
}
