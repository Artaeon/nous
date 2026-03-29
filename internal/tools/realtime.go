package tools

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// RealTimeData provides live data feeds from free public APIs.
// No API keys required. Uses CoinGecko, Yahoo Finance, open.er-api.com,
// and worldtimeapi.org.
type RealTimeData struct {
	client *http.Client
}

// NewRealTimeData creates a RealTimeData tool with sensible defaults.
func NewRealTimeData() *RealTimeData {
	return &RealTimeData{
		client: &http.Client{Timeout: 10 * time.Second},
	}
}

// newRealTimeDataWithClient creates a RealTimeData with a custom HTTP client (for testing).
func newRealTimeDataWithClient(client *http.Client) *RealTimeData {
	return &RealTimeData{client: client}
}

// ---------------------------------------------------------------------------
// Crypto prices — CoinGecko free API
// ---------------------------------------------------------------------------

// coinGeckoResponse holds the JSON shape returned by the simple/price endpoint.
type coinGeckoResponse map[string]struct {
	USD          float64 `json:"usd"`
	USD24hChange float64 `json:"usd_24h_change"`
}

// cryptoAliases maps common shorthand symbols to CoinGecko IDs.
var cryptoAliases = map[string]string{
	"btc":  "bitcoin",
	"eth":  "ethereum",
	"sol":  "solana",
	"doge": "dogecoin",
	"ada":  "cardano",
	"xrp":  "ripple",
	"dot":  "polkadot",
	"matic": "matic-network",
	"avax": "avalanche-2",
	"link": "chainlink",
	"atom": "cosmos",
	"ltc":  "litecoin",
}

// CryptoPrice fetches the current USD price of a cryptocurrency.
// Supports full names (bitcoin, ethereum) and ticker symbols (BTC, ETH).
// Returns a formatted string like "Bitcoin: $67,234.50 USD (24h change: +2.3%)".
func (rt *RealTimeData) CryptoPrice(symbol string) (string, error) {
	symbol = strings.ToLower(strings.TrimSpace(symbol))
	if symbol == "" {
		return "", fmt.Errorf("crypto: symbol is required")
	}

	// Resolve aliases
	if id, ok := cryptoAliases[symbol]; ok {
		symbol = id
	}

	url := fmt.Sprintf(
		"https://api.coingecko.com/api/v3/simple/price?ids=%s&vs_currencies=usd&include_24hr_change=true",
		symbol,
	)

	resp, err := rt.client.Get(url)
	if err != nil {
		return "", fmt.Errorf("crypto: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("crypto: API returned HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", fmt.Errorf("crypto: reading response: %w", err)
	}

	var data coinGeckoResponse
	if err := json.Unmarshal(body, &data); err != nil {
		return "", fmt.Errorf("crypto: invalid JSON: %w", err)
	}

	info, ok := data[symbol]
	if !ok {
		return "", fmt.Errorf("crypto: no data found for %q", symbol)
	}

	// Format the name with uppercase first letter.
	name := strings.ToUpper(symbol[:1]) + symbol[1:]

	changeSign := "+"
	if info.USD24hChange < 0 {
		changeSign = ""
	}

	return fmt.Sprintf("%s: $%s USD (24h change: %s%.1f%%)",
		name,
		formatNumberCommas(info.USD),
		changeSign,
		info.USD24hChange,
	), nil
}

// formatNumberCommas inserts commas into a float for display (e.g., 67234.5 → "67,234.50").
func formatNumberCommas(n float64) string {
	// Format with 2 decimal places.
	s := fmt.Sprintf("%.2f", n)

	// Split integer and decimal parts.
	parts := strings.SplitN(s, ".", 2)
	intPart := parts[0]
	decPart := ""
	if len(parts) > 1 {
		decPart = "." + parts[1]
	}

	// Insert commas.
	if len(intPart) <= 3 {
		return intPart + decPart
	}

	var result []byte
	negative := false
	if intPart[0] == '-' {
		negative = true
		intPart = intPart[1:]
	}

	for i, c := range intPart {
		if i > 0 && (len(intPart)-i)%3 == 0 {
			result = append(result, ',')
		}
		result = append(result, byte(c))
	}

	if negative {
		return "-" + string(result) + decPart
	}
	return string(result) + decPart
}

// ---------------------------------------------------------------------------
// Stock quotes — Yahoo Finance chart API
// ---------------------------------------------------------------------------

// yahooChartResponse holds the relevant fields from Yahoo's chart endpoint.
type yahooChartResponse struct {
	Chart struct {
		Result []struct {
			Meta struct {
				Symbol             string  `json:"symbol"`
				ShortName          string  `json:"shortName"`
				RegularMarketPrice float64 `json:"regularMarketPrice"`
				ChartPreviousClose float64 `json:"chartPreviousClose"`
				PreviousClose      float64 `json:"previousClose"`
			} `json:"meta"`
		} `json:"result"`
		Error *struct {
			Code        string `json:"code"`
			Description string `json:"description"`
		} `json:"error"`
	} `json:"chart"`
}

// StockQuote fetches the current price of a stock by ticker symbol.
// Returns a formatted string like "AAPL (Apple): $178.72 (+1.2%)".
func (rt *RealTimeData) StockQuote(symbol string) (string, error) {
	symbol = strings.ToUpper(strings.TrimSpace(symbol))
	if symbol == "" {
		return "", fmt.Errorf("stock: symbol is required")
	}

	url := fmt.Sprintf(
		"https://query1.finance.yahoo.com/v8/finance/chart/%s?range=1d&interval=1d",
		symbol,
	)

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("stock: creating request: %w", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0")

	resp, err := rt.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("stock: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("stock: API returned HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", fmt.Errorf("stock: reading response: %w", err)
	}

	var data yahooChartResponse
	if err := json.Unmarshal(body, &data); err != nil {
		return "", fmt.Errorf("stock: invalid JSON: %w", err)
	}

	if data.Chart.Error != nil {
		return "", fmt.Errorf("stock: %s", data.Chart.Error.Description)
	}

	if len(data.Chart.Result) == 0 {
		return "", fmt.Errorf("stock: no data found for %s", symbol)
	}

	meta := data.Chart.Result[0].Meta
	price := meta.RegularMarketPrice
	prevClose := meta.ChartPreviousClose
	if prevClose == 0 {
		prevClose = meta.PreviousClose
	}

	name := meta.ShortName
	if name == "" {
		name = meta.Symbol
	}

	var changeStr string
	if prevClose > 0 {
		changePct := ((price - prevClose) / prevClose) * 100
		sign := "+"
		if changePct < 0 {
			sign = ""
		}
		changeStr = fmt.Sprintf(" (%s%.1f%%)", sign, changePct)
	}

	return fmt.Sprintf("%s (%s): $%s%s",
		symbol, name, formatNumberCommas(price), changeStr,
	), nil
}

// ---------------------------------------------------------------------------
// Exchange rates — open.er-api.com (free, no key)
// ---------------------------------------------------------------------------

// erAPIResponse holds the response from the Open Exchange Rates API.
type erAPIResponse struct {
	Result string             `json:"result"`
	Rates  map[string]float64 `json:"rates"`
}

// ExchangeRate fetches the current exchange rate between two currencies.
// Returns a formatted string like "1 USD = 0.92 EUR".
func (rt *RealTimeData) ExchangeRate(from, to string) (string, error) {
	from = strings.ToUpper(strings.TrimSpace(from))
	to = strings.ToUpper(strings.TrimSpace(to))
	if from == "" || to == "" {
		return "", fmt.Errorf("exchange: both 'from' and 'to' currencies are required")
	}

	url := fmt.Sprintf("https://open.er-api.com/v6/latest/%s", from)

	resp, err := rt.client.Get(url)
	if err != nil {
		return "", fmt.Errorf("exchange: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("exchange: API returned HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", fmt.Errorf("exchange: reading response: %w", err)
	}

	var data erAPIResponse
	if err := json.Unmarshal(body, &data); err != nil {
		return "", fmt.Errorf("exchange: invalid JSON: %w", err)
	}

	if data.Result != "success" {
		return "", fmt.Errorf("exchange: API error for %s", from)
	}

	rate, ok := data.Rates[to]
	if !ok {
		return "", fmt.Errorf("exchange: no rate found for %s → %s", from, to)
	}

	return fmt.Sprintf("1 %s = %.4f %s", from, rate, to), nil
}

// ---------------------------------------------------------------------------
// World clock — worldtimeapi.org
// ---------------------------------------------------------------------------

// worldTimeResponse holds the relevant fields from the WorldTimeAPI.
type worldTimeResponse struct {
	Timezone     string `json:"timezone"`
	Datetime     string `json:"datetime"`     // ISO 8601
	Abbreviation string `json:"abbreviation"` // e.g., "EST"
}

// timezoneAliases maps friendly city names to IANA timezone identifiers.
var timezoneAliases = map[string]string{
	"new york":      "America/New_York",
	"nyc":           "America/New_York",
	"los angeles":   "America/Los_Angeles",
	"la":            "America/Los_Angeles",
	"chicago":       "America/Chicago",
	"london":        "Europe/London",
	"paris":         "Europe/Paris",
	"berlin":        "Europe/Berlin",
	"tokyo":         "Asia/Tokyo",
	"sydney":        "Australia/Sydney",
	"mumbai":        "Asia/Kolkata",
	"dubai":         "Asia/Dubai",
	"singapore":     "Asia/Singapore",
	"hong kong":     "Asia/Hong_Kong",
	"shanghai":      "Asia/Shanghai",
	"beijing":       "Asia/Shanghai",
	"moscow":        "Europe/Moscow",
	"sao paulo":     "America/Sao_Paulo",
	"toronto":       "America/Toronto",
	"vancouver":     "America/Vancouver",
	"denver":        "America/Denver",
	"utc":           "Etc/UTC",
	"gmt":           "Etc/GMT",
}

// WorldTime fetches the current time in a given timezone.
// Accepts IANA timezone names (America/New_York) or city names (New York, Tokyo).
// Returns a formatted string like "New York: 2:30 PM EST (March 29, 2026)".
func (rt *RealTimeData) WorldTime(timezone string) (string, error) {
	timezone = strings.TrimSpace(timezone)
	if timezone == "" {
		return "", fmt.Errorf("worldtime: timezone is required")
	}

	displayName := timezone

	// Resolve friendly names to IANA identifiers.
	lower := strings.ToLower(timezone)
	if iana, ok := timezoneAliases[lower]; ok {
		timezone = iana
		displayName = strings.Title(lower)
	}

	url := fmt.Sprintf("http://worldtimeapi.org/api/timezone/%s", timezone)

	resp, err := rt.client.Get(url)
	if err != nil {
		return "", fmt.Errorf("worldtime: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("worldtime: API returned HTTP %d for %q", resp.StatusCode, timezone)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", fmt.Errorf("worldtime: reading response: %w", err)
	}

	var data worldTimeResponse
	if err := json.Unmarshal(body, &data); err != nil {
		return "", fmt.Errorf("worldtime: invalid JSON: %w", err)
	}

	// Parse the ISO 8601 datetime.
	t, err := time.Parse("2006-01-02T15:04:05.999999-07:00", data.Datetime)
	if err != nil {
		// Try without fractional seconds.
		t, err = time.Parse("2006-01-02T15:04:05-07:00", data.Datetime)
		if err != nil {
			return "", fmt.Errorf("worldtime: parsing time %q: %w", data.Datetime, err)
		}
	}

	abbr := data.Abbreviation
	if abbr == "" {
		abbr = "local"
	}

	// Use the last part of the IANA timezone as display name if we didn't alias it.
	if displayName == timezone {
		parts := strings.Split(timezone, "/")
		displayName = strings.ReplaceAll(parts[len(parts)-1], "_", " ")
	}

	return fmt.Sprintf("%s: %s %s (%s)",
		displayName,
		t.Format("3:04 PM"),
		abbr,
		t.Format("January 2, 2006"),
	), nil
}

// ---------------------------------------------------------------------------
// Routing — parse freeform queries into the right sub-command
// ---------------------------------------------------------------------------

// routeQuery inspects a freeform query and dispatches to the right method.
func (rt *RealTimeData) routeQuery(args map[string]string) (string, error) {
	// Explicit sub-command routing.
	if sub := args["sub"]; sub != "" {
		switch strings.ToLower(sub) {
		case "crypto":
			return rt.CryptoPrice(args["symbol"])
		case "stock":
			return rt.StockQuote(args["symbol"])
		case "exchange", "fx":
			return rt.ExchangeRate(args["from"], args["to"])
		case "time", "clock":
			return rt.WorldTime(args["timezone"])
		}
	}

	// Freeform query routing.
	query := strings.ToLower(args["query"])
	if query == "" {
		return "", fmt.Errorf("realtime: provide a query (e.g., 'bitcoin price', 'AAPL stock', 'USD to EUR', 'time in Tokyo')")
	}

	// Crypto: "bitcoin price", "price of ethereum", "eth"
	cryptoKeywords := []string{"bitcoin", "ethereum", "solana", "dogecoin", "cardano",
		"ripple", "polkadot", "avalanche", "chainlink", "cosmos", "litecoin",
		"btc", "eth", "sol", "doge", "ada", "xrp", "dot", "matic", "avax", "link", "atom", "ltc"}
	for _, kw := range cryptoKeywords {
		if strings.Contains(query, kw) {
			return rt.CryptoPrice(kw)
		}
	}
	if strings.Contains(query, "crypto") {
		// Default to bitcoin if generic "crypto price" query.
		return rt.CryptoPrice("bitcoin")
	}

	// Exchange rate: "usd to eur", "convert gbp to jpy", "exchange rate eur usd"
	if strings.Contains(query, " to ") && isLikelyCurrency(query) {
		from, to := parseCurrencyPair(query)
		if from != "" && to != "" {
			return rt.ExchangeRate(from, to)
		}
	}
	if strings.Contains(query, "exchange rate") {
		from, to := parseCurrencyPair(query)
		if from != "" && to != "" {
			return rt.ExchangeRate(from, to)
		}
	}

	// Time: "time in tokyo", "what time is it in london", "world clock new york"
	if strings.Contains(query, "time in") || strings.Contains(query, "time at") ||
		strings.Contains(query, "clock") {
		tz := extractTimezone(query)
		if tz != "" {
			return rt.WorldTime(tz)
		}
	}

	// Stock: "AAPL stock", "stock price TSLA", "quote MSFT"
	if strings.Contains(query, "stock") || strings.Contains(query, "quote") ||
		strings.Contains(query, "share") || strings.Contains(query, "price") {
		sym := extractStockSymbol(query)
		if sym != "" {
			return rt.StockQuote(sym)
		}
	}

	return "", fmt.Errorf("realtime: couldn't understand query %q — try 'bitcoin price', 'AAPL stock', 'USD to EUR', or 'time in Tokyo'", args["query"])
}

// isLikelyCurrency checks if the query contains currency-like 3-letter codes.
func isLikelyCurrency(query string) bool {
	common := []string{"usd", "eur", "gbp", "jpy", "aud", "cad", "chf", "cny",
		"inr", "brl", "krw", "sek", "nok", "dkk", "nzd", "sgd", "hkd", "mxn",
		"dollar", "euro", "pound", "yen", "franc", "rupee"}
	for _, c := range common {
		if strings.Contains(query, c) {
			return true
		}
	}
	return false
}

// parseCurrencyPair extracts "from" and "to" currencies from a query.
func parseCurrencyPair(query string) (string, string) {
	// Handle "X to Y" patterns.
	words := strings.Fields(query)
	for i, w := range words {
		if w == "to" && i > 0 && i < len(words)-1 {
			from := extractCurrencyCode(words[i-1])
			to := extractCurrencyCode(words[i+1])
			if from != "" && to != "" {
				return from, to
			}
		}
	}

	// Handle "exchange rate X Y" patterns.
	var codes []string
	for _, w := range words {
		if c := extractCurrencyCode(w); c != "" {
			codes = append(codes, c)
		}
	}
	if len(codes) >= 2 {
		return codes[0], codes[1]
	}

	return "", ""
}

// extractCurrencyCode tries to resolve a word to a 3-letter currency code.
func extractCurrencyCode(word string) string {
	word = strings.ToLower(strings.TrimRight(word, ".,;:!?"))
	if len(word) == 3 {
		// Assume it's already a code.
		return strings.ToUpper(word)
	}
	aliases := map[string]string{
		"dollar": "USD", "dollars": "USD",
		"euro": "EUR", "euros": "EUR",
		"pound": "GBP", "pounds": "GBP",
		"yen": "JPY",
		"franc": "CHF", "francs": "CHF",
		"rupee": "INR", "rupees": "INR",
	}
	if code, ok := aliases[word]; ok {
		return code
	}
	return ""
}

// extractTimezone pulls a timezone or city name from a freeform query.
func extractTimezone(query string) string {
	// Strip common prefixes.
	for _, prefix := range []string{
		"what time is it in ", "what's the time in ", "time in ",
		"current time in ", "world clock ", "clock in ", "time at ",
	} {
		if idx := strings.Index(query, prefix); idx >= 0 {
			tz := strings.TrimSpace(query[idx+len(prefix):])
			tz = strings.TrimRight(tz, "?!.")
			if tz != "" {
				return tz
			}
		}
	}
	return ""
}

// extractStockSymbol pulls a stock ticker from a freeform query.
func extractStockSymbol(query string) string {
	words := strings.Fields(query)
	skip := map[string]bool{
		"stock": true, "price": true, "quote": true, "share": true,
		"of": true, "the": true, "what": true, "is": true, "get": true,
		"current": true, "show": true, "me": true, "for": true, "a": true,
	}
	for _, w := range words {
		w = strings.TrimRight(w, ".,;:!?")
		if !skip[strings.ToLower(w)] && len(w) >= 1 && len(w) <= 5 {
			// Likely a ticker symbol.
			return strings.ToUpper(w)
		}
	}
	return ""
}

// RegisterRealTimeData adds the real-time data tool to the registry.
func RegisterRealTimeData(reg *Registry) {
	rt := NewRealTimeData()
	reg.Register(Tool{
		Name:        "realtime",
		Description: "Live data: crypto prices, stock quotes, exchange rates, world time. Args: query (freeform, e.g., 'bitcoin price', 'AAPL stock', 'USD to EUR', 'time in Tokyo'), or sub (crypto|stock|exchange|time) + specific args.",
		Execute: func(args map[string]string) (string, error) {
			return rt.routeQuery(args)
		},
	})
}
