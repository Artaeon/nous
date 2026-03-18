package tools

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// currencyCache stores fetched exchange rates with a timestamp.
type currencyCache struct {
	mu      sync.Mutex
	rates   map[string]cachedRates // keyed by base currency (lowercase)
	client  *http.Client
}

type cachedRates struct {
	rates     map[string]float64
	fetchedAt time.Time
}

var (
	defaultCurrencyCache *currencyCache
	currencyCacheOnce    sync.Once
)

func getCurrencyCache() *currencyCache {
	currencyCacheOnce.Do(func() {
		defaultCurrencyCache = &currencyCache{
			rates:  make(map[string]cachedRates),
			client: &http.Client{Timeout: 10 * time.Second},
		}
	})
	return defaultCurrencyCache
}

// newCurrencyCache creates a currency cache with a custom HTTP client (for testing).
func newCurrencyCache(client *http.Client) *currencyCache {
	return &currencyCache{
		rates:  make(map[string]cachedRates),
		client: client,
	}
}

const currencyCacheDuration = 1 * time.Hour

// fetchRates retrieves exchange rates for a base currency, using cache if fresh.
func (cc *currencyCache) fetchRates(base string) (map[string]float64, error) {
	base = strings.ToLower(base)

	cc.mu.Lock()
	if cached, ok := cc.rates[base]; ok && time.Since(cached.fetchedAt) < currencyCacheDuration {
		cc.mu.Unlock()
		return cached.rates, nil
	}
	cc.mu.Unlock()

	url := fmt.Sprintf("https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/%s.json", base)

	resp, err := cc.client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("currency: failed to fetch rates: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("currency: API returned HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("currency: reading response: %w", err)
	}

	// The API returns: {"date": "...", "<base>": {"<target>": rate, ...}}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("currency: invalid JSON: %w", err)
	}

	ratesJSON, ok := raw[base]
	if !ok {
		return nil, fmt.Errorf("currency: no rates found for %s", base)
	}

	var rates map[string]float64
	if err := json.Unmarshal(ratesJSON, &rates); err != nil {
		return nil, fmt.Errorf("currency: invalid rates JSON: %w", err)
	}

	cc.mu.Lock()
	cc.rates[base] = cachedRates{
		rates:     rates,
		fetchedAt: time.Now(),
	}
	cc.mu.Unlock()

	return rates, nil
}

// ConvertCurrency converts an amount from one currency to another.
// Returns the converted amount, a formatted string, and any error.
func ConvertCurrency(amount float64, from, to string) (float64, string, error) {
	return getCurrencyCache().convertCurrency(amount, from, to)
}

func (cc *currencyCache) convertCurrency(amount float64, from, to string) (float64, string, error) {
	from = strings.ToUpper(from)
	to = strings.ToUpper(to)

	if from == to {
		formatted := fmt.Sprintf("%.2f %s = %.2f %s", amount, from, amount, to)
		return amount, formatted, nil
	}

	rates, err := cc.fetchRates(from)
	if err != nil {
		return 0, "", err
	}

	rate, ok := rates[strings.ToLower(to)]
	if !ok {
		return 0, "", fmt.Errorf("currency: no exchange rate found for %s -> %s", from, to)
	}

	result := amount * rate
	formatted := fmt.Sprintf("%.2f %s = %.2f %s", amount, from, result, to)

	return result, formatted, nil
}

// FormatCurrency formats a currency conversion result.
func FormatCurrency(amount float64, from string, result float64, to string) string {
	return fmt.Sprintf("%.2f %s = %.2f %s", amount, strings.ToUpper(from), result, strings.ToUpper(to))
}

// RegisterCurrencyTool adds the currency conversion tool to the registry.
func RegisterCurrencyTool(r *Registry) {
	r.Register(Tool{
		Name:        "currency",
		Description: "Convert between currencies using live exchange rates. Args: amount (required), from (required, e.g. 'USD'), to (required, e.g. 'EUR').",
		Execute: func(args map[string]string) (string, error) {
			amountStr := args["amount"]
			from := args["from"]
			to := args["to"]

			if amountStr == "" || from == "" || to == "" {
				return "", fmt.Errorf("currency requires 'amount', 'from', and 'to' arguments")
			}

			amount, err := strconv.ParseFloat(amountStr, 64)
			if err != nil {
				return "", fmt.Errorf("invalid amount: %s", amountStr)
			}

			_, formatted, err := ConvertCurrency(amount, from, to)
			if err != nil {
				return "", err
			}

			return formatted, nil
		},
	})
}
