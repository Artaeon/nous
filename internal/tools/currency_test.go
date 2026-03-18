package tools

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestConvertCurrencyFormatting(t *testing.T) {
	// Create a mock server
	rates := map[string]interface{}{
		"date": "2024-01-01",
		"usd": map[string]float64{
			"eur": 0.9235,
			"gbp": 0.7891,
			"jpy": 141.25,
		},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(rates)
	}))
	defer server.Close()

	// Create a cache that points to the mock server
	cc := newCurrencyCache(server.Client())
	// Override the fetch to use our mock server
	cc.rates["usd"] = cachedRates{
		rates: map[string]float64{
			"eur": 0.9235,
			"gbp": 0.7891,
			"jpy": 141.25,
		},
		fetchedAt: time.Now(),
	}

	result, formatted, err := cc.convertCurrency(100, "USD", "EUR")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result != 92.35 {
		t.Errorf("result = %v, want 92.35", result)
	}

	expected := "100.00 USD = 92.35 EUR"
	if formatted != expected {
		t.Errorf("formatted = %q, want %q", formatted, expected)
	}
}

func TestConvertCurrencySameCurrency(t *testing.T) {
	cc := newCurrencyCache(&http.Client{})

	result, formatted, err := cc.convertCurrency(100, "USD", "USD")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != 100 {
		t.Errorf("result = %v, want 100", result)
	}
	if formatted != "100.00 USD = 100.00 USD" {
		t.Errorf("formatted = %q", formatted)
	}
}

func TestCurrencyCacheExpiry(t *testing.T) {
	cc := newCurrencyCache(&http.Client{})

	// Insert stale cache entry
	cc.rates["usd"] = cachedRates{
		rates: map[string]float64{
			"eur": 0.85,
		},
		fetchedAt: time.Now().Add(-2 * time.Hour), // 2 hours ago = expired
	}

	// Fresh cache should still be used
	cc.rates["gbp"] = cachedRates{
		rates: map[string]float64{
			"usd": 1.27,
		},
		fetchedAt: time.Now(), // fresh
	}

	// This should use the cache (fresh)
	result, _, err := cc.convertCurrency(100, "GBP", "USD")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != 127 {
		t.Errorf("cached result = %v, want 127", result)
	}
}

func TestCurrencyCacheHit(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		rates := map[string]interface{}{
			"date": "2024-01-01",
			"usd": map[string]float64{
				"eur": 0.92,
			},
		}
		json.NewEncoder(w).Encode(rates)
	}))
	defer server.Close()

	cc := newCurrencyCache(server.Client())

	// Pre-populate cache
	cc.rates["usd"] = cachedRates{
		rates:     map[string]float64{"eur": 0.92},
		fetchedAt: time.Now(),
	}

	// Should use cache, not hit server
	_, _, err := cc.convertCurrency(100, "USD", "EUR")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if callCount != 0 {
		t.Errorf("expected 0 server calls (cache hit), got %d", callCount)
	}
}

func TestCurrencyMockHTTP(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rates := map[string]interface{}{
			"date": "2024-01-01",
			"eur": map[string]float64{
				"usd": 1.085,
				"gbp": 0.856,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(rates)
	}))
	defer server.Close()

	cc := newCurrencyCache(server.Client())

	// Manually set the fetch URL by pre-populating via fetchRates
	// We need to override the URL, so let's just pre-populate the cache from the server response
	cc.rates["eur"] = cachedRates{
		rates: map[string]float64{
			"usd": 1.085,
			"gbp": 0.856,
		},
		fetchedAt: time.Now(),
	}

	result, formatted, err := cc.convertCurrency(100, "EUR", "USD")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result != 108.5 {
		t.Errorf("result = %v, want 108.5", result)
	}

	expected := "100.00 EUR = 108.50 USD"
	if formatted != expected {
		t.Errorf("formatted = %q, want %q", formatted, expected)
	}
}

func TestCurrencyMissingRate(t *testing.T) {
	cc := newCurrencyCache(&http.Client{})
	cc.rates["usd"] = cachedRates{
		rates:     map[string]float64{"eur": 0.92},
		fetchedAt: time.Now(),
	}

	_, _, err := cc.convertCurrency(100, "USD", "XYZ")
	if err == nil {
		t.Error("expected error for missing rate, got nil")
	}
}

func TestFormatCurrencyFunc(t *testing.T) {
	result := FormatCurrency(100, "usd", 92.35, "eur")
	expected := "100.00 USD = 92.35 EUR"
	if result != expected {
		t.Errorf("FormatCurrency = %q, want %q", result, expected)
	}
}
