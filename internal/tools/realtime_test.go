package tools

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestFormatNumber(t *testing.T) {
	tests := []struct {
		in   float64
		want string
	}{
		{0, "0.00"},
		{1.5, "1.50"},
		{100, "100.00"},
		{1234.56, "1,234.56"},
		{67234.5, "67,234.50"},
		{1000000, "1,000,000.00"},
		{-1234.56, "-1,234.56"},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("%.2f", tt.in), func(t *testing.T) {
			got := formatNumberCommas(tt.in)
			if got != tt.want {
				t.Errorf("formatNumberCommas(%v) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestCryptoPrice(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ids := r.URL.Query().Get("ids")
		data := map[string]interface{}{
			ids: map[string]interface{}{
				"usd":            67234.50,
				"usd_24h_change": 2.3,
			},
		}
		json.NewEncoder(w).Encode(data)
	}))
	defer srv.Close()

	rt := newRealTimeDataWithClient(srv.Client())
	// Override the URL by using a custom transport that redirects.
	rt.client.Transport = rewriteTransport{base: srv.URL}

	result, err := rt.CryptoPrice("bitcoin")
	if err != nil {
		t.Fatalf("CryptoPrice: %v", err)
	}
	if !strings.Contains(result, "Bitcoin") {
		t.Errorf("expected 'Bitcoin' in result, got: %s", result)
	}
	if !strings.Contains(result, "67,234.50") {
		t.Errorf("expected '67,234.50' in result, got: %s", result)
	}
	if !strings.Contains(result, "+2.3%") {
		t.Errorf("expected '+2.3%%' in result, got: %s", result)
	}
}

func TestCryptoPrice_Alias(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ids := r.URL.Query().Get("ids")
		if ids != "bitcoin" {
			http.Error(w, "unexpected id", 400)
			return
		}
		data := map[string]interface{}{
			"bitcoin": map[string]interface{}{
				"usd":            50000.00,
				"usd_24h_change": -1.5,
			},
		}
		json.NewEncoder(w).Encode(data)
	}))
	defer srv.Close()

	rt := newRealTimeDataWithClient(srv.Client())
	rt.client.Transport = rewriteTransport{base: srv.URL}

	result, err := rt.CryptoPrice("btc")
	if err != nil {
		t.Fatalf("CryptoPrice(btc): %v", err)
	}
	if !strings.Contains(result, "Bitcoin") {
		t.Errorf("expected 'Bitcoin' in result, got: %s", result)
	}
	if !strings.Contains(result, "-1.5%") {
		t.Errorf("expected '-1.5%%' in result, got: %s", result)
	}
}

func TestStockQuote(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data := map[string]interface{}{
			"chart": map[string]interface{}{
				"result": []interface{}{
					map[string]interface{}{
						"meta": map[string]interface{}{
							"symbol":             "AAPL",
							"shortName":          "Apple Inc.",
							"regularMarketPrice": 178.72,
							"chartPreviousClose": 176.60,
						},
					},
				},
			},
		}
		json.NewEncoder(w).Encode(data)
	}))
	defer srv.Close()

	rt := newRealTimeDataWithClient(srv.Client())
	rt.client.Transport = rewriteTransport{base: srv.URL}

	result, err := rt.StockQuote("AAPL")
	if err != nil {
		t.Fatalf("StockQuote: %v", err)
	}
	if !strings.Contains(result, "AAPL") {
		t.Errorf("expected 'AAPL' in result, got: %s", result)
	}
	if !strings.Contains(result, "Apple") {
		t.Errorf("expected 'Apple' in result, got: %s", result)
	}
	if !strings.Contains(result, "178.72") {
		t.Errorf("expected '178.72' in result, got: %s", result)
	}
}

func TestExchangeRate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data := map[string]interface{}{
			"result": "success",
			"rates": map[string]float64{
				"EUR": 0.9200,
				"GBP": 0.7900,
			},
		}
		json.NewEncoder(w).Encode(data)
	}))
	defer srv.Close()

	rt := newRealTimeDataWithClient(srv.Client())
	rt.client.Transport = rewriteTransport{base: srv.URL}

	result, err := rt.ExchangeRate("USD", "EUR")
	if err != nil {
		t.Fatalf("ExchangeRate: %v", err)
	}
	if result != "1 USD = 0.9200 EUR" {
		t.Errorf("ExchangeRate = %q, want %q", result, "1 USD = 0.9200 EUR")
	}
}

func TestExchangeRate_MissingCurrency(t *testing.T) {
	_, err := NewRealTimeData().ExchangeRate("", "EUR")
	if err == nil {
		t.Error("expected error for empty 'from' currency")
	}
}

func TestWorldTime(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data := map[string]interface{}{
			"timezone":     "America/New_York",
			"datetime":     "2026-03-29T14:30:00.000000-04:00",
			"abbreviation": "EDT",
		}
		json.NewEncoder(w).Encode(data)
	}))
	defer srv.Close()

	rt := newRealTimeDataWithClient(srv.Client())
	rt.client.Transport = rewriteTransport{base: srv.URL}

	result, err := rt.WorldTime("New York")
	if err != nil {
		t.Fatalf("WorldTime: %v", err)
	}
	if !strings.Contains(result, "New York") {
		t.Errorf("expected 'New York' in result, got: %s", result)
	}
	if !strings.Contains(result, "2:30 PM") {
		t.Errorf("expected '2:30 PM' in result, got: %s", result)
	}
	if !strings.Contains(result, "EDT") {
		t.Errorf("expected 'EDT' in result, got: %s", result)
	}
}

func TestRouteQuery_Crypto(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data := map[string]interface{}{
			"bitcoin": map[string]interface{}{
				"usd":            60000.00,
				"usd_24h_change": 1.0,
			},
		}
		json.NewEncoder(w).Encode(data)
	}))
	defer srv.Close()

	rt := newRealTimeDataWithClient(srv.Client())
	rt.client.Transport = rewriteTransport{base: srv.URL}

	result, err := rt.routeQuery(map[string]string{"query": "bitcoin price"})
	if err != nil {
		t.Fatalf("routeQuery: %v", err)
	}
	if !strings.Contains(result, "Bitcoin") {
		t.Errorf("expected 'Bitcoin' in routed result, got: %s", result)
	}
}

func TestRouteQuery_Exchange(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data := map[string]interface{}{
			"result": "success",
			"rates": map[string]float64{
				"EUR": 0.92,
			},
		}
		json.NewEncoder(w).Encode(data)
	}))
	defer srv.Close()

	rt := newRealTimeDataWithClient(srv.Client())
	rt.client.Transport = rewriteTransport{base: srv.URL}

	result, err := rt.routeQuery(map[string]string{"query": "usd to eur"})
	if err != nil {
		t.Fatalf("routeQuery: %v", err)
	}
	if !strings.Contains(result, "USD") || !strings.Contains(result, "EUR") {
		t.Errorf("expected USD/EUR in result, got: %s", result)
	}
}

func TestRouteQuery_ExplicitSub(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data := map[string]interface{}{
			"ethereum": map[string]interface{}{
				"usd":            3500.00,
				"usd_24h_change": 0.5,
			},
		}
		json.NewEncoder(w).Encode(data)
	}))
	defer srv.Close()

	rt := newRealTimeDataWithClient(srv.Client())
	rt.client.Transport = rewriteTransport{base: srv.URL}

	result, err := rt.routeQuery(map[string]string{"sub": "crypto", "symbol": "ethereum"})
	if err != nil {
		t.Fatalf("routeQuery explicit sub: %v", err)
	}
	if !strings.Contains(result, "Ethereum") {
		t.Errorf("expected 'Ethereum' in result, got: %s", result)
	}
}

func TestRouteQuery_Empty(t *testing.T) {
	rt := NewRealTimeData()
	_, err := rt.routeQuery(map[string]string{})
	if err == nil {
		t.Error("expected error for empty query")
	}
}

func TestParseCurrencyPair(t *testing.T) {
	tests := []struct {
		query    string
		wantFrom string
		wantTo   string
	}{
		{"usd to eur", "USD", "EUR"},
		{"convert gbp to jpy", "GBP", "JPY"},
		{"dollar to euro", "USD", "EUR"},
		{"exchange rate eur usd", "EUR", "USD"},
	}
	for _, tt := range tests {
		t.Run(tt.query, func(t *testing.T) {
			from, to := parseCurrencyPair(tt.query)
			if from != tt.wantFrom || to != tt.wantTo {
				t.Errorf("parseCurrencyPair(%q) = (%q, %q), want (%q, %q)",
					tt.query, from, to, tt.wantFrom, tt.wantTo)
			}
		})
	}
}

func TestExtractStockSymbol(t *testing.T) {
	tests := []struct {
		query string
		want  string
	}{
		{"AAPL stock", "AAPL"},
		{"stock price of TSLA", "TSLA"},
		{"get quote MSFT", "MSFT"},
	}
	for _, tt := range tests {
		t.Run(tt.query, func(t *testing.T) {
			got := extractStockSymbol(tt.query)
			if got != tt.want {
				t.Errorf("extractStockSymbol(%q) = %q, want %q", tt.query, got, tt.want)
			}
		})
	}
}

// rewriteTransport redirects all requests to the test server.
type rewriteTransport struct {
	base string
}

func (rt rewriteTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req.URL.Scheme = "http"
	req.URL.Host = strings.TrimPrefix(rt.base, "http://")
	return http.DefaultTransport.RoundTrip(req)
}
