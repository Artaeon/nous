package cognitive

import "testing"

func TestNLUURLRouting(t *testing.T) {
	nlu := NewNLU()
	tests := []struct {
		input      string
		wantAction string
	}{
		{"scrape this website https://stoicera.com", "fetch_url"},
		{"fetch https://example.com", "fetch_url"},
		{"get the content from https://google.com", "fetch_url"},
		{"open https://stoicera.com", "fetch_url"},
		{"summarize https://stoicera.com", "summarize_url"},
		{"give me a summary of https://stoicera.com", "summarize_url"},
		{"tldr https://example.com", "summarize_url"},
	}
	for _, tt := range tests {
		r := nlu.Understand(tt.input)
		if r.Action != tt.wantAction {
			t.Errorf("Understand(%q): want action=%s, got action=%s (intent=%s, conf=%.2f)",
				tt.input, tt.wantAction, r.Action, r.Intent, r.Confidence)
		}
		if r.Entities["url"] == "" {
			t.Errorf("Understand(%q): no URL extracted", tt.input)
		}
	}
}
