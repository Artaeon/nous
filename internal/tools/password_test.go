package tools

import (
	"strings"
	"testing"
	"unicode"
)

func TestGeneratePasswordDefaultLength(t *testing.T) {
	result, err := toolPassword(map[string]string{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 16 {
		t.Errorf("expected 16 chars, got %d: %q", len(result), result)
	}
}

func TestGeneratePasswordCustomLength(t *testing.T) {
	result, err := toolPassword(map[string]string{"length": "24"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 24 {
		t.Errorf("expected 24 chars, got %d", len(result))
	}
}

func TestGeneratePasswordCharacterClasses(t *testing.T) {
	result, err := toolPassword(map[string]string{"length": "32"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	hasUpper, hasLower, hasDigit, hasSymbol := false, false, false, false
	for _, c := range result {
		switch {
		case unicode.IsUpper(c):
			hasUpper = true
		case unicode.IsLower(c):
			hasLower = true
		case unicode.IsDigit(c):
			hasDigit = true
		default:
			hasSymbol = true
		}
	}
	if !hasUpper || !hasLower || !hasDigit || !hasSymbol {
		t.Errorf("missing character class in %q (upper=%v lower=%v digit=%v symbol=%v)",
			result, hasUpper, hasLower, hasDigit, hasSymbol)
	}
}

func TestGeneratePassphrase(t *testing.T) {
	result, err := toolPassword(map[string]string{"type": "passphrase"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	words := strings.Split(result, "-")
	if len(words) != 4 {
		t.Errorf("expected 4 words, got %d: %q", len(words), result)
	}
	for _, w := range words {
		if len(w) == 0 {
			t.Error("empty word in passphrase")
		}
	}
}

func TestGeneratePassphraseCustomLength(t *testing.T) {
	result, err := toolPassword(map[string]string{"type": "passphrase", "length": "6"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	words := strings.Split(result, "-")
	if len(words) != 6 {
		t.Errorf("expected 6 words, got %d", len(words))
	}
}

func TestGeneratePIN(t *testing.T) {
	result, err := toolPassword(map[string]string{"type": "pin"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 6 {
		t.Errorf("expected 6 digits, got %d", len(result))
	}
	for _, c := range result {
		if !unicode.IsDigit(c) {
			t.Errorf("non-digit in PIN: %c", c)
		}
	}
}

func TestGenerateUniqueness(t *testing.T) {
	seen := make(map[string]bool)
	for i := 0; i < 100; i++ {
		result, err := toolPassword(map[string]string{})
		if err != nil {
			t.Fatalf("error on iteration %d: %v", i, err)
		}
		if seen[result] {
			t.Fatalf("duplicate password generated: %q", result)
		}
		seen[result] = true
	}
}

func TestPasswordToolRegistration(t *testing.T) {
	r := NewRegistry()
	RegisterPasswordTools(r)
	tool, err := r.Get("password")
	if err != nil {
		t.Fatalf("tool not registered: %v", err)
	}
	if tool.Name != "password" {
		t.Errorf("expected name 'password', got %q", tool.Name)
	}
}

func TestPasswordMinLength(t *testing.T) {
	result, err := toolPassword(map[string]string{"length": "1"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) < 4 {
		t.Errorf("expected min 4 chars, got %d", len(result))
	}
}

func TestPasswordInvalidType(t *testing.T) {
	_, err := toolPassword(map[string]string{"type": "invalid"})
	if err == nil {
		t.Error("expected error for invalid type")
	}
}
