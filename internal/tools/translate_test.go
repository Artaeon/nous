package tools

import (
	"strings"
	"testing"
)

func TestResolveLanguageCode(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"spanish", "es"},
		{"Spanish", "es"},
		{"FRENCH", "fr"},
		{"german", "de"},
		{"japanese", "ja"},
		{"chinese", "zh"},
		{"korean", "ko"},
		{"italian", "it"},
		{"portuguese", "pt"},
		{"russian", "ru"},
		{"arabic", "ar"},
		{"hindi", "hi"},
		{"english", "en"},
		{"es", "es"},
		{"fr", "fr"},
		{"auto", "auto"},
		{"unknown", "unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := ResolveLanguageCode(tt.input)
			if got != tt.want {
				t.Errorf("ResolveLanguageCode(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestLanguageDisplayName(t *testing.T) {
	tests := []struct {
		code string
		want string
	}{
		{"es", "Spanish"},
		{"fr", "French"},
		{"de", "German"},
		{"ja", "Japanese"},
		{"xx", "xx"},
	}

	for _, tt := range tests {
		t.Run(tt.code, func(t *testing.T) {
			got := LanguageDisplayName(tt.code)
			if got != tt.want {
				t.Errorf("LanguageDisplayName(%q) = %q, want %q", tt.code, got, tt.want)
			}
		})
	}
}

func TestParseTranslateResponse(t *testing.T) {
	valid := []byte(`{"translation":"Hola mundo"}`)
	got, err := ParseTranslateResponse(valid)
	if err != nil {
		t.Fatalf("ParseTranslateResponse error: %v", err)
	}
	if got != "Hola mundo" {
		t.Errorf("got %q, want %q", got, "Hola mundo")
	}
}

func TestParseTranslateResponseEmpty(t *testing.T) {
	empty := []byte(`{"translation":""}`)
	_, err := ParseTranslateResponse(empty)
	if err == nil {
		t.Error("expected error for empty translation")
	}
}

func TestParseTranslateResponseInvalid(t *testing.T) {
	_, err := ParseTranslateResponse([]byte("not json"))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestFormatTranslation(t *testing.T) {
	got := FormatTranslation("Hello", "Hola", "es")
	if got != "Hello → Hola (Spanish)" {
		t.Errorf("FormatTranslation = %q, want %q", got, "Hello → Hola (Spanish)")
	}

	// Unknown language code should use the code itself
	got = FormatTranslation("Hello", "Merhaba", "tr")
	if !strings.Contains(got, "(tr)") {
		t.Errorf("FormatTranslation for unknown code = %q, want to contain '(tr)'", got)
	}
}

func TestRegisterTranslateTools(t *testing.T) {
	r := NewRegistry()
	RegisterTranslateTools(r)

	tool, err := r.Get("translate")
	if err != nil {
		t.Fatalf("translate tool not registered: %v", err)
	}

	if tool.Name != "translate" {
		t.Errorf("tool name = %q, want %q", tool.Name, "translate")
	}

	// Missing text should error
	_, err = tool.Execute(map[string]string{"to": "es"})
	if err == nil {
		t.Error("expected error for missing text arg")
	}

	// Missing to should error
	_, err = tool.Execute(map[string]string{"text": "hello"})
	if err == nil {
		t.Error("expected error for missing to arg")
	}
}

func TestOfflineTranslate(t *testing.T) {
	got, ok := offlineTranslate("hello", "ja")
	if !ok {
		t.Fatal("expected offline translation to exist for hello->ja")
	}
	if !strings.Contains(got, "こんにちは") {
		t.Fatalf("expected japanese greeting in offline translation, got %q", got)
	}
}

func TestTranslateTextOfflineFirst(t *testing.T) {
	got, err := TranslateText("hello", "auto", "japanese")
	if err != nil {
		t.Fatalf("TranslateText offline-first failed: %v", err)
	}
	if !strings.Contains(got, "こんにちは") || !strings.Contains(strings.ToLower(got), "offline") {
		t.Fatalf("expected offline translated output, got %q", got)
	}
}
