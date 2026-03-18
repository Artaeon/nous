package tools

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// langNameToCode maps common language names to ISO 639-1 codes.
var langNameToCode = map[string]string{
	"spanish":    "es",
	"french":     "fr",
	"german":     "de",
	"japanese":   "ja",
	"chinese":    "zh",
	"korean":     "ko",
	"italian":    "it",
	"portuguese": "pt",
	"russian":    "ru",
	"arabic":     "ar",
	"hindi":      "hi",
	"english":    "en",
}

// langCodeToName maps ISO 639-1 codes to human-readable language names.
var langCodeToName = map[string]string{
	"es": "Spanish",
	"fr": "French",
	"de": "German",
	"ja": "Japanese",
	"zh": "Chinese",
	"ko": "Korean",
	"it": "Italian",
	"pt": "Portuguese",
	"ru": "Russian",
	"ar": "Arabic",
	"hi": "Hindi",
	"en": "English",
}

// ResolveLanguageCode converts a language name or code to its ISO 639-1 code.
// If the input is already a valid 2-letter code, it is returned as-is.
// If it's a known language name, the corresponding code is returned.
// Otherwise the original input is returned unchanged.
func ResolveLanguageCode(input string) string {
	lower := strings.ToLower(strings.TrimSpace(input))
	if code, ok := langNameToCode[lower]; ok {
		return code
	}
	return lower
}

// LanguageDisplayName returns a human-readable name for a language code.
// If the code is not recognized, the code itself is returned.
func LanguageDisplayName(code string) string {
	if name, ok := langCodeToName[code]; ok {
		return name
	}
	return code
}

// lingvaResponse represents the JSON response from the Lingva API.
type lingvaResponse struct {
	Translation string `json:"translation"`
}

// TranslateText translates text using the Lingva API.
func TranslateText(text, from, to string) (string, error) {
	if text == "" {
		return "", fmt.Errorf("translate: text is required")
	}
	if to == "" {
		return "", fmt.Errorf("translate: target language is required")
	}

	to = ResolveLanguageCode(to)
	from = ResolveLanguageCode(from)

	encodedText := url.PathEscape(text)
	apiURL := fmt.Sprintf("https://lingva.ml/api/v1/%s/%s/%s", from, to, encodedText)

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(apiURL)
	if err != nil {
		return "", fmt.Errorf("translate: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("translate: HTTP %d from API", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", fmt.Errorf("translate: reading response: %w", err)
	}

	translation, err := ParseTranslateResponse(body)
	if err != nil {
		return "", err
	}

	return FormatTranslation(text, translation, to), nil
}

// ParseTranslateResponse extracts the translation string from a Lingva API JSON response.
func ParseTranslateResponse(data []byte) (string, error) {
	var result lingvaResponse
	if err := json.Unmarshal(data, &result); err != nil {
		return "", fmt.Errorf("translate: invalid JSON response: %w", err)
	}
	if result.Translation == "" {
		return "", fmt.Errorf("translate: empty translation in response")
	}
	return result.Translation, nil
}

// FormatTranslation formats a translation result for display.
func FormatTranslation(original, translation, toLang string) string {
	displayName := LanguageDisplayName(toLang)
	return fmt.Sprintf("%s → %s (%s)", original, translation, displayName)
}

// RegisterTranslateTools adds the translate tool to the registry.
func RegisterTranslateTools(r *Registry) {
	r.Register(Tool{
		Name:        "translate",
		Description: "Translate text between languages. Args: text (required), to (target language code or name, required), from (source language, optional, default 'auto').",
		Execute: func(args map[string]string) (string, error) {
			text := args["text"]
			if text == "" {
				return "", fmt.Errorf("translate requires 'text' argument")
			}

			to := args["to"]
			if to == "" {
				return "", fmt.Errorf("translate requires 'to' argument (target language)")
			}

			from := args["from"]
			if from == "" {
				from = "auto"
			}

			return TranslateText(text, from, to)
		},
	})
}
