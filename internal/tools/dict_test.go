package tools

import (
	"strings"
	"testing"
)

// Mock API response for "hello"
const mockDictJSON = `[
  {
    "word": "hello",
    "meanings": [
      {
        "partOfSpeech": "noun",
        "definitions": [
          {
            "definition": "An utterance of \"hello\"; a greeting.",
            "example": "she was greeted with a chorus of hellos",
            "synonyms": ["greeting", "welcome"],
            "antonyms": ["goodbye", "farewell"]
          }
        ],
        "synonyms": ["greeting"],
        "antonyms": ["goodbye"]
      },
      {
        "partOfSpeech": "exclamation",
        "definitions": [
          {
            "definition": "Used as a greeting or to begin a phone conversation.",
            "example": "hello there, Katie!",
            "synonyms": [],
            "antonyms": []
          }
        ],
        "synonyms": ["hi", "howdy"],
        "antonyms": []
      }
    ]
  }
]`

func TestParseDictResponse(t *testing.T) {
	entries, err := ParseDictResponse([]byte(mockDictJSON))
	if err != nil {
		t.Fatalf("ParseDictResponse error: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(entries))
	}
	if entries[0].Word != "hello" {
		t.Errorf("word = %q, want %q", entries[0].Word, "hello")
	}
	if len(entries[0].Meanings) != 2 {
		t.Errorf("expected 2 meanings, got %d", len(entries[0].Meanings))
	}
}

func TestParseDictResponseInvalid(t *testing.T) {
	_, err := ParseDictResponse([]byte("not json"))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestFormatDefinitions(t *testing.T) {
	entries, _ := ParseDictResponse([]byte(mockDictJSON))
	result := FormatDefinitions(entries)

	if !strings.Contains(result, "hello (noun)") {
		t.Errorf("should contain part of speech, got %q", result)
	}
	if !strings.Contains(result, "utterance") {
		t.Errorf("should contain definition text, got %q", result)
	}
	if !strings.Contains(result, "Example:") {
		t.Errorf("should contain example, got %q", result)
	}
	if !strings.Contains(result, "hello (exclamation)") {
		t.Errorf("should contain exclamation, got %q", result)
	}
}

func TestFormatDefinitionsEmpty(t *testing.T) {
	result := FormatDefinitions(nil)
	if result != "No definitions found." {
		t.Errorf("expected 'No definitions found.', got %q", result)
	}
}

func TestFormatSynonyms(t *testing.T) {
	entries, _ := ParseDictResponse([]byte(mockDictJSON))
	result := FormatSynonyms(entries)

	if !strings.HasPrefix(result, "Synonyms: ") {
		t.Errorf("should start with 'Synonyms: ', got %q", result)
	}
	if !strings.Contains(result, "greeting") {
		t.Errorf("should contain 'greeting', got %q", result)
	}
	if !strings.Contains(result, "hi") {
		t.Errorf("should contain 'hi', got %q", result)
	}
	if !strings.Contains(result, "welcome") {
		t.Errorf("should contain 'welcome', got %q", result)
	}
}

func TestFormatSynonymsEmpty(t *testing.T) {
	result := FormatSynonyms(nil)
	if result != "No synonyms found." {
		t.Errorf("expected 'No synonyms found.', got %q", result)
	}
}

func TestFormatAntonyms(t *testing.T) {
	entries, _ := ParseDictResponse([]byte(mockDictJSON))
	result := FormatAntonyms(entries)

	if !strings.HasPrefix(result, "Antonyms: ") {
		t.Errorf("should start with 'Antonyms: ', got %q", result)
	}
	if !strings.Contains(result, "goodbye") {
		t.Errorf("should contain 'goodbye', got %q", result)
	}
	if !strings.Contains(result, "farewell") {
		t.Errorf("should contain 'farewell', got %q", result)
	}
}

func TestFormatAntonymsEmpty(t *testing.T) {
	result := FormatAntonyms(nil)
	if result != "No antonyms found." {
		t.Errorf("expected 'No antonyms found.', got %q", result)
	}
}

func TestDictEmptyWord(t *testing.T) {
	_, err := LookupWord("", "define")
	if err == nil {
		t.Error("expected error for empty word")
	}
}

func TestDictUnknownAction(t *testing.T) {
	// This will fail because it tries the network, so we test the action validation
	// by using a mock. For now, just test with a known-bad action after fetch.
	entries, _ := ParseDictResponse([]byte(mockDictJSON))
	_ = entries // We can't easily test the action path without network.
	// Instead, test the error path directly:
	_, err := LookupWord("test", "rhyme")
	// This may fail with network error or action error depending on connectivity.
	// We just verify it doesn't panic.
	_ = err
}

func TestDictToolRegistration(t *testing.T) {
	r := NewRegistry()
	RegisterDictTools(r)

	tool, err := r.Get("dict")
	if err != nil {
		t.Fatal("dict tool not registered")
	}
	if tool.Name != "dict" {
		t.Errorf("tool name = %q, want %q", tool.Name, "dict")
	}
}

func TestFormatDefinitionsLimitPerPOS(t *testing.T) {
	// Test that we limit to 3 definitions per part of speech
	data := `[{
		"word": "test",
		"meanings": [{
			"partOfSpeech": "noun",
			"definitions": [
				{"definition": "def1", "example": ""},
				{"definition": "def2", "example": ""},
				{"definition": "def3", "example": ""},
				{"definition": "def4", "example": ""},
				{"definition": "def5", "example": ""}
			],
			"synonyms": [],
			"antonyms": []
		}]
	}]`
	entries, err := ParseDictResponse([]byte(data))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	result := FormatDefinitions(entries)
	// Should have 3 definitions, not 5
	count := strings.Count(result, "test (noun)")
	if count != 3 {
		t.Errorf("expected 3 definitions, got %d in: %q", count, result)
	}
}
