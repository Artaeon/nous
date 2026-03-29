package cognitive

import (
	"testing"
)

func TestExtractMath(t *testing.T) {
	se := NewSmartEntityExtractor()

	tests := []struct {
		input    string
		wantExpr string
	}{
		{"15% of 340", "0.15*340"},
		{"whats 5 + 3", "5+3"},
		{"square root of 144", "sqrt(144)"},
		{"how much is 20% tip on 85", "0.20*85"},
		{"15 times 7", "15*7"},
		{"100 divided by 4", "100/4"},
		{"calculate 12 plus 8", "12+8"},
		{"what is 9 minus 3", "9-3"},
		{"2 to the power of 10", "2^10"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			entities := make(map[string]string)
			se.ExtractForIntent(tt.input, "calculate", entities)
			got := entities["expression"]
			if got != tt.wantExpr {
				t.Errorf("ExtractForIntent(%q, calculate) expression = %q, want %q", tt.input, got, tt.wantExpr)
			}
		})
	}
}

func TestExtractTranslate(t *testing.T) {
	se := NewSmartEntityExtractor()

	tests := []struct {
		input    string
		wantText string
		wantLang string
	}{
		{"translate hello to japanese", "hello", "japanese"},
		{"how do you say goodbye in french", "goodbye", "french"},
		{"whats hello in spanish", "hello", "spanish"},
		{"translate good morning into german", "good morning", "german"},
		{"please translate thank you to italian", "thank you", "italian"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			entities := make(map[string]string)
			se.ExtractForIntent(tt.input, "translate", entities)
			gotText := entities["text"]
			gotLang := entities["to"]
			if gotText != tt.wantText {
				t.Errorf("text = %q, want %q", gotText, tt.wantText)
			}
			if gotLang != tt.wantLang {
				t.Errorf("to = %q, want %q", gotLang, tt.wantLang)
			}
		})
	}
}

func TestExtractTimer(t *testing.T) {
	se := NewSmartEntityExtractor()

	tests := []struct {
		input        string
		wantDuration string
	}{
		{"set a timer for 5 minutes", "5m"},
		{"remind me in 10 min", "10m"},
		{"timer 30 seconds", "30s"},
		{"set timer for 2 hours", "2h"},
		{"countdown 90 secs", "90s"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			entities := make(map[string]string)
			se.ExtractForIntent(tt.input, "timer", entities)
			got := entities["duration"]
			if got != tt.wantDuration {
				t.Errorf("duration = %q, want %q", got, tt.wantDuration)
			}
		})
	}
}

func TestExtractConvert(t *testing.T) {
	se := NewSmartEntityExtractor()

	tests := []struct {
		input      string
		wantValue  string
		wantFrom   string
		wantToUnit string
	}{
		{"convert 5 miles to km", "5", "miles", "km"},
		{"how many feet in a mile", "1", "mile", "feet"},
		{"10 celsius to fahrenheit", "10", "celsius", "fahrenheit"},
		{"convert 100 kg to pounds", "100", "kg", "pounds"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			entities := make(map[string]string)
			se.ExtractForIntent(tt.input, "convert", entities)
			if got := entities["value"]; got != tt.wantValue {
				t.Errorf("value = %q, want %q", got, tt.wantValue)
			}
			if got := entities["from"]; got != tt.wantFrom {
				t.Errorf("from = %q, want %q", got, tt.wantFrom)
			}
			if got := entities["to_unit"]; got != tt.wantToUnit {
				t.Errorf("to_unit = %q, want %q", got, tt.wantToUnit)
			}
		})
	}
}

func TestExtractWeather(t *testing.T) {
	se := NewSmartEntityExtractor()

	tests := []struct {
		input        string
		wantLocation string
		wantDate     string
	}{
		{"whats the weather in paris", "paris", ""},
		{"is it going to rain tomorrow", "", "tomorrow"},
		{"weather for london", "london", ""},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			entities := make(map[string]string)
			se.ExtractForIntent(tt.input, "weather", entities)
			if got := entities["location"]; got != tt.wantLocation {
				t.Errorf("location = %q, want %q", got, tt.wantLocation)
			}
			if tt.wantDate != "" {
				if got := entities["date"]; got != tt.wantDate {
					t.Errorf("date = %q, want %q", got, tt.wantDate)
				}
			}
		})
	}
}

func TestExtractSkipsExistingEntities(t *testing.T) {
	se := NewSmartEntityExtractor()

	// If expression is already set, don't overwrite
	entities := map[string]string{"expression": "2+2"}
	se.ExtractForIntent("15% of 340", "calculate", entities)
	if entities["expression"] != "2+2" {
		t.Errorf("should not overwrite existing expression, got %q", entities["expression"])
	}

	// If text+to are already set, don't overwrite
	entities = map[string]string{"text": "hi", "to": "french"}
	se.ExtractForIntent("translate hello to japanese", "translate", entities)
	if entities["text"] != "hi" || entities["to"] != "french" {
		t.Error("should not overwrite existing translate entities")
	}
}
