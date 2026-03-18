package tools

import (
	"math"
	"testing"
)

func almostEqual(a, b, epsilon float64) bool {
	return math.Abs(a-b) < epsilon
}

func TestConvertLength(t *testing.T) {
	tests := []struct {
		value    float64
		from, to string
		expected float64
	}{
		{1, "km", "m", 1000},
		{1, "mi", "km", 1.609344},
		{12, "in", "ft", 1},
		{1, "yd", "ft", 3},
		{1000, "mm", "m", 1},
		{100, "cm", "m", 1},
		{1, "nm", "m", 1852},
	}

	for _, tt := range tests {
		result, err := ConvertUnits(tt.value, tt.from, tt.to)
		if err != nil {
			t.Errorf("ConvertUnits(%v, %s, %s) error: %v", tt.value, tt.from, tt.to, err)
			continue
		}
		if !almostEqual(result, tt.expected, 0.001) {
			t.Errorf("ConvertUnits(%v, %s, %s) = %v, want %v", tt.value, tt.from, tt.to, result, tt.expected)
		}
	}
}

func TestConvertWeight(t *testing.T) {
	tests := []struct {
		value    float64
		from, to string
		expected float64
	}{
		{1, "kg", "g", 1000},
		{1, "lb", "oz", 16},
		{1000, "mg", "g", 1},
		{1, "kg", "lb", 2.20462},
		{1, "stone", "lb", 14},
	}

	for _, tt := range tests {
		result, err := ConvertUnits(tt.value, tt.from, tt.to)
		if err != nil {
			t.Errorf("ConvertUnits(%v, %s, %s) error: %v", tt.value, tt.from, tt.to, err)
			continue
		}
		if !almostEqual(result, tt.expected, 0.01) {
			t.Errorf("ConvertUnits(%v, %s, %s) = %v, want %v", tt.value, tt.from, tt.to, result, tt.expected)
		}
	}
}

func TestConvertTemperature(t *testing.T) {
	tests := []struct {
		value    float64
		from, to string
		expected float64
	}{
		{0, "c", "f", 32},
		{100, "c", "f", 212},
		{32, "f", "c", 0},
		{212, "f", "c", 100},
		{0, "c", "k", 273.15},
		{273.15, "k", "c", 0},
		{-40, "c", "f", -40},
		{72, "f", "c", 22.2222},
	}

	for _, tt := range tests {
		result, err := ConvertUnits(tt.value, tt.from, tt.to)
		if err != nil {
			t.Errorf("ConvertUnits(%v, %s, %s) error: %v", tt.value, tt.from, tt.to, err)
			continue
		}
		if !almostEqual(result, tt.expected, 0.01) {
			t.Errorf("ConvertUnits(%v, %s, %s) = %v, want %v", tt.value, tt.from, tt.to, result, tt.expected)
		}
	}
}

func TestConvertVolume(t *testing.T) {
	tests := []struct {
		value    float64
		from, to string
		expected float64
	}{
		{1, "l", "ml", 1000},
		{1, "gal", "l", 3.78541},
		{1, "cup", "tbsp", 16},
		{1, "tbsp", "tsp", 3},
	}

	for _, tt := range tests {
		result, err := ConvertUnits(tt.value, tt.from, tt.to)
		if err != nil {
			t.Errorf("ConvertUnits(%v, %s, %s) error: %v", tt.value, tt.from, tt.to, err)
			continue
		}
		if !almostEqual(result, tt.expected, 0.01) {
			t.Errorf("ConvertUnits(%v, %s, %s) = %v, want %v", tt.value, tt.from, tt.to, result, tt.expected)
		}
	}
}

func TestConvertSpeed(t *testing.T) {
	tests := []struct {
		value    float64
		from, to string
		expected float64
	}{
		{100, "km/h", "mph", 62.1371},
		{60, "mph", "km/h", 96.5606},
		{1, "m/s", "km/h", 3.6},
		{1, "knots", "km/h", 1.852},
	}

	for _, tt := range tests {
		result, err := ConvertUnits(tt.value, tt.from, tt.to)
		if err != nil {
			t.Errorf("ConvertUnits(%v, %s, %s) error: %v", tt.value, tt.from, tt.to, err)
			continue
		}
		if !almostEqual(result, tt.expected, 0.01) {
			t.Errorf("ConvertUnits(%v, %s, %s) = %v, want %v", tt.value, tt.from, tt.to, result, tt.expected)
		}
	}
}

func TestConvertData(t *testing.T) {
	tests := []struct {
		value    float64
		from, to string
		expected float64
	}{
		{1, "kb", "b", 1024},
		{1, "mb", "kb", 1024},
		{1, "gb", "mb", 1024},
		{1, "tb", "gb", 1024},
		{1, "pb", "tb", 1024},
	}

	for _, tt := range tests {
		result, err := ConvertUnits(tt.value, tt.from, tt.to)
		if err != nil {
			t.Errorf("ConvertUnits(%v, %s, %s) error: %v", tt.value, tt.from, tt.to, err)
			continue
		}
		if !almostEqual(result, tt.expected, 0.001) {
			t.Errorf("ConvertUnits(%v, %s, %s) = %v, want %v", tt.value, tt.from, tt.to, result, tt.expected)
		}
	}
}

func TestConvertTime(t *testing.T) {
	tests := []struct {
		value    float64
		from, to string
		expected float64
	}{
		{1, "h", "min", 60},
		{1, "day", "h", 24},
		{1, "week", "day", 7},
		{60, "s", "min", 1},
		{1, "year", "day", 365},
		{1, "month", "day", 30},
	}

	for _, tt := range tests {
		result, err := ConvertUnits(tt.value, tt.from, tt.to)
		if err != nil {
			t.Errorf("ConvertUnits(%v, %s, %s) error: %v", tt.value, tt.from, tt.to, err)
			continue
		}
		if !almostEqual(result, tt.expected, 0.001) {
			t.Errorf("ConvertUnits(%v, %s, %s) = %v, want %v", tt.value, tt.from, tt.to, result, tt.expected)
		}
	}
}

func TestConvertArea(t *testing.T) {
	tests := []struct {
		value    float64
		from, to string
		expected float64
	}{
		{1, "km2", "m2", 1e6},
		{1, "hectare", "m2", 10000},
		{1, "acre", "m2", 4046.8564224},
		{1, "acre", "hectare", 0.404686},
	}

	for _, tt := range tests {
		result, err := ConvertUnits(tt.value, tt.from, tt.to)
		if err != nil {
			t.Errorf("ConvertUnits(%v, %s, %s) error: %v", tt.value, tt.from, tt.to, err)
			continue
		}
		if !almostEqual(result, tt.expected, 0.01) {
			t.Errorf("ConvertUnits(%v, %s, %s) = %v, want %v", tt.value, tt.from, tt.to, result, tt.expected)
		}
	}
}

func TestConvertSameUnit(t *testing.T) {
	result, err := ConvertUnits(42, "km", "km")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != 42 {
		t.Errorf("same unit conversion: got %v, want 42", result)
	}
}

func TestConvertIncompatibleUnits(t *testing.T) {
	_, err := ConvertUnits(1, "km", "kg")
	if err == nil {
		t.Error("expected error for incompatible units, got nil")
	}
}

func TestConvertUnknownUnit(t *testing.T) {
	_, err := ConvertUnits(1, "foobar", "m")
	if err == nil {
		t.Error("expected error for unknown unit, got nil")
	}
}

func TestConvertTemperatureToNonTemperature(t *testing.T) {
	_, err := ConvertUnits(1, "c", "km")
	if err == nil {
		t.Error("expected error for temperature to non-temperature, got nil")
	}
}

func TestParseConversion(t *testing.T) {
	tests := []struct {
		input         string
		wantValue     float64
		wantFrom      string
		wantTo        string
	}{
		{"100 km to miles", 100, "km", "mi"},
		{"5 pounds in kg", 5, "lb", "kg"},
		{"72 fahrenheit to celsius", 72, "f", "c"},
		{"3.5 liters to gallons", 3.5, "l", "gal"},
		{"1024 bytes to kb", 1024, "b", "kb"},
	}

	for _, tt := range tests {
		value, from, to, err := ParseConversion(tt.input)
		if err != nil {
			t.Errorf("ParseConversion(%q) error: %v", tt.input, err)
			continue
		}
		if !almostEqual(value, tt.wantValue, 0.001) {
			t.Errorf("ParseConversion(%q) value = %v, want %v", tt.input, value, tt.wantValue)
		}
		if from != tt.wantFrom {
			t.Errorf("ParseConversion(%q) from = %q, want %q", tt.input, from, tt.wantFrom)
		}
		if to != tt.wantTo {
			t.Errorf("ParseConversion(%q) to = %q, want %q", tt.input, to, tt.wantTo)
		}
	}
}

func TestParseConversionInvalid(t *testing.T) {
	_, _, _, err := ParseConversion("hello world")
	if err == nil {
		t.Error("expected error for unparseable input")
	}
}

func TestFormatConversion(t *testing.T) {
	result := FormatConversion(100, "km", 62.1371, "mi")
	if result != "100 km = 62.1371 mi" {
		t.Errorf("FormatConversion got %q", result)
	}
}

func TestCaseInsensitiveUnits(t *testing.T) {
	result, err := ConvertUnits(1, "KM", "M")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !almostEqual(result, 1000, 0.001) {
		t.Errorf("case insensitive: got %v, want 1000", result)
	}
}
