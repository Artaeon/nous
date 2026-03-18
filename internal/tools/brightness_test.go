package tools

import (
	"testing"
)

func TestCalculateBrightnessPercent(t *testing.T) {
	tests := []struct {
		name    string
		current int
		max     int
		want    int
	}{
		{name: "half", current: 500, max: 1000, want: 50},
		{name: "full", current: 1000, max: 1000, want: 100},
		{name: "zero", current: 0, max: 1000, want: 0},
		{name: "quarter", current: 250, max: 1000, want: 25},
		{name: "max_zero", current: 100, max: 0, want: 0},
		{name: "over_max", current: 1500, max: 1000, want: 100},
		{name: "negative_current", current: -10, max: 1000, want: 0},
		{name: "small_values", current: 1, max: 3, want: 33},
		{name: "exact_max", current: 255, max: 255, want: 100},
		{name: "typical_laptop", current: 120, max: 255, want: 47},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CalculateBrightnessPercent(tt.current, tt.max)
			if got != tt.want {
				t.Errorf("CalculateBrightnessPercent(%d, %d) = %d, want %d", tt.current, tt.max, got, tt.want)
			}
		})
	}
}

func TestBrightnessToolRegistration(t *testing.T) {
	r := NewRegistry()
	RegisterBrightnessTools(r)

	tool, err := r.Get("brightness")
	if err != nil {
		t.Fatal("brightness tool not registered")
	}

	if tool.Name != "brightness" {
		t.Errorf("tool name = %q, want %q", tool.Name, "brightness")
	}
}

func TestBrightnessToolSetRequiresLevel(t *testing.T) {
	r := NewRegistry()
	RegisterBrightnessTools(r)

	tool, _ := r.Get("brightness")

	_, err := tool.Execute(map[string]string{"action": "set"})
	if err == nil {
		t.Error("expected error when set action has no level")
	}
}

func TestBrightnessToolSetInvalidLevel(t *testing.T) {
	r := NewRegistry()
	RegisterBrightnessTools(r)

	tool, _ := r.Get("brightness")

	_, err := tool.Execute(map[string]string{"action": "set", "level": "abc"})
	if err == nil {
		t.Error("expected error for non-numeric level")
	}
}

func TestBrightnessToolInvalidStep(t *testing.T) {
	r := NewRegistry()
	RegisterBrightnessTools(r)

	tool, _ := r.Get("brightness")

	_, err := tool.Execute(map[string]string{"action": "up", "step": "notanumber"})
	if err == nil {
		t.Error("expected error for non-numeric step")
	}
}

func TestBrightnessToolUnknownAction(t *testing.T) {
	r := NewRegistry()
	RegisterBrightnessTools(r)

	tool, _ := r.Get("brightness")

	_, err := tool.Execute(map[string]string{"action": "explode"})
	if err == nil {
		t.Error("expected error for unknown action")
	}
}
