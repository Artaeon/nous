package tools

import (
	"reflect"
	"strings"
	"testing"
)

func TestValidateNotifyUrgency(t *testing.T) {
	tests := []struct {
		urgency string
		wantErr bool
	}{
		{"low", false},
		{"normal", false},
		{"critical", false},
		{"high", true},
		{"", true},
		{"NORMAL", true}, // case-sensitive, caller should lowercase
	}

	for _, tt := range tests {
		t.Run(tt.urgency, func(t *testing.T) {
			err := ValidateNotifyUrgency(tt.urgency)
			if tt.wantErr && err == nil {
				t.Errorf("ValidateNotifyUrgency(%q) expected error", tt.urgency)
			}
			if !tt.wantErr && err != nil {
				t.Errorf("ValidateNotifyUrgency(%q) unexpected error: %v", tt.urgency, err)
			}
		})
	}
}

func TestBuildNotifyCommand(t *testing.T) {
	tests := []struct {
		name    string
		title   string
		body    string
		urgency string
		want    []string
	}{
		{
			name:    "title_only",
			title:   "Hello",
			body:    "",
			urgency: "normal",
			want:    []string{"-u", "normal", "Hello"},
		},
		{
			name:    "title_and_body",
			title:   "Alert",
			body:    "Something happened",
			urgency: "critical",
			want:    []string{"-u", "critical", "Alert", "Something happened"},
		},
		{
			name:    "low_urgency",
			title:   "Info",
			body:    "Just FYI",
			urgency: "low",
			want:    []string{"-u", "low", "Info", "Just FYI"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BuildNotifyCommand(tt.title, tt.body, tt.urgency)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("BuildNotifyCommand(%q, %q, %q) = %v, want %v", tt.title, tt.body, tt.urgency, got, tt.want)
			}
		})
	}
}

func TestNotifyToolRegistration(t *testing.T) {
	r := NewRegistry()
	RegisterNotifyTools(r)

	tool, err := r.Get("notify")
	if err != nil {
		t.Fatal("notify tool not registered")
	}

	if tool.Name != "notify" {
		t.Errorf("tool name = %q, want %q", tool.Name, "notify")
	}
}

func TestNotifyToolRequiresTitle(t *testing.T) {
	r := NewRegistry()
	RegisterNotifyTools(r)

	tool, _ := r.Get("notify")

	_, err := tool.Execute(map[string]string{})
	if err == nil {
		t.Error("expected error when title is missing")
	}
	if err != nil && !strings.Contains(err.Error(), "title") {
		t.Errorf("expected title-related error, got: %v", err)
	}
}

func TestNotifyToolEmptyTitle(t *testing.T) {
	r := NewRegistry()
	RegisterNotifyTools(r)

	tool, _ := r.Get("notify")

	_, err := tool.Execute(map[string]string{"title": ""})
	if err == nil {
		t.Error("expected error when title is empty")
	}
}

func TestNotifyToolInvalidUrgency(t *testing.T) {
	r := NewRegistry()
	RegisterNotifyTools(r)

	tool, _ := r.Get("notify")

	_, err := tool.Execute(map[string]string{"title": "Test", "urgency": "extreme"})
	if err == nil {
		t.Error("expected error for invalid urgency")
	}
}
