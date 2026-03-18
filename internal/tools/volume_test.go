package tools

import (
	"testing"
)

func TestParseVolumeOutput(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    int
		wantErr bool
	}{
		{
			name:  "typical_stereo",
			input: "Volume: front-left: 32768 /  50% / -18.06 dB,   front-right: 32768 /  50% / -18.06 dB",
			want:  50,
		},
		{
			name:  "full_volume",
			input: "Volume: front-left: 65536 / 100% / 0.00 dB,   front-right: 65536 / 100% / 0.00 dB",
			want:  100,
		},
		{
			name:  "zero_volume",
			input: "Volume: front-left: 0 /   0% / -inf dB,   front-right: 0 /   0% / -inf dB",
			want:  0,
		},
		{
			name:  "mono",
			input: "Volume: mono: 42598 /  65% / -11.22 dB",
			want:  65,
		},
		{
			name:    "empty",
			input:   "",
			wantErr: true,
		},
		{
			name:    "no_percentage",
			input:   "Volume: front-left: 32768",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseVolumeOutput(tt.input)
			if tt.wantErr {
				if err == nil {
					t.Errorf("ParseVolumeOutput(%q) expected error, got %d", tt.input, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseVolumeOutput(%q) error: %v", tt.input, err)
			}
			if got != tt.want {
				t.Errorf("ParseVolumeOutput(%q) = %d, want %d", tt.input, got, tt.want)
			}
		})
	}
}

func TestParseMuteOutput(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    bool
		wantErr bool
	}{
		{
			name:  "muted",
			input: "Mute: yes",
			want:  true,
		},
		{
			name:  "unmuted",
			input: "Mute: no",
			want:  false,
		},
		{
			name:  "muted_with_whitespace",
			input: "  Mute: yes  \n",
			want:  true,
		},
		{
			name:    "empty",
			input:   "",
			wantErr: true,
		},
		{
			name:    "garbage",
			input:   "something unexpected",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseMuteOutput(tt.input)
			if tt.wantErr {
				if err == nil {
					t.Errorf("ParseMuteOutput(%q) expected error, got %v", tt.input, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseMuteOutput(%q) error: %v", tt.input, err)
			}
			if got != tt.want {
				t.Errorf("ParseMuteOutput(%q) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestVolumeToolRegistration(t *testing.T) {
	r := NewRegistry()
	RegisterVolumeTools(r)

	tool, err := r.Get("volume")
	if err != nil {
		t.Fatal("volume tool not registered")
	}

	if tool.Name != "volume" {
		t.Errorf("tool name = %q, want %q", tool.Name, "volume")
	}
}

func TestVolumeToolSetRequiresLevel(t *testing.T) {
	r := NewRegistry()
	RegisterVolumeTools(r)

	tool, _ := r.Get("volume")

	_, err := tool.Execute(map[string]string{"action": "set"})
	if err == nil {
		t.Error("expected error when set action has no level")
	}
}

func TestVolumeToolSetInvalidLevel(t *testing.T) {
	r := NewRegistry()
	RegisterVolumeTools(r)

	tool, _ := r.Get("volume")

	_, err := tool.Execute(map[string]string{"action": "set", "level": "abc"})
	if err == nil {
		t.Error("expected error for non-numeric level")
	}
}

func TestVolumeToolUnknownAction(t *testing.T) {
	r := NewRegistry()
	RegisterVolumeTools(r)

	tool, _ := r.Get("volume")

	_, err := tool.Execute(map[string]string{"action": "explode"})
	if err == nil {
		t.Error("expected error for unknown action")
	}
}
