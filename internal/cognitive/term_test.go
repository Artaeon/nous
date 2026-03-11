package cognitive

import (
	"strings"
	"testing"
)

func TestStyledWrapsTextWithColorCodes(t *testing.T) {
	result := Styled(ColorRed, "error")
	want := ColorRed + "error" + ColorReset
	if result != want {
		t.Errorf("Styled(ColorRed, \"error\") = %q, want %q", result, want)
	}
}

func TestStyledWithDifferentColors(t *testing.T) {
	tests := []struct {
		color string
		text  string
	}{
		{ColorGreen, "success"},
		{ColorBlue, "info"},
		{ColorYellow, "warning"},
		{ColorCyan, "highlight"},
		{ColorMagenta, "special"},
		{ColorGray, "dimmed"},
		{ColorBold, "bold text"},
	}

	for _, tt := range tests {
		result := Styled(tt.color, tt.text)
		if !strings.HasPrefix(result, tt.color) {
			t.Errorf("Styled result should start with color code for %q", tt.text)
		}
		if !strings.HasSuffix(result, ColorReset) {
			t.Errorf("Styled result should end with reset code for %q", tt.text)
		}
		if !strings.Contains(result, tt.text) {
			t.Errorf("Styled result should contain the text %q", tt.text)
		}
	}
}

func TestBannerContainsModelName(t *testing.T) {
	banner := Banner("0.3.0", "qwen2.5:1.5b", "localhost:11434", 9, 64)

	if !strings.Contains(banner, "qwen2.5:1.5b") {
		t.Error("banner should contain model name")
	}
}

func TestBannerContainsHost(t *testing.T) {
	banner := Banner("0.3.0", "qwen2.5:1.5b", "localhost:11434", 9, 64)

	if !strings.Contains(banner, "localhost:11434") {
		t.Error("banner should contain host")
	}
}

func TestBannerContainsToolCount(t *testing.T) {
	banner := Banner("0.3.0", "qwen2.5:1.5b", "localhost:11434", 9, 64)

	if !strings.Contains(banner, "9 built-ins") {
		t.Error("banner should contain tool count")
	}
}

func TestBannerContainsVersion(t *testing.T) {
	banner := Banner("0.3.0", "qwen2.5:1.5b", "localhost:11434", 9, 64)

	if !strings.Contains(banner, "v0.3.0") {
		t.Error("banner should contain version with 'v' prefix")
	}
}

func TestBannerContainsMemorySlots(t *testing.T) {
	banner := Banner("0.3.0", "qwen2.5:1.5b", "localhost:11434", 9, 64)

	if !strings.Contains(banner, "64 working slots") {
		t.Error("banner should contain memory slot count")
	}
}

func TestBannerContainsNousName(t *testing.T) {
	banner := Banner("0.3.0", "qwen2.5:1.5b", "localhost:11434", 9, 64)

	if !strings.Contains(banner, "nous") {
		t.Error("banner should contain nous")
	}
}

func TestBannerHasSeparator(t *testing.T) {
	banner := Banner("0.3.0", "qwen2.5:1.5b", "localhost:11434", 9, 64)

	if !strings.Contains(banner, "╭") || !strings.Contains(banner, "╰") {
		t.Error("banner should have separator line")
	}
}

func TestPromptContainsNousName(t *testing.T) {
	prompt := Prompt()
	if !strings.Contains(prompt, "nous") {
		t.Error("prompt should contain the product name")
	}
}

func TestSectionIncludesTitle(t *testing.T) {
	section := Section("Status")
	if !strings.Contains(section, "Status") {
		t.Error("section should contain title")
	}
	if !strings.Contains(section, "─") {
		t.Error("section should include separator glyphs")
	}
}

func TestPanelIncludesTitleAndLines(t *testing.T) {
	panel := Panel("Quickstart", []string{"/help", "/dashboard"})
	if !strings.Contains(panel, "Quickstart") {
		t.Error("panel should contain title")
	}
	if !strings.Contains(panel, "/help") || !strings.Contains(panel, "/dashboard") {
		t.Error("panel should contain provided lines")
	}
	if !strings.Contains(panel, "╭") || !strings.Contains(panel, "╰") {
		t.Error("panel should render a box")
	}
}

func TestVisibleLenCountsVisibleCharacters(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  int
	}{
		{"plain text", "hello world", 11},
		{"empty string", "", 0},
		{"colored text", "\033[31mred\033[0m", 3},
		{"bold colored", "\033[1m\033[36mtext\033[0m", 4},
		{"multiple escapes", "\033[31ma\033[0m\033[32mb\033[0m", 2},
		{"escape only", "\033[31m\033[0m", 0},
		{"unicode text", "νοῦς", 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := visibleLen(tt.input)
			if got != tt.want {
				t.Errorf("visibleLen(%q) = %d, want %d", tt.input, got, tt.want)
			}
		})
	}
}

func TestNewSpinnerCreatesValidSpinner(t *testing.T) {
	s := NewSpinner()

	if s == nil {
		t.Fatal("NewSpinner should not return nil")
	}
	if len(s.frames) == 0 {
		t.Error("spinner should have at least one frame")
	}
	if s.frames[0] != "⠋" {
		t.Errorf("first frame should be ⠋, got %q", s.frames[0])
	}
	// done and stopped channels are lazily created in Start()
	if s.running {
		t.Error("spinner should not be running initially")
	}
	if s.current != 0 {
		t.Error("spinner current index should start at 0")
	}
}
