package cognitive

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

// Terminal color codes for consistent styling across Nous.

const (
	ColorReset   = "\033[0m"
	ColorBold    = "\033[1m"
	ColorDim     = "\033[2m"
	ColorWhite   = "\033[97m"
	ColorRed     = "\033[31m"
	ColorGreen   = "\033[32m"
	ColorYellow  = "\033[33m"
	ColorBlue    = "\033[34m"
	ColorMagenta = "\033[35m"
	ColorCyan    = "\033[36m"
	ColorGray    = "\033[90m"
)

// Styled returns text wrapped in the given color code.
func Styled(color, text string) string {
	return color + text + ColorReset
}

// Banner returns the Nous startup banner with version info.
// Clean minimal design inspired by modern CLI tools.
func Banner(version, model, host string, toolCount int, memSlots int) string {
	var b strings.Builder
	b.WriteString("\n")
	b.WriteString(fmt.Sprintf("  %s╭──────────────────────────────────────────────╮%s\n", ColorDim, ColorReset))
	b.WriteString(fmt.Sprintf("  %s│%s %sNous%s %sv%s%-31s%s│%s\n", ColorDim, ColorReset, ColorBold, ColorReset, ColorDim, version, "", ColorDim, ColorReset))
	b.WriteString(fmt.Sprintf("  %s│%s %smodel%s   %s%-35s%s│%s\n", ColorDim, ColorReset, ColorGray, ColorReset, ColorCyan, truncateDisplay(model, 35), ColorDim, ColorReset))
	b.WriteString(fmt.Sprintf("  %s│%s %shost%s    %s%-35s%s│%s\n", ColorDim, ColorReset, ColorGray, ColorReset, ColorGray, truncateDisplay(host, 35), ColorDim, ColorReset))
	b.WriteString(fmt.Sprintf("  %s│%s %stools%s   %s%-35s%s│%s\n", ColorDim, ColorReset, ColorGray, ColorReset, ColorWhite, fmt.Sprintf("%d built-ins", toolCount), ColorDim, ColorReset))
	b.WriteString(fmt.Sprintf("  %s│%s %smemory%s  %s%-35s%s│%s\n", ColorDim, ColorReset, ColorGray, ColorReset, ColorWhite, fmt.Sprintf("%d working slots", memSlots), ColorDim, ColorReset))
	b.WriteString(fmt.Sprintf("  %s╰──────────────────────────────────────────────╯%s\n", ColorDim, ColorReset))
	b.WriteString("\n")
	return b.String()
}

// Prompt returns the REPL prompt string.
func Prompt() string {
	return ColorGray + "  nous" + ColorReset + ColorCyan + " › " + ColorReset
}

// Separator returns a thin horizontal line for visual breaks.
func Separator() string {
	return "  " + ColorDim + strings.Repeat("─", 36) + ColorReset + "\n"
}

// Section returns a titled section header for CLI screens.
func Section(title string) string {
	return fmt.Sprintf("  %s%s%s %s%s%s\n", ColorBold, title, ColorReset, ColorDim, strings.Repeat("─", max(1, 36-len(title))), ColorReset)
}

// KeyValue formats a labeled value row.
func KeyValue(label, value string) string {
	return fmt.Sprintf("  %s%-14s%s %s\n", ColorGray, label, ColorReset, value)
}

// Panel renders a lightweight box with a title and lines.
func Panel(title string, lines []string) string {
	width := 44
	for _, line := range lines {
		if l := visibleLen(line); l+2 > width {
			width = l + 2
		}
	}

	var b strings.Builder
	b.WriteString(fmt.Sprintf("  %s╭─ %s%s%s %s╮%s\n", ColorDim, ColorBold, title, ColorReset, strings.Repeat("─", max(1, width-visibleLen(title)-2)), ColorReset))
	for _, line := range lines {
		padding := width - visibleLen(line)
		if padding < 0 {
			padding = 0
		}
		b.WriteString(fmt.Sprintf("  %s│%s %s%s %s│%s\n", ColorDim, ColorReset, line, strings.Repeat(" ", padding), ColorDim, ColorReset))
	}
	b.WriteString(fmt.Sprintf("  %s╰%s╯%s\n", ColorDim, strings.Repeat("─", width+2), ColorReset))
	return b.String()
}

// ToolStatus formats a tool call status line.
// Format: "  tool_name  args  duration"
func ToolStatus(name, args string, duration time.Duration) string {
	d := formatDurationShort(duration)
	if args != "" {
		// Truncate long args
		if len(args) > 40 {
			args = args[:40] + "..."
		}
		return fmt.Sprintf("  %s%s%s  %s%s%s  %s%s%s", ColorMagenta, name, ColorReset, ColorGray, args, ColorReset, ColorDim, d, ColorReset)
	}
	return fmt.Sprintf("  %s%s%s  %s%s%s", ColorMagenta, name, ColorReset, ColorDim, d, ColorReset)
}

// TimingFooter returns a compact timing line.
func TimingFooter(duration time.Duration) string {
	return fmt.Sprintf("  %s%s%s", ColorDim, formatDurationShort(duration), ColorReset)
}

func formatDurationShort(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	return fmt.Sprintf("%.1fs", d.Seconds())
}

func truncateDisplay(s string, maxLen int) string {
	if visibleLen(s) <= maxLen {
		return s
	}
	if maxLen <= 1 {
		return s[:maxLen]
	}
	return s[:maxLen-1] + "…"
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// visibleLen returns the display width of a string, stripping ANSI escapes.
func visibleLen(s string) int {
	n := 0
	inEsc := false
	for _, r := range s {
		if r == '\033' {
			inEsc = true
			continue
		}
		if inEsc {
			if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
				inEsc = false
			}
			continue
		}
		n++
	}
	return n
}

// Spinner provides a simple terminal spinner for long operations.
type Spinner struct {
	frames  []string
	current int
	done    chan struct{}
	stopped chan struct{}
	mu      sync.Mutex
	running bool
}

// NewSpinner creates a new Spinner with braille-dot frames.
func NewSpinner() *Spinner {
	return &Spinner{
		frames: []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"},
	}
}

// Start begins spinning in a background goroutine with the given label.
func (s *Spinner) Start(label string) {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return
	}
	s.running = true
	s.done = make(chan struct{})
	s.stopped = make(chan struct{})
	s.mu.Unlock()

	go func() {
		ticker := time.NewTicker(80 * time.Millisecond)
		defer ticker.Stop()
		defer close(s.stopped)
		for {
			select {
			case <-s.done:
				// Clear the spinner line
				fmt.Print("\r\033[K")
				return
			case <-ticker.C:
				s.mu.Lock()
				frame := s.frames[s.current%len(s.frames)]
				s.current++
				s.mu.Unlock()
				fmt.Printf("\r  %s%s %s%s", ColorCyan, frame, label, ColorReset)
			}
		}
	}()
}

// Stop halts the spinner, waits for cleanup, then returns.
func (s *Spinner) Stop() {
	s.mu.Lock()
	if !s.running {
		s.mu.Unlock()
		return
	}
	s.running = false
	close(s.done)
	stopped := s.stopped
	s.mu.Unlock()
	// Wait for the goroutine to finish clearing the line
	<-stopped
}
