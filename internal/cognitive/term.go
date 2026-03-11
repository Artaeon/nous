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
func Banner(version, model, host string, toolCount int, memSlots int) string {
	const width = 39 // inner width between side borders

	border := func(left, mid, right string) string {
		return "  " + ColorCyan + left + strings.Repeat("─", width) + right + ColorReset + "\n"
	}
	row := func(content string) string {
		// content is already styled; compute visible length for padding
		visible := visibleLen(content)
		pad := width - 2 - visible // 2 for leading/trailing space
		if pad < 0 {
			pad = 0
		}
		return "  " + ColorCyan + "│" + ColorReset + " " + content + strings.Repeat(" ", pad) + " " + ColorCyan + "│" + ColorReset + "\n"
	}

	var b strings.Builder
	b.WriteString("\n")
	b.WriteString(border("╭", "─", "╮"))
	b.WriteString(row(ColorBold + "νοῦς" + ColorReset + " v" + version))
	b.WriteString(row(ColorDim + "Native Orchestration of Unified" + ColorReset))
	b.WriteString(row(ColorDim + "Streams" + ColorReset))
	b.WriteString("  " + ColorCyan + "├" + strings.Repeat("─", width) + "┤" + ColorReset + "\n")
	b.WriteString(row(fmt.Sprintf("Model:  %s", model)))
	b.WriteString(row(fmt.Sprintf("Host:   %s", host)))
	b.WriteString(row(fmt.Sprintf("Tools:  %d available", toolCount)))
	b.WriteString(row(fmt.Sprintf("Memory: %d slots", memSlots)))
	b.WriteString(border("╰", "─", "╯"))
	return b.String()
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
