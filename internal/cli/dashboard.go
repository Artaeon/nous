package cli

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// ANSI color constants (mirrored from cognitive/term.go to avoid import cycle).
const (
	colorReset   = "\033[0m"
	colorBold    = "\033[1m"
	colorDim     = "\033[2m"
	colorRed     = "\033[31m"
	colorGreen   = "\033[32m"
	colorYellow  = "\033[33m"
	colorCyan    = "\033[36m"
	colorGray    = "\033[90m"
)

// HandInfo carries the data needed to render a hand in the dashboard.
type HandInfo struct {
	Name             string
	Enabled          bool
	State            string // "idle", "running", "paused", "failed", etc.
	LastRun          time.Time
	TotalRuns        int
	RequiresApproval bool
}

// MemoryInfo carries memory subsystem stats for the dashboard.
type MemoryInfo struct {
	WorkingUsed  int
	WorkingMax   int
	LongTerm     int
	Episodic     int
	ProjectFacts int
}

// DashboardData bundles everything the dashboard renderer needs.
type DashboardData struct {
	Model      string
	Stats      *SessionStats
	Hands      []HandInfo
	Memory     MemoryInfo
	ChannelCnt int // number of active channels
}

// RenderDashboard builds the full-width ANSI dashboard panel.
func RenderDashboard(d DashboardData) string {
	const width = 56

	var b strings.Builder

	// Top border
	b.WriteString(fmt.Sprintf("  %s╭─ %sNous Dashboard%s %s╮%s\n",
		colorDim, colorBold, colorReset,
		strings.Repeat("─", width-17), colorReset))

	// Header line: model, uptime, memory
	snap := d.Stats.Snapshot()
	memMB := MemoryUsageMB()
	headerLine := fmt.Sprintf("Model: %s%s%s  Uptime: %s%s%s  Memory: %s%.0fMB%s",
		colorCyan, truncate(d.Model, 20), colorReset,
		colorGreen, d.Stats.FormatUptime(), colorReset,
		colorDim, memMB, colorReset)
	b.WriteString(dashRow(headerLine, width))

	// Session stats
	sessLine := fmt.Sprintf("Session: %d messages  %d tool calls  %d tokens",
		snap.Messages, snap.ToolCalls, snap.TokensUsed)
	b.WriteString(dashRow(sessLine, width))

	// Hands section
	b.WriteString(dashSep("Hands", width))
	if len(d.Hands) == 0 {
		b.WriteString(dashRow(colorDim+"(none registered)"+colorReset, width))
	} else {
		// Sort by name for stable output
		sorted := make([]HandInfo, len(d.Hands))
		copy(sorted, d.Hands)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })

		for _, h := range sorted {
			b.WriteString(dashRow(formatHand(h), width))
		}
	}

	// Memory section
	b.WriteString(dashSep("Memory", width))
	b.WriteString(dashRow(fmt.Sprintf("Working: %d/%d slots  Long-term: %d facts",
		d.Memory.WorkingUsed, d.Memory.WorkingMax, d.Memory.LongTerm), width))
	b.WriteString(dashRow(fmt.Sprintf("Episodic: %d memories  Project: %d facts",
		d.Memory.Episodic, d.Memory.ProjectFacts), width))

	// Channels section
	b.WriteString(dashSep("Channels", width))
	if d.ChannelCnt == 0 {
		b.WriteString(dashRow(colorDim+"(none active)"+colorReset, width))
	} else {
		b.WriteString(dashRow(fmt.Sprintf("%d active", d.ChannelCnt), width))
	}

	// Bottom border
	b.WriteString(fmt.Sprintf("  %s╰%s╯%s\n",
		colorDim, strings.Repeat("─", width+2), colorReset))

	return b.String()
}

func dashRow(content string, width int) string {
	pad := width - visibleLen(content)
	if pad < 0 {
		pad = 0
	}
	return fmt.Sprintf("  %s│%s %s%s %s│%s\n",
		colorDim, colorReset, content, strings.Repeat(" ", pad), colorDim, colorReset)
}

func dashSep(title string, width int) string {
	fill := width - visibleLen(title) - 1
	if fill < 1 {
		fill = 1
	}
	return fmt.Sprintf("  %s├─ %s%s%s %s│%s\n",
		colorDim, colorBold, title, colorReset,
		strings.Repeat("─", fill), colorReset)
}

func formatHand(h HandInfo) string {
	// Bullet: filled for enabled, empty for disabled
	bullet := colorGreen + "\u25cf" + colorReset // ●
	if !h.Enabled {
		bullet = colorGray + "\u25cb" + colorReset // ○
	}

	state := h.State
	if state == "" {
		state = "idle"
	}

	// Color the state
	switch state {
	case "running":
		state = colorYellow + state + colorReset
	case "failed":
		state = colorRed + state + colorReset
	default:
		state = colorDim + state + colorReset
	}

	lastRun := ""
	if !h.LastRun.IsZero() {
		ago := time.Since(h.LastRun)
		lastRun = fmt.Sprintf("last: %s ago", formatDurationShort(ago))
	}

	runs := ""
	if h.TotalRuns > 0 {
		runs = fmt.Sprintf("%s\u2713 %d run", colorGreen, h.TotalRuns)
		if h.TotalRuns != 1 {
			runs += "s"
		}
		runs += colorReset
	}

	approval := ""
	if h.RequiresApproval && !h.Enabled {
		approval = colorDim + "(requires approval)" + colorReset
	}

	// Assemble: "● name    state    last: Xs ago   ✓ N runs"
	parts := []string{fmt.Sprintf("%s %-12s %s", bullet, h.Name, state)}
	if lastRun != "" {
		parts = append(parts, lastRun)
	}
	if runs != "" {
		parts = append(parts, runs)
	}
	if approval != "" {
		parts = append(parts, approval)
	}

	return strings.Join(parts, "   ")
}

func formatDurationShort(d time.Duration) string {
	d = d.Truncate(time.Second)
	h := int(d.Hours())
	m := int(d.Minutes()) % 60
	s := int(d.Seconds()) % 60

	switch {
	case h > 0:
		return fmt.Sprintf("%dh", h)
	case m > 0:
		return fmt.Sprintf("%dm", m)
	default:
		return fmt.Sprintf("%ds", s)
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	if maxLen <= 1 {
		return s[:maxLen]
	}
	return s[:maxLen-1] + "…"
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
