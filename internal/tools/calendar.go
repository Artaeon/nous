package tools

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// CalEvent represents a parsed calendar event.
type CalEvent struct {
	Summary     string
	Start       time.Time
	End         time.Time
	Location    string
	Description string
}

// ReadCalendar reads an ICS file and returns formatted upcoming events.
func ReadCalendar(path string, daysAhead int) (string, error) {
	if path == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("calendar: cannot determine home directory: %w", err)
		}
		path = filepath.Join(home, ".nous", "calendar.ics")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Sprintf("No calendar file found at %s\nCreate one or specify a path: calendar path=/path/to/file.ics", path), nil
		}
		return "", fmt.Errorf("calendar: %w", err)
	}

	events, err := ParseICS(string(data))
	if err != nil {
		return "", err
	}

	return FormatCalendar(events, daysAhead), nil
}

// ParseICS parses ICS content into a slice of CalEvent.
func ParseICS(data string) ([]CalEvent, error) {
	var events []CalEvent

	// Split into VEVENT blocks
	blocks := strings.Split(data, "BEGIN:VEVENT")
	for _, block := range blocks[0:] { // skip preamble naturally
		endIdx := strings.Index(block, "END:VEVENT")
		if endIdx < 0 {
			continue
		}
		block = block[:endIdx]

		var ev CalEvent
		ev.Summary = icsField(block, "SUMMARY")
		ev.Location = icsField(block, "LOCATION")
		ev.Description = icsField(block, "DESCRIPTION")

		dtStart := icsField(block, "DTSTART")
		if dtStart == "" {
			continue
		}
		var err error
		ev.Start, err = parseICSDateTime(dtStart)
		if err != nil {
			continue
		}

		dtEnd := icsField(block, "DTEND")
		if dtEnd != "" {
			ev.End, _ = parseICSDateTime(dtEnd)
		}

		events = append(events, ev)
	}

	return events, nil
}

// icsField extracts a field value from an ICS block.
// Handles both "FIELD:value" and "FIELD;params:value" forms.
func icsField(block, field string) string {
	for _, line := range strings.Split(block, "\n") {
		line = strings.TrimRight(line, "\r")
		if strings.HasPrefix(line, field+":") {
			return strings.TrimSpace(line[len(field)+1:])
		}
		if strings.HasPrefix(line, field+";") {
			// DTSTART;VALUE=DATE:20260318
			idx := strings.Index(line, ":")
			if idx >= 0 {
				return strings.TrimSpace(line[idx+1:])
			}
		}
	}
	return ""
}

// parseICSDateTime parses ICS date/datetime formats:
// 20260318, 20260318T140000Z, 20260318T140000
func parseICSDateTime(s string) (time.Time, error) {
	s = strings.TrimSpace(s)

	switch len(s) {
	case 8:
		// DATE only: 20260318
		return time.ParseInLocation("20060102", s, time.Local)
	case 15:
		// DATETIME local: 20260318T140000
		return time.ParseInLocation("20060102T150405", s, time.Local)
	case 16:
		// DATETIME UTC: 20260318T140000Z
		return time.Parse("20060102T150405Z", s)
	default:
		// Try common formats
		if t, err := time.Parse("20060102T150405Z", s); err == nil {
			return t, nil
		}
		if t, err := time.ParseInLocation("20060102T150405", s, time.Local); err == nil {
			return t, nil
		}
		return time.Time{}, fmt.Errorf("calendar: cannot parse datetime %q", s)
	}
}

// FormatCalendar filters events to the next N days and formats them grouped by date.
func FormatCalendar(events []CalEvent, daysAhead int) string {
	if daysAhead <= 0 {
		daysAhead = 7
	}

	now := time.Now()
	today := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, now.Location())
	cutoff := today.AddDate(0, 0, daysAhead)

	// Filter and sort
	var upcoming []CalEvent
	for _, ev := range events {
		if ev.Start.Before(cutoff) && !ev.Start.Before(today) {
			upcoming = append(upcoming, ev)
		}
	}

	if len(upcoming) == 0 {
		return fmt.Sprintf("No upcoming events in the next %d days.", daysAhead)
	}

	sort.Slice(upcoming, func(i, j int) bool {
		return upcoming[i].Start.Before(upcoming[j].Start)
	})

	// Group by date
	var sb strings.Builder
	var currentDate string
	for _, ev := range upcoming {
		dateKey := ev.Start.Format("2006-01-02")
		if dateKey != currentDate {
			currentDate = dateKey
			label := dayLabel(ev.Start, today)
			sb.WriteString(fmt.Sprintf("\n\U0001F4C5 %s (%s):\n", label, ev.Start.Format("Monday, January 2")))
		}

		startTime := ev.Start.Format("15:04")
		endTime := ""
		if !ev.End.IsZero() {
			endTime = "-" + ev.End.Format("15:04")
		}

		line := fmt.Sprintf("  %s%s  %s", startTime, endTime, ev.Summary)
		if ev.Location != "" {
			line += fmt.Sprintf(" (%s)", ev.Location)
		}
		sb.WriteString(line + "\n")
	}

	return strings.TrimLeft(sb.String(), "\n")
}

// dayLabel returns "Today", "Tomorrow", or the weekday name.
func dayLabel(date, today time.Time) string {
	d := date.Format("2006-01-02")
	t := today.Format("2006-01-02")
	tom := today.AddDate(0, 0, 1).Format("2006-01-02")

	switch d {
	case t:
		return "Today"
	case tom:
		return "Tomorrow"
	default:
		return date.Format("Monday")
	}
}

// RegisterCalendarTools adds the calendar tool to the registry.
func RegisterCalendarTools(r *Registry) {
	r.Register(Tool{
		Name:        "calendar",
		Description: "Check upcoming calendar events from an ICS file. Args: path (optional, default ~/.nous/calendar.ics), days (optional, default 7).",
		Execute: func(args map[string]string) (string, error) {
			path := args["path"]
			days := 7
			if v, ok := args["days"]; ok {
				if n, err := strconv.Atoi(v); err == nil && n > 0 {
					days = n
				}
			}
			return ReadCalendar(path, days)
		},
	})
}
