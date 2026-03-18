package tools

import (
	"strings"
	"testing"
	"time"
)

const sampleICS = `BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Test//Test//EN
BEGIN:VEVENT
DTSTART:20260318T090000
DTEND:20260318T100000
SUMMARY:Team standup
LOCATION:Room 3
DESCRIPTION:Daily sync
END:VEVENT
BEGIN:VEVENT
DTSTART:20260318T140000Z
DTEND:20260318T153000Z
SUMMARY:Design review
END:VEVENT
BEGIN:VEVENT
DTSTART;VALUE=DATE:20260319
SUMMARY:All day event
END:VEVENT
BEGIN:VEVENT
DTSTART:20260320T110000
DTEND:20260320T120000
SUMMARY:Lunch with Alex
LOCATION:Cafe
END:VEVENT
END:VCALENDAR`

func TestParseICS(t *testing.T) {
	events, err := ParseICS(sampleICS)
	if err != nil {
		t.Fatalf("ParseICS error: %v", err)
	}

	if len(events) != 4 {
		t.Fatalf("expected 4 events, got %d", len(events))
	}

	// Check first event
	ev := events[0]
	if ev.Summary != "Team standup" {
		t.Errorf("expected summary 'Team standup', got %q", ev.Summary)
	}
	if ev.Location != "Room 3" {
		t.Errorf("expected location 'Room 3', got %q", ev.Location)
	}
	if ev.Description != "Daily sync" {
		t.Errorf("expected description 'Daily sync', got %q", ev.Description)
	}
	if ev.Start.Hour() != 9 || ev.Start.Minute() != 0 {
		t.Errorf("expected start 09:00, got %s", ev.Start.Format("15:04"))
	}
	if ev.End.Hour() != 10 || ev.End.Minute() != 0 {
		t.Errorf("expected end 10:00, got %s", ev.End.Format("15:04"))
	}

	// Check UTC event
	ev2 := events[1]
	if ev2.Summary != "Design review" {
		t.Errorf("expected summary 'Design review', got %q", ev2.Summary)
	}

	// Check DATE-only event
	ev3 := events[2]
	if ev3.Summary != "All day event" {
		t.Errorf("expected summary 'All day event', got %q", ev3.Summary)
	}
	if ev3.Start.Year() != 2026 || ev3.Start.Month() != 3 || ev3.Start.Day() != 19 {
		t.Errorf("expected date 2026-03-19, got %s", ev3.Start.Format("2006-01-02"))
	}
}

func TestParseICSDateTime(t *testing.T) {
	tests := []struct {
		input string
		year  int
		month time.Month
		day   int
	}{
		{"20260318", 2026, time.March, 18},
		{"20260318T140000", 2026, time.March, 18},
		{"20260318T140000Z", 2026, time.March, 18},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			parsed, err := parseICSDateTime(tt.input)
			if err != nil {
				t.Fatalf("parseICSDateTime(%q) error: %v", tt.input, err)
			}
			if parsed.Year() != tt.year || parsed.Month() != tt.month || parsed.Day() != tt.day {
				t.Errorf("expected %d-%02d-%02d, got %s", tt.year, tt.month, tt.day, parsed.Format("2006-01-02"))
			}
		})
	}
}

func TestFormatCalendar(t *testing.T) {
	now := time.Now()
	today := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, now.Location())

	events := []CalEvent{
		{
			Summary:  "Morning meeting",
			Start:    today.Add(9 * time.Hour),
			End:      today.Add(10 * time.Hour),
			Location: "Room A",
		},
		{
			Summary: "Afternoon work",
			Start:   today.Add(14 * time.Hour),
			End:     today.Add(15*time.Hour + 30*time.Minute),
		},
		{
			Summary: "Tomorrow event",
			Start:   today.Add(24*time.Hour + 11*time.Hour),
			End:     today.Add(24*time.Hour + 12*time.Hour),
		},
	}

	result := FormatCalendar(events, 7)

	if !strings.Contains(result, "Today") {
		t.Error("expected output to contain 'Today'")
	}
	if !strings.Contains(result, "Tomorrow") {
		t.Error("expected output to contain 'Tomorrow'")
	}
	if !strings.Contains(result, "Morning meeting") {
		t.Error("expected output to contain 'Morning meeting'")
	}
	if !strings.Contains(result, "Room A") {
		t.Error("expected output to contain location 'Room A'")
	}
	if !strings.Contains(result, "09:00-10:00") {
		t.Error("expected output to contain time range '09:00-10:00'")
	}
}

func TestFormatCalendarEmpty(t *testing.T) {
	result := FormatCalendar(nil, 7)
	if !strings.Contains(result, "No upcoming events") {
		t.Errorf("expected 'No upcoming events', got %q", result)
	}
}

func TestIcsField(t *testing.T) {
	block := "SUMMARY:Test Event\nLOCATION:Office\nDTSTART;VALUE=DATE:20260318\n"

	if v := icsField(block, "SUMMARY"); v != "Test Event" {
		t.Errorf("expected 'Test Event', got %q", v)
	}
	if v := icsField(block, "LOCATION"); v != "Office" {
		t.Errorf("expected 'Office', got %q", v)
	}
	if v := icsField(block, "DTSTART"); v != "20260318" {
		t.Errorf("expected '20260318', got %q", v)
	}
	if v := icsField(block, "MISSING"); v != "" {
		t.Errorf("expected empty, got %q", v)
	}
}
