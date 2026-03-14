package hands

import (
	"testing"
	"time"
)

func TestParseScheduleShortcuts(t *testing.T) {
	tests := []struct {
		input string
		check func(Schedule) bool
		desc  string
	}{
		{"@hourly", func(s Schedule) bool { return s.Minute != nil && s.Minute[0] && len(s.Minute) == 1 && s.Hour == nil }, "hourly fires at minute 0 every hour"},
		{"@daily", func(s Schedule) bool { return s.Minute[0] && s.Hour[0] && s.DayOfMonth == nil }, "daily fires at 00:00"},
		{"@weekly", func(s Schedule) bool { return s.DayOfWeek[0] && s.Minute[0] && s.Hour[0] }, "weekly fires on Sunday at 00:00"},
		{"@startup", func(s Schedule) bool { return s.AtStartup }, "startup flag is set"},
	}

	for _, tt := range tests {
		sched, err := ParseSchedule(tt.input)
		if err != nil {
			t.Fatalf("ParseSchedule(%q) error: %v", tt.input, err)
		}
		if !tt.check(sched) {
			t.Errorf("ParseSchedule(%q): %s", tt.input, tt.desc)
		}
	}
}

func TestParseScheduleStandard(t *testing.T) {
	// Every 15 minutes
	sched, err := ParseSchedule("*/15 * * * *")
	if err != nil {
		t.Fatalf("ParseSchedule */15: %v", err)
	}
	if sched.Minute == nil {
		t.Fatal("expected non-nil Minute for */15")
	}
	for _, m := range []int{0, 15, 30, 45} {
		if !sched.Minute[m] {
			t.Errorf("expected minute %d in */15 schedule", m)
		}
	}
	if sched.Minute[10] {
		t.Error("minute 10 should not be in */15 schedule")
	}

	// Specific time: 9:30 on weekdays
	sched, err = ParseSchedule("30 9 * * 1-5")
	if err != nil {
		t.Fatalf("ParseSchedule 30 9 * * 1-5: %v", err)
	}
	if !sched.Minute[30] || len(sched.Minute) != 1 {
		t.Error("expected minute 30 only")
	}
	if !sched.Hour[9] || len(sched.Hour) != 1 {
		t.Error("expected hour 9 only")
	}
	for _, d := range []int{1, 2, 3, 4, 5} {
		if !sched.DayOfWeek[d] {
			t.Errorf("expected day %d in weekday schedule", d)
		}
	}
	if sched.DayOfWeek[0] || sched.DayOfWeek[6] {
		t.Error("weekend days should not be in weekday schedule")
	}
}

func TestParseScheduleList(t *testing.T) {
	sched, err := ParseSchedule("0,30 8,12,18 * * *")
	if err != nil {
		t.Fatalf("ParseSchedule with lists: %v", err)
	}
	if len(sched.Minute) != 2 || !sched.Minute[0] || !sched.Minute[30] {
		t.Error("expected minutes 0 and 30")
	}
	if len(sched.Hour) != 3 || !sched.Hour[8] || !sched.Hour[12] || !sched.Hour[18] {
		t.Error("expected hours 8, 12, 18")
	}
}

func TestParseScheduleInvalid(t *testing.T) {
	invalids := []string{
		"",
		"* *",
		"* * * *",
		"60 * * * *",
		"* 25 * * *",
		"* * 32 * *",
		"* * * 13 *",
		"* * * * 7",
		"abc * * * *",
	}
	for _, expr := range invalids {
		_, err := ParseSchedule(expr)
		if err == nil {
			t.Errorf("ParseSchedule(%q) should have returned an error", expr)
		}
	}
}

func TestNextRun(t *testing.T) {
	// Hourly schedule — next run from 10:30 should be 11:00
	sched, _ := ParseSchedule("@hourly")
	ref := time.Date(2026, 3, 14, 10, 30, 0, 0, time.UTC)
	next := NextRun(sched, ref)
	expected := time.Date(2026, 3, 14, 11, 0, 0, 0, time.UTC)
	if !next.Equal(expected) {
		t.Errorf("NextRun(@hourly from 10:30) = %v, want %v", next, expected)
	}

	// Daily schedule — next run from 00:30 should be next day 00:00
	sched, _ = ParseSchedule("@daily")
	ref = time.Date(2026, 3, 14, 0, 30, 0, 0, time.UTC)
	next = NextRun(sched, ref)
	expected = time.Date(2026, 3, 15, 0, 0, 0, 0, time.UTC)
	if !next.Equal(expected) {
		t.Errorf("NextRun(@daily from 00:30) = %v, want %v", next, expected)
	}
}

func TestNextRunStartup(t *testing.T) {
	sched, _ := ParseSchedule("@startup")
	next := NextRun(sched, time.Now())
	if !next.IsZero() {
		t.Errorf("NextRun(@startup) should return zero time, got %v", next)
	}
}

func TestNextRunEvery15(t *testing.T) {
	sched, _ := ParseSchedule("*/15 * * * *")
	ref := time.Date(2026, 3, 14, 10, 3, 0, 0, time.UTC)
	next := NextRun(sched, ref)
	expected := time.Date(2026, 3, 14, 10, 15, 0, 0, time.UTC)
	if !next.Equal(expected) {
		t.Errorf("NextRun(*/15 from 10:03) = %v, want %v", next, expected)
	}
}

func TestMatchesSchedule(t *testing.T) {
	// Monday at 9:00
	sched, _ := ParseSchedule("0 9 * * 1")

	// 2026-03-16 is a Monday
	monday9am := time.Date(2026, 3, 16, 9, 0, 0, 0, time.UTC)
	if !matchesSchedule(sched, monday9am) {
		t.Error("Monday 9:00 should match '0 9 * * 1'")
	}

	// Tuesday 9:00 should not match
	tuesday9am := time.Date(2026, 3, 17, 9, 0, 0, 0, time.UTC)
	if matchesSchedule(sched, tuesday9am) {
		t.Error("Tuesday 9:00 should not match '0 9 * * 1'")
	}

	// Monday 10:00 should not match
	monday10am := time.Date(2026, 3, 16, 10, 0, 0, 0, time.UTC)
	if matchesSchedule(sched, monday10am) {
		t.Error("Monday 10:00 should not match '0 9 * * 1'")
	}
}

func TestParseFieldStep(t *testing.T) {
	// */5 in range 0-59
	result, err := parseField("*/5", 0, 59)
	if err != nil {
		t.Fatalf("parseField(*/5): %v", err)
	}
	for i := 0; i <= 55; i += 5 {
		if !result[i] {
			t.Errorf("expected %d in */5", i)
		}
	}
	if result[3] {
		t.Error("3 should not be in */5")
	}
}

func TestParseFieldRange(t *testing.T) {
	result, err := parseField("1-5", 0, 6)
	if err != nil {
		t.Fatalf("parseField(1-5): %v", err)
	}
	for i := 1; i <= 5; i++ {
		if !result[i] {
			t.Errorf("expected %d in 1-5", i)
		}
	}
	if result[0] || result[6] {
		t.Error("0 and 6 should not be in 1-5")
	}
}
