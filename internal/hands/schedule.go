package hands

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Schedule represents a parsed cron expression.
// Fields: Minute, Hour, DayOfMonth, Month, DayOfWeek
// Each field is a set of valid values. A nil set means "any" (wildcard).
type Schedule struct {
	Minute     map[int]bool
	Hour       map[int]bool
	DayOfMonth map[int]bool
	Month      map[int]bool
	DayOfWeek  map[int]bool
	AtStartup  bool // true for @startup — triggers once on registration
}

// ParseSchedule parses a cron expression or shortcut into a Schedule.
// Supported shortcuts: @hourly, @daily, @weekly, @startup
// Standard format: "minute hour day-of-month month day-of-week"
// Supports: single values, ranges (1-5), lists (1,3,5), step (*/15), and * (any).
func ParseSchedule(expr string) (Schedule, error) {
	expr = strings.TrimSpace(expr)

	// Shortcuts
	switch strings.ToLower(expr) {
	case "@hourly":
		return ParseSchedule("0 * * * *")
	case "@daily":
		return ParseSchedule("0 0 * * *")
	case "@weekly":
		return ParseSchedule("0 0 * * 0")
	case "@startup":
		return Schedule{AtStartup: true}, nil
	}

	fields := strings.Fields(expr)
	if len(fields) != 5 {
		return Schedule{}, fmt.Errorf("invalid cron expression: expected 5 fields, got %d", len(fields))
	}

	minute, err := parseField(fields[0], 0, 59)
	if err != nil {
		return Schedule{}, fmt.Errorf("invalid minute field: %w", err)
	}
	hour, err := parseField(fields[1], 0, 23)
	if err != nil {
		return Schedule{}, fmt.Errorf("invalid hour field: %w", err)
	}
	dom, err := parseField(fields[2], 1, 31)
	if err != nil {
		return Schedule{}, fmt.Errorf("invalid day-of-month field: %w", err)
	}
	month, err := parseField(fields[3], 1, 12)
	if err != nil {
		return Schedule{}, fmt.Errorf("invalid month field: %w", err)
	}
	dow, err := parseField(fields[4], 0, 6)
	if err != nil {
		return Schedule{}, fmt.Errorf("invalid day-of-week field: %w", err)
	}

	return Schedule{
		Minute:     minute,
		Hour:       hour,
		DayOfMonth: dom,
		Month:      month,
		DayOfWeek:  dow,
	}, nil
}

// NextRun computes the next time the schedule fires after the given time.
// Returns zero Time if the schedule is @startup (fires immediately).
func NextRun(sched Schedule, after time.Time) time.Time {
	if sched.AtStartup {
		return time.Time{}
	}

	// Start from the next full minute after 'after'
	t := after.Truncate(time.Minute).Add(time.Minute)

	// Search up to 366 days ahead (covers all valid cron patterns)
	limit := t.Add(366 * 24 * time.Hour)

	for t.Before(limit) {
		if matchesSchedule(sched, t) {
			return t
		}
		t = t.Add(time.Minute)
	}

	return time.Time{} // no match found within horizon
}

// matchesSchedule checks if a time matches all schedule fields.
func matchesSchedule(sched Schedule, t time.Time) bool {
	if sched.Minute != nil && !sched.Minute[t.Minute()] {
		return false
	}
	if sched.Hour != nil && !sched.Hour[t.Hour()] {
		return false
	}
	if sched.DayOfMonth != nil && !sched.DayOfMonth[t.Day()] {
		return false
	}
	if sched.Month != nil && !sched.Month[int(t.Month())] {
		return false
	}
	if sched.DayOfWeek != nil && !sched.DayOfWeek[int(t.Weekday())] {
		return false
	}
	return true
}

// parseField parses a single cron field into a set of valid values.
// Returns nil for wildcard (*).
func parseField(field string, min, max int) (map[int]bool, error) {
	if field == "*" {
		return nil, nil // wildcard — matches any value
	}

	result := make(map[int]bool)

	// Handle lists (e.g. "1,5,10")
	parts := strings.Split(field, ",")
	for _, part := range parts {
		// Handle step values (e.g. "*/15" or "0-30/5")
		if strings.Contains(part, "/") {
			stepParts := strings.SplitN(part, "/", 2)
			step, err := strconv.Atoi(stepParts[1])
			if err != nil || step <= 0 {
				return nil, fmt.Errorf("invalid step: %s", part)
			}

			rangeStart, rangeEnd := min, max
			if stepParts[0] != "*" {
				rangeStart, rangeEnd, err = parseRange(stepParts[0], min, max)
				if err != nil {
					return nil, err
				}
			}

			for i := rangeStart; i <= rangeEnd; i += step {
				result[i] = true
			}
			continue
		}

		// Handle ranges (e.g. "1-5")
		if strings.Contains(part, "-") {
			start, end, err := parseRange(part, min, max)
			if err != nil {
				return nil, err
			}
			for i := start; i <= end; i++ {
				result[i] = true
			}
			continue
		}

		// Single value
		val, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("invalid value: %s", part)
		}
		if val < min || val > max {
			return nil, fmt.Errorf("value %d out of range [%d, %d]", val, min, max)
		}
		result[val] = true
	}

	return result, nil
}

// parseRange parses "start-end" into two integers.
func parseRange(s string, min, max int) (int, int, error) {
	parts := strings.SplitN(s, "-", 2)
	if len(parts) != 2 {
		return 0, 0, fmt.Errorf("invalid range: %s", s)
	}
	start, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0, 0, fmt.Errorf("invalid range start: %s", parts[0])
	}
	end, err := strconv.Atoi(parts[1])
	if err != nil {
		return 0, 0, fmt.Errorf("invalid range end: %s", parts[1])
	}
	if start < min || end > max || start > end {
		return 0, 0, fmt.Errorf("range %d-%d out of bounds [%d, %d]", start, end, min, max)
	}
	return start, end, nil
}
