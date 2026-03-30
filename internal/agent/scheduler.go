package agent

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ScheduledJob represents a recurring agent goal.
type ScheduledJob struct {
	ID       string    `json:"id"`
	Goal     string    `json:"goal"`
	Schedule string    `json:"schedule"` // "daily 9:00", "hourly", "every 30m", "weekdays 8:00"
	LastRun  time.Time `json:"last_run,omitempty"`
	NextRun  time.Time `json:"next_run"`
	Enabled  bool      `json:"enabled"`
}

// Scheduler runs agent tasks on a schedule.
type Scheduler struct {
	agent   *Agent
	mu      sync.RWMutex
	jobs    []ScheduledJob
	running bool
	stopCh  chan struct{}
	nextID  int
	path    string // persistence path
}

// NewScheduler creates a scheduler backed by the agent.
func NewScheduler(agent *Agent) *Scheduler {
	path := filepath.Join(agent.Config.Workspace, "scheduler.json")
	s := &Scheduler{
		agent:  agent,
		path:   path,
		nextID: 1,
	}
	s.load()
	return s
}

// AddJob schedules a recurring goal and returns the job ID.
func (s *Scheduler) AddJob(goal, schedule string) (string, error) {
	next, err := nextRunTime(schedule, time.Now())
	if err != nil {
		return "", fmt.Errorf("invalid schedule %q: %w", schedule, err)
	}

	s.mu.Lock()
	id := fmt.Sprintf("job-%d", s.nextID)
	s.nextID++
	s.jobs = append(s.jobs, ScheduledJob{
		ID:       id,
		Goal:     goal,
		Schedule: schedule,
		NextRun:  next,
		Enabled:  true,
	})
	s.mu.Unlock()

	s.save()
	return id, nil
}

// RemoveJob removes a scheduled job by ID.
func (s *Scheduler) RemoveJob(id string) {
	s.mu.Lock()
	for i, j := range s.jobs {
		if j.ID == id {
			s.jobs = append(s.jobs[:i], s.jobs[i+1:]...)
			break
		}
	}
	s.mu.Unlock()
	s.save()
}

// ListJobs returns all scheduled jobs.
func (s *Scheduler) ListJobs() []ScheduledJob {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]ScheduledJob, len(s.jobs))
	copy(out, s.jobs)
	return out
}

// Start begins the scheduling loop.
func (s *Scheduler) Start() {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return
	}
	s.running = true
	s.stopCh = make(chan struct{})
	s.mu.Unlock()

	go s.loop()
}

// Stop halts the scheduling loop.
func (s *Scheduler) Stop() {
	s.mu.Lock()
	if !s.running {
		s.mu.Unlock()
		return
	}
	close(s.stopCh)
	s.running = false
	s.mu.Unlock()
}

// loop checks for due jobs every 30 seconds.
func (s *Scheduler) loop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-s.stopCh:
			return
		case now := <-ticker.C:
			s.checkJobs(now)
		}
	}
}

// checkJobs runs any jobs that are due.
func (s *Scheduler) checkJobs(now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for i := range s.jobs {
		job := &s.jobs[i]
		if !job.Enabled || now.Before(job.NextRun) {
			continue
		}

		// Run the job only if the agent is idle. If busy, skip this run.
		if err := s.agent.Start(job.Goal); err != nil {
			// Agent is busy — skip this run, it'll fire next cycle.
			continue
		}

		job.LastRun = now
		next, err := nextRunTime(job.Schedule, now)
		if err == nil {
			job.NextRun = next
		}
	}

	s.saveUnlocked()
}

// nextRunTime computes the next run time for a schedule string.
// Supported formats:
//
//	"daily 9:00"      - every day at 9:00 AM
//	"daily 21:30"     - every day at 9:30 PM
//	"hourly"          - every hour on the hour
//	"every 30m"       - every 30 minutes
//	"every 2h"        - every 2 hours
//	"weekdays 8:00"   - Monday-Friday at 8:00 AM
//	"weekly monday 9:00" - every Monday at 9:00 AM
func nextRunTime(schedule string, from time.Time) (time.Time, error) {
	lower := strings.ToLower(strings.TrimSpace(schedule))

	// "every Nm" or "every Nh"
	if strings.HasPrefix(lower, "every ") {
		durStr := strings.TrimPrefix(lower, "every ")
		dur, err := parseSimpleDuration(durStr)
		if err != nil {
			return time.Time{}, err
		}
		return from.Add(dur), nil
	}

	// "hourly"
	if lower == "hourly" {
		next := from.Truncate(time.Hour).Add(time.Hour)
		return next, nil
	}

	// "daily HH:MM"
	if strings.HasPrefix(lower, "daily") {
		timeStr := strings.TrimSpace(strings.TrimPrefix(lower, "daily"))
		if timeStr == "" {
			timeStr = "9:00"
		}
		return nextDailyAt(from, timeStr)
	}

	// "weekdays HH:MM"
	if strings.HasPrefix(lower, "weekdays") {
		timeStr := strings.TrimSpace(strings.TrimPrefix(lower, "weekdays"))
		if timeStr == "" {
			timeStr = "9:00"
		}
		return nextWeekdayAt(from, timeStr)
	}

	// "weekly DAY HH:MM"
	if strings.HasPrefix(lower, "weekly") {
		rest := strings.TrimSpace(strings.TrimPrefix(lower, "weekly"))
		parts := strings.Fields(rest)
		if len(parts) < 2 {
			return time.Time{}, fmt.Errorf("weekly schedule needs day and time: %q", schedule)
		}
		return nextWeeklyAt(from, parts[0], parts[1])
	}

	return time.Time{}, fmt.Errorf("unrecognized schedule format: %q", schedule)
}

func nextDailyAt(from time.Time, timeStr string) (time.Time, error) {
	hour, min, err := parseTimeOfDay(timeStr)
	if err != nil {
		return time.Time{}, err
	}
	next := time.Date(from.Year(), from.Month(), from.Day(), hour, min, 0, 0, from.Location())
	if !next.After(from) {
		next = next.AddDate(0, 0, 1)
	}
	return next, nil
}

func nextWeekdayAt(from time.Time, timeStr string) (time.Time, error) {
	hour, min, err := parseTimeOfDay(timeStr)
	if err != nil {
		return time.Time{}, err
	}
	next := time.Date(from.Year(), from.Month(), from.Day(), hour, min, 0, 0, from.Location())
	if !next.After(from) {
		next = next.AddDate(0, 0, 1)
	}
	// Skip weekends
	for next.Weekday() == time.Saturday || next.Weekday() == time.Sunday {
		next = next.AddDate(0, 0, 1)
	}
	return next, nil
}

func nextWeeklyAt(from time.Time, dayStr, timeStr string) (time.Time, error) {
	hour, min, err := parseTimeOfDay(timeStr)
	if err != nil {
		return time.Time{}, err
	}
	targetDay, err := parseWeekday(dayStr)
	if err != nil {
		return time.Time{}, err
	}

	next := time.Date(from.Year(), from.Month(), from.Day(), hour, min, 0, 0, from.Location())
	// Advance to the target weekday
	for next.Weekday() != targetDay || !next.After(from) {
		next = next.AddDate(0, 0, 1)
	}
	return next, nil
}

func parseTimeOfDay(s string) (int, int, error) {
	parts := strings.SplitN(s, ":", 2)
	if len(parts) != 2 {
		return 0, 0, fmt.Errorf("invalid time %q (expected HH:MM)", s)
	}
	hour, err := strconv.Atoi(strings.TrimSpace(parts[0]))
	if err != nil || hour < 0 || hour > 23 {
		return 0, 0, fmt.Errorf("invalid hour in %q", s)
	}
	min, err := strconv.Atoi(strings.TrimSpace(parts[1]))
	if err != nil || min < 0 || min > 59 {
		return 0, 0, fmt.Errorf("invalid minute in %q", s)
	}
	return hour, min, nil
}

func parseWeekday(s string) (time.Weekday, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "sunday", "sun":
		return time.Sunday, nil
	case "monday", "mon":
		return time.Monday, nil
	case "tuesday", "tue":
		return time.Tuesday, nil
	case "wednesday", "wed":
		return time.Wednesday, nil
	case "thursday", "thu":
		return time.Thursday, nil
	case "friday", "fri":
		return time.Friday, nil
	case "saturday", "sat":
		return time.Saturday, nil
	default:
		return 0, fmt.Errorf("unknown weekday: %q", s)
	}
}

func parseSimpleDuration(s string) (time.Duration, error) {
	s = strings.TrimSpace(s)
	if strings.HasSuffix(s, "m") {
		n, err := strconv.Atoi(strings.TrimSuffix(s, "m"))
		if err != nil || n <= 0 {
			return 0, fmt.Errorf("invalid duration: %q", s)
		}
		return time.Duration(n) * time.Minute, nil
	}
	if strings.HasSuffix(s, "h") {
		n, err := strconv.Atoi(strings.TrimSuffix(s, "h"))
		if err != nil || n <= 0 {
			return 0, fmt.Errorf("invalid duration: %q", s)
		}
		return time.Duration(n) * time.Hour, nil
	}
	if strings.HasSuffix(s, "s") {
		n, err := strconv.Atoi(strings.TrimSuffix(s, "s"))
		if err != nil || n <= 0 {
			return 0, fmt.Errorf("invalid duration: %q", s)
		}
		return time.Duration(n) * time.Second, nil
	}
	return 0, fmt.Errorf("invalid duration: %q (use 30m, 2h, etc.)", s)
}

// save persists the job list to disk.
func (s *Scheduler) save() {
	s.mu.RLock()
	defer s.mu.RUnlock()
	s.saveUnlocked()
}

func (s *Scheduler) saveUnlocked() {
	data, err := json.MarshalIndent(s.jobs, "", "  ")
	if err != nil {
		return
	}
	dir := filepath.Dir(s.path)
	os.MkdirAll(dir, 0o755)
	os.WriteFile(s.path, data, 0o644)
}

// load reads the job list from disk.
func (s *Scheduler) load() {
	data, err := os.ReadFile(s.path)
	if err != nil {
		return
	}
	var jobs []ScheduledJob
	if err := json.Unmarshal(data, &jobs); err != nil {
		return
	}
	s.jobs = jobs
	for _, j := range s.jobs {
		id := 0
		fmt.Sscanf(j.ID, "job-%d", &id)
		if id >= s.nextID {
			s.nextID = id + 1
		}
	}
}
