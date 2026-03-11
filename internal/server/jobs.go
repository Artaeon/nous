package server

import (
	"fmt"
	"sync"
	"time"
)

type JobStatus string

const (
	JobQueued    JobStatus = "queued"
	JobRunning   JobStatus = "running"
	JobCompleted JobStatus = "completed"
	JobFailed    JobStatus = "failed"
	JobCanceled  JobStatus = "canceled"
)

// Job represents a background task submitted to a long-running Nous server.
type Job struct {
	ID         string    `json:"id"`
	Message    string    `json:"message"`
	Status     JobStatus `json:"status"`
	Result     string    `json:"result,omitempty"`
	Error      string    `json:"error,omitempty"`
	DurationMs int64     `json:"duration_ms,omitempty"`
	CreatedAt  time.Time `json:"created_at"`
	StartedAt  time.Time `json:"started_at,omitempty"`
	FinishedAt time.Time `json:"finished_at,omitempty"`
}

// JobManager stores submitted jobs and processes them sequentially.
type JobManager struct {
	mu      sync.RWMutex
	jobs    map[string]*Job
	order   []string
	queue   chan string
	nextID  uint64
	maxJobs int
}

func NewJobManager() *JobManager {
	return &JobManager{
		jobs:    make(map[string]*Job),
		queue:   make(chan string, 64),
		maxJobs: 200,
	}
}

func (m *JobManager) Submit(message string) Job {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.nextID++
	id := fmt.Sprintf("job-%06d", m.nextID)
	job := &Job{
		ID:        id,
		Message:   message,
		Status:    JobQueued,
		CreatedAt: time.Now(),
	}
	m.jobs[id] = job
	m.order = append(m.order, id)
	m.pruneLocked()
	m.queue <- id
	return *job
}

func (m *JobManager) Get(id string) (Job, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	job, ok := m.jobs[id]
	if !ok {
		return Job{}, false
	}
	return *job, true
}

func (m *JobManager) List() []Job {
	m.mu.RLock()
	defer m.mu.RUnlock()

	out := make([]Job, 0, len(m.order))
	for i := len(m.order) - 1; i >= 0; i-- {
		if job, ok := m.jobs[m.order[i]]; ok {
			out = append(out, *job)
		}
	}
	return out
}

func (m *JobManager) Cancel(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()

	job, ok := m.jobs[id]
	if !ok || job.Status != JobQueued {
		return false
	}
	job.Status = JobCanceled
	job.FinishedAt = time.Now()
	return true
}

func (m *JobManager) Stats() (queued, running, completed, failed int) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	for _, job := range m.jobs {
		switch job.Status {
		case JobQueued:
			queued++
		case JobRunning:
			running++
		case JobCompleted:
			completed++
		case JobFailed:
			failed++
		}
	}
	return queued, running, completed, failed
}

func (m *JobManager) StartWorker(run func(message string) (string, int64)) {
	go func() {
		for id := range m.queue {
			m.mu.Lock()
			job, ok := m.jobs[id]
			if !ok || job.Status == JobCanceled {
				m.mu.Unlock()
				continue
			}
			job.Status = JobRunning
			job.StartedAt = time.Now()
			message := job.Message
			m.mu.Unlock()

			result, duration := run(message)

			m.mu.Lock()
			job, ok = m.jobs[id]
			if ok {
				job.DurationMs = duration
				job.FinishedAt = time.Now()
				if result == "(timeout waiting for response)" {
					job.Status = JobFailed
					job.Error = result
				} else {
					job.Status = JobCompleted
					job.Result = result
				}
			}
			m.mu.Unlock()
		}
	}()
}

func (m *JobManager) pruneLocked() {
	if len(m.order) <= m.maxJobs {
		return
	}

	trimmed := m.order[:0]
	for _, id := range m.order {
		job := m.jobs[id]
		if len(trimmed) < m.maxJobs || (job != nil && job.Status == JobRunning) {
			trimmed = append(trimmed, id)
			continue
		}
		delete(m.jobs, id)
	}
	m.order = trimmed
}