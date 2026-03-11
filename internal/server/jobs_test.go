package server

import (
	"strings"
	"testing"
	"time"
)

func TestJobManagerSubmitAndListNewestFirst(t *testing.T) {
	m := NewJobManager()
	m.queue = make(chan string, 10)

	first := m.Submit("first task")
	second := m.Submit("second task")

	if first.Status != JobQueued || second.Status != JobQueued {
		t.Fatalf("expected queued jobs, got %q and %q", first.Status, second.Status)
	}

	jobs := m.List()
	if len(jobs) != 2 {
		t.Fatalf("expected 2 jobs, got %d", len(jobs))
	}
	if jobs[0].ID != second.ID {
		t.Fatalf("expected newest job first, got %q want %q", jobs[0].ID, second.ID)
	}
	if jobs[1].ID != first.ID {
		t.Fatalf("expected oldest job last, got %q want %q", jobs[1].ID, first.ID)
	}
	if !strings.HasPrefix(first.ID, "job-") {
		t.Fatalf("expected generated job id, got %q", first.ID)
	}
}

func TestJobManagerCancelQueuedJob(t *testing.T) {
	m := NewJobManager()
	m.queue = make(chan string, 10)
	job := m.Submit("cancel me")

	if !m.Cancel(job.ID) {
		t.Fatal("expected cancel to succeed for queued job")
	}

	canceled, ok := m.Get(job.ID)
	if !ok {
		t.Fatal("expected canceled job to remain addressable")
	}
	if canceled.Status != JobCanceled {
		t.Fatalf("expected canceled status, got %q", canceled.Status)
	}
	if canceled.FinishedAt.IsZero() {
		t.Fatal("expected canceled job to record finish time")
	}
	if m.Cancel(job.ID) {
		t.Fatal("expected second cancel to fail")
	}
}

func TestJobManagerStartWorkerCompletesJob(t *testing.T) {
	m := NewJobManager()
	done := make(chan struct{}, 1)
	m.StartWorker(func(message string) (string, int64) {
		done <- struct{}{}
		return strings.ToUpper(message), 42
	})

	job := m.Submit("hello")
	<-done

	completed := waitForJobStatus(t, m, job.ID, JobCompleted)
	if completed.Result != "HELLO" {
		t.Fatalf("expected uppercase result, got %q", completed.Result)
	}
	if completed.DurationMs != 42 {
		t.Fatalf("expected duration 42, got %d", completed.DurationMs)
	}
	if completed.StartedAt.IsZero() || completed.FinishedAt.IsZero() {
		t.Fatal("expected completed job timestamps to be set")
	}
}

func TestJobManagerStartWorkerMarksTimeoutAsFailure(t *testing.T) {
	m := NewJobManager()
	m.StartWorker(func(message string) (string, int64) {
		return "(timeout waiting for response)", 99
	})

	job := m.Submit("slow request")
	failed := waitForJobStatus(t, m, job.ID, JobFailed)

	if failed.Error != "(timeout waiting for response)" {
		t.Fatalf("expected timeout error, got %q", failed.Error)
	}
	if failed.Result != "" {
		t.Fatalf("expected no result on failure, got %q", failed.Result)
	}
}

func TestJobManagerPruneKeepsMostRecentJobs(t *testing.T) {
	m := NewJobManager()
	m.maxJobs = 2
	m.queue = make(chan string, 10)

	first := m.Submit("first")
	second := m.Submit("second")
	third := m.Submit("third")

	if _, ok := m.Get(first.ID); ok {
		t.Fatalf("expected oldest job %q to be pruned", first.ID)
	}
	if _, ok := m.Get(second.ID); !ok {
		t.Fatalf("expected second job %q to remain", second.ID)
	}
	if _, ok := m.Get(third.ID); !ok {
		t.Fatalf("expected newest job %q to remain", third.ID)
	}
}

func waitForJobStatus(t *testing.T, m *JobManager, id string, want JobStatus) Job {
	t.Helper()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		job, ok := m.Get(id)
		if ok && job.Status == want {
			return job
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job %s did not reach status %q in time", id, want)
	return Job{}
}
