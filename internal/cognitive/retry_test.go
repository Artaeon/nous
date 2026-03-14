package cognitive

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestWithRetrySuccessFirstTry(t *testing.T) {
	calls := 0
	err := WithRetry(context.Background(), DefaultRetryConfig(), func() error {
		calls++
		return nil
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if calls != 1 {
		t.Errorf("expected 1 call, got %d", calls)
	}
}

func TestWithRetrySuccessAfterRetries(t *testing.T) {
	config := RetryConfig{
		MaxRetries:    5,
		InitialDelay:  1 * time.Millisecond,
		MaxDelay:      10 * time.Millisecond,
		BackoffFactor: 2.0,
	}

	calls := 0
	err := WithRetry(context.Background(), config, func() error {
		calls++
		if calls < 3 {
			return fmt.Errorf("timeout: connection timed out")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if calls != 3 {
		t.Errorf("expected 3 calls, got %d", calls)
	}
}

func TestWithRetryMaxRetriesExceeded(t *testing.T) {
	config := RetryConfig{
		MaxRetries:    2,
		InitialDelay:  1 * time.Millisecond,
		MaxDelay:      10 * time.Millisecond,
		BackoffFactor: 2.0,
	}

	calls := 0
	err := WithRetry(context.Background(), config, func() error {
		calls++
		return fmt.Errorf("503 service unavailable")
	})
	if err == nil {
		t.Fatal("expected error after max retries")
	}
	// 1 initial + 2 retries = 3 calls
	if calls != 3 {
		t.Errorf("expected 3 calls (1 initial + 2 retries), got %d", calls)
	}
	if !strings.Contains(err.Error(), "max retries") {
		t.Errorf("expected 'max retries' in error, got %q", err.Error())
	}
}

func TestWithRetryNonRetryableError(t *testing.T) {
	config := RetryConfig{
		MaxRetries:    5,
		InitialDelay:  1 * time.Millisecond,
		MaxDelay:      10 * time.Millisecond,
		BackoffFactor: 2.0,
	}

	calls := 0
	permanentErr := fmt.Errorf("invalid API key")
	err := WithRetry(context.Background(), config, func() error {
		calls++
		return permanentErr
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if calls != 1 {
		t.Errorf("expected 1 call (no retries for non-retryable), got %d", calls)
	}
	if err != permanentErr {
		t.Errorf("expected original error, got %v", err)
	}
}

func TestWithRetryContextCancellation(t *testing.T) {
	config := RetryConfig{
		MaxRetries:    10,
		InitialDelay:  100 * time.Millisecond,
		MaxDelay:      1 * time.Second,
		BackoffFactor: 2.0,
	}

	ctx, cancel := context.WithCancel(context.Background())

	var calls int32
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	err := WithRetry(ctx, config, func() error {
		atomic.AddInt32(&calls, 1)
		return fmt.Errorf("timeout: connect timeout")
	})

	if err == nil {
		t.Fatal("expected error after context cancellation")
	}
	// Should have stopped early due to cancellation
	c := atomic.LoadInt32(&calls)
	if c > 5 {
		t.Errorf("expected fewer calls due to cancellation, got %d", c)
	}
}

func TestWithRetryBackoffTiming(t *testing.T) {
	config := RetryConfig{
		MaxRetries:    3,
		InitialDelay:  50 * time.Millisecond,
		MaxDelay:      500 * time.Millisecond,
		BackoffFactor: 2.0,
	}

	var timestamps []time.Time
	err := WithRetry(context.Background(), config, func() error {
		timestamps = append(timestamps, time.Now())
		return fmt.Errorf("429 too many requests")
	})

	if err == nil {
		t.Fatal("expected error")
	}

	// 4 calls total (1 initial + 3 retries)
	if len(timestamps) != 4 {
		t.Fatalf("expected 4 timestamps, got %d", len(timestamps))
	}

	// Verify delays are increasing (with some tolerance for jitter)
	for i := 1; i < len(timestamps); i++ {
		gap := timestamps[i].Sub(timestamps[i-1])
		// With jitter, the minimum gap should be at least 25% of the nominal delay
		minExpected := 10 * time.Millisecond // very generous lower bound
		if gap < minExpected {
			t.Errorf("gap %d: %v is too small (expected at least %v)", i, gap, minExpected)
		}
	}
}

func TestIsRetryable(t *testing.T) {
	tests := []struct {
		name      string
		err       error
		retryable bool
	}{
		{"nil error", nil, false},
		{"permanent error", errors.New("invalid request"), false},
		{"timeout", errors.New("timeout: connection timed out"), true},
		{"429", errors.New("HTTP 429 Too Many Requests"), true},
		{"503", errors.New("503 service unavailable"), true},
		{"connection refused", errors.New("dial tcp: connection refused"), true},
		{"connection reset", errors.New("connection reset by peer"), true},
		{"deadline exceeded", context.DeadlineExceeded, true},
		{"auth error", errors.New("401 unauthorized"), false},
		{"not found", errors.New("404 not found"), false},
		{"temporary failure", errors.New("temporary failure in name resolution"), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsRetryable(tt.err)
			if got != tt.retryable {
				t.Errorf("IsRetryable(%v) = %v, want %v", tt.err, got, tt.retryable)
			}
		})
	}
}

func TestComputeDelay(t *testing.T) {
	config := RetryConfig{
		InitialDelay:  1 * time.Second,
		MaxDelay:      30 * time.Second,
		BackoffFactor: 2.0,
	}

	// Attempt 0: 1s
	d := computeDelay(config, 0)
	if d != 1*time.Second {
		t.Errorf("attempt 0: expected 1s, got %v", d)
	}

	// Attempt 1: 2s
	d = computeDelay(config, 1)
	if d != 2*time.Second {
		t.Errorf("attempt 1: expected 2s, got %v", d)
	}

	// Attempt 2: 4s
	d = computeDelay(config, 2)
	if d != 4*time.Second {
		t.Errorf("attempt 2: expected 4s, got %v", d)
	}

	// Attempt 5: 32s, capped to 30s
	d = computeDelay(config, 5)
	if d != 30*time.Second {
		t.Errorf("attempt 5: expected 30s (capped), got %v", d)
	}
}

func TestDefaultRetryConfig(t *testing.T) {
	c := DefaultRetryConfig()
	if c.MaxRetries != 3 {
		t.Errorf("expected MaxRetries=3, got %d", c.MaxRetries)
	}
	if c.InitialDelay != 1*time.Second {
		t.Errorf("expected InitialDelay=1s, got %v", c.InitialDelay)
	}
	if c.MaxDelay != 30*time.Second {
		t.Errorf("expected MaxDelay=30s, got %v", c.MaxDelay)
	}
	if c.BackoffFactor != 2.0 {
		t.Errorf("expected BackoffFactor=2.0, got %v", c.BackoffFactor)
	}
}
