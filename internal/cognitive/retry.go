package cognitive

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"net"
	"strings"
	"time"
)

// RetryConfig controls the retry behavior for transient failures.
type RetryConfig struct {
	MaxRetries    int           // maximum number of retry attempts (default 3)
	InitialDelay  time.Duration // delay before the first retry (default 1s)
	MaxDelay      time.Duration // maximum delay between retries (default 30s)
	BackoffFactor float64       // multiplier applied to delay after each retry (default 2.0)
}

// DefaultRetryConfig returns a RetryConfig with sensible defaults.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:    3,
		InitialDelay:  1 * time.Second,
		MaxDelay:      30 * time.Second,
		BackoffFactor: 2.0,
	}
}

// WithRetry executes fn and retries on retryable errors with exponential backoff and jitter.
// It respects context cancellation and returns the last error if all retries are exhausted.
func WithRetry(ctx context.Context, config RetryConfig, fn func() error) error {
	if config.MaxRetries <= 0 {
		config.MaxRetries = 3
	}
	if config.InitialDelay <= 0 {
		config.InitialDelay = 1 * time.Second
	}
	if config.MaxDelay <= 0 {
		config.MaxDelay = 30 * time.Second
	}
	if config.BackoffFactor <= 0 {
		config.BackoffFactor = 2.0
	}

	var lastErr error
	delay := config.InitialDelay

	for attempt := 0; attempt <= config.MaxRetries; attempt++ {
		// Check context before each attempt
		if err := ctx.Err(); err != nil {
			if lastErr != nil {
				return fmt.Errorf("%w (context: %v)", lastErr, err)
			}
			return err
		}

		lastErr = fn()
		if lastErr == nil {
			return nil
		}

		// Don't retry non-retryable errors
		if !IsRetryable(lastErr) {
			return lastErr
		}

		// Don't sleep after the last attempt
		if attempt == config.MaxRetries {
			break
		}

		// Add jitter: delay * (0.5 to 1.5)
		jitter := 0.5 + rand.Float64()
		sleepDuration := time.Duration(float64(delay) * jitter)
		if sleepDuration > config.MaxDelay {
			sleepDuration = config.MaxDelay
		}

		select {
		case <-ctx.Done():
			return fmt.Errorf("%w (context: %v)", lastErr, ctx.Err())
		case <-time.After(sleepDuration):
		}

		// Increase delay for next iteration
		delay = time.Duration(float64(delay) * config.BackoffFactor)
		if delay > config.MaxDelay {
			delay = config.MaxDelay
		}
	}

	return fmt.Errorf("max retries (%d) exceeded: %w", config.MaxRetries, lastErr)
}

// IsRetryable returns true for errors that indicate a transient failure:
// timeouts, connection refused, HTTP 429 (Too Many Requests), and HTTP 503 (Service Unavailable).
func IsRetryable(err error) bool {
	if err == nil {
		return false
	}

	// Context deadline exceeded (timeout)
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}

	// Network errors
	var netErr net.Error
	if errors.As(err, &netErr) {
		if netErr.Timeout() {
			return true
		}
	}

	// Connection refused
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		return true
	}

	// Check error message for common retryable patterns
	msg := strings.ToLower(err.Error())

	// HTTP status codes in error messages
	if strings.Contains(msg, "429") || strings.Contains(msg, "too many requests") {
		return true
	}
	if strings.Contains(msg, "503") || strings.Contains(msg, "service unavailable") {
		return true
	}

	// Connection errors
	if strings.Contains(msg, "connection refused") {
		return true
	}
	if strings.Contains(msg, "connection reset") {
		return true
	}
	if strings.Contains(msg, "timeout") {
		return true
	}
	if strings.Contains(msg, "temporary failure") {
		return true
	}

	return false
}

// computeDelay calculates the backoff delay for a given attempt.
// Exported for testing only.
func computeDelay(config RetryConfig, attempt int) time.Duration {
	delay := float64(config.InitialDelay) * math.Pow(config.BackoffFactor, float64(attempt))
	if time.Duration(delay) > config.MaxDelay {
		return config.MaxDelay
	}
	return time.Duration(delay)
}
