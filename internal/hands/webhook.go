package hands

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"time"
)

// WebhookConfig holds configuration for webhook-triggered hand execution.
type WebhookConfig struct {
	Secret         string   `json:"secret"`          // HMAC-SHA256 secret for signature validation
	AllowedSources []string `json:"allowed_sources"` // optional IP/origin allowlist
}

// WebhookPayload is the JSON body for an incoming webhook trigger.
type WebhookPayload struct {
	HandName string            `json:"hand_name"`
	Trigger  string            `json:"trigger"`
	Data     map[string]string `json:"data"`
}

// ValidateWebhook checks an HMAC-SHA256 signature against the given payload.
// The signature should be a hex-encoded HMAC-SHA256 of the payload using the secret.
func ValidateWebhook(secret string, payload []byte, signature string) bool {
	if secret == "" || signature == "" {
		return false
	}

	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(payload)
	expected := hex.EncodeToString(mac.Sum(nil))

	return hmac.Equal([]byte(expected), []byte(signature))
}

// WebhookRateLimiter enforces per-hand rate limiting for webhook triggers.
type WebhookRateLimiter struct {
	mu       sync.Mutex
	lastFire map[string]time.Time
	interval time.Duration
}

// NewWebhookRateLimiter creates a rate limiter with the given minimum interval between triggers.
func NewWebhookRateLimiter(interval time.Duration) *WebhookRateLimiter {
	return &WebhookRateLimiter{
		lastFire: make(map[string]time.Time),
		interval: interval,
	}
}

// Allow returns true if the hand is allowed to be triggered.
// It records the current time if allowed.
func (rl *WebhookRateLimiter) Allow(handName string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	if last, ok := rl.lastFire[handName]; ok {
		if now.Sub(last) < rl.interval {
			return false
		}
	}
	rl.lastFire[handName] = now
	return true
}

// TimeUntilAllowed returns how long until the next trigger is allowed for a hand.
// Returns 0 if the hand can be triggered immediately.
func (rl *WebhookRateLimiter) TimeUntilAllowed(handName string) time.Duration {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	last, ok := rl.lastFire[handName]
	if !ok {
		return 0
	}
	remaining := rl.interval - time.Since(last)
	if remaining < 0 {
		return 0
	}
	return remaining
}

// WebhookTriggerResult is returned after a webhook trigger attempt.
type WebhookTriggerResult struct {
	RunID    string `json:"run_id"`
	HandName string `json:"hand_name"`
	Status   string `json:"status"`
}

// GenerateWebhookRunID creates a unique run ID for a webhook trigger.
func GenerateWebhookRunID(handName string) string {
	return fmt.Sprintf("wh_%s_%d", handName, time.Now().UnixNano())
}
