package hands

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"testing"
	"time"
)

func TestValidateWebhook(t *testing.T) {
	secret := "test-secret-key"
	payload := []byte(`{"hand_name":"daily-report","trigger":"push"}`)

	// Generate valid signature
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(payload)
	validSig := hex.EncodeToString(mac.Sum(nil))

	tests := []struct {
		name      string
		secret    string
		payload   []byte
		signature string
		want      bool
	}{
		{
			name:      "valid signature",
			secret:    secret,
			payload:   payload,
			signature: validSig,
			want:      true,
		},
		{
			name:      "invalid signature",
			secret:    secret,
			payload:   payload,
			signature: "deadbeef",
			want:      false,
		},
		{
			name:      "empty secret",
			secret:    "",
			payload:   payload,
			signature: validSig,
			want:      false,
		},
		{
			name:      "empty signature",
			secret:    secret,
			payload:   payload,
			signature: "",
			want:      false,
		},
		{
			name:      "wrong secret",
			secret:    "wrong-secret",
			payload:   payload,
			signature: validSig,
			want:      false,
		},
		{
			name:      "tampered payload",
			secret:    secret,
			payload:   []byte(`{"hand_name":"evil","trigger":"push"}`),
			signature: validSig,
			want:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ValidateWebhook(tt.secret, tt.payload, tt.signature)
			if got != tt.want {
				t.Errorf("ValidateWebhook() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestWebhookRateLimiter(t *testing.T) {
	rl := NewWebhookRateLimiter(60 * time.Second)

	// First trigger should be allowed
	if !rl.Allow("test-hand") {
		t.Error("first trigger should be allowed")
	}

	// Immediate second trigger should be denied
	if rl.Allow("test-hand") {
		t.Error("immediate second trigger should be denied")
	}

	// Different hand should be allowed
	if !rl.Allow("other-hand") {
		t.Error("different hand should be allowed")
	}
}

func TestWebhookRateLimiterTimeUntilAllowed(t *testing.T) {
	rl := NewWebhookRateLimiter(60 * time.Second)

	// Untracked hand should have zero wait
	if d := rl.TimeUntilAllowed("unknown"); d != 0 {
		t.Errorf("untracked hand should have 0 wait, got %v", d)
	}

	rl.Allow("test-hand")

	// Should have non-zero wait after trigger
	wait := rl.TimeUntilAllowed("test-hand")
	if wait <= 0 {
		t.Errorf("expected positive wait after trigger, got %v", wait)
	}
	if wait > 60*time.Second {
		t.Errorf("wait should not exceed interval, got %v", wait)
	}
}

func TestWebhookRateLimiterExpiry(t *testing.T) {
	// Use a very short interval for testing
	rl := NewWebhookRateLimiter(1 * time.Millisecond)

	if !rl.Allow("test-hand") {
		t.Fatal("first trigger should be allowed")
	}

	// Wait for the interval to elapse
	time.Sleep(5 * time.Millisecond)

	if !rl.Allow("test-hand") {
		t.Error("trigger should be allowed after interval elapses")
	}
}

func TestGenerateWebhookRunID(t *testing.T) {
	id1 := GenerateWebhookRunID("test")
	id2 := GenerateWebhookRunID("test")

	if id1 == "" {
		t.Error("run ID should not be empty")
	}
	if id1 == id2 {
		t.Error("run IDs should be unique")
	}
	if len(id1) < 10 {
		t.Error("run ID seems too short")
	}
}

func TestWebhookPayloadStruct(t *testing.T) {
	p := WebhookPayload{
		HandName: "daily-report",
		Trigger:  "github-push",
		Data: map[string]string{
			"repo":   "myorg/myrepo",
			"branch": "main",
		},
	}

	if p.HandName != "daily-report" {
		t.Errorf("HandName = %q, want %q", p.HandName, "daily-report")
	}
	if p.Data["repo"] != "myorg/myrepo" {
		t.Errorf("Data[repo] = %q, want %q", p.Data["repo"], "myorg/myrepo")
	}
}

func TestWebhookConfigStruct(t *testing.T) {
	cfg := WebhookConfig{
		Secret:         "my-secret",
		AllowedSources: []string{"192.168.1.0/24", "10.0.0.0/8"},
	}

	if cfg.Secret != "my-secret" {
		t.Errorf("Secret = %q, want %q", cfg.Secret, "my-secret")
	}
	if len(cfg.AllowedSources) != 2 {
		t.Errorf("AllowedSources length = %d, want 2", len(cfg.AllowedSources))
	}
}
