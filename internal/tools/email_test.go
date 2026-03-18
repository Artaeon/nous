package tools

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestLoadEmailConfigNotExist(t *testing.T) {
	// Save and restore HOME
	origHome := os.Getenv("HOME")
	tmpDir := t.TempDir()
	os.Setenv("HOME", tmpDir)
	defer os.Setenv("HOME", origHome)

	config, err := LoadEmailConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if config != nil {
		t.Fatal("expected nil config when file doesn't exist")
	}
}

func TestLoadEmailConfigValid(t *testing.T) {
	tmpDir := t.TempDir()
	nousDir := filepath.Join(tmpDir, ".nous")
	os.MkdirAll(nousDir, 0755)

	configJSON := `{
		"host": "imap.example.com",
		"port": 993,
		"username": "user@example.com",
		"password": "secret",
		"use_tls": true
	}`
	os.WriteFile(filepath.Join(nousDir, "email.json"), []byte(configJSON), 0644)

	origHome := os.Getenv("HOME")
	os.Setenv("HOME", tmpDir)
	defer os.Setenv("HOME", origHome)

	config, err := LoadEmailConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if config == nil {
		t.Fatal("expected non-nil config")
	}
	if config.Host != "imap.example.com" {
		t.Errorf("expected host 'imap.example.com', got %q", config.Host)
	}
	if config.Port != 993 {
		t.Errorf("expected port 993, got %d", config.Port)
	}
	if config.Username != "user@example.com" {
		t.Errorf("expected username 'user@example.com', got %q", config.Username)
	}
	if !config.UseTLS {
		t.Error("expected UseTLS to be true")
	}
	if config.Mailbox != "INBOX" {
		t.Errorf("expected default mailbox 'INBOX', got %q", config.Mailbox)
	}
}

func TestLoadEmailConfigDefaults(t *testing.T) {
	tmpDir := t.TempDir()
	nousDir := filepath.Join(tmpDir, ".nous")
	os.MkdirAll(nousDir, 0755)

	// Minimal config, let defaults fill in
	configJSON := `{"host": "mail.test.com", "username": "test", "password": "pass"}`
	os.WriteFile(filepath.Join(nousDir, "email.json"), []byte(configJSON), 0644)

	origHome := os.Getenv("HOME")
	os.Setenv("HOME", tmpDir)
	defer os.Setenv("HOME", origHome)

	config, err := LoadEmailConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if config.Port != 143 {
		t.Errorf("expected default port 143, got %d", config.Port)
	}
	if config.Mailbox != "INBOX" {
		t.Errorf("expected default mailbox 'INBOX', got %q", config.Mailbox)
	}
}

func TestFormatEmailSummary(t *testing.T) {
	messages := []EmailMessage{
		{From: "John <john@example.com>", Subject: "Meeting tomorrow", Date: time.Now().Add(-2 * time.Hour)},
		{From: "jane@example.com", Subject: "Project update", Date: time.Now().Add(-30 * time.Minute)},
	}

	result := FormatEmailSummary(messages, 3)

	if !strings.Contains(result, "3 unread") {
		t.Error("expected '3 unread' in output")
	}
	if !strings.Contains(result, "John") {
		t.Error("expected 'John' in output")
	}
	if !strings.Contains(result, "Meeting tomorrow") {
		t.Error("expected 'Meeting tomorrow' in output")
	}
	if !strings.Contains(result, "2h ago") {
		t.Error("expected '2h ago' in output")
	}
	if !strings.Contains(result, "30m ago") {
		t.Error("expected '30m ago' in output")
	}
}

func TestFormatEmailSummaryEmpty(t *testing.T) {
	result := FormatEmailSummary(nil, 0)
	if result != "No unread emails." {
		t.Errorf("expected 'No unread emails.', got %q", result)
	}
}

func TestFormatTimeAgo(t *testing.T) {
	tests := []struct {
		duration time.Duration
		expected string
	}{
		{30 * time.Second, "just now"},
		{5 * time.Minute, "5m ago"},
		{1 * time.Minute, "1m ago"},
		{1 * time.Hour, "1h ago"},
		{3 * time.Hour, "3h ago"},
		{24 * time.Hour, "1d ago"},
		{72 * time.Hour, "3d ago"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			result := formatTimeAgo(time.Now().Add(-tt.duration))
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestCheckEmailNilConfig(t *testing.T) {
	result, err := CheckEmail(nil, 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result, "not configured") {
		t.Errorf("expected setup message, got %q", result)
	}
}

func TestExtractNthQuoted(t *testing.T) {
	s := `"first" "second" "third"`
	if v := extractNthQuoted(s, 1); v != "first" {
		t.Errorf("expected 'first', got %q", v)
	}
	if v := extractNthQuoted(s, 2); v != "second" {
		t.Errorf("expected 'second', got %q", v)
	}
	if v := extractNthQuoted(s, 3); v != "third" {
		t.Errorf("expected 'third', got %q", v)
	}
	if v := extractNthQuoted(s, 4); v != "" {
		t.Errorf("expected empty, got %q", v)
	}
}

func TestEmailSetupMessage(t *testing.T) {
	msg := emailSetupMessage()
	if !strings.Contains(msg, "email.json") {
		t.Error("setup message should mention email.json")
	}
	if !strings.Contains(msg, "imap") {
		t.Error("setup message should mention imap")
	}
}
