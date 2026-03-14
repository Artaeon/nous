package channels

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// --- Config tests ---

func TestLoadConfigFromFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "channels.json")

	cfg := Config{
		Telegram: &TelegramConfig{
			Enabled:      true,
			Token:        "tg-token-123",
			AllowedUsers: []string{"111", "222"},
		},
		Discord: &DiscordConfig{
			Enabled:      false,
			Token:        "dc-token-456",
			AllowedUsers: []string{"333"},
		},
	}

	data, _ := json.Marshal(cfg)
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}

	loaded, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig error: %v", err)
	}

	if loaded.Telegram == nil || loaded.Telegram.Token != "tg-token-123" {
		t.Fatal("telegram token not loaded")
	}
	if !loaded.Telegram.Enabled {
		t.Fatal("telegram should be enabled")
	}
	if len(loaded.Telegram.AllowedUsers) != 2 {
		t.Fatalf("expected 2 allowed users, got %d", len(loaded.Telegram.AllowedUsers))
	}
	if loaded.Discord == nil || loaded.Discord.Enabled {
		t.Fatal("discord should be disabled")
	}
}

func TestLoadConfigEnvOverride(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "channels.json")

	// Write a minimal config without tokens
	cfg := Config{
		Telegram: &TelegramConfig{
			Enabled:      true,
			AllowedUsers: []string{"111"},
		},
	}
	data, _ := json.Marshal(cfg)
	os.WriteFile(path, data, 0644)

	t.Setenv("NOUS_TELEGRAM_TOKEN", "env-tg-token")

	loaded, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig error: %v", err)
	}

	if loaded.Telegram.Token != "env-tg-token" {
		t.Fatalf("expected env override token, got %q", loaded.Telegram.Token)
	}
}

func TestLoadConfigEnvCreatesSection(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "channels.json")
	os.WriteFile(path, []byte("{}"), 0644)

	t.Setenv("NOUS_DISCORD_TOKEN", "env-dc-token")

	loaded, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig error: %v", err)
	}

	if loaded.Discord == nil {
		t.Fatal("discord section should be created from env")
	}
	if loaded.Discord.Token != "env-dc-token" {
		t.Fatalf("discord token = %q, want env-dc-token", loaded.Discord.Token)
	}
}

func TestConfigExists(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "channels.json")

	if ConfigExists(path) {
		t.Fatal("should not exist yet")
	}

	os.WriteFile(path, []byte("{}"), 0644)

	if !ConfigExists(path) {
		t.Fatal("should exist now")
	}
}

// --- ChannelConfig (allowlist) tests ---

func TestIsAllowedEmptyList(t *testing.T) {
	cfg := ChannelConfig{}
	if cfg.IsAllowed("user123") {
		t.Fatal("empty allowlist should deny all users")
	}
}

func TestIsAllowedWithMatch(t *testing.T) {
	cfg := ChannelConfig{AllowedUsers: []string{"user1", "user2"}}
	if !cfg.IsAllowed("user1") {
		t.Fatal("user1 should be allowed")
	}
	if !cfg.IsAllowed("user2") {
		t.Fatal("user2 should be allowed")
	}
	if cfg.IsAllowed("user3") {
		t.Fatal("user3 should not be allowed")
	}
}

func TestIsRoomAllowedEmptyList(t *testing.T) {
	cfg := ChannelConfig{}
	if !cfg.IsRoomAllowed("any-room") {
		t.Fatal("empty room allowlist should allow all rooms")
	}
}

func TestIsRoomAllowedWithMatch(t *testing.T) {
	cfg := ChannelConfig{AllowedRooms: []string{"!room1:example.com"}}
	if !cfg.IsRoomAllowed("!room1:example.com") {
		t.Fatal("room1 should be allowed")
	}
	if cfg.IsRoomAllowed("!room2:example.com") {
		t.Fatal("room2 should not be allowed")
	}
}

// --- Manager tests ---

func TestManagerRegisterAndList(t *testing.T) {
	m := NewManager(nil)
	m.Register(&mockChannel{name: "telegram"})
	m.Register(&mockChannel{name: "discord"})

	infos := m.List()
	if len(infos) != 2 {
		t.Fatalf("expected 2 channels, got %d", len(infos))
	}

	names := map[string]bool{}
	for _, info := range infos {
		names[info.Name] = true
	}
	if !names["telegram"] || !names["discord"] {
		t.Fatal("expected telegram and discord in list")
	}
}

func TestManagerStartStopChannels(t *testing.T) {
	started := make(chan string, 2)
	ch1 := &mockChannel{name: "ch1", onStart: func() { started <- "ch1" }}
	ch2 := &mockChannel{name: "ch2", onStart: func() { started <- "ch2" }}

	m := NewManager(nil)
	m.Register(ch1)
	m.Register(ch2)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	m.Start(ctx)

	// Wait for both channels to start
	timer := time.After(2 * time.Second)
	for i := 0; i < 2; i++ {
		select {
		case <-started:
		case <-timer:
			t.Fatal("channels did not start in time")
		}
	}

	m.Stop()
	if !ch1.stopped || !ch2.stopped {
		t.Fatal("expected both channels to be stopped")
	}
}

func TestManagerHandleMessageAllowlist(t *testing.T) {
	var handled bool
	m := NewManager(func(channel, chatID, userID, text string) (string, error) {
		handled = true
		return "ok", nil
	})

	ch := &mockChannel{name: "test"}
	cfg := ChannelConfig{AllowedUsers: []string{"allowed-user"}}

	// Non-allowlisted user should be rejected
	m.HandleMessage(ch, cfg, "chat1", "unknown-user", "hello")
	if handled {
		t.Fatal("message from non-allowlisted user should be rejected")
	}

	// Allowlisted user should be processed
	m.HandleMessage(ch, cfg, "chat1", "allowed-user", "hello")
	if !handled {
		t.Fatal("message from allowlisted user should be processed")
	}
}

func TestManagerHandleMessageSendsResponse(t *testing.T) {
	m := NewManager(func(channel, chatID, userID, text string) (string, error) {
		return "response to: " + text, nil
	})

	ch := &mockChannel{name: "test"}
	cfg := ChannelConfig{AllowedUsers: []string{"user1"}}

	m.HandleMessage(ch, cfg, "chat1", "user1", "question")

	if ch.lastSentChat != "chat1" {
		t.Fatalf("expected send to chat1, got %q", ch.lastSentChat)
	}
	if ch.lastSentMsg != "response to: question" {
		t.Fatalf("expected response, got %q", ch.lastSentMsg)
	}
}

func TestManagerStartStopLifecycle(t *testing.T) {
	started := make(chan string, 3)
	ch1 := &mockChannel{name: "a", onStart: func() { started <- "a" }}
	ch2 := &mockChannel{name: "b", onStart: func() { started <- "b" }}
	ch3 := &mockChannel{name: "c", onStart: func() { started <- "c" }}

	m := NewManager(nil)
	m.Register(ch1)
	m.Register(ch2)
	m.Register(ch3)

	ctx, cancel := context.WithCancel(context.Background())
	m.Start(ctx)

	// Wait for all 3 channels to start
	timer := time.After(3 * time.Second)
	for i := 0; i < 3; i++ {
		select {
		case <-started:
		case <-timer:
			t.Fatal("not all channels started in time")
		}
	}

	// Verify all are active
	infos := m.List()
	activeCount := 0
	for _, info := range infos {
		if info.Active {
			activeCount++
		}
	}
	if activeCount != 3 {
		t.Errorf("expected 3 active channels, got %d", activeCount)
	}

	// Stop
	m.Stop()
	cancel()

	if !ch1.stopped || !ch2.stopped || !ch3.stopped {
		t.Error("expected all channels to be stopped")
	}
}

func TestManagerHandleMessageRouting(t *testing.T) {
	var gotChannel, gotChatID, gotUserID, gotText string
	m := NewManager(func(channel, chatID, userID, text string) (string, error) {
		gotChannel = channel
		gotChatID = chatID
		gotUserID = userID
		gotText = text
		return "response", nil
	})

	ch := &mockChannel{name: "telegram"}
	cfg := ChannelConfig{AllowedUsers: []string{"user42"}}

	m.HandleMessage(ch, cfg, "chat99", "user42", "hello world")

	if gotChannel != "telegram" {
		t.Errorf("channel = %q, want telegram", gotChannel)
	}
	if gotChatID != "chat99" {
		t.Errorf("chatID = %q, want chat99", gotChatID)
	}
	if gotUserID != "user42" {
		t.Errorf("userID = %q, want user42", gotUserID)
	}
	if gotText != "hello world" {
		t.Errorf("text = %q, want hello world", gotText)
	}
	if ch.lastSentMsg != "response" {
		t.Errorf("sent message = %q, want response", ch.lastSentMsg)
	}
}

func TestManagerHandleMessageAllowlistDeniesMultipleUsers(t *testing.T) {
	callCount := 0
	m := NewManager(func(channel, chatID, userID, text string) (string, error) {
		callCount++
		return "ok", nil
	})

	ch := &mockChannel{name: "test"}
	cfg := ChannelConfig{AllowedUsers: []string{"alice", "bob"}}

	// Allowed users
	m.HandleMessage(ch, cfg, "c", "alice", "msg")
	m.HandleMessage(ch, cfg, "c", "bob", "msg")
	if callCount != 2 {
		t.Errorf("expected 2 calls for allowed users, got %d", callCount)
	}

	// Denied users
	m.HandleMessage(ch, cfg, "c", "eve", "msg")
	m.HandleMessage(ch, cfg, "c", "mallory", "msg")
	if callCount != 2 {
		t.Errorf("expected still 2 calls after denied users, got %d", callCount)
	}
}

func TestManagerHandleMessageNoHandler(t *testing.T) {
	m := NewManager(nil)
	ch := &mockChannel{name: "test"}
	cfg := ChannelConfig{AllowedUsers: []string{"user1"}}

	// Should not panic when handler is nil
	m.HandleMessage(ch, cfg, "c", "user1", "hello")

	// No message should be sent since handler is nil
	if ch.lastSentMsg != "" {
		t.Errorf("expected no message sent when handler is nil, got %q", ch.lastSentMsg)
	}
}

func TestManagerHandleMessageHandlerError(t *testing.T) {
	m := NewManager(func(channel, chatID, userID, text string) (string, error) {
		return "", fmt.Errorf("something broke")
	})

	ch := &mockChannel{name: "test"}
	cfg := ChannelConfig{AllowedUsers: []string{"user1"}}

	m.HandleMessage(ch, cfg, "c", "user1", "hello")

	// Should send error message to user
	if ch.lastSentMsg != "I encountered an error processing your message." {
		t.Errorf("expected error message, got %q", ch.lastSentMsg)
	}
}

func TestManagerListEmpty(t *testing.T) {
	m := NewManager(nil)
	infos := m.List()
	if len(infos) != 0 {
		t.Errorf("expected empty list, got %d channels", len(infos))
	}
}

func TestManagerRegisterOverwrites(t *testing.T) {
	m := NewManager(nil)
	m.Register(&mockChannel{name: "test"})
	m.Register(&mockChannel{name: "test"}) // overwrite

	infos := m.List()
	if len(infos) != 1 {
		t.Errorf("expected 1 channel after overwrite, got %d", len(infos))
	}
}

// --- mock channel ---

type mockChannel struct {
	name         string
	onStart      func()
	stopped      bool
	lastSentChat string
	lastSentMsg  string
	mu           sync.Mutex
}

func (m *mockChannel) Name() string { return m.name }

func (m *mockChannel) Start(ctx context.Context) error {
	if m.onStart != nil {
		m.onStart()
	}
	<-ctx.Done()
	return nil
}

func (m *mockChannel) Stop() error {
	m.stopped = true
	return nil
}

func (m *mockChannel) Send(chatID string, message string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.lastSentChat = chatID
	m.lastSentMsg = message
	return nil
}
