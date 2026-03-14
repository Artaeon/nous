package channels

import (
	"context"
	"log"
	"sync"
)

// Channel is the interface that all chat platform adapters must implement.
type Channel interface {
	// Name returns the channel identifier (e.g. "telegram", "discord", "matrix").
	Name() string
	// Start connects to the platform and begins receiving messages.
	Start(ctx context.Context) error
	// Stop gracefully disconnects from the platform.
	Stop() error
	// Send delivers a message to the specified chat/room.
	Send(chatID string, message string) error
}

// MessageHandler processes an incoming message and returns a response.
// Parameters: channel name, chat ID, user ID, message text.
// Returns: response text and any error.
type MessageHandler func(channel string, chatID string, userID string, text string) (string, error)

// Manager registers and orchestrates multiple chat channels.
// It routes incoming messages through the shared handler (cognitive pipeline)
// and manages the lifecycle of all channels.
type Manager struct {
	handler  MessageHandler
	channels map[string]Channel
	active   map[string]bool
	mu       sync.RWMutex
}

// NewManager creates a channel manager with the given message handler.
func NewManager(handler MessageHandler) *Manager {
	return &Manager{
		handler:  handler,
		channels: make(map[string]Channel),
		active:   make(map[string]bool),
	}
}

// Register adds a channel adapter to the manager.
func (m *Manager) Register(ch Channel) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.channels[ch.Name()] = ch
}

// Start launches all registered channels. Channels that fail to connect
// log a warning and are skipped — the manager continues with the rest.
func (m *Manager) Start(ctx context.Context) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	for name, ch := range m.channels {
		ch := ch
		name := name
		go func() {
			log.Printf("channels: starting %s", name)
			m.mu.Lock()
			m.active[name] = true
			m.mu.Unlock()
			err := ch.Start(ctx)
			m.mu.Lock()
			m.active[name] = false
			m.mu.Unlock()
			if err != nil {
				log.Printf("channels: %s failed: %v", name, err)
			}
		}()
	}

	return nil
}

// Stop shuts down all registered channels.
func (m *Manager) Stop() {
	m.mu.RLock()
	defer m.mu.RUnlock()

	for name, ch := range m.channels {
		if err := ch.Stop(); err != nil {
			log.Printf("channels: error stopping %s: %v", name, err)
		}
	}
}

// List returns the names and connection status of all registered channels.
func (m *Manager) List() []ChannelInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var infos []ChannelInfo
	for name, ch := range m.channels {
		infos = append(infos, ChannelInfo{
			Name:   ch.Name(),
			Active: m.active[name],
		})
	}
	return infos
}

// ChannelInfo describes a registered channel for status reporting.
type ChannelInfo struct {
	Name   string `json:"name"`
	Active bool   `json:"active"`
}

// HandleMessage is called by channel adapters when a message arrives.
// It checks the allowlist, routes through the handler, and sends the response.
func (m *Manager) HandleMessage(ch Channel, cfg ChannelConfig, chatID, userID, text string) {
	if !cfg.IsAllowed(userID) {
		log.Printf("channels: %s: rejected message from non-allowlisted user %s", ch.Name(), userID)
		return
	}

	if m.handler == nil {
		log.Printf("channels: %s: no handler configured", ch.Name())
		return
	}

	response, err := m.handler(ch.Name(), chatID, userID, text)
	if err != nil {
		log.Printf("channels: %s: handler error for user %s in %s: %v", ch.Name(), userID, chatID, err)
		response = "I encountered an error processing your message."
	}

	if err := ch.Send(chatID, response); err != nil {
		log.Printf("channels: %s: send error: %v", ch.Name(), err)
	}
}
