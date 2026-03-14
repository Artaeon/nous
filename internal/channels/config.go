package channels

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Config holds the top-level channels configuration loaded from ~/.nous/channels.json.
type Config struct {
	Telegram *TelegramConfig `json:"telegram,omitempty"`
	Discord  *DiscordConfig  `json:"discord,omitempty"`
	Matrix   *MatrixConfig   `json:"matrix,omitempty"`
}

// TelegramConfig holds Telegram-specific settings.
type TelegramConfig struct {
	Enabled      bool     `json:"enabled"`
	Token        string   `json:"token"`
	AllowedUsers []string `json:"allowed_users"`
}

// DiscordConfig holds Discord-specific settings.
type DiscordConfig struct {
	Enabled      bool     `json:"enabled"`
	Token        string   `json:"token"`
	AllowedUsers []string `json:"allowed_users"`
}

// MatrixConfig holds Matrix-specific settings.
type MatrixConfig struct {
	Enabled      bool     `json:"enabled"`
	Token        string   `json:"token"`
	Homeserver   string   `json:"homeserver"`
	AllowedUsers []string `json:"allowed_users"`
	AllowedRooms []string `json:"allowed_rooms"`
}

// ChannelConfig is a common interface for per-channel security settings.
type ChannelConfig struct {
	AllowedUsers []string
	AllowedRooms []string
}

// IsAllowed checks whether a user ID is on the allowlist.
// An empty allowlist means nobody is allowed (secure by default).
func (c ChannelConfig) IsAllowed(userID string) bool {
	for _, allowed := range c.AllowedUsers {
		if allowed == userID {
			return true
		}
	}
	return false
}

// IsRoomAllowed checks whether a room/chat ID is on the allowlist.
// An empty allowlist means all rooms are allowed (for Telegram/Discord where
// user-level filtering is sufficient).
func (c ChannelConfig) IsRoomAllowed(roomID string) bool {
	if len(c.AllowedRooms) == 0 {
		return true
	}
	for _, allowed := range c.AllowedRooms {
		if allowed == roomID {
			return true
		}
	}
	return false
}

// DefaultConfigPath returns the default path for channels.json.
func DefaultConfigPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".nous", "channels.json")
}

// LoadConfig reads and parses the channels configuration file.
// Environment variables override file-based tokens:
//   - NOUS_TELEGRAM_TOKEN
//   - NOUS_DISCORD_TOKEN
//   - NOUS_MATRIX_TOKEN
//   - NOUS_MATRIX_HOMESERVER
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read channels config: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse channels config: %w", err)
	}

	// Environment variable overrides
	if token := os.Getenv("NOUS_TELEGRAM_TOKEN"); token != "" {
		if cfg.Telegram == nil {
			cfg.Telegram = &TelegramConfig{Enabled: true}
		}
		cfg.Telegram.Token = token
	}

	if token := os.Getenv("NOUS_DISCORD_TOKEN"); token != "" {
		if cfg.Discord == nil {
			cfg.Discord = &DiscordConfig{Enabled: true}
		}
		cfg.Discord.Token = token
	}

	if token := os.Getenv("NOUS_MATRIX_TOKEN"); token != "" {
		if cfg.Matrix == nil {
			cfg.Matrix = &MatrixConfig{Enabled: true}
		}
		cfg.Matrix.Token = token
	}

	if hs := os.Getenv("NOUS_MATRIX_HOMESERVER"); hs != "" {
		if cfg.Matrix == nil {
			cfg.Matrix = &MatrixConfig{Enabled: true}
		}
		cfg.Matrix.Homeserver = hs
	}

	return &cfg, nil
}

// ConfigExists checks whether a channels configuration file is present.
func ConfigExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
