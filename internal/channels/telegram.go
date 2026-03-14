package channels

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	telegramAPIBase      = "https://api.telegram.org/bot"
	telegramPollTimeout  = 30 // seconds for long-polling
	telegramRateLimit    = 30 // messages per second (Telegram limit)
	telegramRateInterval = time.Second
)

// Telegram implements the Channel interface for Telegram Bot API.
// It uses long polling via getUpdates — no webhooks needed.
type Telegram struct {
	token   string
	manager *Manager
	config  ChannelConfig
	client  *http.Client

	mu     sync.Mutex
	cancel context.CancelFunc

	// Rate limiter: token bucket allowing telegramRateLimit sends per second.
	rateMu    sync.Mutex
	rateSlots int
	rateReset time.Time
}

// NewTelegram creates a Telegram channel adapter.
func NewTelegram(token string, manager *Manager, cfg ChannelConfig) *Telegram {
	return &Telegram{
		token:   token,
		manager: manager,
		config:  cfg,
		client: &http.Client{
			Timeout: time.Duration(telegramPollTimeout+10) * time.Second,
		},
		rateSlots: telegramRateLimit,
		rateReset: time.Now(),
	}
}

func (t *Telegram) Name() string { return "telegram" }

// Start begins long-polling for updates from the Telegram Bot API.
func (t *Telegram) Start(ctx context.Context) error {
	t.mu.Lock()
	ctx, t.cancel = context.WithCancel(ctx)
	t.mu.Unlock()
	offset := 0

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		updates, err := t.getUpdates(ctx, offset)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			log.Printf("telegram: getUpdates error: %v", err)
			time.Sleep(5 * time.Second)
			continue
		}

		for _, update := range updates {
			if update.Message == nil || update.Message.Text == "" {
				offset = update.UpdateID + 1
				continue
			}

			msg := update.Message
			chatID := strconv.FormatInt(msg.Chat.ID, 10)
			userID := strconv.FormatInt(msg.From.ID, 10)

			go t.manager.HandleMessage(t, t.config, chatID, userID, msg.Text)

			offset = update.UpdateID + 1
		}
	}
}

// Stop terminates the polling loop.
func (t *Telegram) Stop() error {
	t.mu.Lock()
	cancel := t.cancel
	t.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	return nil
}

// Send delivers a message to a Telegram chat using sendMessage.
// Supports Markdown formatting.
func (t *Telegram) Send(chatID string, message string) error {
	t.waitForRateSlot()

	params := url.Values{
		"chat_id":    {chatID},
		"text":       {message},
		"parse_mode": {"Markdown"},
	}

	apiURL := telegramAPIBase + t.token + "/sendMessage"
	resp, err := t.client.PostForm(apiURL, params)
	if err != nil {
		return fmt.Errorf("telegram send: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		// Retry without Markdown if parsing fails
		if strings.Contains(string(body), "can't parse") {
			params.Set("parse_mode", "")
			resp2, err2 := t.client.PostForm(apiURL, params)
			if err2 != nil {
				return fmt.Errorf("telegram send (retry): %w", err2)
			}
			resp2.Body.Close()
			return nil
		}
		return fmt.Errorf("telegram send: HTTP %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// getUpdates calls the Telegram getUpdates API with long polling.
func (t *Telegram) getUpdates(ctx context.Context, offset int) ([]tgUpdate, error) {
	params := url.Values{
		"offset":  {strconv.Itoa(offset)},
		"timeout": {strconv.Itoa(telegramPollTimeout)},
	}

	apiURL := telegramAPIBase + t.token + "/getUpdates?" + params.Encode()
	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, err
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result tgResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("telegram: decode response: %w", err)
	}

	if !result.OK {
		return nil, fmt.Errorf("telegram: API error: %s", result.Description)
	}

	return result.Result, nil
}

// waitForRateSlot blocks until a rate limit slot is available.
func (t *Telegram) waitForRateSlot() {
	for {
		t.rateMu.Lock()
		now := time.Now()
		if now.After(t.rateReset) {
			t.rateSlots = telegramRateLimit
			t.rateReset = now.Add(telegramRateInterval)
		}
		if t.rateSlots > 0 {
			t.rateSlots--
			t.rateMu.Unlock()
			return
		}
		waitUntil := t.rateReset
		t.rateMu.Unlock()
		time.Sleep(time.Until(waitUntil))
	}
}

// --- Telegram API types ---

type tgResponse struct {
	OK          bool       `json:"ok"`
	Result      []tgUpdate `json:"result"`
	Description string     `json:"description"`
}

type tgUpdate struct {
	UpdateID int        `json:"update_id"`
	Message  *tgMessage `json:"message"`
}

type tgMessage struct {
	MessageID int    `json:"message_id"`
	From      tgUser `json:"from"`
	Chat      tgChat `json:"chat"`
	Text      string `json:"text"`
}

type tgUser struct {
	ID       int64  `json:"id"`
	Username string `json:"username"`
}

type tgChat struct {
	ID   int64  `json:"id"`
	Type string `json:"type"`
}
