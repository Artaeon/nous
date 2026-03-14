package channels

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Matrix implements the Channel interface for the Matrix Client-Server API.
// It uses long-polling via the /sync endpoint and sends messages via the
// PUT /rooms/{roomId}/send endpoint.
type Matrix struct {
	token      string
	homeserver string
	manager    *Manager
	config     ChannelConfig
	client     *http.Client
	userID     string
	txnID      int64 // accessed atomically
	synced     bool  // true after first sync response is processed

	mu     sync.Mutex
	cancel context.CancelFunc
}

// NewMatrix creates a Matrix channel adapter.
func NewMatrix(token, homeserver string, manager *Manager, cfg ChannelConfig) *Matrix {
	// Normalize homeserver URL
	homeserver = strings.TrimRight(homeserver, "/")

	return &Matrix{
		token:      token,
		homeserver: homeserver,
		manager:    manager,
		config:     cfg,
		client:     &http.Client{Timeout: 90 * time.Second},
	}
}

func (m *Matrix) Name() string { return "matrix" }

// Start begins syncing with the Matrix homeserver using long polling.
func (m *Matrix) Start(ctx context.Context) error {
	m.mu.Lock()
	ctx, m.cancel = context.WithCancel(ctx)
	m.mu.Unlock()

	// Fetch our own user ID so we can ignore our own messages
	if err := m.resolveUserID(ctx); err != nil {
		log.Printf("matrix: could not resolve own user ID: %v", err)
	}

	since := ""
	// Do initial sync with short timeout to get the sync token,
	// then switch to long polling with 30s timeout.
	filterID := ""

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		syncResp, err := m.sync(ctx, since, filterID)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			log.Printf("matrix: sync error: %v", err)
			time.Sleep(5 * time.Second)
			continue
		}

		// Skip events from the first sync to avoid processing historical messages.
		if !m.synced {
			m.synced = true
			since = syncResp.NextBatch
			continue
		}

		// Process joined room events
		for roomID, room := range syncResp.Rooms.Join {
			if !m.config.IsRoomAllowed(roomID) {
				continue
			}

			for _, event := range room.Timeline.Events {
				if event.Type != "m.room.message" {
					continue
				}
				if event.Sender == m.userID {
					continue // skip our own messages
				}

				// Extract message body
				body, ok := event.Content["body"].(string)
				if !ok || body == "" {
					continue
				}

				msgType, _ := event.Content["msgtype"].(string)
				if msgType != "m.text" {
					continue
				}

				go m.manager.HandleMessage(m, m.config, roomID, event.Sender, body)
			}
		}

		since = syncResp.NextBatch
	}
}

// Stop terminates the sync loop.
func (m *Matrix) Stop() error {
	m.mu.Lock()
	cancel := m.cancel
	m.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	return nil
}

// Send delivers a message to a Matrix room.
func (m *Matrix) Send(chatID string, message string) error {
	txnSeq := atomic.AddInt64(&m.txnID, 1)
	txn := fmt.Sprintf("nous_%d_%d", time.Now().UnixMilli(), txnSeq)

	eventType := "m.room.message"
	apiURL := fmt.Sprintf("%s/_matrix/client/v3/rooms/%s/send/%s/%s",
		m.homeserver,
		url.PathEscape(chatID),
		eventType,
		url.PathEscape(txn),
	)

	payload := map[string]string{
		"msgtype": "m.text",
		"body":    message,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("matrix send: marshal: %w", err)
	}

	req, err := http.NewRequest("PUT", apiURL, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("matrix send: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+m.token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := m.client.Do(req)
	if err != nil {
		return fmt.Errorf("matrix send: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return fmt.Errorf("matrix send: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	return nil
}

// resolveUserID fetches our own user ID from the homeserver.
func (m *Matrix) resolveUserID(ctx context.Context) error {
	apiURL := m.homeserver + "/_matrix/client/v3/account/whoami"
	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+m.token)

	resp, err := m.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	var result struct {
		UserID string `json:"user_id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return err
	}

	m.userID = result.UserID
	log.Printf("matrix: connected as %s", m.userID)
	return nil
}

// sync calls the Matrix /sync endpoint with long polling.
func (m *Matrix) sync(ctx context.Context, since, filterID string) (*matrixSyncResponse, error) {
	params := url.Values{
		"timeout": {"30000"},
	}
	if since != "" {
		params.Set("since", since)
	}
	if filterID != "" {
		params.Set("filter", filterID)
	}

	apiURL := m.homeserver + "/_matrix/client/v3/sync?" + params.Encode()
	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+m.token)

	resp, err := m.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("matrix sync: HTTP %d: %s", resp.StatusCode, string(body))
	}

	var syncResp matrixSyncResponse
	if err := json.NewDecoder(resp.Body).Decode(&syncResp); err != nil {
		return nil, fmt.Errorf("matrix sync: decode: %w", err)
	}

	return &syncResp, nil
}

// --- Matrix API types ---

type matrixSyncResponse struct {
	NextBatch string          `json:"next_batch"`
	Rooms     matrixRoomSync  `json:"rooms"`
}

type matrixRoomSync struct {
	Join map[string]matrixJoinedRoom `json:"join"`
}

type matrixJoinedRoom struct {
	Timeline matrixTimeline `json:"timeline"`
}

type matrixTimeline struct {
	Events []matrixEvent `json:"events"`
}

type matrixEvent struct {
	Type    string                 `json:"type"`
	Sender  string                 `json:"sender"`
	Content map[string]interface{} `json:"content"`
}
