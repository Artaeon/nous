package channels

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/tls"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

const (
	discordAPIBase    = "https://discord.com/api/v10"
	discordGatewayURL = "wss://gateway.discord.gg/?v=10&encoding=json"
)

// Discord gateway opcodes.
const (
	discordOpDispatch        = 0
	discordOpHeartbeat       = 1
	discordOpIdentify        = 2
	discordOpResume          = 6
	discordOpReconnect       = 7
	discordOpInvalidSession  = 9
	discordOpHello           = 10
	discordOpHeartbeatAck    = 11
)

// Discord implements the Channel interface for the Discord Bot Gateway.
// It uses a pure stdlib WebSocket implementation for the Gateway and
// REST API calls for sending messages.
type Discord struct {
	token     string
	manager   *Manager
	config    ChannelConfig
	client    *http.Client
	cancel    context.CancelFunc
	botUserID string

	mu        sync.Mutex
	sessionID string
	seq       *int
	conn      io.ReadWriteCloser
}

// NewDiscord creates a Discord channel adapter.
func NewDiscord(token string, manager *Manager, cfg ChannelConfig) *Discord {
	return &Discord{
		token:   token,
		manager: manager,
		config:  cfg,
		client:  &http.Client{Timeout: 30 * time.Second},
	}
}

func (d *Discord) Name() string { return "discord" }

// Start connects to the Discord Gateway via WebSocket and listens for events.
func (d *Discord) Start(ctx context.Context) error {
	d.mu.Lock()
	ctx, d.cancel = context.WithCancel(ctx)
	d.mu.Unlock()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if err := d.connectAndListen(ctx); err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			log.Printf("discord: connection error: %v, reconnecting in 5s", err)
			time.Sleep(5 * time.Second)
		}
	}
}

// Stop terminates the gateway connection.
func (d *Discord) Stop() error {
	d.mu.Lock()
	cancel := d.cancel
	if d.conn != nil {
		d.conn.Close()
	}
	d.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	return nil
}

// Send delivers a message to a Discord channel via the REST API.
func (d *Discord) Send(chatID string, message string) error {
	// Discord message limit is 2000 characters
	if len(message) > 2000 {
		message = message[:1997] + "..."
	}

	payload := map[string]string{"content": message}
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("discord send: marshal: %w", err)
	}

	url := discordAPIBase + "/channels/" + chatID + "/messages"
	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("discord send: %w", err)
	}
	req.Header.Set("Authorization", "Bot "+d.token)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "Nous (https://github.com/artaeon/nous, 0.6.0)")

	resp, err := d.client.Do(req)
	if err != nil {
		return fmt.Errorf("discord send: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return fmt.Errorf("discord send: HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	return nil
}

// connectAndListen establishes a WebSocket connection and processes events.
// This uses a minimal WebSocket client built on net/http (stdlib only).
func (d *Discord) connectAndListen(ctx context.Context) error {
	conn, err := dialWebSocket(ctx, discordGatewayURL)
	if err != nil {
		return fmt.Errorf("discord: websocket dial: %w", err)
	}
	defer conn.Close()

	d.mu.Lock()
	d.conn = conn
	d.mu.Unlock()

	// Read Hello
	helloData, err := wsRead(conn)
	if err != nil {
		return fmt.Errorf("discord: read hello: %w", err)
	}

	var hello discordPayload
	if err := json.Unmarshal(helloData, &hello); err != nil {
		return fmt.Errorf("discord: parse hello: %w", err)
	}

	if hello.Op != discordOpHello {
		return fmt.Errorf("discord: expected hello (op 10), got op %d", hello.Op)
	}

	var helloBody struct {
		HeartbeatInterval int `json:"heartbeat_interval"`
	}
	json.Unmarshal(hello.D, &helloBody)
	heartbeatInterval := time.Duration(helloBody.HeartbeatInterval) * time.Millisecond

	// Send Identify
	identify := discordPayload{
		Op: discordOpIdentify,
		D: mustMarshal(map[string]interface{}{
			"token":   d.token,
			"intents": 1 << 9 | 1 << 15, // GUILD_MESSAGES | MESSAGE_CONTENT
			"properties": map[string]string{
				"os":      "linux",
				"browser": "nous",
				"device":  "nous",
			},
		}),
	}
	if err := wsWrite(conn, mustMarshal(identify)); err != nil {
		return fmt.Errorf("discord: send identify: %w", err)
	}

	// Start heartbeat goroutine
	heartbeatCtx, heartbeatCancel := context.WithCancel(ctx)
	defer heartbeatCancel()

	go d.heartbeatLoop(heartbeatCtx, conn, heartbeatInterval)

	// Event loop
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		data, err := wsRead(conn)
		if err != nil {
			return fmt.Errorf("discord: read: %w", err)
		}

		var payload discordPayload
		if err := json.Unmarshal(data, &payload); err != nil {
			continue
		}

		// Track sequence number for heartbeats
		if payload.S != nil {
			d.mu.Lock()
			d.seq = payload.S
			d.mu.Unlock()
		}

		switch payload.Op {
		case discordOpDispatch:
			d.handleDispatch(payload)
		case discordOpReconnect:
			return fmt.Errorf("server requested reconnect")
		case discordOpInvalidSession:
			d.mu.Lock()
			d.sessionID = ""
			d.seq = nil
			d.mu.Unlock()
			return fmt.Errorf("invalid session, re-identifying")
		case discordOpHeartbeatAck:
			// heartbeat acknowledged
		case discordOpHeartbeat:
			// server requested immediate heartbeat
			d.sendHeartbeat(conn)
		}
	}
}

// heartbeatLoop sends periodic heartbeats to keep the connection alive.
func (d *Discord) heartbeatLoop(ctx context.Context, conn io.ReadWriteCloser, interval time.Duration) {
	// Add jitter to first heartbeat
	jitter := make([]byte, 2)
	rand.Read(jitter)
	firstDelay := time.Duration(binary.LittleEndian.Uint16(jitter)%1000) * time.Millisecond
	select {
	case <-ctx.Done():
		return
	case <-time.After(firstDelay):
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		d.sendHeartbeat(conn)

		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func (d *Discord) sendHeartbeat(conn io.ReadWriteCloser) {
	d.mu.Lock()
	seq := d.seq
	d.mu.Unlock()

	hb := discordPayload{Op: discordOpHeartbeat}
	if seq != nil {
		hb.D = mustMarshal(*seq)
	} else {
		hb.D = json.RawMessage("null")
	}

	wsWrite(conn, mustMarshal(hb))
}

// handleDispatch processes Discord dispatch events (op 0).
func (d *Discord) handleDispatch(payload discordPayload) {
	switch payload.T {
	case "READY":
		var ready struct {
			SessionID string `json:"session_id"`
			User      struct {
				ID string `json:"id"`
			} `json:"user"`
		}
		json.Unmarshal(payload.D, &ready)
		d.mu.Lock()
		d.sessionID = ready.SessionID
		d.botUserID = ready.User.ID
		d.mu.Unlock()
		log.Printf("discord: connected as %s", ready.User.ID)

	case "MESSAGE_CREATE":
		var msg discordMessage
		if err := json.Unmarshal(payload.D, &msg); err != nil {
			return
		}

		// Ignore messages from the bot itself
		d.mu.Lock()
		botID := d.botUserID
		d.mu.Unlock()
		if msg.Author.ID == botID {
			return
		}

		// Ignore messages from other bots
		if msg.Author.Bot {
			return
		}

		go d.manager.HandleMessage(d, d.config, msg.ChannelID, msg.Author.ID, msg.Content)
	}
}

func mustMarshal(v interface{}) json.RawMessage {
	data, _ := json.Marshal(v)
	return data
}

// --- Discord API types ---

type discordPayload struct {
	Op int              `json:"op"`
	D  json.RawMessage  `json:"d,omitempty"`
	S  *int             `json:"s,omitempty"`
	T  string           `json:"t,omitempty"`
}

type discordMessage struct {
	ID        string        `json:"id"`
	ChannelID string        `json:"channel_id"`
	Content   string        `json:"content"`
	Author    discordAuthor `json:"author"`
}

type discordAuthor struct {
	ID       string `json:"id"`
	Username string `json:"username"`
	Bot      bool   `json:"bot"`
}

// --- Minimal WebSocket client (pure stdlib) ---
// This implements just enough of RFC 6455 for Discord Gateway communication.

// wsConn wraps a net.Conn with a bufio.Reader so that bytes consumed
// during the HTTP upgrade handshake are not lost.
type wsConn struct {
	net.Conn
	br *bufio.Reader
}

func (w *wsConn) Read(p []byte) (int, error) {
	return w.br.Read(p)
}

func dialWebSocket(ctx context.Context, urlStr string) (io.ReadWriteCloser, error) {
	u, err := url.Parse(urlStr)
	if err != nil {
		return nil, err
	}
	host := u.Host
	if !strings.Contains(host, ":") {
		if u.Scheme == "wss" {
			host += ":443"
		} else {
			host += ":80"
		}
	}

	var conn net.Conn
	dialer := &net.Dialer{}
	if u.Scheme == "wss" {
		conn, err = tls.DialWithDialer(dialer, "tcp", host, &tls.Config{ServerName: u.Hostname()})
	} else {
		conn, err = dialer.DialContext(ctx, "tcp", host)
	}
	if err != nil {
		return nil, err
	}

	// Generate WebSocket key (base64-encoded 16 random bytes per RFC 6455)
	keyBytes := make([]byte, 16)
	rand.Read(keyBytes)
	key := base64.StdEncoding.EncodeToString(keyBytes)

	// Send upgrade request
	path := u.RequestURI()
	req := fmt.Sprintf("GET %s HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: %s\r\n\r\n", path, u.Host, key)
	if _, err := conn.Write([]byte(req)); err != nil {
		conn.Close()
		return nil, err
	}

	// Read response - verify 101 status
	reader := bufio.NewReader(conn)
	statusLine, err := reader.ReadString('\n')
	if err != nil {
		conn.Close()
		return nil, err
	}
	if !strings.Contains(statusLine, "101") {
		conn.Close()
		return nil, fmt.Errorf("websocket upgrade failed: %s", strings.TrimSpace(statusLine))
	}
	// Consume remaining headers
	for {
		line, err := reader.ReadString('\n')
		if err != nil || strings.TrimSpace(line) == "" {
			break
		}
	}

	return &wsConn{Conn: conn, br: reader}, nil
}

// wsRead reads a WebSocket text frame. This is a simplified reader
// that handles the Discord Gateway's typical frame sizes.
func wsRead(conn io.Reader) ([]byte, error) {
	// Read frame header
	header := make([]byte, 2)
	if _, err := io.ReadFull(conn, header); err != nil {
		return nil, err
	}

	// Parse payload length
	payloadLen := int(header[1] & 0x7F)
	masked := header[1]&0x80 != 0

	switch payloadLen {
	case 126:
		ext := make([]byte, 2)
		if _, err := io.ReadFull(conn, ext); err != nil {
			return nil, err
		}
		payloadLen = int(binary.BigEndian.Uint16(ext))
	case 127:
		ext := make([]byte, 8)
		if _, err := io.ReadFull(conn, ext); err != nil {
			return nil, err
		}
		payloadLen = int(binary.BigEndian.Uint64(ext))
	}

	var mask []byte
	if masked {
		mask = make([]byte, 4)
		if _, err := io.ReadFull(conn, mask); err != nil {
			return nil, err
		}
	}

	payload := make([]byte, payloadLen)
	if _, err := io.ReadFull(conn, payload); err != nil {
		return nil, err
	}

	if masked {
		for i := range payload {
			payload[i] ^= mask[i%4]
		}
	}

	return payload, nil
}

// wsWrite writes a masked WebSocket text frame.
func wsWrite(conn io.Writer, data []byte) error {
	// Text frame with FIN bit
	frame := []byte{0x81}

	// Payload length with mask bit set (client must mask)
	length := len(data)
	switch {
	case length <= 125:
		frame = append(frame, byte(length)|0x80)
	case length <= 65535:
		frame = append(frame, 126|0x80)
		lenBytes := make([]byte, 2)
		binary.BigEndian.PutUint16(lenBytes, uint16(length))
		frame = append(frame, lenBytes...)
	default:
		frame = append(frame, 127|0x80)
		lenBytes := make([]byte, 8)
		binary.BigEndian.PutUint64(lenBytes, uint64(length))
		frame = append(frame, lenBytes...)
	}

	// Generate mask key
	mask := make([]byte, 4)
	rand.Read(mask)
	frame = append(frame, mask...)

	// Mask the payload
	masked := make([]byte, length)
	for i := range data {
		masked[i] = data[i] ^ mask[i%4]
	}
	frame = append(frame, masked...)

	_, err := conn.Write(frame)
	return err
}
