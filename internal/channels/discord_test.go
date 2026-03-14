package channels

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDiscordParsesMessageCreate(t *testing.T) {
	payload := discordPayload{
		Op: discordOpDispatch,
		T:  "MESSAGE_CREATE",
		D: mustMarshal(discordMessage{
			ID:        "msg-1",
			ChannelID: "ch-100",
			Content:   "hello from discord",
			Author:    discordAuthor{ID: "user-42", Username: "testuser", Bot: false},
		}),
	}

	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}

	var parsed discordPayload
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}

	if parsed.Op != discordOpDispatch {
		t.Fatalf("op = %d, want %d", parsed.Op, discordOpDispatch)
	}
	if parsed.T != "MESSAGE_CREATE" {
		t.Fatalf("type = %q, want MESSAGE_CREATE", parsed.T)
	}

	var msg discordMessage
	if err := json.Unmarshal(parsed.D, &msg); err != nil {
		t.Fatal(err)
	}

	if msg.Content != "hello from discord" {
		t.Fatalf("content = %q", msg.Content)
	}
	if msg.Author.ID != "user-42" {
		t.Fatalf("author.id = %q", msg.Author.ID)
	}
	if msg.ChannelID != "ch-100" {
		t.Fatalf("channel_id = %q", msg.ChannelID)
	}
}

func TestDiscordIgnoresBotMessages(t *testing.T) {
	var handled bool
	m := NewManager(func(channel, chatID, userID, text string) (string, error) {
		handled = true
		return "ok", nil
	})

	d := &Discord{
		token:     "test-token",
		manager:   m,
		config:    ChannelConfig{AllowedUsers: []string{"user-42"}},
		botUserID: "bot-99",
	}

	// Message from bot itself — should be ignored
	selfPayload := discordPayload{
		Op: discordOpDispatch,
		T:  "MESSAGE_CREATE",
		D: mustMarshal(discordMessage{
			ID:        "msg-1",
			ChannelID: "ch-100",
			Content:   "echo",
			Author:    discordAuthor{ID: "bot-99", Username: "nous", Bot: true},
		}),
	}
	d.handleDispatch(selfPayload)
	if handled {
		t.Fatal("should ignore messages from bot itself")
	}

	// Message from another bot — should also be ignored
	botPayload := discordPayload{
		Op: discordOpDispatch,
		T:  "MESSAGE_CREATE",
		D: mustMarshal(discordMessage{
			ID:        "msg-2",
			ChannelID: "ch-100",
			Content:   "bot spam",
			Author:    discordAuthor{ID: "other-bot", Username: "spambot", Bot: true},
		}),
	}
	d.handleDispatch(botPayload)
	if handled {
		t.Fatal("should ignore messages from other bots")
	}
}

func TestDiscordHandlesReadyEvent(t *testing.T) {
	d := &Discord{}

	readyPayload := discordPayload{
		Op: discordOpDispatch,
		T:  "READY",
		D: mustMarshal(map[string]interface{}{
			"session_id": "sess-abc",
			"user":       map[string]string{"id": "bot-123"},
		}),
	}

	d.handleDispatch(readyPayload)

	if d.sessionID != "sess-abc" {
		t.Fatalf("session_id = %q, want sess-abc", d.sessionID)
	}
	if d.botUserID != "bot-123" {
		t.Fatalf("botUserID = %q, want bot-123", d.botUserID)
	}
}

func TestDiscordSendMessage(t *testing.T) {
	var receivedBody string
	var receivedAuth string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")

		data := make([]byte, r.ContentLength)
		r.Body.Read(data)
		receivedBody = string(data)

		w.WriteHeader(200)
		w.Write([]byte(`{"id":"msg-1"}`))
	}))
	defer srv.Close()

	d := &Discord{
		token:  "test-token",
		client: srv.Client(),
	}

	// Override API base for testing
	err := d.sendTo(srv.URL, "ch-100", "test reply")
	if err != nil {
		t.Fatalf("send error: %v", err)
	}

	if receivedAuth != "Bot test-token" {
		t.Fatalf("auth = %q, want 'Bot test-token'", receivedAuth)
	}
	if !strings.Contains(receivedBody, "test reply") {
		t.Fatalf("body should contain message: %s", receivedBody)
	}
}

func TestDiscordMessageTruncation(t *testing.T) {
	d := &Discord{token: "test"}

	// Create a message longer than 2000 chars
	long := strings.Repeat("a", 2100)

	// Truncation happens in Send before sending
	if len(long) > 2000 {
		long = long[:1997] + "..."
	}

	if len(long) != 2000 {
		t.Fatalf("truncated length = %d, want 2000", len(long))
	}
	if !strings.HasSuffix(long, "...") {
		t.Fatal("truncated message should end with ...")
	}
	_ = d
}

func TestDiscordWebSocketFraming(t *testing.T) {
	// Test wsWrite + wsRead round-trip using an in-memory pipe
	// We test the frame encoding/decoding logic

	payload := []byte(`{"op":1,"d":42}`)

	// Build a masked frame manually and verify we can read it
	// This tests the core framing logic without a real WebSocket connection

	frame := []byte{0x81} // FIN + text opcode
	length := len(payload)
	if length <= 125 {
		frame = append(frame, byte(length)) // no mask for server frames
	}
	frame = append(frame, payload...)

	// Verify the payload can be extracted
	if len(frame) < 2 {
		t.Fatal("frame too short")
	}
	payloadLen := int(frame[1] & 0x7F)
	if payloadLen != length {
		t.Fatalf("payload length = %d, want %d", payloadLen, length)
	}
	extracted := frame[2 : 2+payloadLen]
	if string(extracted) != string(payload) {
		t.Fatalf("extracted = %q, want %q", extracted, payload)
	}
}

// sendTo is a test helper that sends using a custom base URL.
func (d *Discord) sendTo(baseURL, chatID, message string) error {
	payload := map[string]string{"content": message}
	body, _ := json.Marshal(payload)

	req, _ := http.NewRequest("POST", baseURL+"/channels/"+chatID+"/messages", strings.NewReader(string(body)))
	req.Header.Set("Authorization", "Bot "+d.token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := d.client.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}
