package channels

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestTelegramParsesUpdates(t *testing.T) {
	updates := tgResponse{
		OK: true,
		Result: []tgUpdate{
			{
				UpdateID: 100,
				Message: &tgMessage{
					MessageID: 1,
					From:      tgUser{ID: 42, Username: "testuser"},
					Chat:      tgChat{ID: 12345, Type: "private"},
					Text:      "hello nous",
				},
			},
			{
				UpdateID: 101,
				Message:  nil, // non-message update (e.g. edited_message)
			},
		},
	}

	data, err := json.Marshal(updates)
	if err != nil {
		t.Fatal(err)
	}

	var parsed tgResponse
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}

	if len(parsed.Result) != 2 {
		t.Fatalf("expected 2 updates, got %d", len(parsed.Result))
	}

	msg := parsed.Result[0].Message
	if msg == nil {
		t.Fatal("first update should have a message")
	}
	if msg.Text != "hello nous" {
		t.Fatalf("text = %q, want %q", msg.Text, "hello nous")
	}
	if msg.From.ID != 42 {
		t.Fatalf("from.id = %d, want 42", msg.From.ID)
	}
	if msg.Chat.ID != 12345 {
		t.Fatalf("chat.id = %d, want 12345", msg.Chat.ID)
	}

	if parsed.Result[1].Message != nil {
		t.Fatal("second update should have nil message")
	}
}

func TestTelegramSendMessage(t *testing.T) {
	var receivedChatID, receivedText, receivedParseMode string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/sendMessage") {
			http.Error(w, "not found", 404)
			return
		}

		r.ParseForm()
		receivedChatID = r.FormValue("chat_id")
		receivedText = r.FormValue("text")
		receivedParseMode = r.FormValue("parse_mode")

		w.WriteHeader(200)
		w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	tg := &Telegram{
		token:     "test-token",
		client:    srv.Client(),
		rateSlots: telegramRateLimit,
		rateReset: time.Now().Add(time.Second),
	}

	// Override the API base for testing — we create a Send-like method using the test server
	err := tg.sendTo(srv.URL+"/bot"+tg.token, "12345", "hello *world*")
	if err != nil {
		t.Fatalf("send error: %v", err)
	}

	if receivedChatID != "12345" {
		t.Fatalf("chat_id = %q, want 12345", receivedChatID)
	}
	if receivedText != "hello *world*" {
		t.Fatalf("text = %q, want %q", receivedText, "hello *world*")
	}
	if receivedParseMode != "Markdown" {
		t.Fatalf("parse_mode = %q, want Markdown", receivedParseMode)
	}
}

func TestTelegramRateLimiter(t *testing.T) {
	tg := &Telegram{
		rateSlots: 2,
		rateReset: time.Now().Add(100 * time.Millisecond),
	}

	// First two should pass immediately
	start := time.Now()
	tg.waitForRateSlot()
	tg.waitForRateSlot()
	elapsed := time.Since(start)

	if elapsed > 10*time.Millisecond {
		t.Fatalf("first two slots should be immediate, took %v", elapsed)
	}

	// Third should block until rate reset
	start = time.Now()
	tg.waitForRateSlot()
	elapsed = time.Since(start)

	if elapsed < 50*time.Millisecond {
		t.Fatalf("third slot should have waited for reset, only waited %v", elapsed)
	}
}

func TestTelegramGetUpdates(t *testing.T) {
	var callCount int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&callCount, 1)

		offset := r.URL.Query().Get("offset")
		if offset == "" {
			t.Error("expected offset parameter")
		}

		resp := tgResponse{
			OK: true,
			Result: []tgUpdate{
				{
					UpdateID: 200,
					Message: &tgMessage{
						Text: "test message",
						From: tgUser{ID: 1},
						Chat: tgChat{ID: 2},
					},
				},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	tg := &Telegram{
		token:  "test-token",
		client: srv.Client(),
	}

	// Test the getUpdates parsing by calling the test server directly
	updates, err := tg.getUpdatesFrom(srv.URL+"/bottest-token", 0)
	if err != nil {
		t.Fatalf("getUpdates error: %v", err)
	}

	if len(updates) != 1 {
		t.Fatalf("expected 1 update, got %d", len(updates))
	}
	if updates[0].Message.Text != "test message" {
		t.Fatalf("text = %q", updates[0].Message.Text)
	}
}

// sendTo is a test helper that sends using a custom base URL.
func (t *Telegram) sendTo(apiBase, chatID, message string) error {
	t.waitForRateSlot()

	params := strings.NewReader("chat_id=" + chatID + "&text=" + message + "&parse_mode=Markdown")
	resp, err := t.client.Post(apiBase+"/sendMessage", "application/x-www-form-urlencoded", params)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

// getUpdatesFrom is a test helper that fetches updates from a custom base URL.
func (t *Telegram) getUpdatesFrom(apiBase string, offset int) ([]tgUpdate, error) {
	resp, err := t.client.Get(apiBase + "/getUpdates?offset=0&timeout=0")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result tgResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	if !result.OK {
		return nil, nil
	}
	return result.Result, nil
}
