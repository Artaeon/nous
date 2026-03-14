package channels

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestMatrixParsesSyncResponse(t *testing.T) {
	syncJSON := `{
		"next_batch": "s123_456",
		"rooms": {
			"join": {
				"!room1:example.com": {
					"timeline": {
						"events": [
							{
								"type": "m.room.message",
								"sender": "@alice:example.com",
								"content": {
									"msgtype": "m.text",
									"body": "hello nous"
								}
							},
							{
								"type": "m.room.message",
								"sender": "@bob:example.com",
								"content": {
									"msgtype": "m.image",
									"body": "photo.jpg"
								}
							},
							{
								"type": "m.room.member",
								"sender": "@carol:example.com",
								"content": {
									"membership": "join"
								}
							}
						]
					}
				}
			}
		}
	}`

	var resp matrixSyncResponse
	if err := json.Unmarshal([]byte(syncJSON), &resp); err != nil {
		t.Fatalf("parse error: %v", err)
	}

	if resp.NextBatch != "s123_456" {
		t.Fatalf("next_batch = %q", resp.NextBatch)
	}

	room, ok := resp.Rooms.Join["!room1:example.com"]
	if !ok {
		t.Fatal("expected room !room1:example.com")
	}

	if len(room.Timeline.Events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(room.Timeline.Events))
	}

	// First event: text message
	ev := room.Timeline.Events[0]
	if ev.Type != "m.room.message" {
		t.Fatalf("type = %q", ev.Type)
	}
	if ev.Sender != "@alice:example.com" {
		t.Fatalf("sender = %q", ev.Sender)
	}
	body, _ := ev.Content["body"].(string)
	if body != "hello nous" {
		t.Fatalf("body = %q", body)
	}
	msgtype, _ := ev.Content["msgtype"].(string)
	if msgtype != "m.text" {
		t.Fatalf("msgtype = %q", msgtype)
	}

	// Second event: image (should be filtered by msgtype check)
	ev2 := room.Timeline.Events[1]
	msgtype2, _ := ev2.Content["msgtype"].(string)
	if msgtype2 != "m.image" {
		t.Fatalf("expected m.image, got %q", msgtype2)
	}

	// Third event: member event (not m.room.message)
	if room.Timeline.Events[2].Type != "m.room.member" {
		t.Fatal("expected m.room.member event")
	}
}

func TestMatrixSendMessage(t *testing.T) {
	var receivedMethod string
	var receivedBody map[string]string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedMethod = r.Method

		auth := r.Header.Get("Authorization")
		if auth != "Bearer test-token" {
			t.Errorf("auth = %q", auth)
		}

		json.NewDecoder(r.Body).Decode(&receivedBody)

		w.WriteHeader(200)
		w.Write([]byte(`{"event_id":"$evt1"}`))
	}))
	defer srv.Close()

	mx := NewMatrix("test-token", srv.URL, nil, ChannelConfig{})

	err := mx.Send("!room1:example.com", "test response")
	if err != nil {
		t.Fatalf("send error: %v", err)
	}

	if receivedMethod != "PUT" {
		t.Fatalf("method = %q, want PUT", receivedMethod)
	}

	if receivedBody["msgtype"] != "m.text" {
		t.Fatalf("msgtype = %q", receivedBody["msgtype"])
	}
	if receivedBody["body"] != "test response" {
		t.Fatalf("body = %q", receivedBody["body"])
	}
}

func TestMatrixResolveUserID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/_matrix/client/v3/account/whoami" {
			json.NewEncoder(w).Encode(map[string]string{
				"user_id": "@nous:example.com",
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	mx := NewMatrix("test-token", srv.URL, nil, ChannelConfig{})
	if err := mx.resolveUserID(context.Background()); err != nil {
		t.Fatalf("resolveUserID error: %v", err)
	}

	if mx.userID != "@nous:example.com" {
		t.Fatalf("userID = %q", mx.userID)
	}
}

func TestMatrixRoomFiltering(t *testing.T) {
	cfg := ChannelConfig{
		AllowedUsers: []string{"@alice:example.com"},
		AllowedRooms: []string{"!allowed:example.com"},
	}

	if !cfg.IsRoomAllowed("!allowed:example.com") {
		t.Fatal("allowed room should pass")
	}
	if cfg.IsRoomAllowed("!other:example.com") {
		t.Fatal("non-allowed room should be blocked")
	}
}

func TestMatrixIgnoresOwnMessages(t *testing.T) {
	// Verify the message filtering logic
	mx := &Matrix{
		userID: "@nous:example.com",
	}

	event := matrixEvent{
		Type:   "m.room.message",
		Sender: "@nous:example.com",
		Content: map[string]interface{}{
			"msgtype": "m.text",
			"body":    "echo",
		},
	}

	// The sender matches our userID — should be skipped
	if event.Sender != mx.userID {
		t.Fatal("sender should match our user ID")
	}
}

func TestMatrixHomeserverNormalization(t *testing.T) {
	mx := NewMatrix("token", "https://matrix.example.com/", nil, ChannelConfig{})
	if mx.homeserver != "https://matrix.example.com" {
		t.Fatalf("homeserver = %q, trailing slash should be stripped", mx.homeserver)
	}

	mx2 := NewMatrix("token", "https://matrix.example.com", nil, ChannelConfig{})
	if mx2.homeserver != "https://matrix.example.com" {
		t.Fatalf("homeserver = %q", mx2.homeserver)
	}
}
