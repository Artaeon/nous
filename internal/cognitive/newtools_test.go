package cognitive

import (
    "fmt"
    "testing"
)

func TestNewToolRouting(t *testing.T) {
    nlu := NewNLU()
    tests := []struct{q, intent, action string}{
        {"define serendipity", "dict", "dict"},
        {"definition of serendipity", "dict", "dict"},
        {"list processes", "process", "process"},
        {"top processes", "process", "process"},
        {"base64 encode hello", "hash", "hash"},
        {"sha256 hash of hello", "hash", "hash"},
        {"translate hello to spanish", "translate", "translate"},
        {"am i online", "network", "network"},
        {"disk usage of home", "disk_usage", "disk_usage"},
        {"set volume to 50", "volume", "volume"},
        {"brightness", "brightness", "brightness"},
        {"set a timer for 5 minutes", "timer", "timer"},
        {"generate qr code for hello", "qrcode", "qrcode"},
        {"compress this folder", "archive", "archive"},
    }
    for _, tt := range tests {
        r := nlu.Understand(tt.q)
        fmt.Printf("%-35s → intent=%-15s action=%-20s conf=%.2f\n", tt.q, r.Intent, r.Action, r.Confidence)
        if r.Intent != tt.intent {
            t.Errorf("%q: want intent=%q got %q", tt.q, tt.intent, r.Intent)
        }
        if r.Action != tt.action {
            t.Errorf("%q: want action=%q got %q", tt.q, tt.action, r.Action)
        }
    }
}
