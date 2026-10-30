package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAPIKeyFile(t *testing.T) {
	key := strings.Repeat("synthetic-", 5)
	file := filepath.Join(t.TempDir(), "key")
	if err := os.WriteFile(file, []byte(key+"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	got, err := loadAPIKey("", file)
	if err != nil || got != key {
		t.Fatal("private key file was not accepted")
	}
	if _, err := loadAPIKey("legacy", file); err == nil {
		t.Fatal("ambiguous key sources accepted")
	}
	if err := os.Chmod(file, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadAPIKey("", file); err == nil {
		t.Fatal("publicly readable key accepted")
	}
	for _, data := range []string{"", "short", strings.Repeat("x", 4097), key + "\n" + key} {
		if err := os.Chmod(file, 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(file, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := loadAPIKey("", file); err == nil || strings.Contains(err.Error(), file) {
			t.Fatal("invalid key accepted or path disclosed")
		}
	}
	if _, err := loadAPIKey("", t.TempDir()); err == nil {
		t.Fatal("directory accepted")
	}
	if got, err := loadAPIKey("legacy", ""); err != nil || got != "legacy" {
		t.Fatal("legacy flag changed")
	}
}
