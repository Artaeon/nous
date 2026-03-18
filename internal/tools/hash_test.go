package tools

import (
	"strings"
	"testing"
)

func TestHashMD5(t *testing.T) {
	got, err := HashEncodeDecode("md5", "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// md5("hello") = 5d41402abc4b2a76b9719d911017c592
	if got != "md5: 5d41402abc4b2a76b9719d911017c592" {
		t.Errorf("md5 got %q", got)
	}
}

func TestHashSHA1(t *testing.T) {
	got, err := HashEncodeDecode("sha1", "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// sha1("hello") = aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d
	if got != "sha1: aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d" {
		t.Errorf("sha1 got %q", got)
	}
}

func TestHashSHA256(t *testing.T) {
	got, err := HashEncodeDecode("sha256", "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// sha256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
	if got != "sha256: 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824" {
		t.Errorf("sha256 got %q", got)
	}
}

func TestHashSHA512(t *testing.T) {
	got, err := HashEncodeDecode("sha512", "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// sha512("hello") = 9b71d224bd62f3785d96d46ad3ea3d73319bfbc2890caadae2dff72519673ca72323c3d99ba5c11d7c7acc6e14b8c5da0c4663475c2e5c3adef46f73bcdec043
	if !strings.HasPrefix(got, "sha512: 9b71d224bd62f378") {
		t.Errorf("sha512 got %q", got)
	}
}

func TestHashDefaultAction(t *testing.T) {
	got, err := HashEncodeDecode("", "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.HasPrefix(got, "sha256: ") {
		t.Errorf("default action should be sha256, got %q", got)
	}
}

func TestBase64Encode(t *testing.T) {
	got, err := HashEncodeDecode("base64encode", "hello world")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "base64encode: aGVsbG8gd29ybGQ=" {
		t.Errorf("base64encode got %q", got)
	}
}

func TestBase64Decode(t *testing.T) {
	got, err := HashEncodeDecode("base64decode", "aGVsbG8gd29ybGQ=")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "base64decode: hello world" {
		t.Errorf("base64decode got %q", got)
	}
}

func TestBase64DecodeInvalid(t *testing.T) {
	_, err := HashEncodeDecode("base64decode", "!!!invalid!!!")
	if err == nil {
		t.Error("expected error for invalid base64")
	}
}

func TestURLEncode(t *testing.T) {
	got, err := HashEncodeDecode("urlencode", "hello world&foo=bar")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "urlencode: hello+world%26foo%3Dbar" {
		t.Errorf("urlencode got %q", got)
	}
}

func TestURLDecode(t *testing.T) {
	got, err := HashEncodeDecode("urldecode", "hello+world%26foo%3Dbar")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "urldecode: hello world&foo=bar" {
		t.Errorf("urldecode got %q", got)
	}
}

func TestURLDecodeInvalid(t *testing.T) {
	_, err := HashEncodeDecode("urldecode", "%zz")
	if err == nil {
		t.Error("expected error for invalid url encoding")
	}
}

func TestHexEncode(t *testing.T) {
	got, err := HashEncodeDecode("hex", "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "hex: 68656c6c6f" {
		t.Errorf("hex got %q", got)
	}
}

func TestHexDecode(t *testing.T) {
	got, err := HashEncodeDecode("unhex", "68656c6c6f")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "unhex: hello" {
		t.Errorf("unhex got %q", got)
	}
}

func TestHexDecodeInvalid(t *testing.T) {
	_, err := HashEncodeDecode("unhex", "zzzz")
	if err == nil {
		t.Error("expected error for invalid hex")
	}
}

func TestHashEmptyInput(t *testing.T) {
	_, err := HashEncodeDecode("md5", "")
	if err == nil {
		t.Error("expected error for empty input")
	}
}

func TestHashUnknownAction(t *testing.T) {
	_, err := HashEncodeDecode("rot13", "hello")
	if err == nil {
		t.Error("expected error for unknown action")
	}
}

func TestHashToolRegistration(t *testing.T) {
	r := NewRegistry()
	RegisterHashTools(r)

	tool, err := r.Get("hash")
	if err != nil {
		t.Fatal("hash tool not registered")
	}
	if tool.Name != "hash" {
		t.Errorf("tool name = %q, want %q", tool.Name, "hash")
	}
}
