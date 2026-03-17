package cognitive

import (
	"testing"
	"time"
)

func TestConstrainedDecoderNilLLM(t *testing.T) {
	cd := NewConstrainedDecoder(nil)

	if idx := cd.AskChoice("pick one", []string{"a", "b"}); idx != -1 {
		t.Errorf("AskChoice with nil LLM should return -1, got %d", idx)
	}

	if result := cd.AskYesNo("is this true?", true); result != true {
		t.Error("AskYesNo with nil LLM should return default value")
	}

	if result := cd.AskYesNo("is this true?", false); result != false {
		t.Error("AskYesNo with nil LLM should return default value")
	}

	if err := cd.AskJSON("give me json", &struct{}{}); err == nil {
		t.Error("AskJSON with nil LLM should return error")
	}
}

func TestConstrainedDecoderEmptyOptions(t *testing.T) {
	cd := NewConstrainedDecoder(nil)

	if idx := cd.AskChoice("pick one", nil); idx != -1 {
		t.Errorf("AskChoice with no options should return -1, got %d", idx)
	}
	if idx := cd.AskChoice("pick one", []string{}); idx != -1 {
		t.Errorf("AskChoice with empty options should return -1, got %d", idx)
	}
}

func TestConstrainedDecoderTimeout(t *testing.T) {
	cd := &ConstrainedDecoder{
		LLM:     nil,
		Timeout: 1 * time.Millisecond,
	}
	if cd.Timeout != 1*time.Millisecond {
		t.Error("timeout should be configurable")
	}
}

func TestBinaryCascadeNilLLM(t *testing.T) {
	cd := NewConstrainedDecoder(nil)
	results := cd.BinaryCascade("context", []string{"q1?", "q2?"})
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
	// With nil LLM, all should be default (false)
	for i, r := range results {
		if r {
			t.Errorf("result[%d] should be false with nil LLM", i)
		}
	}
}
