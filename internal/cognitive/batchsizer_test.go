package cognitive

import (
	"testing"
	"time"
)

func TestNewAdaptiveBatchSizer(t *testing.T) {
	abs := NewAdaptiveBatchSizer()
	if abs == nil {
		t.Fatal("NewAdaptiveBatchSizer returned nil")
	}
	if abs.baseNumPredict != 1024 {
		t.Errorf("baseNumPredict = %d, want 1024", abs.baseNumPredict)
	}
}

func TestParamsForQueryFastPath(t *testing.T) {
	abs := NewAdaptiveBatchSizer()
	params := abs.ParamsForQuery("hello", PathFast)

	if params.NumPredict > 512 {
		t.Errorf("fast path NumPredict = %d, should be small", params.NumPredict)
	}
	if params.NumCtx > 4096 {
		t.Errorf("fast path NumCtx = %d, should be 4096", params.NumCtx)
	}
}

func TestParamsForQueryMediumPath(t *testing.T) {
	abs := NewAdaptiveBatchSizer()
	params := abs.ParamsForQuery("what is quantum computing?", PathMedium)

	if params.NumPredict < 256 {
		t.Errorf("medium path NumPredict = %d, should be moderate", params.NumPredict)
	}
}

func TestParamsForQueryFullPathComplex(t *testing.T) {
	abs := NewAdaptiveBatchSizer()
	params := abs.ParamsForQuery(
		"analyze the architecture of this project, compare the different modules, explain the trade-offs, and refactor the code step by step",
		PathFull,
	)

	if params.NumPredict < 1024 {
		t.Errorf("complex full path NumPredict = %d, should be large", params.NumPredict)
	}
}

func TestParamsForQuerySimple(t *testing.T) {
	abs := NewAdaptiveBatchSizer()
	params := abs.ParamsForQuery("hi", PathFast)

	if params.NumPredict > 256 {
		t.Errorf("simple query NumPredict = %d, should be very small", params.NumPredict)
	}
}

func TestRecordLatency(t *testing.T) {
	abs := NewAdaptiveBatchSizer()
	abs.RecordLatency(2 * time.Second)
	abs.RecordLatency(3 * time.Second)

	avg, _, samples := abs.Stats()
	if samples != 2 {
		t.Errorf("samples = %d, want 2", samples)
	}
	if avg < 2000 || avg > 3000 {
		t.Errorf("avgLatencyMs = %f, want between 2000-3000", avg)
	}
}

func TestHighLatencyReducesBatch(t *testing.T) {
	abs := NewAdaptiveBatchSizer()

	// Record high latencies
	for i := 0; i < 5; i++ {
		abs.RecordLatency(10 * time.Second)
	}

	params := abs.ParamsForQuery("explain something", PathMedium)
	normalParams := NewAdaptiveBatchSizer().ParamsForQuery("explain something", PathMedium)

	if params.NumPredict >= normalParams.NumPredict {
		t.Errorf("high latency should reduce NumPredict: got %d >= normal %d",
			params.NumPredict, normalParams.NumPredict)
	}
}

func TestEstimateComplexity(t *testing.T) {
	tests := []struct {
		query   string
		wantMin float64
		wantMax float64
	}{
		{"hi", 0.0, 0.1},
		{"what is Go?", 0.0, 0.3},
		{"explain the architecture of this system step by step", 0.3, 1.0},
		{"analyze main.go and compare the different approaches", 0.3, 1.0},
	}

	for _, tt := range tests {
		got := estimateComplexity(tt.query)
		if got < tt.wantMin || got > tt.wantMax {
			t.Errorf("estimateComplexity(%q) = %f, want [%f, %f]",
				tt.query, got, tt.wantMin, tt.wantMax)
		}
	}
}

func TestParamsNeverBelowMinimum(t *testing.T) {
	abs := NewAdaptiveBatchSizer()

	// High latency + simple query = minimum batch
	for i := 0; i < 10; i++ {
		abs.RecordLatency(20 * time.Second)
	}

	params := abs.ParamsForQuery("ok", PathFast)
	if params.NumPredict < abs.minNumPredict {
		t.Errorf("NumPredict %d below minimum %d", params.NumPredict, abs.minNumPredict)
	}
}

func TestStatsConcurrencySafe(t *testing.T) {
	abs := NewAdaptiveBatchSizer()
	done := make(chan struct{})
	go func() {
		for i := 0; i < 100; i++ {
			abs.RecordLatency(time.Duration(i) * time.Millisecond)
		}
		close(done)
	}()

	for i := 0; i < 100; i++ {
		abs.ParamsForQuery("test", PathMedium)
	}
	<-done

	_, _, samples := abs.Stats()
	if samples != 100 {
		t.Errorf("samples = %d, want 100", samples)
	}
}
