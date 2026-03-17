package cognitive

import (
	"sync"
	"testing"
	"time"
)

func TestEmbedCacheBasic(t *testing.T) {
	ec := NewEmbedCache(10, 120*time.Second)
	if ec.Size() != 0 {
		t.Errorf("new cache should be empty, got %d", ec.Size())
	}

	// Miss on empty cache
	if vec := ec.Get("hello"); vec != nil {
		t.Error("expected nil on cache miss")
	}

	// Put and get
	ec.Put("hello", []float64{0.1, 0.2, 0.3})
	if ec.Size() != 1 {
		t.Errorf("expected size 1, got %d", ec.Size())
	}

	vec := ec.Get("hello")
	if vec == nil {
		t.Fatal("expected cache hit")
	}
	if len(vec) != 3 || vec[0] != 0.1 {
		t.Errorf("unexpected vector: %v", vec)
	}

	hits, misses := ec.Stats()
	if hits != 1 || misses != 1 {
		t.Errorf("expected 1 hit and 1 miss, got %d/%d", hits, misses)
	}
}

func TestEmbedCacheTTL(t *testing.T) {
	ec := NewEmbedCache(10, 1*time.Millisecond)
	ec.Put("hello", []float64{1.0})

	time.Sleep(5 * time.Millisecond)
	if vec := ec.Get("hello"); vec != nil {
		t.Error("expected nil after TTL expiration")
	}
}

func TestEmbedCacheLRUEviction(t *testing.T) {
	ec := NewEmbedCache(3, 120*time.Second)

	ec.Put("a", []float64{1.0})
	ec.Put("b", []float64{2.0})
	ec.Put("c", []float64{3.0})
	ec.Put("d", []float64{4.0}) // should evict "a"

	if ec.Size() != 3 {
		t.Errorf("expected size 3, got %d", ec.Size())
	}
	if vec := ec.Get("a"); vec != nil {
		t.Error("expected 'a' to be evicted")
	}
	if vec := ec.Get("d"); vec == nil {
		t.Error("expected 'd' to be present")
	}
}

func TestEmbedCachePutEmpty(t *testing.T) {
	ec := NewEmbedCache(10, 120*time.Second)
	ec.Put("hello", nil)
	ec.Put("hello", []float64{})
	if ec.Size() != 0 {
		t.Error("should not cache empty vectors")
	}
}

func TestEmbedCacheConcurrency(t *testing.T) {
	ec := NewEmbedCache(100, 120*time.Second)
	var wg sync.WaitGroup

	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			key := string(rune('a' + n%26))
			ec.Put(key, []float64{float64(n)})
			ec.Get(key)
		}(i)
	}
	wg.Wait()

	// Just verify no panic/race
	if ec.Size() > 100 {
		t.Error("should not exceed max size")
	}
}

func BenchmarkEmbedCacheGet(b *testing.B) {
	ec := NewEmbedCache(128, 120*time.Second)
	ec.Put("test query", []float64{0.1, 0.2, 0.3, 0.4, 0.5})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ec.Get("test query")
	}
}

func BenchmarkEmbedCachePut(b *testing.B) {
	ec := NewEmbedCache(128, 120*time.Second)
	vec := make([]float64, 768)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ec.Put("test query", vec)
	}
}
