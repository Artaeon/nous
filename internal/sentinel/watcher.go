package sentinel

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

// EventType classifies a filesystem change.
type EventType int

const (
	EventCreated  EventType = iota
	EventModified
	EventDeleted
	EventRenamed
)

func (e EventType) String() string {
	switch e {
	case EventCreated:
		return "created"
	case EventModified:
		return "modified"
	case EventDeleted:
		return "deleted"
	case EventRenamed:
		return "renamed"
	default:
		return "unknown"
	}
}

// FileEvent represents a single filesystem change detected by the sentinel.
type FileEvent struct {
	Path      string    // relative path from project root
	Type      EventType
	Timestamp time.Time
}

// Callback is invoked when file changes are detected.
// Events are batched and debounced — the callback receives all changes
// accumulated during the debounce window.
type Callback func(events []FileEvent)

// Watcher monitors a project directory for file changes using Linux inotify.
// It recursively watches all non-hidden directories and coalesces rapid changes
// into batched notifications with a configurable debounce window.
type Watcher struct {
	mu       sync.Mutex
	rootDir  string
	fd       int // inotify file descriptor
	watches  map[int]string // watch descriptor → directory path
	callback Callback
	debounce time.Duration
	pending  []FileEvent
	timer    *time.Timer
	stopped  bool

	// Filtering
	ignoreExts  map[string]bool
	ignoreDirs  map[string]bool
}

// NewWatcher creates a sentinel watching the given root directory.
// Changes are debounced by the given duration before invoking the callback.
func NewWatcher(rootDir string, debounce time.Duration, cb Callback) (*Watcher, error) {
	fd, err := syscall.InotifyInit1(syscall.IN_NONBLOCK | syscall.IN_CLOEXEC)
	if err != nil {
		return nil, fmt.Errorf("inotify_init1: %w", err)
	}

	w := &Watcher{
		rootDir:  rootDir,
		fd:       fd,
		watches:  make(map[int]string),
		callback: cb,
		debounce: debounce,
		ignoreExts: map[string]bool{
			".swp": true, ".swo": true, ".swn": true,
			".tmp": true, ".bak": true, "~": true,
		},
		ignoreDirs: map[string]bool{
			".git": true, ".nous": true, ".idea": true, ".vscode": true,
			"node_modules": true, "vendor": true, "__pycache__": true,
			".cache": true, "dist": true, "build": true,
		},
	}

	// Add recursive watches
	if err := w.addRecursive(rootDir); err != nil {
		syscall.Close(fd)
		return nil, err
	}

	return w, nil
}

// addRecursive adds inotify watches for dir and all its subdirectories.
func (w *Watcher) addRecursive(dir string) error {
	return filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if !info.IsDir() {
			return nil
		}

		name := info.Name()
		if w.ignoreDirs[name] && path != w.rootDir {
			return filepath.SkipDir
		}
		if strings.HasPrefix(name, ".") && path != w.rootDir {
			return filepath.SkipDir
		}

		return w.addWatch(path)
	})
}

// addWatch registers a single directory for inotify monitoring.
func (w *Watcher) addWatch(dir string) error {
	const mask = syscall.IN_CREATE | syscall.IN_MODIFY | syscall.IN_DELETE |
		syscall.IN_MOVED_FROM | syscall.IN_MOVED_TO | syscall.IN_CLOSE_WRITE

	wd, err := syscall.InotifyAddWatch(w.fd, dir, mask)
	if err != nil {
		return nil // non-fatal: permission denied, too many watches, etc.
	}

	w.mu.Lock()
	w.watches[wd] = dir
	w.mu.Unlock()

	return nil
}

// Run starts the event loop. Blocks until Stop() is called or an error occurs.
func (w *Watcher) Run() error {
	buf := make([]byte, 4096)
	epfd, err := syscall.EpollCreate1(0)
	if err != nil {
		return fmt.Errorf("epoll_create1: %w", err)
	}
	defer syscall.Close(epfd)

	event := syscall.EpollEvent{Events: syscall.EPOLLIN, Fd: int32(w.fd)}
	if err := syscall.EpollCtl(epfd, syscall.EPOLL_CTL_ADD, w.fd, &event); err != nil {
		return fmt.Errorf("epoll_ctl: %w", err)
	}

	events := make([]syscall.EpollEvent, 1)

	for {
		w.mu.Lock()
		stopped := w.stopped
		w.mu.Unlock()
		if stopped {
			return nil
		}

		// Wait with 500ms timeout so we can check stopped flag
		n, err := syscall.EpollWait(epfd, events, 500)
		if err != nil {
			if err == syscall.EINTR {
				continue
			}
			return fmt.Errorf("epoll_wait: %w", err)
		}

		if n == 0 {
			continue
		}

		// Read inotify events
		nread, err := syscall.Read(w.fd, buf)
		if err != nil {
			if err == syscall.EAGAIN {
				continue
			}
			return fmt.Errorf("read inotify: %w", err)
		}

		w.processEvents(buf[:nread])
	}
}

// processEvents parses raw inotify event data and queues FileEvents.
func (w *Watcher) processEvents(buf []byte) {
	offset := 0
	for offset < len(buf) {
		// Parse inotify_event struct
		if offset+syscall.SizeofInotifyEvent > len(buf) {
			break
		}
		raw := (*syscall.InotifyEvent)(unsafe.Pointer(&buf[offset]))
		nameLen := raw.Len

		var name string
		if nameLen > 0 {
			nameBytes := buf[offset+syscall.SizeofInotifyEvent : offset+syscall.SizeofInotifyEvent+int(nameLen)]
			// Trim null bytes
			end := 0
			for end < len(nameBytes) && nameBytes[end] != 0 {
				end++
			}
			name = string(nameBytes[:end])
		}

		offset += syscall.SizeofInotifyEvent + int(nameLen)

		if name == "" {
			continue
		}

		// Skip filtered extensions
		ext := filepath.Ext(name)
		if w.ignoreExts[ext] {
			continue
		}

		// Determine directory
		w.mu.Lock()
		dir := w.watches[int(raw.Wd)]
		w.mu.Unlock()
		if dir == "" {
			continue
		}

		absPath := filepath.Join(dir, name)
		relPath, err := filepath.Rel(w.rootDir, absPath)
		if err != nil {
			continue
		}

		// Classify event type
		var evType EventType
		mask := raw.Mask
		switch {
		case mask&syscall.IN_CREATE != 0:
			evType = EventCreated
			// If a new directory was created, watch it recursively
			if mask&syscall.IN_ISDIR != 0 && !w.ignoreDirs[name] {
				w.addRecursive(absPath)
			}
		case mask&syscall.IN_CLOSE_WRITE != 0, mask&syscall.IN_MODIFY != 0:
			evType = EventModified
		case mask&syscall.IN_DELETE != 0:
			evType = EventDeleted
		case mask&(syscall.IN_MOVED_FROM|syscall.IN_MOVED_TO) != 0:
			evType = EventRenamed
		default:
			continue
		}

		fe := FileEvent{
			Path:      relPath,
			Type:      evType,
			Timestamp: time.Now(),
		}

		w.queueEvent(fe)
	}
}

// queueEvent adds an event to the pending batch and resets the debounce timer.
func (w *Watcher) queueEvent(ev FileEvent) {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Deduplicate: if the same path already pending, update type
	for i, p := range w.pending {
		if p.Path == ev.Path {
			w.pending[i] = ev
			w.resetTimer()
			return
		}
	}

	w.pending = append(w.pending, ev)
	w.resetTimer()
}

// resetTimer resets the debounce timer. Must be called with mu held.
func (w *Watcher) resetTimer() {
	if w.timer != nil {
		w.timer.Stop()
	}
	w.timer = time.AfterFunc(w.debounce, w.flush)
}

// flush sends all pending events to the callback.
func (w *Watcher) flush() {
	w.mu.Lock()
	events := w.pending
	w.pending = nil
	w.mu.Unlock()

	if len(events) > 0 && w.callback != nil {
		w.callback(events)
	}
}

// Stop shuts down the watcher and releases resources.
func (w *Watcher) Stop() {
	w.mu.Lock()
	w.stopped = true
	if w.timer != nil {
		w.timer.Stop()
	}
	w.mu.Unlock()

	// Flush any pending events
	w.flush()

	syscall.Close(w.fd)
}

// WatchCount returns the number of directories being monitored.
func (w *Watcher) WatchCount() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.watches)
}

// ChangedGoFiles filters events to only .go file changes (for index updates).
func ChangedGoFiles(events []FileEvent) []string {
	var paths []string
	seen := make(map[string]bool)
	for _, ev := range events {
		if strings.HasSuffix(ev.Path, ".go") && !seen[ev.Path] {
			seen[ev.Path] = true
			paths = append(paths, ev.Path)
		}
	}
	return paths
}
