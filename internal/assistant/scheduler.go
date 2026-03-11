package assistant

import (
	"context"
	"time"

	"github.com/artaeon/nous/internal/blackboard"
)

type Scheduler struct {
	Store        *Store
	Board        *blackboard.Blackboard
	Interval     time.Duration
	OnNotify     func(Notification)
}

func NewScheduler(store *Store, board *blackboard.Blackboard) *Scheduler {
	return &Scheduler{
		Store:    store,
		Board:    board,
		Interval: 30 * time.Second,
		OnNotify: func(Notification) {},
	}
}

func (s *Scheduler) Run(ctx context.Context) error {
	if s.Store == nil {
		<-ctx.Done()
		return ctx.Err()
	}
	interval := s.Interval
	if interval <= 0 {
		interval = 30 * time.Second
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			notifications, err := s.Store.TriggerDue(time.Now())
			if err != nil {
				if s.Board != nil {
					s.Board.Set("assistant_error", err.Error())
				}
				continue
			}
			for _, note := range notifications {
				if s.Board != nil {
					s.Board.Set("assistant_notification", note.Message)
				}
				s.OnNotify(note)
			}
		}
	}
}
