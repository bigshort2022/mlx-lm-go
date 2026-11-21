package mlx

import (
	"runtime"
	"testing"
)

// MLX streams are thread affine, so the package must pick up the default GPU
// stream of whichever thread it is called on rather than the stream of the
// thread that loaded it.
func TestOperationsRunOnAnyPinnedThread(t *testing.T) {
	// Two goroutines hold two different OS threads at the same time and take
	// turns, because the package is single-threaded by contract.
	turns := [2]chan struct{}{make(chan struct{}), make(chan struct{})}
	ready := make(chan struct{})
	errs := make(chan error)
	for _, turn := range turns {
		go func() {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			ready <- struct{}{}
			<-turn
			errs <- Scoped(func() error {
				a := FromFloat32([]float32{1, 2}, 2)
				Eval(Add(a, a))
				return nil
			})
		}()
	}
	<-ready
	<-ready
	for i, turn := range turns {
		close(turn)
		if err := <-errs; err != nil {
			t.Fatalf("thread %d: %v", i, err)
		}
	}
}
