package micromodel

import "math"

// -----------------------------------------------------------------------
// Selective State Space Model (S6) — core Mamba operation.
//
// Implements the selective scan from "Mamba: Linear-Time Sequence Modeling
// with Selective State Spaces" (Gu & Dao, 2023).
//
// The key insight: B, C, and Δ (dt) are input-dependent, making the SSM
// selective — it can choose what to remember and what to ignore.
//
// Recurrence per timestep t:
//   h_t = Ā·h_{t-1} + B̄·x_t
//   y_t = C_t·h_t + D·x_t
//
// Where Ā = exp(Δ·A), B̄ = Δ·B (zero-order hold discretization).
//
// Complexity: O(seqLen × dInner × dState) — linear in sequence length.
// Memory: O(dInner × dState) for the hidden state — constant per step.
// -----------------------------------------------------------------------

// SelectiveScan runs the core SSM recurrence.
//
//   x:  (seqLen, dInner) — input after conv1d + SiLU
//   dt: (seqLen, dInner) — discretization timestep (after softplus)
//   A:  (dInner, dState) — state transition (negative, from -exp(ALog))
//   B:  (seqLen, dState) — input-dependent input matrix
//   C:  (seqLen, dState) — input-dependent output matrix
//   D:  (dInner)         — skip connection
//
// Returns: (seqLen, dInner)
func SelectiveScan(x []float32, seqLen, dInner int, dt []float32, A []float32, dState int, B, C, D []float32) []float32 {
	y := make([]float32, seqLen*dInner)

	// Hidden state: (dInner, dState) — persistent across timesteps
	h := make([]float32, dInner*dState)

	for t := 0; t < seqLen; t++ {
		tOff := t * dInner
		tSOff := t * dState

		for i := 0; i < dInner; i++ {
			dtVal := dt[tOff+i]
			xVal := x[tOff+i]
			iSOff := i * dState

			var yVal float32
			for s := 0; s < dState; s++ {
				// Discretize: Ā = exp(dt * A), B̄ = dt * B
				aBar := float32(math.Exp(float64(dtVal * A[iSOff+s])))
				bBar := dtVal * B[tSOff+s]

				// State update: h = Ā·h + B̄·x
				h[iSOff+s] = aBar*h[iSOff+s] + bBar*xVal

				// Output: y += C·h
				yVal += C[tSOff+s] * h[iSOff+s]
			}

			// Skip connection: y += D·x
			y[tOff+i] = yVal + D[i]*xVal
		}
	}

	return y
}

// SelectiveScanBackward computes gradients for the selective scan.
// Returns gradients for x, dt, B, C, D.
func SelectiveScanBackward(
	dy []float32, // (seqLen, dInner) — upstream gradient
	x []float32, seqLen, dInner int,
	dt []float32,
	A []float32, dState int,
	B, C, D []float32,
) (dx, ddt, dA, dB, dC, dD []float32) {
	dx = make([]float32, seqLen*dInner)
	ddt = make([]float32, seqLen*dInner)
	dA = make([]float32, dInner*dState)
	dB = make([]float32, seqLen*dState)
	dC = make([]float32, seqLen*dState)
	dD = make([]float32, dInner)

	// Forward pass to reconstruct hidden states
	hAll := make([]float32, (seqLen+1)*dInner*dState) // h[t] for t=0..seqLen
	for t := 0; t < seqLen; t++ {
		tOff := t * dInner
		tSOff := t * dState
		for i := 0; i < dInner; i++ {
			dtVal := dt[tOff+i]
			xVal := x[tOff+i]
			iSOff := i * dState
			hPrev := t * dInner * dState
			hCur := (t + 1) * dInner * dState
			for s := 0; s < dState; s++ {
				aBar := float32(math.Exp(float64(dtVal * A[iSOff+s])))
				bBar := dtVal * B[tSOff+s]
				hAll[hCur+iSOff+s] = aBar*hAll[hPrev+iSOff+s] + bBar*xVal
			}
		}
	}

	// Backward pass: reverse through time
	dh := make([]float32, dInner*dState) // gradient for hidden state

	for t := seqLen - 1; t >= 0; t-- {
		tOff := t * dInner
		tSOff := t * dState
		hCur := (t + 1) * dInner * dState
		hPrev := t * dInner * dState

		for i := 0; i < dInner; i++ {
			dtVal := dt[tOff+i]
			xVal := x[tOff+i]
			iSOff := i * dState
			dyVal := dy[tOff+i]

			// Gradient from skip connection
			dD[i] += dyVal * xVal
			dx[tOff+i] += dyVal * D[i]

			for s := 0; s < dState; s++ {
				aBar := float32(math.Exp(float64(dtVal * A[iSOff+s])))
				bBar := dtVal * B[tSOff+s]

				// dy/dC: y = C·h → dC += dy * h
				dC[tSOff+s] += dyVal * hAll[hCur+iSOff+s]

				// dy/dh: y = C·h → dh += dy * C
				dhVal := dyVal*C[tSOff+s] + dh[iSOff+s]

				// dh/dh_prev: h = aBar*h_prev + ... → dh_prev += dhVal * aBar
				dh[iSOff+s] = dhVal * aBar

				// dh/dA: h = exp(dt*A)*h_prev → dA += dhVal * h_prev * aBar * dt
				hPrevVal := hAll[hPrev+iSOff+s]
				dA[iSOff+s] += dhVal * hPrevVal * aBar * dtVal

				// dh/ddt: via aBar and bBar
				// aBar = exp(dt*A) → d_aBar/d_dt = A*aBar
				// bBar = dt*B → d_bBar/d_dt = B
				ddt[tOff+i] += dhVal * (hPrevVal*aBar*A[iSOff+s] + B[tSOff+s]*xVal)

				// dh/dB: h = ... + dt*B*x → dB += dhVal * dt * x
				dB[tSOff+s] += dhVal * dtVal * xVal

				// dh/dx: h = ... + bBar*x → dx += dhVal * bBar
				dx[tOff+i] += dhVal * bBar
			}
		}
	}

	return
}

// -----------------------------------------------------------------------
// Activation functions
// -----------------------------------------------------------------------

// silu computes SiLU (Swish) activation: x * sigmoid(x).
// SiLU is smoother than ReLU and used throughout Mamba.
func silu(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * sigmoid(v)
	}
	return out
}

// siluBackward computes the gradient of SiLU.
// d/dx[x·σ(x)] = σ(x) + x·σ(x)·(1-σ(x)) = σ(x)·(1 + x·(1-σ(x)))
func siluBackward(x, dy []float32) []float32 {
	dx := make([]float32, len(x))
	for i, v := range x {
		s := sigmoid(v)
		dx[i] = dy[i] * s * (1 + v*(1-s))
	}
	return dx
}

// sigmoid computes 1 / (1 + exp(-x)).
func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

// softplus computes log(1 + exp(x)), a smooth approximation of ReLU.
// Used for ensuring dt (discretization timestep) is positive.
func softplus(x []float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		if v > 20 {
			out[i] = v // avoid overflow
		} else {
			out[i] = float32(math.Log1p(math.Exp(float64(v))))
		}
	}
	return out
}

// softplusBackward computes gradient of softplus: sigmoid(x).
func softplusBackward(x, dy []float32) []float32 {
	dx := make([]float32, len(x))
	for i, v := range x {
		dx[i] = dy[i] * sigmoid(v)
	}
	return dx
}

// -----------------------------------------------------------------------
// 1D Causal Convolution
// -----------------------------------------------------------------------

// conv1D applies a causal 1D convolution along the sequence dimension.
// Each of the dInner channels has its own kernel of size kSize.
//
//   x:      (seqLen, dInner) — input
//   weight: (dInner, kSize)  — per-channel kernels
//   bias:   (dInner)         — per-channel bias
//
// Causal: only uses current and past positions (no future leakage).
// Returns: (seqLen, dInner)
func conv1D(x []float32, seqLen, dInner int, weight []float32, bias []float32, kSize int) []float32 {
	out := make([]float32, seqLen*dInner)

	for t := 0; t < seqLen; t++ {
		for ch := 0; ch < dInner; ch++ {
			var sum float32
			for k := 0; k < kSize; k++ {
				pos := t - k // causal: look back k steps
				if pos >= 0 {
					sum += x[pos*dInner+ch] * weight[ch*kSize+k]
				}
			}
			out[t*dInner+ch] = sum + bias[ch]
		}
	}

	return out
}

// conv1DBackward computes gradients for the causal 1D convolution.
// Returns gradients for input x, weights, and bias.
func conv1DBackward(dy []float32, x []float32, seqLen, dInner int, weight []float32, kSize int) (dx, dw, db []float32) {
	dx = make([]float32, seqLen*dInner)
	dw = make([]float32, dInner*kSize)
	db = make([]float32, dInner)

	for t := 0; t < seqLen; t++ {
		for ch := 0; ch < dInner; ch++ {
			g := dy[t*dInner+ch]
			db[ch] += g
			for k := 0; k < kSize; k++ {
				pos := t - k
				if pos >= 0 {
					dw[ch*kSize+k] += g * x[pos*dInner+ch]
					dx[pos*dInner+ch] += g * weight[ch*kSize+k]
				}
			}
		}
	}

	return
}

// -----------------------------------------------------------------------
// Element-wise operations
// -----------------------------------------------------------------------

// vecMul multiplies two vectors element-wise.
func vecMul(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] * b[i]
	}
	return out
}

// vecScale multiplies a vector by a scalar.
func vecScale(x []float32, s float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * s
	}
	return out
}
