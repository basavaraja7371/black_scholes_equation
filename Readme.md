# Black–Scholes Equation — Crank–Nicolson Method (Matrix Formulation)


## 1. Black–Scholes Equation

The Black–Scholes PDE for an option price $V(S,t)$ is

$$
\frac{\partial V}{\partial t} =
-\frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
- rS \frac{\partial V}{\partial S}
+ rV,
\quad t \in [0,T]
$$

Introduce time-to-maturity

$$
\tau = T - t
$$

Then the PDE becomes

$$
\frac{\partial V}{\partial \tau}
= \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
+ rS \frac{\partial V}{\partial S}
- rV
$$

We now evolve **forward in $\tau$** from $0 \to T$.

---

## 2. Initial and Boundary Conditions

### Initial Condition (at $ \tau = 0 $, i.e. maturity)

- **European Call**
$$
V(S,0) = \max(S-K,0)
$$

- **European Put**
$$
V(S,0) = \max(K-S,0)
$$

---

### Spatial Domain

$$
S \in [0, S_{\max}], \quad S_{\max} = 2K
$$

---

### Boundary Conditions

#### Call Option

$$
V(0,\tau) = 0
$$

$$
V(S_{\max},\tau) \approx S_{\max} - K e^{-r\tau}
$$

#### Put Option

$$
V(0,\tau) = K e^{-r\tau}
$$

$$
V(S_{\max},\tau) = 0
$$

---

## 3. Spatial Discretization

Let

$$
S_i = i\Delta S, \quad i = 0,1,\dots,N
$$

Central differences:

$$
\frac{\partial V}{\partial S}
\approx
\frac{V_{i+1} - V_{i-1}}{2\Delta S}
$$

$$
\frac{\partial^2 V}{\partial S^2}
\approx
\frac{V_{i+1} - 2V_i + V_{i-1}}{\Delta S^2}
$$

---

## 4. Crank–Nicolson Discretization

The Crank–Nicolson scheme averages the spatial operator between time levels $n$ and $n+1$:

If
$$\frac{\partial V}{\partial t} = F(t, S, V_s, V_{ss})$$

then CN method gives,
$$
\frac{V_i^{n+1} - V_i^n}{\Delta \tau}
= \frac{1}{2} \ (F^{n+1} + F^{n})
$$

Substituting finite differences yields

$$
\frac{V_i^{n+1} - V_i^n}{\Delta \tau}
= \frac{1}{2}
\Big[
a_i (V_{i-1}^{n+1} + V_{i-1}^n)
+ b_i (V_i^{n+1} + V_i^n)
+ c_i (V_{i+1}^{n+1} + V_{i+1}^n)
\Big]
$$

---

## 5. Coefficients $a_i, b_i, c_i$

$$
a_i
= \frac{1}{2}\frac{\sigma^2 S_i^2}{\Delta S^2}
- \frac{rS_i}{2\Delta S}
$$

$$
b_i
= -\frac{\sigma^2 S_i^2}{\Delta S^2}
- r
$$

$$
c_i
= \frac{1}{2}\frac{\sigma^2 S_i^2}{\Delta S^2}
+ \frac{rS_i}{2\Delta S}
$$

---

## 6. Rearranged CN Equation

Multiply both sides by $\Delta \tau$ and rearrange:

$$
V_i^{n+1}
- \frac{\Delta \tau}{2}
\left(
a_i V_{i-1}^{n+1}
+ b_i V_i^{n+1}
+ c_i V_{i+1}^{n+1}
\right)
=
V_i^n
+ \frac{\Delta \tau}{2}
\left(
a_i V_{i-1}^n
+ b_i V_i^n
+ c_i V_{i+1}^n
\right)
$$

---

## 7. Matrix Form — Definition of $A$

Let the interior solution vector be

$$
\mathbf{V}^n = [V_1^n, V_2^n, \dots, V_{N-1}^n]^T
$$

Define the tridiagonal matrix $A$:

$$
A =
\frac{\Delta \tau}{2}
\begin{bmatrix}
b_1 & c_1 & 0 & \cdots \\
a_2 & b_2 & c_2 & \cdots \\
0 & a_3 & b_3 & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{bmatrix}
$$

---

## 8. Final Crank–Nicolson System

$$
\boxed{
(I - A)\mathbf{V}^{n+1}
= (I + A)\mathbf{V}^n
+ \mathbf{b}
}
$$

---

## 9. Boundary Vector $ \mathbf{b} $

Boundary conditions introduce additional terms:

$$
\mathbf{b} =
\begin{bmatrix}
b_1 \\
0 \\
\vdots \\
b_{N-1}
\end{bmatrix}
$$

where

$$
b_1 = \frac{\Delta \tau}{2} a_1 (V_0^{n+1} + V_0^n)
$$

$$
b_{N-1} = \frac{\Delta \tau}{2} c_{N-1} (V_N^{n+1} + V_N^n)
$$

All interior components are zero.

---

## 10. Time-Stepping Algorithm

1. Initialize $ \mathbf{V}^0 $ from payoff
2. For each time step $n$:
   - Evaluate boundary values $V_0^n, V_N^n$
   - Assemble boundary vector $\mathbf{b}$
   - Solve
     $$
     (I - A)\mathbf{V}^{n+1} = (I + A)\mathbf{V}^n + \mathbf{b}
     $$
3. Reconstruct full solution including boundaries

---

## 11. Analytical Solution (Validation)

European Call:

$$
V(S,t) = S\Phi(d_1) - K e^{-r(T-t)}\Phi(d_2)
$$
European Put:
$$
V(S,t) =K e^{-r(T-t)} \Phi(-d_2) -S \Phi(-d_1)
$$

Where, 
$$
d_1 = \frac{\ln(S/K) + (r + \tfrac{1}{2}\sigma^2)(T-t)}{\sigma\sqrt{T-t}},
\quad
d_2 = d_1 - \sigma\sqrt{T-t}
$$


Used to verify numerical accuracy.
