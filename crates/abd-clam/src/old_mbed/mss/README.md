# The Mass-Spring System

The Mass-Spring System (MSS) is a simple physics simulation that models the behavior of masses connected by springs.
We use the MSS to perform a dimensionality reduction from an arbitrary metric space to a 3D Euclidean space.
We use a mass in 3D to represent a point or a cluster of points in the original metric space.
We then strategically attach springs between the masses to preserve the structure of the original metric space.
Each spring has a rest length equal to the distance between the two masses it connects in the original metric space.
The MSS is then allowed to relax to a stable configuration, and the 3D positions of the masses are used as the reduced representation of the original metric space.

The underlying algorithm relies on the Damped Harmonic Oscillator to simulate the behavior of the masses and springs.

## The Damped Harmonic Oscillator

A Harmonic Oscillator is a system that, when displaced from its equilibrium position, experiences a restoring force proportional to the displacement.
The Damped Harmonic Oscillator is a Harmonic Oscillator with a damping force proportional to the velocity of the system.
See [Wikipedia](https://en.wikipedia.org/wiki/Harmonic_oscillator) for more information.

The Damped Harmonic Oscillator is described by the following differential equation:

$$
m x'' + c x' + k x = 0
$$

where:

- $m$ is the mass of the system,
- $c$ is the damping coefficient,
- $k$ is the spring constant,
- $x$ is the displacement of the system from its equilibrium position,
- $x'$ is the velocity of the system, and
- $x''$ is the acceleration of the system.

This can be rewritten into the form:

$$
x'' + 2 \zeta \omega_{0} x' + \omega_{0}^2 x = 0
$$

where:

- $\zeta = \frac{c}{2\sqrt{m k}}$ is the damping ratio, and
- $\omega_{0} = \sqrt{\frac{k}{m}}$ is the natural frequency of the system.

For the MSS, we want an under-damped system, where $\zeta < 1$.
This means that the system will oscillate around the equilibrium position before coming to rest.

The solution to the Damped Harmonic Oscillator is:

$$
x(t) = A e^{-\zeta \omega_{0} t} \sin(\sqrt{1 - \zeta^2} \omega_{0} t + \phi)
$$

where:

- $A$ is the amplitude of the oscillation, and
- $\phi$ is the phase of the oscillation.

The amplitude and phase are chosen to satisfy the initial conditions of the system.

From the solution, we can see that the velocity of the system is:

$$
x'(t) = A \omega_{0} e^{-\zeta \omega_{0} t} \left( \sqrt{1 - \zeta^2} \cos(\sqrt{1 - \zeta^2} \omega_{0} t + \phi) - \zeta  \sin(\sqrt{1 - \zeta^2} \omega_{0} t + \phi) \right)
$$

Using the initial conditions that $x(0) = 0$ and $x'(0) = 0$, we can solve for $A$ and $\phi$:

$$
0 = A \sin(\phi) \Rightarrow \phi = 0, \pm \pi, \pm 2\pi, \ldots
$$

We choose $\phi = 0$ for simplicity.
Thus, we have:

$$
A = \frac{1}{\omega_{0} \sqrt{1 - \zeta^2}}
$$

Thus, the solution to the Damped Harmonic Oscillator with our initial conditions is:

$$
x(t) = \frac{1}{\omega_{0} \sqrt{1 - \zeta^2}} e^{-\zeta \omega_{0} t} \sin(\sqrt{1 - \zeta^2} \omega_{0} t)
