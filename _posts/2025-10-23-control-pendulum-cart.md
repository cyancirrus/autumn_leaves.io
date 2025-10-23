---
layout: post
title: "An Introductory Project for Control Theory"
date: 2025-10-23
tags: [c, c++, control-theory, backend, learning]
---

**Implementations**
- [Cart Pendulum](https://github.com/cyancirrus/stablestar/blob/main/src/cart_pendulum.cpp)
- [Inverted Pendulum](https://github.com/cyancirrus/stablestar/blob/main/src/pendulum.cpp)
- [Visualization](https://github.com/cyancirrus/stablestar/blob/main/src/main.cpp)

## Control Theory in Action
![Model Performance](./assets/control_simulation.gif)

What _is_ control, and beyond this, what is control theory?

Mathematically, control is taking an action such that a previously unstable state becomes stable after some form of intervention.  
Today, I'll go through one of the most basic examples often seen in a control theory course and discuss how to model the problem both mathematically and programmatically.

The problem we’ll explore today is the inverted pendulum. Imagine that while raking leaves this fall, you try to balance the rake upside down, with the heavy end in the air and the handle balanced on your palm.

As the rake tips forward, you move your hand forward to ‘right’ it again, and if it tips backward, you pull back. This is a simple, intuitive example of feedback control — a perfect introduction to several ideas within the control space.

## A Simplified Model

*Simplifying Assumptions*

Let’s start with a few simplifying assumptions:
- All of the mass is at the top of the “rake” where it picks up leaves.
- We’re balancing the rake in 2D, as if it were sandwiched between sheets of glass.
- We can magically apply a corrective force in any direction with no side effects.

### Derivation of Inverted Pendulum

Similarly to $F = M \cdot a$, for the pendulum we have:

>> $\tau = r \times F = L \cdot m g \cdot \sin(\theta)$

Using $\tau = I \cdot \alpha$ where $\alpha = \ddot{\theta}$ and $I = m L^2$ for a point mass gives:

>> $mL^2 * \ddot{\theta} = m g L * \sin(\theta)$

>> $\ddot{\theta} = \frac{g}{L} * \sin(\theta)$

We can ask ourselves what kind of function produces this form. Expressing the solution as:

>> $\lambda = \pm\sqrt{g/L}$

>> $a \cdot e^{\lambda t} + b \cdot e^{-\lambda t}$

Note that this function has no stable equilibrium — deviations grow exponentially. Unlike the normal pendulum, there’s no stable point except for perfectly balancing it.  
However, we can utilize control to ensure the system remains stable and follows our objectives.

### Control Dynamics

Let’s consider **Proportional Derivative (PD)** control.

We define:
- $k_p$ := proportional gain  
- $k_d$ := derivative gain  

Essentially, $k_p$ scales based on the absolute deviation, and $k_d$ helps prevent overshoot. We apply a control torque opposing the pendulum’s motion:

>> $\tau_{control} = k_p \cdot \theta + k_d \cdot \dot{\theta}$

The dynamics become:

>> $mL^2 \cdot \ddot{\theta}(t) = m g L \cdot \sin(\theta(t)) - \tau_{control}$

Dividing by $mL^2$ and linearizing with $\sin(\theta) \approx \theta$:

>> $\ddot{\theta}(t) = \frac{g}{L} \cdot \theta(t) - \frac{k_p}{mL^2} \cdot \theta(t) - \frac{k_d}{mL^2} \cdot \dot{\theta}(t)$

The exact scaling of $k_p$ and $k_d$ depends on the system’s physical parameters, but the key insight is that we oppose both position and velocity deviations to achieve stability.

### Coupled Control — How Do We Actually Balance the Rake?

The above derivation gives us a sketch of the system, but it won’t actually balance our rake — our favorite fall activity!  
Our hand doesn’t move at the speed of light, and it will also drift. How can we express the dynamics of our balance (the hand) alongside the dynamics of the rake (the pendulum) that’s trying to fall?

To do this, we introduce the standard state-space model from control theory:

>> $\dot{x} = A\vec{x} + B\vec{u}$

We now represent the state with both the pendulum and the cart:

>> $\vec{x} = [\theta, \dot{\theta}, x, \dot{x}]^T$

We want to find $A$ so that the derivatives match. Trivially:

>> $\frac{d}{dt} \theta(t) = \dot{\theta}(t)$

>> $\frac{d}{dt} x(t) = \dot{x}(t)$

So we need to determine what $\ddot{\theta}$ and $\ddot{x}$ are. Let’s consider where the pendulum tip is:

>> $x_p = x_0 + L \cdot \sin(\theta)$

>> $\ddot{x}_p = \ddot{x}_0 + L \cdot [\ddot{\theta} * \cos(\theta) - \dot{\theta}^2 * \sin(\theta)]$

Linearizing for small $\theta$:

>> $\ddot{x}_p \approx \ddot{x_0} + L \cdot \ddot{\theta}$

Finally, include the effect of the pendulum pushing on the cart as it falls:

>> $\ddot{\theta}(t) = \frac{g}{L} \cdot \sin(\theta) - \frac{\ddot{x}}{L} \cdot \cos(\theta)$

Substituting the expression for $\ddot{x}$ into the pendulum equation and linearizing $\cos(\theta) \approx 1$, we obtain the coupled system:

>> $\ddot{\theta} = \frac{(M + m)g}{ML} \cdot \theta - \frac{F}{ML}$

>> $\ddot{x} = -\frac{mg}{M} \cdot \theta + \frac{1}{M} \cdot F$

Finally, we have:

>> $\dot{x} = A\vec{x} + B\vec{u}$

>> $u = F$

>> $\vec{B} = [0, -\frac{1}{ML}, 0, \frac{1}{M}]^T$

### Back to Programming

Now that we’ve derived our model, the next step is to implement it in code.  
If we note that $\ddot{\theta}$ doesn’t depend directly on $x$, we can first determine the force, integrate for $\theta$, and then do the same for $x$.

```cpp
void PendulumCart::control(float dt, float kp, float kd) {

    float force = kp * theta + kd * theta_dot;

    theta_dot += dt * ((mass_cart + mass_pendulum) * g * \sin(theta) - force) / (mass_cart * length_pendulum);

    theta += dt * theta_dot;

    // friction to help with the visualization; does not exist in the ideal system
    x_dot *= 0.99;

    x_dot += dt * (force - mass_pendulum * g * theta) / mass_cart;

    x += dt * x_dot;
}
```

That’s essentially the main control loop within the simulation!
After deriving the state-space model, it’s straightforward to obtain a numerical solution for the dynamics.
While one could use autodiff or matrix solvers ($O(n^3)$ operations), this system is simple enough to iterate directly as shown.

Of course, not all control systems are this transparent. In many cases, models are derived automatically or optimized using advanced techniques.
But this example makes for a great primer to brush up on control concepts and experiment with different control gains to see their effect.

### Summary

This was a fun break from deep implementation work — even though it still felt quite hands-on. It gave me a chance to brush up on my C++ and revisit some core control theory ideas.

Control theory is a fascinating and deep field, encompassing concepts like **observability**, **controllability**, **optimal control**, and **stochastic control**.
I hope you’ve enjoyed this quick dive as much as I did!

Next week, I’m hoping to extend these results and tackle something more challenging — perhaps building a fully asynchronous Kalman Filter or expanding on these ideas further. We’ll see — I’m sure I’ll find something fun!
