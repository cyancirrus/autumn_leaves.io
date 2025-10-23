---
layout: post
title: "A Introductory Project for Control Theory"
date: 2025-10-23
tags: [c, c++, control-theory, backend, learning] ---
---

**Implementations**
- [Cart Pendulum](https://github.com/cyancirrus/stablestar/blob/main/src/cart_pendulum.cpp)
- [Inverted Pendulum](https://github.com/cyancirrus/stablestar/blob/main/src/pendulum.cpp)
- [Visualization](https://github.com/cyancirrus/stablestar/blob/main/src/main.cpp)

## Control Theory in Action
![Model Performance](./assets/control_simulation.gif)

What _is_ control and beyond this what is control theory?

Control mathematically is taking an action such that a previously unstable state has now become stable after some sort of intervention.
Today I'll be going through one of the most basic examples which are seen in a control theory course and talking through how to both model the problem mathematically and also how to model the problem programatically.

The problem we'll be exploring today is the inverted pendulum -- Imagine as you're raking leaves this fall, you try to balance the rake upside down where the heavy end is in the air and ur balancing the rake on your palm.

As the rake tips forward you can swing your arm forward in order to 'right' the rake again, and if the rake is falling backwards you can pull back and right the rake as well. This will be a great intro several ideas within the Control Space.

## A Simplified Model

*Simplifying Assumptions*

First lets make a couple of simplifying assumptions
- All of the mass is at the top of the 'rake' where it picks up leaves
- Lets assume that we somehow balancing the rake in 2-dimensions, almost like it were sandwhiched between plates of glass
- Lets first assume we can magically just apply a force to a direction to correct the position with no side effects

### Derivation of Inverted Pendulum

Similarly to the $F = M \cdot a$ equation for motion for pendulums we have the following identity $Tau = r \times F$.

Differentiating wr.t. t we have 

$Tau = \dot {I} \vec {w} + I \dot {\vec {w}} vec{n}$


$w = \lVert\vec{r}\rVert * \lVert\vec{w}\rVert sin {Theta } \vec {n} \div \lVert\vec{r}\rVert ^ 2
Carefully note the sine theta. Well be utilizing this cross product result.

Where 'n' denotes the normal vector and 2-d we can treat as a scaler, after a bit of algebra we get that 

After a bit of recductions we obtain the first main result we'll be using.
$\ddot\Theta(t) = g \div l \cdot sin {Theta}$


We simply can ask ourselves what function would produce this form. We'll express solutions as
$a \cdot exp{w \cdot Theta{t}} k + b \cdot exp{-w \cdot Theta{t}}$

Note that the above function has no stable equilibria and deviations will increase. Unlike the normal pendulum there is no stable point except for perfectly balancing the pendulum.
However, we utilize control to ensure that our system remains in valid territory and abides by our objective.

### Control Dynamics

Lets consider Proporitional Derivative (PD) control.

We'll define two terms
$k_p := \text{proportional gain}$
$k_d := \text{derivative gain}$

Essentially $k_p$ will scale based upon the absolute deviation and $k_d$ will help to prevent the system from overshooting.
We'll now revise our dynamics to show these new forces in our system.


$\ddot\Theta{t} = g \div l \cdot sin {Theta{t}} - k_p \cdot \Theta{t} - k_d \cdot \dot\Theta{t}$

Linearizing around theta bc $sin{Theta} \approx {Theta}$ for small theta, which is our objective, ie balancing the rake straight up in the air. 

$\ddot\Theta{t} = Theta{t} \cdot (g \div l  - alpha\cdot\k_p ) - alpha\cdot\k_d \cdot \dot\Theta{t}$

There are finer ways one can scale these factors but this is enough depth for an entry point, just remember $alpha$ and $beta$ can be finitely determined. One simply needs to put back $Theta{t}$ in terms of torque so that the control which is also phrased in torque are on the same footing.

### Coupled Control - How do we actually balance the rake?


Above we've derived a quick sketch on how we can express our system - but this won't let us balance our rake -- which is our favourite fall time activity! Our hand doesn't move at the speed of light, and our hand will also drift through the space. How can we express the dynamics of our balance alongside the dynamics of our rake - which so eagerly wishes to fall?


Perfect that's fine for idealized control but how do we do control with systems? Introduce the standard state space model for control theory
$\dot{x} = \matrix{A}\vec{x} + \matrix{B}\vec{u}$

We seek a representation for our data which will explain all of our dynamics, we have a new item we'll be tracking our hand _the cart_. So we'll be needing to include both $x$ and $\dot{x}$ for our new state space.

$\dot{vec{x}}=[Theta, \dot\Theta, x, \dot\x]'$

We then wish to find $\matrix{A}$ so that the derivatives are equal. Trivially we'll have
$d\div{dt} Theta{t} = \dot Theta{t}$
$d\div{dt} x{t} = \dot x{t}$

So we are  essentially we're trying to find what both $\ddot\Theta$ and $\ddot\x$ equal.

so lets slow down and think of where the pendulum for x is ie

$x_p = x_0 + l \cdot sin{theta}$

$\ddot\x_p = \ddot{x_0} + l\cdot\( \ddot{theta} *cos(theta) - (theta')^2 * sin(theta))$

Then we'll linearize with our approximations for $theta$.

$\ddot{x_p} \approx \ddot{x_0} + l \cdot \ddot\theta$

And finally we'll include a term for how the pendulum when it falls it will push on the cart in the opposite direction that the pendulum falls.
$\ddot{theta{t}} = g/l * sin{theta} - \ddot\x\div\l\cdot\cos{theta}$


We'll perform some algebra and arrive at the following for both $\ddot\x$ and $\ddot\theta$.

$\ddot{theta} = (M + mg)\div{Ml} - F\div{ML}$
$\ddot{x} = -{mg}\div{M}\cdot\Theta  + 1\div{M} \cdot F$

Finally we have solved and determined that 
$\dot{x} = \matrix{A}\vec{x} + \matrix{B}\vec{u}$

$u = F$

$vec{B} = [0, 1\div{m}, 0, -1\div{Ml}]$

### Back to Programming

We are *so* close to be doing so now the challenge is to implement our findings within code. If we note that $\ddot{Theta}$ doesn't depend upon $x$ we can simply determine the force and then find first $Theta$ and simply integrate. Afterwards we can apply the the same procedure for $x$.


```cpp
void PendulumCart::control(float dt, float kp, float kd) {

	float force = kp * theta + kd * theta_dot;

	theta_dot += dt * ((mass_cart + mass_pendulum) * g * sin(theta) - force) /(mass_cart * length_pendulum);

	theta += dt * theta_dot;

	// friction

	x_dot *= 0.99;

	x_dot += dt * (force -mass_pendulum * g * theta ) / mass_cart;

	x += dt * x_dot;

}
```

Essentially that's the main control within the simulation! It's quite trivially after the state space model derivation in order to get a solution. for the dynamics. If we were to use some sort of autodiff we'd have $O{n^3}$ opperations which for something so straight forward I simply iterated via the procedure above. 

There definitely do exist models where control is automatically derived and not all dynamics will be so transparent such that we utilize these methods to find closed form solutions. This is a special case of several simplifying assumptions and the fact that these are ODE's with known dynamics. However, everything was a great primer to brush up on several control theory concepts and to work with different proportions of control values and see how they interact.

Fun enough one can use many of the loss forms similar to statistics in order to perform a metaparameter search for optimization underneath your data.
 
### Summary

This was a fun break from deep diving into implementations, although even this felt a bit light on the implementation for myself, although provided me a chance to brush up on my C++. Control presents a super interesting form and there are incredibly deep topics not presented here - observability, controlability, optimal control, stochastic control. It's a very interesting field and I hope you have enjoyed as much as I have this quick dive!

Next week I hope to extend these results, and do something a bit more challanging, it would be nice to get up a fully async Kahlman Filter, or to extend on some of these ideas here. I"m not sure, I'm sure I'll find something fun!
