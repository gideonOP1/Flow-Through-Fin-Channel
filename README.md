# Flow-Through-A-Fin-Channel
This is a simulation of an incompressible fluid flowing through a set of channels written in Python by using NumPy and MatPlotLib libraries

## Brief Details
A *projection method* which decouples the pressure gradient and the advection/diffusion terms of the Navier Stokes Equation is used to solve the problem. A *Forward in time and Central in Space (FTCS) scheme* is employed to solve ifferential equations numerically. A *five point stencil* is used to implement the laplace operator. A time step limit is implemented which prevents the numerical solution to become unstable. *Drichlet Boundary Conditions* are used for Velocity and *Neumann Boundary Conditions* are used for Pressure. *tqdm* library is used to monitor the iterations.

## References
[12 Steps to Navier Stokes by Prof Lorena Barba](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/)

[uCFD - 4 Steps To Navier Stokes by Prof Tony Saad](http://www.tonysaad.net/ucfd/)

[Solving the Navier-Stokes equations in Python](https://www.youtube.com/watch?v=BQLvNLgMTQE&ab_channel=MachineLearning%26Simulation)
