from gdopt import *

# MWE with u*(t) = 1 constant, min int u(t)² dt s.t. int u(t) dt = 1.

hw = Model("Hello World")

x = hw.addState(start=0)
u = hw.addControl(
    lb=0, ub=2, guess=(-2 + 2 * sqrt(exp(1))) / (1 + (-1 + sqrt(exp(1))) * t)
)

hw.addDynamic(x, 2 * t - u)

hw.addFinal(x, eq=0)

hw.addLagrange(u**2, obj=Objective.MINIMIZE)

hw.hasLinearConstraints()
hw.hasQuadraticObjective()

# steps=1, rksteps=2 would be sufficient for the analytic solution!
hw.solve(tf=1, steps=25, rksteps=3, flags={"tolerance": 1e-14})

hw.plot(dots=Dots.ALL)
