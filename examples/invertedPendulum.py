from gdopt import *

# choose implicit Euler, since it's the most stable
# maybe the mu globalization needs to be disabled here

model = Model("invertedPendulum")

Ms = model.addRuntimeParameter(default=1, symbol="Ms")
Mp = model.addRuntimeParameter(default=0.5, symbol="Mp")
R = model.addRuntimeParameter(default=1, symbol="R")
G = -9.81

s = model.addState(start=0)
v = model.addState(start=0)
phi = model.addState(start=pi)
omega = model.addState(start=0)

u = model.addInput(lb=-15, ub=15, guess=0.1 * t)

dvdt = u + (sin(-phi) * Mp * R * omega**2 - cos(phi) * sin(-phi) * Mp * G) / (
    Ms + Mp * sin(-phi) ** 2
)

model.addDynamic(s, v)
model.addDynamic(v, dvdt)
model.addDynamic(phi, omega)
model.addDynamic(omega, (sin(-phi) * G - cos(phi) * dvdt) / R)

# this seems like a dirty hack
# the problem has MANY local optima. Therefore it is hard to converge to the global optimum, if only a poor guess is provided.
# adding this contraints FORCES the global optimal solution,
# since the lagrange intergrand has to be bounded
lagrange = sin(phi / 2) ** 2
model.addPath(piecewise((lagrange, t > 1)), ub=0.1)
model.addLagrange(u**2, Objective.MINIMIZE)

model.generate()

model.setValue(Ms, 1.5)

model.optimize(
    steps=5000,
    rksteps=1,
    tf=5,
    flags={"linearSolver": LinearSolver.MA27, "kktMuGlobalization": False},
)

model.plot(meshIteration=0)
