from gdopt import *

model = Model("orbitRaising")

r = model.addState(start=1, lb=0, ub=2)
theta = model.addState(start=0, lb=0, ub=pi)
vr = model.addState(start=0, lb=0, ub=2)
vtheta = model.addState(start=1, lb=0, ub=2)

u1 = model.addControl(lb=-1, ub=1, guess=1)
u2 = model.addControl(lb=-1, ub=1, guess=1)

a = 0.1405 / (1 - 0.0749 * t)
model.addPath(u1**2 + u2**2, eq=1)

model.addF(r, vr)
model.addF(theta, vtheta / r)
model.addF(vr, vtheta**2 / r - 1 / r**2 + a * u1)
model.addF(vtheta, -vtheta * vr / r + a * u2)

model.addFinal(vr, eq=0)
model.addFinal(1 / r - vtheta**2, eq=0)

model.addMayer(r, Objective.MAXIMIZE)

model.setTolerance(1e-10)
model.setMeshIterations(5)
model.setMeshAlgorithm(MeshAlgorithm.L2_BOUNDARY_NORM)
model.setMuStrategyRefinement(MuStrategy.MONOTONE)
model.setMuInitRefinement(1e-14)

model.solve(tf=3.32, steps=25, rksteps=5)
model.plot(dots=Dots.ALL)
