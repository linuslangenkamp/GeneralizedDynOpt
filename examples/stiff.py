from gdopt import *

model = Model("stiff")

x = model.addState(start=0.15)
y = model.addState(start=1)

u = model.addInput(lb=0, ub=1)

# TODO: fix me! This runtime parameter doesnt work properly somehow.
# maxStiffness = model.addRuntimeParameter(default=5.5, symbol="maxStiffness")

stiffness = model.addParameter(lb=-250, ub=-11.5)

model.addDynamic(x, stiffness * (x + y * cos(t)) + u)
model.addDynamic(y, x * exp(u) / (u**2 + 1))

model.addMayer(x**2 + y**2, Objective.MINIMIZE)

model.generate()

model.optimize(
    tf=0.25,
    steps=100,
    rksteps=5,
    flags={"linearSolver": LinearSolver.MUMPS, "initVars": InitVars.SOLVE},
    meshFlags={"algorithm": MeshAlgorithm.L2_BOUNDARY_NORM, "iterations": 5},
)

model.plotVarsAndRefinement(dotsMesh=Dots.BASE)
