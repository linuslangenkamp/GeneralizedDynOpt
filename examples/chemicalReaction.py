from gdopt import *

"""
* Chemical Reaction from Adaptive mesh refinement method for solving
* optimal control problems using interpolation error analysis and improved data compression
* Zhao, Li
* DOI:10.13140/RG.2.2.28764.33922
"""

model = Model("chemicalReaction")

rho = 2.5
k = 1.5
umax = model.addRuntimeParameter(default=0.5, symbol="umax")

x = model.addX(start=1)
y = model.addX(start=0.01)

u = model.addU(lb=0.1, ub=umax)

model.addF(x, -u * x)
model.addF(y, u * x - rho * u**k * y)

model.addMayer(y, obj=Objective.MAXIMIZE)

model.generate()

model.optimize(
    tf=2,
    steps=10,
    rksteps=3,
    flags={
        "linearSolver": LinearSolver.MUMPS,
        "initVars": InitVars.SOLVE,
        "ipoptPrintLevel": 0,
    },
    meshFlags={
        "algorithm": MeshAlgorithm.L2_BOUNDARY_NORM,
        "iterations": 6,
        "muStrategyRefinement": MuStrategy.MONOTONE,
        "muInitRefinement": 1e-15,
    },
)

model.plot(dots=Dots.BASE)
