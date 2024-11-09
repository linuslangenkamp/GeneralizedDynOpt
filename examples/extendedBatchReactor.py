from gdopt import *

model = Model("extendedBatchReactor")

x1 = model.addState(symbol="Reactant", start=1)
depl = model.addState(symbol="deplReactant", start=0)
x2 = model.addState(symbol="Product", start=0)

# guesses can be quite complicated
u = model.addInput(
    symbol="u",
    lb=0,
    ub=5,
    guess=guessPiecewise((0.6, t <= 1 / 2), (guessQuadratic(0.7, 0.8, 5), 1 / 2 < t)),
)

R_v = model.addRuntimeParameter(default=1, symbol="REACT_SPEED")
D_v = model.addRuntimeParameter(default=1, symbol="DEPLETION_SPEED")

model.addDynamic(depl, u**2 / 2 * D_v * x1)
model.addDynamic(x1, -(u * R_v + u**2 / 2 * D_v) * x1)
model.addDynamic(x2, u * x1 * R_v)

model.addMayer(x2, Objective.MAXIMIZE)

model.hasLinearObjective()

model.generate()

model.optimize(
    tf=1,
    steps=250,
    rksteps=3,
    flags={
        "linearSolver": LinearSolver.MA57,
        "initVars": InitVars.SOLVE,
        "exportJacobianPath": "/tmp",
    },
    meshFlags={"algorithm": MeshAlgorithm.L2_BOUNDARY_NORM, "iterations": 6},
)

model.plotInputs(dots=Dots.ALL)
model.plotSparseMatrix(MatrixType.JACOBIAN)
model.printResults()
model.plotInputsAndRefinement()
