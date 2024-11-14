from gdopt import *

model = Model("hypersensitive")

x = model.addState(start=1)

u = model.addInput()

model.addDynamic(x, -(x**3) + u)

model.addFinal(1.5 - x, eq=0)

model.addLagrange(0.5 * (x**2 + u**2))

print(model)

model.generate()

model.optimize(
    tf=10000,
    steps=30,
    rksteps=9,
    flags={"linearSolver": LinearSolver.MA57},
    meshFlags={
        "algorithm": MeshAlgorithm.L2_BOUNDARY_NORM,
        "refinementMethod": RefinementMethod.LINEAR_SPLINE,
        "iterations": 20,
        "muStrategyRefinement": MuStrategy.MONOTONE,
        "muInitRefinement": 1e-12,
    },
)

model.plotInputs(interval=[9980, 10000], dots=Dots.ALL)
model.plotInputsAndRefinement(interval=[9900, 10000], dotsMesh=Dots.BASE)
