from gdopt import *

model = Model("oilShalePyrolysis")

x1 = model.addState(start=1, symbol="kerogen")
x2 = model.addState(start=0, symbol="pyrolytic bitumen")
x3 = model.addState(start=0, symbol="oil & gas")
x4 = model.addState(start=0, symbol="organic carbon")

T = model.addInput(lb=698.15, ub=748.15, symbol="temperature")

k1 = exp(8.86 - (20300 / 1.9872) / T)
k2 = exp(24.25 - (37400 / 1.9872) / T)
k3 = exp(23.67 - (33800 / 1.9872) / T)
k4 = exp(18.75 - (28200 / 1.9872) / T)
k5 = exp(20.70 - (31000 / 1.9872) / T)

# its possible to remove states x3, x4 cause they are 'waste products' and used in the dynamic
model.addDynamic(x1, -k1 * x1 - (k3 + k4 + k5) * x1 * x2)
model.addDynamic(x2, k1 * x1 - k2 * x2 + k3 * x1 * x2)
model.addDynamic(x3, k2 * x2 + k4 * x1 * x2)
model.addDynamic(x4, k5 * x1 * x2)

model.addMayer(x2, Objective.MAXIMIZE)

model.generate()

model.optimize(
    tf=8,
    steps=50,
    rksteps=3,
    flags={"tolerance": 1e-15, "linearSolver": LinearSolver.MA57},
    meshFlags={
        "algorithm": MeshAlgorithm.L2_BOUNDARY_NORM,
        "iterations": 5,
        "muStrategyRefinement": MuStrategy.MONOTONE,
        "muInitRefinement": 1e-16,
    },
)

model.getResults(5)
model.plot(specifCols=["pyrolytic bitumen", "temperature"])
for i in range(len(model.resultHistory[5]["time"])):
    print(model.resultHistory[5]["time"][i], model.resultHistory[5]["temperature"][i] ,model.resultHistory[5]["pyrolytic bitumen"][i])
model.plotInputsAndRefinement(dotsGraph=Dots.BASE)
