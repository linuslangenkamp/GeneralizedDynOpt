from gdopt import *

model = Model("rayleigh")

x = model.addState(start=-5, symbol="x")
y = model.addState(start=-5, symbol="y")

u = model.addInput()

model.addDynamic(x, y)
model.addDynamic(y, -x + y * (1.4 - 0.14 * y**2) + 4 * u)

model.addLagrange(x**2 + u**2)

model.generate()

model.optimize(
    tf=4.5,
    steps=1,
    rksteps=70,
    flags={
        "linearSolver": LinearSolver.MA57,
        "initVars": InitVars.SOLVE_EXPLICIT,
        "exportJacobianPath": "/tmp",
    },
    meshFlags={
        "algorithm": MeshAlgorithm.L2_BOUNDARY_NORM,
    },
)

model.plot(dots=Dots.ALL, specifCols=["x"])
model.plotSparseMatrix(MatrixType.JACOBIAN)
