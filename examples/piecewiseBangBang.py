from gdopt import *

model = Model("piecewiseBangBang")

x = model.addState(start=0, symbol="position")
v = model.addState(start=0, symbol="velocity")

a = model.addInput(lb=-100, ub=100, symbol="acceleration")

# x'' = a
model.addDynamic(x, v)
model.addDynamic(v, a)

model.addPath(
    v * a * piecewise((1, t < 0.25), (0, t >= 0.25)), lb=-30, ub=30
)  # constraint only has to hold for time < 0.25

model.addFinal(v, eq=0)

model.addMayer(x, Objective.MAXIMIZE)

model.generate()

model.hasLinearObjective()

# optimizer attributes can be set directly as well
model.meshAlgorithm = MeshAlgorithm.L2_BOUNDARY_NORM
model.meshIterations = 6
model.tolerance = 1e-12

model.optimize(tf=0.5, steps=100, rksteps=3, flags={"exportJacobianPath": "/tmp"})

model.plot(dots=Dots.BASE)
model.plotSparseMatrix(matrixType=MatrixType.JACOBIAN)
model.plotMeshRefinement()
