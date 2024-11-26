from gdopt import *

model = Model("bangBang")

x1 = model.addState(start=0)
x2 = model.addState(start=0)

u = model.addInput(lb=-5, ub=10)

# x1'' = u
model.addDynamic(x1, x2)
model.addDynamic(x2, u)

model.addPath(x2 * u, lb=-30, ub=30)

model.addFinal(x2, eq=0)

model.addMayer(x1, Objective.MAXIMIZE)

model.generate()

# optimizer attributes can be set directly as well
model.meshAlgorithm = MeshAlgorithm.L2_BOUNDARY_NORM
model.meshIterations = 5

model.optimize(tf=0.5, steps=15, rksteps=3)

model.plot(dots=Dots.BASE)
model.plotMeshRefinement()
