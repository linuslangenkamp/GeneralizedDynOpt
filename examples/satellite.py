from gdopt import *

model = Model("satellite")

I1 = 1000000
I2 = 833333
I3 = 916667

T1S = 550
T2S = 50
T3S = 550

M1 = 0.70106
M2 = 0.0923
M3 = 0.56098
M4 = 0.43047

x1 = model.addState(start=0)
x2 = model.addState(start=0)
x3 = model.addState(start=0)
x4 = model.addState(start=1)
x5 = model.addState(start=0.01)
x6 = model.addState(start=0.005)
x7 = model.addState(start=0.001)

u1 = model.addInput(nominal=0.005)
u2 = model.addInput(nominal=0.00005)
u3 = model.addInput(nominal=0.005)

model.addDynamic(x1, 0.5 * (x5 * x4 - x6 * x3 + x7 * x2))
model.addDynamic(x2, 0.5 * (x5 * x3 + x6 * x4 - x7 * x1))
model.addDynamic(x3, 0.5 * (-x5 * x2 + x6 * x1 + x7 * x4))
model.addDynamic(x4, -0.5 * (x5 * x1 + x6 * x2 + x7 * x3))
model.addDynamic(x5, ((I2 - I3) * x6 * x7 + T1S * u1) / I1)
model.addDynamic(x6, ((I3 - I1) * x7 * x5 + T2S * u2) / I2)
model.addDynamic(x7, ((I1 - I2) * x5 * x6 + T3S * u3) / I3)

model.addMayer(
    (x1 - M1) ** 2
    + (x2 - M2) ** 2
    + (x3 - M3) ** 2
    + (x4 - M4) ** 2
    + x5**2
    + x6**2
    + x7**2,
    Objective.MINIMIZE,
)

model.addLagrange(0.5 * (u1**2 + u2**2 + u3**2), Objective.MINIMIZE)

model.generate()

model.optimize(
    steps=1000,
    rksteps=3,
    tf=100,
    flags={
        "linearSolver": LinearSolver.MUMPS,
        "tolerance": 1e-12,
    },
)

model.plot()
