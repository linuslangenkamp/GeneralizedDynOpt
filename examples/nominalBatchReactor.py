from gdopt import *

model = Model("nominalBatchReactor")

NOM_SCALED = model.addRuntimeParameter(default=1e-10, symbol="NOM_SCALED")

y1 = model.addState(symbol="Reactant", start=NOM_SCALED, nominal=NOM_SCALED)
y2 = model.addState(symbol="Product", start=0, nominal=1 / NOM_SCALED)

u = model.addInput(symbol="u", lb=0, ub=5, guess=1)

x1 = y1 / NOM_SCALED
x2 = y2 * NOM_SCALED

model.addDynamic(y1, -(u + u**2 / 2) * x1 * NOM_SCALED, nominal=NOM_SCALED)
model.addDynamic(y2, u * x1 / NOM_SCALED, nominal=1 / NOM_SCALED)

model.addMayer(y2, Objective.MAXIMIZE, nominal=1 / NOM_SCALED)

model.hasLinearObjective()

model.generate()

model.optimize(
    tf=1,
    steps=250,
    rksteps=3,
    flags={
        "linearSolver": LinearSolver.MA57,
        "initVars": InitVars.SOLVE,
        "tolerance": 1e-14,
    },
)

model.setValue(NOM_SCALED, 1e10)
model.optimize(resimulate=True)

model.plot(dots=Dots.BASE)
