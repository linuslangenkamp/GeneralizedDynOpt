from gdopt import *
import random

model = Model("pureParameter")

# by creating purely parametric models, it is possible to solve
# standard NLPs: in this case a knapsack problem as a NLP, which is conditioned poorly :)

Items = 12

P = [model.addBinaryParameter(symbol=f"p{i}", guess=0.8) for i in range(Items)]

Weights = [random.uniform(0, 2) for i in range(Items)]
Values = [random.uniform(0, 2) for i in range(Items)]

model.addParametric(sum(Weights[i] * P[i] for i in range(Items)), ub=5)

model.addMayer(sum(Values[i] * P[i] for i in range(Items)), Objective.MAXIMIZE)

model.generate()

model.optimize(
    flags={"linearSolver": LinearSolver.MA57, "tolerance": 1e-14}, meshFlags={}
)

model.printResultParameters()
weights = 0
for i in range(Items):
    if model.resultHistory[0][f"p{i}"].iloc[0] > 0.5:
        print(f"Weight {Weights[i]}, Value {Values[i]}: chosen.")
        weights += Weights[i]
    else:
        print(f"Weight {Weights[i]}, Value {Values[i]}: not chosen.")

print(f"Total weights: {weights}")
