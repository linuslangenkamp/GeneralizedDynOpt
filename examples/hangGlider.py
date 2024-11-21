from gdopt import *

model = Model("hangGlider")

x = model.addState(start=0)
y = model.addState(start=1000)
vx = model.addState(start=13.2275675)
vy = model.addState(start=-1.28750052)

CL = model.addControl(lb=0, ub=1.4, guess=1)

uM = 2.5
R = 100
C0 = 0.034
k = 0.069662
m = 100
S = 14
rho = 1.13
g = 9.80665

X = (x/R - 2.5)**2
ua = uM * (1 - X) * exp(-X)
Vy = vy - ua
vr = sqrt(vx**2 + Vy**2)
CD = C0 + k * CL**2

D = 1/2 * CD * rho * S * vr ** 2
L = 1/2 * CL * rho * S * vr ** 2

sinEta = Vy / vr 
cosEta = vx / vr

model.addF(x, vx)
model.addF(y, vy)
model.addF(vx, (-L * sinEta - D * cosEta) / m)
model.addF(vy, (L * cosEta - D * sinEta - m * g) / m)

model.addFinal(y, eq=900)
model.addFinal(vx, eq=13.2275675)
model.addFinal(vy, eq=-1.28750052)

model.addMayer(x, Objective.MAXIMIZE)

model.setKKTMuGlobalization(False)
model.setTolerance(1e-10)
model.setMeshIterations(5)
model.setMeshAlgorithm(MeshAlgorithm.L2_BOUNDARY_NORM)
model.setMuStrategyRefinement(MuStrategy.MONOTONE)
model.setMuInitRefinement(1e-14)

model.solve(tf=98.4, steps=25, rksteps=9)
model.plot(dots=Dots.ALL)