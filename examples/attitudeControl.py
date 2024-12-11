from gdopt import *

"""
From Practical Methods for Optimal Control and Estimation Using Nonlinear Programming, Second Edition
John T. Betts, (2010), 978-0-89871-688-7, Pages: 293ff
"""

m = Model("attitudeControlSpaceStation")

J = Matrix(
    [
        [2.80701911616e7, 4.822509936e5, -1.71675094448e7],
        [4.822509936e5, 9.5144639344e7, 6.02604448e4],
        [-1.71675094448e7, 6.02604448e4, 7.6594401336e7],
    ]
)

hmax = 10000
omegaOrb = 0.00113638387

omega = m.addStates(
    3,
    start=Vector([-9.5380685844896e-6, -1.1363312657036e-3, 5.3472801108427e-6]),
    nominal=1e-4,
)

r = m.addStates(
    3,
    start=Vector([2.9963689649816e-3, 1.5334477761054e-1, 3.8359805613992e-3]),
    nominal=1e-2,
)

h = m.addStates(3, start=Vector([5000, 5000, 5000]), nominal=5000)

u = m.addControls(3, nominal=1e2)

C = Matrix.I(3) + 2 / (1 + r.T * r) * ((r.skew * r.skew) - r.skew)
C2 = C.col(1)
C3 = C.col(2)
omega0r = -omegaOrb * C2
taugg = 3 * omegaOrb**2 * C3.skew * J * C3

m.addDynamics(omega, J.inv * (taugg - omega.skew * (J * omega + h) - u), nominal=1e-4)
m.addDynamics(
    r, 0.5 * (r * r.T + Matrix.I(3) + r.skew) * (omega - omega0r), nominal=1e-2
)
m.addDynamics(h, u, nominal=1e4)

m.addPath(h.T * h, ub=hmax**2, nominal=hmax**2)

# all components of the vector constraint get set to 0, since no eq=Vector() is provided
m.addFinals(J.inv * (taugg - omega.skew * (J * omega + h)), eq=0, nominal=1e-4)
m.addFinals(
    0.5 * (r * r.T + Matrix.I(3) + r.skew) * (omega - omega0r), eq=0, nominal=1e-2
)
m.addFinals(h, eq=0, nominal=1e4)

m.addLagrange(u.T * u, nominal=1e6)

m.generate()

m.optimize(
    tf=1800,
    steps=10,
    rksteps=5,
    flags={"linearSolver": LinearSolver.MUMPS, "tolerance": 1e-11},
    meshFlags={
        "algorithm": MeshAlgorithm.L2_BOUNDARY_NORM,
        "iterations": 8,
        "fullBisections": 3,
        "muStrategyRefinement": MuStrategy.MONOTONE,
        "muInitRefinement": 1e-14,
        "cTol": 1,
    },
)

m.plot(dots=Dots.ALL)
m.plotInputsAndRefinement()
