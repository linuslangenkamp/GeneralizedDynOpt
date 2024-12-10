from gdopt import *

"""
From Practical Methods for Optimal Control and Estimation Using Nonlinear Programming, Second Edition
John T. Betts, (2010), 978-0-89871-688-7, Pages: 293ff
"""


m = Model("attitudeControlSpaceStation")


# TODO: write matrix class with op overloading :)
def vtv(v, w):
    # OUT = v^T w
    return sum(v[i] * w[i] for i in range(len(v)))


def vvt(v, w):
    # OUT = v * w^T
    return [[v[i] * w[j] for j in range(len(w))] for i in range(len(v))]


def vpv(v, w):
    # OUT = v + w
    return [v[i] + w[i] for i in range(len(v))]


def vmv(v, w):
    # OUT = v - w
    return [v[i] - w[i] for i in range(len(v))]


def Mv(A, v):
    # OUT = A * v
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def fM(f, A):
    return [[f * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def fv(f, v):
    return [f * v[i] for i in range(len(v))]


def MM(A, B):
    # OUT = A * B
    return [
        [sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
        for i in range(len(A))
    ]


def MpM(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def MmM(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def colJ(M, j):
    return [M[i][j] for i in range(len(M))]


def skew(v):
    return [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]


J = [
    [2.80701911616e7, 4.822509936e5, -1.71675094448e7],
    [4.822509936e5, 9.5144639344e7, 6.02604448e4],
    [-1.71675094448e7, 6.02604448e4, 7.6594401336e7],
]
Jinv = [
    [
        8371809455422778750 / 202763241149423378503723283,
        -43621893064086875 / 202763241149423378503723283,
        1876452378552155000 / 202763241149423378503723283,
    ],
    [
        -43621893064086875 / 202763241149423378503723283,
        2131333614522456875 / 202763241149423378503723283,
        -11454027415888125 / 202763241149423378503723283,
    ],
    [
        1876452378552155000 / 202763241149423378503723283,
        -11454027415888125 / 202763241149423378503723283,
        3067821423068949375 / 202763241149423378503723283,
    ],
]

hmax = 10000
omegaOrb = 0.00113638387  # rad/sec
omega0bar = [-9.5380685844896e-6, -1.1363312657036e-3, 5.3472801108427e-6]
r0bar = [2.9963689649816e-3, 1.5334477761054e-1, 3.8359805613992e-3]
h0bar = [5000, 5000, 5000]
I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

omega = []
for i in range(3):
    omega.append(m.addState(start=omega0bar[i], nominal=1e-4))

r = []
for i in range(3):
    r.append(m.addState(start=r0bar[i]))

h = []
for i in range(3):
    h.append(m.addState(start=h0bar[i]))


u = []
for i in range(3):
    u.append(m.addControl(guess=0))


skewR = skew(r)
C = MpM(I, fM(2 / (1 + vtv(r, r)), MmM(MM(skewR, skewR), skewR)))
C2 = colJ(C, 1)
C3 = colJ(C, 2)
omega0r = fv(-omegaOrb, C2)
taugg = fv(3 * omegaOrb**2, Mv(MM(skew(C3), J), C3))
sub1 = Mv(skew(omega), vpv(Mv(J, omega), h))


omegaDOT = Mv(Jinv, vmv(taugg, vpv(sub1, u)))
rDOT = fv(0.5, Mv(MpM(MpM(vvt(r, r), I), skewR), vmv(omega, omega0r)))
omegaDOTnoU = Mv(Jinv, vmv(taugg, sub1))


for i in range(3):
    m.addDynamic(omega[i], omegaDOT[i], nominal=1e-4)
    m.addDynamic(r[i], rDOT[i])
    m.addDynamic(h[i], u[i])

m.addPath(vtv(h, h), ub=hmax**2, nominal=hmax**2)

for i in range(3):
    m.addFinal(omegaDOTnoU[i], eq=0, nominal=1e-4)
    m.addFinal(rDOT[i], eq=0)
    m.addFinal(h[i], eq=0, nominal=1e4)

m.addLagrange(vtv(u, u), nominal=1e6)

m.generate()

m.optimize(
    tf=1800,
    steps=30,
    rksteps=4,
    flags={"linearSolver": LinearSolver.MUMPS, "tolerance": 1e-10},
    meshFlags={
        "algorithm": MeshAlgorithm.L2_BOUNDARY_NORM,
        "iterations": 7,
        "fullBisections": 2,
        "muStrategyRefinement": MuStrategy.MONOTONE,
        "muInitRefinement": 1e-14,
        "cTol": 5,
    },
)

m.plotInputsAndRefinement()
