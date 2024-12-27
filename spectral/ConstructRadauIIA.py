from mpmath import *
from sympy import symbols, diff, expand, Poly, solve, N, re, im
import time


def genCppCode(clist, c0list, blist, Dlist, dps=250, mindps=40):
    mp.dps = mindps

    with open("radauConstantsC.txt", "w") as f:
        # c
        outStr = "{"
        for c in clist:
            outStr += "{"
            for i in range(len(c)):
                outStr += str(c[i]) + ","
            outStr = outStr[:-1] + "},\n"
        outStr = outStr[:-2] + "}\n"
        f.write(outStr)

    # c0
    with open("radauConstantsC0.txt", "w") as f:
        outStr = "{"
        for c0 in c0list:
            outStr += "{"
            for i in range(len(c0)):
                outStr += str(c0[i]) + ","
            outStr = outStr[:-1] + "},\n"
        outStr = outStr[:-2] + "}\n"
        f.write(outStr)

    # b
    with open("radauConstantsB.txt", "w") as f:
        outStr = "{"
        for b in blist:
            outStr += "{"
            for i in range(len(b)):
                outStr += str(b[i]) + ","
            outStr = outStr[:-1] + "},\n"
        outStr = outStr[:-2] + "}\n"
        f.write(outStr)

    # D1 matrix
    with open("radauConstantsD.txt", "w") as f:
        outStr = "{"
        for D1 in Dlist:
            outStr += "{"
            for i in range(len(D1)):
                outStr += "{"
                for j in range(len(D1[0])):
                    outStr += str(D1[i][j]) + ","
                outStr = outStr[:-1] + "},\n"
            outStr = outStr[:-2] + "},\n"
        outStr = outStr[:-2] + "}\n"
        f.write(outStr)

    return outStr


x = symbols("x")


def fixRoots(roots, dps=150):
    allRoots = []
    for r in roots:
        rDps = N(r, dps)
        if rDps.is_real:
            allRoots.append(rDps)
        elif abs(im(rDps)) <= 1e-150:
            allRoots.append(re(rDps))
        else:
            rDps = N(r, dps)
            allRoots.append(re(rDps))

    allRoots.sort()
    return allRoots


def lagrange(c, j, dps=150):
    p = mpf(1)
    for i in range(len(c)):
        if i != j:
            p *= (x - c[i]) / (c[j] - c[i])
    return p


def lagrangeToCoeffs(p, dps=150):
    return [mpf(coeff) for coeff in Poly(expand(p)).all_coeffs()[::-1]]


def integrate(poly, roots, i, dps=150):
    poly = lagrangeToCoeffs(poly, dps=dps)
    S = mpf(0)
    for j in range(len(roots)):
        S += (
            N(1, dps)
            / N(j + 1, dps)
            * poly[j]
            * (pow(N(1, dps), j + 1) - pow(N(-1, dps), j + 1))
        )
    return S


def weight(c, i, dps=150):
    mp.dps = dps
    p = mpf(1)
    for j in range(len(c)):
        if i != j:
            p *= c[i] - c[j]
    return mpf(1) / p


def tridiagonal_eigenvalues(n, dps=150):
    mp.dps = dps

    alpha = mp.mpf(1)

    # Define coefficients
    a_0 = -mp.mpf(1) / mp.mpf(3)
    a_j = lambda j: -mp.mpf(1) / (
        (mp.mpf(2) * j + mp.mpf(1)) * (mp.mpf(2) * j + mp.mpf(3))
    )

    b_1 = mp.sqrt(mp.mpf(8) / (mp.mpf(9) * (mp.mpf(3) + alpha)))
    b_j = lambda j: mp.sqrt(
        (mp.mpf(4) * j * j * (j + mp.mpf(1)) * (j + mp.mpf(1)))
        / (
            (mp.mpf(2) * j)
            * (mp.mpf(2) * j + mp.mpf(1))
            * (mp.mpf(2) * j + mp.mpf(1))
            * (mp.mpf(2) * j + mp.mpf(2))
        )
    )

    # Create the tridiagonal matrix
    matrix = mp.zeros(n)
    for i in range(n):
        if i == 0:
            matrix[i, i] = a_0  # First diagonal element
            if n > 1:
                matrix[i, i + 1] = b_1
                matrix[i + 1, i] = b_1
        else:
            matrix[i, i] = a_j(i)  # Diagonal element
            if i < n - 1:  # Off-diagonal elements
                b_val = b_j(i + 1)
                matrix[i, i + 1] = b_val
                matrix[i + 1, i] = b_val

    # Compute eigenvalues
    eigenvalues = mp.eigsy(matrix)[0]
    return eigenvalues


def generate(s, dps=150):
    mp.dps = dps

    # spectrum calculations
    roots = []

    if s > 1:
        roots = (sorted(tridiagonal_eigenvalues(s - 1, dps=dps))) + [mp.mpf(1)]

    # radau nodes
    c = [mpf(N(r, dps)) for r in roots]

    # radau nodes with 0
    c0 = [mpf(-1)] + c

    # quadrature weights
    b = []

    # helper for derivative matrices D1, D2
    weights = [weight(c0, j) for j in range(len(c0))]
    D1 = [[None for _ in range(len(c0))] for _ in range(len(c0))]  # diff matrix at c0

    # Barycentric Formulas from http://richard.baltensp.home.hefr.ch/Publications/3.pdf

    # eval d/dx L
    for i in range(s + 1):
        lagr = lagrange(c0, i)
        if i > 0:
            if s == 1:
                b = [mpf(2)]
            else:
                b.append(integrate(lagr, c0, s))
        for j in range(s + 1):
            if i != j:
                D1[i][j] = weights[j] / (weights[i] * (c0[i] - c0[j]))
        D1[i][i] = -sum(D1[i][j] for j in range(s + 1) if j != i)
    return c, c0, b, D1


clist, c0list, blist, Dlist = [], [], [], []

for m in range(2, 101):
    startTime = time.time()
    c, c0, b, D = generate(m, 150)
    clist.append(c)
    c0list.append(c0)
    blist.append(b)
    Dlist.append(D)
    print(f"{m} {time.time() - startTime}")

genCppCode(clist, c0list, blist, Dlist)
