###############################################################################
# GDOPT - General Dynamic Optimizer
# Copyright (C) 2024  Linus Langenkamp
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
###############################################################################

import os
import subprocess
from .variables import *
from .expressions import *
from .functions import *
from .structures import *
from .radauHandling import *
from .matrix import *
from scipy.integrate import solve_ivp
import time as timer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib.resources as resources

# set global pd settings
pd.set_option("display.precision", 8)


# TODO: add vectorized variable adds and matrix vector / vector vector operations on them


class Model:
    def __init__(self, name="DummyName"):
        self.creationTime = timer.process_time()
        self.xVars = []
        self.uVars = []
        self.pVars = []
        self.rpVars = []
        self.M = None
        self.L = None
        self.F = []
        self.G = []
        self.R = []
        self.A = []
        self.name = name.replace(" ", "")
        self._addedDummy = False

        # additional stuff for running the model
        self._tf = None
        self._steps = None
        self._rksteps = None
        self._outputFilePath = f".generated/{self.name}"
        self._maxIterations = 5000
        self._ipoptPrintLevel = None
        self._kktMuGlobalization = None
        self._muInit = None
        self._muInitRefinement = None
        self._muStrategy = None
        self._muStrategyRefinement = None
        self._ivpSolver = IVPSolver.RADAU
        self._initVars = InitVars.SOLVE
        self._refinementMethod = RefinementMethod.LINEAR_SPLINE
        self._linearSolver = LinearSolver.MUMPS
        self._meshAlgorithm = MeshAlgorithm.NONE
        self._meshIterations = 0
        self._tolerance = 1e-14
        self._exportHessianPath = None
        self._exportJacobianPath = None
        self._initialStatesPath = f".generated/{self.name}"
        self._meshLevel = None
        self._meshCTol = None
        self._meshSigma = None
        self._detectStatesAndControl = None
        self._fullBisections = None
        self._quadraticObjective = False
        self._linearObjective = False
        self._linearConstraints = False
        self._autoVariableNominals = False  # TODO: not in use: automatically scale variables and equations that dont have a nominal value
        self._userScaling = False

        # stuff for analyzing the results
        self.resultHistory = {}
        self.modelInfo = {}
        self.xVarNames = []
        self.uVarNames = []
        self.pVarNames = []
        self.rpVarNames = []

    def addState(
        self, start, symbol=None, lb=-float("inf"), ub=float("inf"), nominal=None
    ):
        # adds a state x for optimization, must a have starting value
        # specify lb or ub if needed

        info = StateStruct(start, symbol=symbol, lb=lb, ub=ub, nominal=nominal)
        variable = Symbol(f"x[{info.id}]")
        varInfo[variable] = info
        self.xVars.append(variable)
        self.checkNominalNone(nominal)
        return variable

    def addX(self, start, symbol=None, lb=-float("inf"), ub=float("inf"), nominal=None):
        # alias for addState()

        return self.addState(start, symbol=symbol, lb=lb, ub=ub, nominal=nominal)

    def addStates(
        self, size, start, symbol=None, lb=-float("inf"), ub=float("inf"), nominal=None
    ):
        newStates = []

        lb = [lb] * size if not isinstance(lb, (list, Vector)) else lb
        ub = [ub] * size if not isinstance(ub, (list, Vector)) else ub
        nominal = (
            [nominal] * size if not isinstance(nominal, (list, Vector)) else nominal
        )

        for i in range(size):
            newStates.append(
                self.addState(
                    start=start[i] if isinstance(start, (list, Vector)) else start,
                    symbol=(
                        symbol[i]
                        if symbol and isinstance(symbol, (list, Vector))
                        else None
                    ),
                    lb=lb[i],
                    ub=ub[i],
                    nominal=nominal[i] if nominal else None,
                )
            )
        return Vector(newStates)

    def addXs(
        self, size, start, symbol=None, lb=-float("inf"), ub=float("inf"), nominal=None
    ):
        return self.addStates(size, start, symbol=symbol, lb=lb, ub=ub, nominal=nominal)

    def addInput(
        self, symbol=None, lb=-float("inf"), ub=float("inf"), guess=0, nominal=None
    ):
        # adds an input / control u for optimization
        # specify lb or ub, start/initialGuess if needed
        # only provide a guess if you are certain that it will benefit

        info = InputStruct(
            symbol=symbol, lb=lb, ub=ub, initialGuess=guess, nominal=nominal
        )
        variable = Symbol(f"u[{info.id}]")
        varInfo[variable] = info
        self.uVars.append(variable)
        self.checkNominalNone(nominal)
        return variable

    def addControl(
        self, symbol=None, lb=-float("inf"), ub=float("inf"), guess=0, nominal=None
    ):
        # alias for addInput()

        return self.addInput(symbol=symbol, lb=lb, ub=ub, guess=guess, nominal=nominal)

    def addContinuous(
        self, symbol=None, lb=-float("inf"), ub=float("inf"), guess=0, nominal=None
    ):
        # alias for addInput()

        return self.addInput(symbol=symbol, lb=lb, ub=ub, guess=guess, nominal=nominal)

    def addU(
        self, symbol=None, lb=-float("inf"), ub=float("inf"), guess=0, nominal=None
    ):
        # alias for addInput()

        return self.addInput(symbol=symbol, lb=lb, ub=ub, guess=guess, nominal=nominal)

    def addInputs(
        self,
        size,
        symbol=None,
        lb=-float("inf"),
        ub=float("inf"),
        guess=0,
        nominal=None,
    ):
        newInputs = []

        # Handle scalar inputs for lb, ub, guess, and nominal by converting them to lists

        lb = [lb] * size if not isinstance(lb, (list, Vector)) else lb
        ub = [ub] * size if not isinstance(ub, (list, Vector)) else ub
        nominal = (
            [nominal] * size if not isinstance(nominal, (list, Vector)) else nominal
        )
        guess = [guess] * size if not isinstance(guess, (list, Vector)) else guess

        for i in range(size):
            newInputs.append(
                self.addInput(
                    symbol=(
                        symbol[i]
                        if symbol and isinstance(symbol, (list, Vector))
                        else None
                    ),
                    lb=lb[i],
                    ub=ub[i],
                    guess=guess[i] if guess else 0,
                    nominal=nominal[i] if nominal else None,
                )
            )
        return Vector(newInputs)

    def addUs(
        self,
        size,
        symbol=None,
        lb=-float("inf"),
        ub=float("inf"),
        guess=0,
        nominal=None,
    ):
        return self.addInputs(
            size, symbol=symbol, lb=lb, ub=ub, guess=guess, nominal=nominal
        )

    def addControls(
        self,
        size,
        symbol=None,
        lb=-float("inf"),
        ub=float("inf"),
        guess=0,
        nominal=None,
    ):
        return self.addInputs(
            size, symbol=symbol, lb=lb, ub=ub, guess=guess, nominal=nominal
        )

    def addParameter(
        self, symbol=None, lb=-float("inf"), ub=float("inf"), guess=0, nominal=None
    ):
        # adds a parameter p for optimization
        # specify lb or ub if needed, start/initialGuess if needed

        info = ParameterStruct(
            symbol=symbol, lb=lb, ub=ub, initialGuess=guess, nominal=nominal
        )
        variable = Symbol(f"p[{info.id}]")
        varInfo[variable] = info
        self.pVars.append(variable)
        self.checkNominalNone(nominal)
        return variable

    def addP(
        self, symbol=None, lb=-float("inf"), ub=float("inf"), guess=0, nominal=None
    ):
        # alias for addParameter()

        return self.addParameter(
            symbol=symbol, lb=lb, ub=ub, guess=guess, nominal=nominal
        )

    def addParameters(
        self,
        size,
        symbol=None,
        lb=-float("inf"),
        ub=float("inf"),
        guess=0,
        nominal=None,
    ):
        newParameters = []

        # Handle scalar inputs for lb, ub, guess, and nominal by converting them to lists
        lb = [lb] * size if not isinstance(lb, (list, Vector)) else lb
        ub = [ub] * size if not isinstance(ub, (list, Vector)) else ub
        nominal = (
            [nominal] * size if not isinstance(nominal, (list, Vector)) else nominal
        )
        guess = [guess] * size if not isinstance(guess, (list, Vector)) else guess

        for i in range(size):
            newParameters.append(
                self.addParameter(
                    symbol=(
                        symbol[i]
                        if symbol and isinstance(symbol, (list, Vector))
                        else None
                    ),
                    lb=lb[i],
                    ub=ub[i],
                    guess=guess[i] if guess else 0,
                    nominal=nominal[i] if nominal else None,
                )
            )
        return Vector(newParameters)

    def addPs(
        self,
        size,
        symbol=None,
        lb=-float("inf"),
        ub=float("inf"),
        guess=0,
        nominal=None,
    ):
        return self.addParameters(
            size, symbol=symbol, lb=lb, ub=ub, guess=guess, nominal=nominal
        )

    def addBinaryParameter(self, lb=0, ub=1, symbol=None, guess=None, nominal=None):
        # dangerous, might break the code
        # adds a binary parameter p for optimization
        # specify lb or ub, start/initialGuess if needed
        # to ensure the binary type, adds a parametric equation (p - lb) * (p - ub) == 0 to the model

        if guess is None:
            guess = (lb + ub) / 2
        info = ParameterStruct(
            symbol=symbol, lb=lb, ub=ub, initialGuess=guess, nominal=nominal
        )
        variable = Symbol(f"p[{info.id}]")
        varInfo[variable] = info
        self.pVars.append(variable)
        self.addParametric(
            variable**2 - variable * (lb + ub) + lb * ub, eq=0
        )  # forces binary type
        self.checkNominalNone(nominal)
        return variable

    def addRuntimeParameter(self, default, symbol=None):
        # adds a runtime parameter: can be changed for optimization
        # will be handled symbolically -> thus quite slow
        # for actual constants simply write pythonic var = value

        info = RuntimeParameterStruct(default=default, symbol=symbol)
        variable = Symbol(f"{str(info.symbol).upper().strip(' ')}_VALUE")
        varInfo[variable] = info
        self.rpVars.append(variable)
        return variable

    def addRP(self, default, symbol=None):
        # alias for addRuntimeParameter()

        return self.addRuntimeParameter(default, symbol=symbol)

    def setValue(self, runtimeParameter, value):
        # sets a runtime parameter value

        varInfo[runtimeParameter].value = value

    def addObjective(self, mayer, lagrange, obj=Objective.MINIMIZE, nominal=None):
        # adds the mayer and lagrange term
        # if mayer and lagrange have a nominal -> sum is used as nominal value

        self.addMayer(mayer, obj, nominal=nominal / 2)
        self.addLagrange(lagrange, obj, nominal=nominal / 2)

    def addMayer(self, expr, obj=Objective.MINIMIZE, nominal=None):
        # adds the mayer term: min/max expr(tf)
        # if mayer and lagrange have a nominal -> sum is used as nominal value
        if self.M is not None:
            print(
                "[GDOPT - ATTENTION] Mayer term already set! Overwriting previous Mayer term."
            )

        if obj == Objective.MAXIMIZE:
            self.M = Expression(-1 * expr, nominal=nominal)
            print(
                "[GDOPT - ATTENTION] Setting Mayer term as -1 * mayer, since maximization is chosen."
            )
        else:
            self.M = Expression(expr, nominal=nominal)
        self.checkNominalNone(nominal)

    def addM(self, expr, obj=Objective.MINIMIZE, nominal=None):
        # alias for addMayer()

        self.addMayer(expr, obj=obj, nominal=nominal)

    def addLagrange(self, expr, obj=Objective.MINIMIZE, nominal=None):
        # adds the lagrange term: min/max integral_0_tf expr dt
        # if mayer and lagrange have a nominal -> sum is used as nominal value

        if self.L is not None:
            print(
                "[GDOPT - ATTENTION] Lagrange term already set! Overwriting previous Lagrange term."
            )

        if obj == Objective.MAXIMIZE:
            self.L = Expression(-1 * expr, nominal=nominal)
            print(
                "[GDOPT - ATTENTION] Setting Lagrange term as -1 * lagrange, since maximization is chosen."
            )
        else:
            self.L = Expression(expr, nominal=nominal)
        self.checkNominalNone(nominal)

    def addL(self, expr, obj=Objective.MINIMIZE, nominal=None):
        # alias for addLagrange()

        self.addLagrange(expr, obj=obj, nominal=nominal)

    def addDynamic(self, diffVar, expr, nominal=None):
        # adds a dynamic constraint: diffVar' = expr
        if isinstance(expr, (list, Vector)):
            raise ValueError("Expression must be scalar")

        for f in self.F:
            if diffVar == f.diffVar:
                fail = f"[GDOPT - ERROR] Equation for diff({diffVar}) has been added already"
                raise InvalidModel(fail)
        self.F.append(DynExpression(diffVar, expr, nominal=nominal))
        self.checkNominalNone(nominal)

    def addF(self, diffVar, expr, nominal=None):
        # alias for addDynamic()

        self.addDynamic(diffVar, expr, nominal=nominal)

    def addOde(self, diffVar, expr, nominal=None):
        # alias for addDynamic()

        self.addDynamic(diffVar, expr, nominal=nominal)

    def addDynamics(self, diffVar, expr, nominal=None):
        if not isinstance(diffVar, (list, Vector)) and not isinstance(
            expr, (list, Vector)
        ):
            raise ValueError(
                "addDynamics expect lists or Vectors for diffVar and expr."
            )

        size = len(diffVar)
        nominal = (
            [nominal] * size if not isinstance(nominal, (list, Vector)) else nominal
        )
        for i in range(size):
            self.addDynamic(diffVar[i], expr[i], nominal=nominal[i])

    def addOdes(self, diffVar, expr, nominal=None):
        self.addDynamics(diffVar, expr, nominal=nominal)

    def addFs(self, diffVar, expr, nominal=None):
        self.addDynamics(diffVar, expr, nominal=nominal)

    def addPath(self, expr, lb=-float("inf"), ub=float("inf"), eq=None, nominal=None):
        # adds a path constraint: lb <= g(.(t)) <= ub or g(.(t)) == eq
        if isinstance(expr, (list, Vector)):
            raise ValueError("Expression must be scalar")
        self.checkBounds(lb, ub, eq)
        self.G.append(Constraint(expr, lb=lb, ub=ub, eq=eq, nominal=nominal))
        self.checkNominalNone(nominal)

    def addG(self, expr, lb=-float("inf"), ub=float("inf"), eq=None, nominal=None):
        # alias for addPath()

        self.addPath(expr, lb=lb, ub=ub, eq=eq, nominal=nominal)

    def addPaths(self, expr, lb=-float("inf"), ub=float("inf"), eq=None, nominal=None):
        if not isinstance(expr, (list, Vector)):
            raise ValueError("addPaths expect lists or Vectors for expr.")

        size = len(expr)
        nominal = (
            [nominal] * size if not isinstance(nominal, (list, Vector)) else nominal
        )
        lb = [lb] * size if not isinstance(lb, (list, Vector)) else lb
        ub = [ub] * size if not isinstance(ub, (list, Vector)) else ub
        eq = [eq] * size if not isinstance(eq, (list, Vector)) else eq
        for i in range(size):
            self.addPath(expr[i], lb=lb[i], ub=ub[i], eq=eq[i], nominal=nominal[i])

    def addGs(self, expr, lb=-float("inf"), ub=float("inf"), eq=None, nominal=None):
        return self.addPaths(expr, lb=lb, ub=ub, eq=eq, nominal=nominal)

    def addFinal(self, expr, lb=-float("inf"), ub=float("inf"), eq=None, nominal=None):
        # adds a final constraint: lb <= r(.(tf)) <= ub or r(.(tf)) == eq
        if isinstance(expr, (list, Vector)):
            raise ValueError("Expression must be scalar")
        self.checkBounds(lb, ub, eq)
        self.R.append(Constraint(expr, lb=lb, ub=ub, eq=eq, nominal=nominal))
        self.checkNominalNone(nominal)

    def addR(self, expr, lb=-float("inf"), ub=float("inf"), eq=None, nominal=None):
        # alias for addFinal()

        self.addFinal(expr, lb=lb, ub=ub, eq=eq, nominal=nominal)

    def addFinals(self, expr, lb=-float("inf"), ub=float("inf"), eq=None, nominal=None):
        if not isinstance(expr, (list, Vector)):
            raise ValueError("addFinals expect lists or Vectors for expr.")

        size = len(expr)
        nominal = (
            [nominal] * size if not isinstance(nominal, (list, Vector)) else nominal
        )
        lb = [lb] * size if not isinstance(lb, (list, Vector)) else lb
        ub = [ub] * size if not isinstance(ub, (list, Vector)) else ub
        eq = [eq] * size if not isinstance(eq, (list, Vector)) else eq
        for i in range(size):
            self.addFinal(expr[i], lb=lb[i], ub=ub[i], eq=eq[i], nominal=nominal[i])

    def addRs(self, expr, lb=-float("inf"), ub=float("inf"), eq=None, nominal=None):
        return self.addFinals(expr, lb=lb, ub=ub, eq=eq, nominal=nominal)

    def addParametric(
        self, expr, lb=-float("inf"), ub=float("inf"), eq=None, nominal=None
    ):
        # adds a parametric constraint: lb <= a(p) <= ub or a(p) == eq
        if isinstance(expr, (list, Vector)):
            raise ValueError("Expression must be scalar")
        if not set(self.pVars).issuperset(expr.free_symbols):
            raise InvalidModel(
                "[GDOPT - ERROR] Parametric constraints only allow parametric variables"
            )

        self.checkBounds(lb, ub, eq)
        self.checkNominalNone(nominal)
        self.A.append(ParametricConstraint(expr, lb=lb, ub=ub, eq=eq, nominal=nominal))

    def addA(self, expr, lb=-float("inf"), ub=float("inf"), eq=None, nominal=None):
        # alias for addParametric()

        self.addParametric(expr, lb=lb, ub=ub, eq=eq, nominal=nominal)

    def addParametrics(
        self, expr, lb=-float("inf"), ub=float("inf"), eq=None, nominal=None
    ):
        if not isinstance(expr, (list, Vector)):
            raise ValueError("addParametrics expect lists or Vectors for expr.")

        size = len(expr)
        nominal = (
            [nominal] * size if not isinstance(nominal, (list, Vector)) else nominal
        )
        lb = [lb] * size if not isinstance(lb, (list, Vector)) else lb
        ub = [ub] * size if not isinstance(ub, (list, Vector)) else ub
        eq = [eq] * size if not isinstance(eq, (list, Vector)) else eq
        for i in range(size):
            self.addParametric(
                expr[i], lb=lb[i], ub=ub[i], eq=eq[i], nominal=nominal[i]
            )

    def addAs(self, expr, lb=-float("inf"), ub=float("inf"), eq=None, nominal=None):
        return self.addParametrics(expr, lb=lb, ub=ub, eq=eq, nominal=nominal)

    def checkBounds(self, lb, ub, eq):
        if eq is not None and (lb != -float("inf") or ub != float("inf")):
            raise InvalidModel("[GDOPT - ERROR] Can't set eq and lb or ub.")
        elif lb > ub:
            raise InvalidModel("[GDOPT - ERROR] lb > ub.")
        else:
            return

    def _addDummy(self):
        # adds a dummy state for purely parametric models

        x = self.addState(start=0)
        self.addDynamic(x, 0)
        self._addedDummy = True

    def __str__(self):
        M = str(self.M) if self.M != None else ""
        L = str(self.L) if self.L != None else ""
        if self.M and self.L:
            out = f"min {M}(tf) + ∫ {L} dt\n"
        elif self.M:
            out = f"min {M}(tf)\n"
        elif self.L:
            out = f"min ∫ {L} dt\n"
        else:
            out = f"min 0\n"

        out += "s.t.\n"

        for f in self.F:
            out += str(f) + f", {f.diffVar}(0) = {varInfo[f.diffVar].start}\n"
        if len(self.F) > 0:
            out += "\n"

        for g in self.G:
            out += str(g) + "\n"
        if len(self.G) > 0:
            out += "\n"

        for r in self.R:
            out += (
                f"{r.lb} <= {str(r.expr)}(tf) <= {r.ub}\n"
                if r.lb != r.ub
                else f"{str(r.expr)}(tf) = {r.lb}\n"
            )
        if len(self.R) > 0:
            out += "\n"

        for a in self.A:
            out += str(a) + "\n"

        return out

    def __repr__(self):
        return str(self)

    def solveDynamic(self):
        # solves the dynamic system with the given initial u(t), p, x0

        # define the rhs, uCallback function
        rhs = [
            Lambdify(
                [t]
                + [x for x in self.xVars]
                + [u for u in self.uVars]
                + [p for p in self.pVars]
                + [rp for rp in self.rpVars],
                (
                    dynEq.expr.subs(FINAL_TIME_SYMBOL, self._tf)
                    if hasattr(dynEq.expr, "subs")
                    else dynEq.expr
                ),
            )
            for dynEq in self.F
        ]

        # dirty uFuncs hack, if standard functions for the guess are provided
        uFuncs = [
            Lambdify(
                [t],
                (
                    varInfo[u].initialGuess.subs(FINAL_TIME_SYMBOL, self._tf)
                    if hasattr(varInfo[u].initialGuess, "subs")
                    else varInfo[u].initialGuess
                ),
            )
            for u in self.uVars
        ]

        # define rhs as an actual function
        ode = lambda T, x: [
            r(
                *(
                    [T]
                    + [elem for elem in x]
                    + [uFuncs[i](T) for i in range(len(self.uVars))]
                    + [varInfo[pVar].initialGuess for pVar in self.pVars]
                    + [varInfo[rpVar].value for rpVar in self.rpVars]
                )
            )
            for r in rhs
        ]

        runtimeConstantsDict = {rpVar: varInfo[rpVar].value for rpVar in self.rpVars}
        runtimeConstantsDict[FINAL_TIME_SYMBOL] = self._tf
        x0 = [
            (
                varInfo[x].start.subs(runtimeConstantsDict)
                if hasattr(varInfo[x].start, "subs")
                else varInfo[x].start
            )
            for x in self.xVars
        ]
        timeHorizon = [0, self._tf]

        # scipy.solve_ivp
        solution = solve_ivp(
            ode,
            timeHorizon,
            x0,
            method=self._ivpSolver.name,
            dense_output=True,
            rtol=1e-10,
        )

        # get solution at the RadauIIA nodes (based on self.rksteps)
        timeVals = generateRadauNodes(timeHorizon, self._steps, self._rksteps)
        stateVals = solution.sol(timeVals)
        return timeVals, stateVals

    @property
    def tf(self) -> float:
        return self._tf

    @tf.setter
    def tf(self, tf: float):
        self._tf = tf

    @property
    def steps(self) -> int:
        return self._steps

    @steps.setter
    def steps(self, steps: int):
        self._steps = steps

    @property
    def rksteps(self) -> int:
        return self._rksteps

    @rksteps.setter
    def rksteps(self, rksteps: int):
        self._rksteps = rksteps

    @property
    def maxIterations(self) -> int:
        return self._maxIterations

    @maxIterations.setter
    def maxIterations(self, iterations: int):
        self._maxIterations = iterations

    @property
    def ipoptPrintLevel(self) -> int:
        return self._ipoptPrintLevel

    @ipoptPrintLevel.setter
    def ipoptPrintLevel(self, printLevel: int):
        self._ipoptPrintLevel = printLevel

    @property
    def muInit(self) -> float:
        return self._muInit

    @muInit.setter
    def muInit(self, muInit: float):
        self._muInit = muInit

    @property
    def muInitRefinement(self) -> float:
        return self._muInitRefinement

    @muInitRefinement.setter
    def muInitRefinement(self, muInitRefinement: float):
        self._muInitRefinement = muInitRefinement

    @property
    def muStrategy(self) -> str:
        return self._muStrategy

    @muStrategy.setter
    def muStrategy(self, muStrategy: str):
        self._muStrategy = muStrategy

    @property
    def muStrategyRefinement(self) -> str:
        return self._muStrategyRefinement

    @muStrategyRefinement.setter
    def muStrategyRefinement(self, muStrategyRefinement: str):
        self._muStrategyRefinement = muStrategyRefinement

    @property
    def outputFilePath(self) -> str:
        return self._outputFilePath

    @outputFilePath.setter
    def outputFilePath(self, path: str):
        self._outputFilePath = path

    @property
    def linearSolver(self):
        return self._linearSolver

    @linearSolver.setter
    def linearSolver(self, solver):
        self._linearSolver = solver

    @property
    def initVars(self):
        return self._initVars

    @initVars.setter
    def initVars(self, initVars):
        self._initVars = initVars

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance: float):
        self._tolerance = tolerance

    @property
    def kktMuGlobalization(self) -> bool:
        return self._kktMuGlobalization

    @kktMuGlobalization.setter
    def kktMuGlobalization(self, kktMuGlobalization: bool):
        self._kktMuGlobalization = kktMuGlobalization

    @property
    def exportHessianPath(self) -> str:
        return self._exportHessianPath

    @exportHessianPath.setter
    def exportHessianPath(self, path: str):
        self._exportHessianPath = path

    @property
    def exportJacobianPath(self) -> str:
        return self._exportJacobianPath

    @exportJacobianPath.setter
    def exportJacobianPath(self, path: str):
        self._exportJacobianPath = path

    @property
    def initialStatesPath(self) -> str:
        return self._initialStatesPath

    @initialStatesPath.setter
    def initialStatesPath(self, path: str):
        self._initialStatesPath = path

    @property
    def ivpSolver(self):
        return self._ivpSolver

    @ivpSolver.setter
    def ivpSolver(self, solver):
        self._ivpSolver = solver

    @property
    def refinementMethod(self):
        return self._refinementMethod

    @refinementMethod.setter
    def refinementMethod(self, refinementMethod):
        self._refinementMethod = refinementMethod

    @property
    def meshAlgorithm(self):
        return self._meshAlgorithm

    @meshAlgorithm.setter
    def meshAlgorithm(self, meshAlgorithm):
        self._meshAlgorithm = meshAlgorithm

    @property
    def fullBisections(self) -> int:
        return self.fullBisections

    @fullBisections.setter
    def fullBisections(self, fullBisections: int):
        self._fullBisections = fullBisections

    @property
    def meshIterations(self) -> int:
        return self._meshIterations

    @meshIterations.setter
    def meshIterations(self, meshIterations: int):
        self._meshIterations = meshIterations

    @property
    def meshLevel(self) -> float:
        return self._meshLevel

    @meshLevel.setter
    def meshLevel(self, meshLevel: float):
        self._meshLevel = meshLevel

    @property
    def meshCTol(self) -> float:
        return self._meshCTol

    @meshCTol.setter
    def meshCTol(self, meshCTol: float):
        self._meshCTol = meshCTol

    @property
    def meshSigma(self) -> float:
        return self._meshSigma

    @meshSigma.setter
    def meshSigma(self, meshSigma: float):
        self._meshSigma = meshSigma

    @property
    def userScaling(self) -> bool:
        return self._userScaling

    @userScaling.setter
    def userScaling(self, userScaling: bool):
        self._userScaling = userScaling

    @property
    def detectStatesAndControl(self):
        return self._detectStatesAndControl

    @detectStatesAndControl.setter
    def detectStatesAndControl(self, detect):
        self._detectStatesAndControl = detect

    def hasLinearObjective(self, linearObjective=True):
        # set this if both Mayer and Lagrange are linear

        self._linearObjective = linearObjective

    def hasQuadraticObjective(self, quadraticObjective=True):
        # set this if both Mayer and Lagrange are quadratic

        self._quadraticObjective = quadraticObjective

    def hasLinearConstraints(self, linearConstraints=True):
        # set this if all constraints, i.e. f, g, r and a are linear

        self._linearConstraints = linearConstraints

    def setExpressionSimplification(self, simp):
        # turn initial simplification of expression at generation on / off, standard = off, good for large models

        Expression.simplification = simp

    def disableOutputPath(self, disable):
        if disable:
            print(
                "[GDOPT - ATTENTION] Output of optimal solution has been disabled. Thus, the analysis can not be used."
            )
            self._outputFilePath = None

    def setFlags(self, flags):
        if "outputPath" in flags:
            self.outputFilePath = flags["outputPath"]
        if "disableOutput" in flags:
            self.disableOutputPath = flags["disableOutput"]
        if "linearSolver" in flags:
            self.linearSolver = flags["linearSolver"]
        if "tolerance" in flags:
            self.tolerance = flags["tolerance"]
        if "exportHessianPath" in flags:
            self.exportHessianPath = flags["exportHessianPath"]
        if "exportJacobianPath" in flags:
            self.exportJacobianPath = flags["exportJacobianPath"]
        if "initialStatesPath" in flags:
            self.initialStatesPath = flags["initialStatesPath"]
        if "ivpSolver" in flags:
            self.ivpSolver = flags["ivpSolver"]
        if "initVars" in flags:
            self.initVars = flags["initVars"]
        if "maxIterations" in flags:
            self.maxIterations = flags["maxIterations"]
        if "ipoptPrintLevel" in flags:
            self.ipoptPrintLevel = flags["ipoptPrintLevel"]
        if "kktMuGlobalization" in flags:
            self.kktMuGlobalization = flags["kktMuGlobalization"]
        if "muStrategy" in flags:
            self.muStrategy = flags["muStrategy"]
        if "muInit" in flags:
            self.muInit = flags["muInit"]
        if "linearObjective" in flags:
            self.hasLinearObjective(flags["linearObjective"])
        if "quadraticObjective" in flags:
            self.hasQuadraticObjective(flags["quadraticObjective"])
        if "linearConstraints" in flags:
            self.hasLinearConstraints(flags["linearConstraints"])

    def setMeshFlags(self, meshFlags):
        if "algorithm" in meshFlags:
            self.meshAlgorithm = meshFlags["algorithm"]
        if "iterations" in meshFlags:
            self.meshIterations = meshFlags["iterations"]
        if "refinementMethod" in meshFlags:
            self.refinementMethod = meshFlags["refinementMethod"]
        if "level" in meshFlags:
            self.meshLevel = meshFlags["level"]
        if "cTol" in meshFlags:
            self.meshCTol = meshFlags["cTol"]
        if "sigma" in meshFlags:
            self.meshSigma = meshFlags["sigma"]
        if "muStrategyRefinement" in meshFlags:
            self.muStrategyRefinement = meshFlags["muStrategyRefinement"]
        if "muInitRefinement" in meshFlags:
            self.muInitRefinement = meshFlags["muInitRefinement"]
        if "detectStatesAndControl" in meshFlags:
            self.detectStatesAndControl = meshFlags["detectStatesAndControl"]
        if "fullBisections" in meshFlags:
            self.fullBisections = meshFlags["fullBisections"]

    def uInitialGuessCodegen(self):
        out = "std::vector<double> initialGuessU(double t) {\n"
        out += f"\t return {{{', '.join(str(toCode(varInfo[u].initialGuess)) for u in self.uVars)}}};"
        out += "\n};\n\n"
        return out

    def objectiveNominal(self):
        nom = 1
        if self.M and self.L:
            if self.M.nominal is not None and self.L.nominal is not None:
                nom = self.M.nominal + self.L.nominal
            elif self.M.nominal is not None:
                nom = self.M.nominal
            elif self.L.nominal is not None:
                nom = self.L.nominal
        elif self.M:
            nom = self.M.nominal if self.M.nominal is not None else 1
        elif self.L:
            nom = self.L.nominal if self.L.nominal is not None else 1
        return nom

    def checkNominalNone(self, nominal):
        # forces userScaling to True, if a single nominal is provided in the model -> else its False
        if nominal is not None:
            self._userScaling = True

    def initAnalysis(self):
        # clear result history for new optimization
        self.resultHistory = {}

        with open("/tmp/modelinfo.txt", "r") as file:
            for line in file:
                line = line.strip()
                key, value = line.split(":")
                key = key.strip()
                value = value.strip()
                if key in [
                    "maxMeshIteration",
                    "initialIntervals",
                    "ipoptIterationsOverall",
                ]:
                    # int singleton
                    self.modelInfo[key] = int(value)
                elif key in [
                    "objective",
                    "ipoptTimeTotal",
                    "ipoptTimeNonfunc",
                    "ipoptTimeFunc",
                    "gdoptTimeAlgorithms",
                    "gdoptTimeIO",
                    "totalTime",
                ]:
                    # float singleton
                    self.modelInfo[key] = float(value)
                elif key in ["intervalHistory", "ipoptIterationHistory"]:
                    # list int
                    values = value.split(",")
                    self.modelInfo[key] = list(map(int, values))
                elif key in [
                    "objectiveHistory",
                    "ipoptTimeTotalHistory",
                    "ipoptTimeNonfuncHistory",
                    "ipoptTimeFuncHistory",
                ]:
                    # list float
                    values = value.split(",")
                    self.modelInfo[key] = list(map(float, values))

    def generate(self, compiler="g++", compileFlags=["-O3", "-ffast-math"]):

        # does the entire code generation and compilation of the model

        timeStartGenerate = timer.time()
        if len(self.F) != len(self.xVars):
            raise InvalidModel("#states != #differential equations")
        elif len(self.xVars) == 0:
            # adding a dummy var: x(0)=0, x'=0 for purely parametric models
            self._addDummy()

        # preparations

        self.F.sort(key=lambda eq: varInfo[eq.diffVar].id, reverse=False)

        # codegen
        filename = self.name + "Generated"

        print("[GDOPT - INFO] Starting .cpp codegen...")

        OUTPUT = f"""// CODEGEN FOR MODEL "{self.name}"\n
// includes
#define _USE_MATH_DEFINES
#include <cmath>
#include <string>
#include <libgdopt/constants.h>
#include <libgdopt/problem.h>
#include <libgdopt/integrator.h>
#include <libgdopt/mesh.h>
#include <libgdopt/gdop.h>
#include <libgdopt/solver.h>
#include <libgdopt/config.h>
\n\n"""

        rpStrings = [
            f"{str(varInfo[rp].symbol).upper().strip(' ')}_VALUE" for rp in self.rpVars
        ]
        if len(self.rpVars) != 0:
            OUTPUT += (
                "// declaration of global runtime parameters (read from .config)\n"
            )
        for rpString in rpStrings:
            OUTPUT += f"double {rpString};\n"

        OUTPUT += "void setGlobalRuntimeParameters(const std::unordered_map<std::string, std::string>& configMap) {\n"
        for rpString in rpStrings:
            OUTPUT += f'\t{rpString} = std::stod(configMap.at("{rpString}"));\n'
        OUTPUT += "}\n\n"

        if self.M:
            OUTPUT += "// mayer term\n"
            OUTPUT += self.M.codegen("Mayer" + self.name)
            print("[GDOPT - INFO] Mayer: codegen done.")

        if self.L:
            OUTPUT += "// lagrange term\n"
            OUTPUT += self.L.codegen("Lagrange" + self.name)
            print("[GDOPT - INFO] Lagrange: codegen done.")

        if self.F:
            OUTPUT += "// dynamic constraints\n"

        for n, f in enumerate(self.F):
            OUTPUT += f.codegen("F" + str(n) + self.name)
            print(f"[GDOPT - INFO] Dynamic constraint {n}: codegen done.")

        if self.G:
            OUTPUT += "// path constraints\n"

        for n, g in enumerate(self.G):
            OUTPUT += g.codegen("G" + str(n) + self.name)
            print(f"[GDOPT - INFO] Path constraint {n}: codegen done.")

        if self.R:
            OUTPUT += "// final constraints\n"

        for n, r in enumerate(self.R):
            OUTPUT += r.codegen("R" + str(n) + self.name)
            print(f"[GDOPT - INFO] Final constraint {n}: codegen done.")

        if self.A:
            OUTPUT += "// parametric constraints\n"

        for n, a in enumerate(self.A):
            OUTPUT += a.codegen("A" + str(n) + self.name)
            print(f"[GDOPT - INFO] Parametric constraints {n}: codegen done.")

        OUTPUT += self.uInitialGuessCodegen()

        pushF = "\n    ".join(
            "F.push_back(" + "F" + str(n) + self.name + "::create());"
            for n in range(len(self.F))
        )
        pushG = "\n    ".join(
            "G.push_back(" + "G" + str(n) + self.name + "::create());"
            for n in range(len(self.G))
        )
        pushR = "\n    ".join(
            "R.push_back(" + "R" + str(n) + self.name + "::create());"
            for n in range(len(self.R))
        )
        pushA = "\n    ".join(
            "A.push_back(" + "A" + str(n) + self.name + "::create());"
            for n in range(len(self.A))
        )

        OUTPUT += f"""Problem createProblem{self.name}() {{

    std::vector<std::unique_ptr<Expression>> F;
    {pushF}
    
    std::vector<std::unique_ptr<Constraint>> G;
    {pushG}
    
    std::vector<std::unique_ptr<Constraint>> R;
    {pushR}
    
    std::vector<std::unique_ptr<ParamConstraint>> A;
    {pushA}

    Problem problem(
            {len(self.xVars)}, {len(self.uVars)}, {len(self.pVars)},  // #vars
            {{{', '.join(str(toCode(varInfo[x].start)) for x in self.xVars)}}},  // x0
            {{{', '.join(str(toCode(varInfo[x].lb) if varInfo[x].lb != -float('inf') else "MINUS_INFINITY") for x in self.xVars)}}},  // lb x
            {{{', '.join(str(toCode(varInfo[x].ub) if varInfo[x].ub != float('inf') else "PLUS_INFINITY") for x in self.xVars)}}},  // ub x
            &initialGuessU,  // u0 initial guesses for optimization
            {{{', '.join(str(toCode(varInfo[u].lb) if varInfo[u].lb != -float('inf') else "MINUS_INFINITY") for u in self.uVars)}}},  // lb u
            {{{', '.join(str(toCode(varInfo[u].ub) if varInfo[u].ub != float('inf') else "PLUS_INFINITY") for u in self.uVars)}}},  // ub u
            {{{', '.join(str(toCode(varInfo[p].initialGuess)) for p in self.pVars)}}},  // p0 initial guesses for optimization
            {{{', '.join(str(toCode(varInfo[p].lb) if varInfo[p].lb != -float('inf') else "MINUS_INFINITY") for p in self.pVars)}}},  // lb p
            {{{', '.join(str(toCode(varInfo[p].ub) if varInfo[p].ub != float('inf') else "PLUS_INFINITY") for p in self.pVars)}}},  // ub p
            {"Mayer" + self.name + "::create()" if self.M else "{}"},
            {"Lagrange" + self.name + "::create()" if self.L else "{}"},
            std::move(F),
            std::move(G),
            std::move(R),
            std::move(A),
            "{self.name}");
    
    if (USER_SCALING) {{
        problem.nominalsX = {{{', '.join(str(toCode("1" if varInfo[x].nominal is None else varInfo[x].nominal)) for x in self.xVars)}}};  // nominal states
        problem.nominalsU = {{{', '.join(str(toCode("1" if varInfo[u].nominal is None else varInfo[u].nominal)) for u in self.uVars)}}};  // nominal inputs
        problem.nominalsP = {{{', '.join(str(toCode("1" if varInfo[p].nominal is None else varInfo[p].nominal)) for p in self.pVars)}}};  // nominal inputs
        
        problem.nominalObjective = {toCode(self.objectiveNominal())};  // nominal objective
        problem.nominalsF = {{{', '.join(str(toCode("1" if f.nominal is None else f.nominal)) for f in self.F)}}};  // nominal dynamic
        problem.nominalsG = {{{', '.join(str(toCode("1" if g.nominal is None else g.nominal)) for g in self.G)}}};  // nominal path
        problem.nominalsR = {{{', '.join(str(toCode("1" if r.nominal is None else r.nominal)) for r in self.R)}}};  // nominal final
        problem.nominalsA = {{{', '.join(str(toCode("1" if a.nominal is None else a.nominal)) for a in self.A)}}};  // nominal parametric
    }}
    
    if (INITIAL_STATES_PATH != "") {{
        problem.initialStatesPath = INITIAL_STATES_PATH + "/initialValues.csv";
    }}

    return problem;
}};\n"""

        OUTPUT += f"""
int main(int argc, char** argv) {{
    // setting the global configuration variables from file
    auto config = readConfig(".generated/{self.name}/{self.name}.config");
    setGlobalStandardConfiguration(config);
    setGlobalRuntimeParameters(config);

    // TODO: set argv, basically all runtimeParameters 

    // problem specific structures
    auto problem = std::make_shared<const Problem>(createProblem{self.name}());
    Integrator radau = Integrator::radauIIA(RADAU_INTEGRATOR);
    Mesh mesh = Mesh::createEquidistantMesh(INTERVALS, FINAL_TIME);

    // create the solver
    Solver solver = Solver(createGDOP(problem, mesh, radau, INIT_VARS));

    // optimize
    int status = solver.solve();

    return status;
}}        
        """

        os.makedirs(f".generated/{self.name}", exist_ok=True)

        with open(f".generated/{self.name}/{filename}.cpp", "w") as file:
            file.write(OUTPUT)

        print(
            f"[GDOPT - INFO] Generated model to .generated/{self.name}/{filename}.cpp."
        )
        print(
            f"[GDOPT - TIMING] Derivative calculations and code generation took {round(timer.time() - timeStartGenerate, 4)} seconds."
        )

        self.compile(compiler=compiler, compileFlags=compileFlags)

    def compile(self, compiler="g++", compileFlags=["-O3", "-ffast-math"]):
        print("[GDOPT - INFO] Compiling generated code...")
        compileStart = timer.time()

        with resources.as_file(resources.files(__package__)) as package_path:
            compileFlags += [
                f"-L{package_path / 'lib'}",
                f"-I{package_path / 'include'}",
                f"-Wl,-rpath",
                f"-Wl,{package_path / 'lib'}",
                "-lstdc++",
                "-lm",
            ]

        compileResult = subprocess.run(
            [
                compiler,
                "-std=c++17",
                f".generated/{self.name}/{self.name}Generated.cpp",
                "-lgdopt",
                f"-o.generated/{self.name}/{self.name}",
            ]
            + compileFlags,
            capture_output=True,
        )
        if compileResult.returncode != 0:
            with open(f"compile_{self.name}_err.log", "w+") as errorFile:
                decoded = compileResult.stderr.decode()
                errorFile.write(decoded)
                raise RuntimeError(
                    f"[GDOPT - ERROR] Compilation failed! Check compile_{self.name}_err.log!\n\n"
                    + decoded
                )
        print(
            f"[GDOPT - TIMING] Compiling generated C++ code took {round(timer.time() - compileStart, 4)} seconds."
        )

    def solve(
        self,
        tf=1,
        steps=1,
        rksteps=1,
        flags={},
        meshFlags={},
        resimulate=False,
        compiler="g++",
        compileFlags=["-O3", "-ffast-math"],
    ):
        # generate and optimize pipelines sequentially

        self.generate(compiler=compiler, compileFlags=compileFlags)
        self.optimize(
            tf=tf,
            steps=steps,
            rksteps=rksteps,
            flags=flags,
            meshFlags=meshFlags,
            resimulate=resimulate,
        )

    def optimize(
        self, tf=1, steps=1, rksteps=1, flags={}, meshFlags={}, resimulate=False
    ):
        # sets the runtime configuration with flags, mesh, refinement, runtime parameters
        # get and write initial states, if set InitVars.SOLVE
        # run the code

        if not resimulate:  # use everything from the previous optimization
            if not self._addedDummy:  # use provided values / standard values
                self.tf = tf
                self.steps = steps
                self.rksteps = rksteps
            else:  # purely parametric
                print(
                    "[GDOPT - ATTENTION] Setting tf = 0, steps = 1, rksteps = 1, since the model is purely parametric."
                )
                self.tf = 0
                self.steps = 1
                self.rksteps = 1

            self.setFlags(flags)
            self.setMeshFlags(meshFlags)

        # solve the IVP with scipy if set
        if self._initVars == InitVars.SOLVE:
            self.initialStatesCode()

        # configuration codegen
        print(
            f"[GDOPT - INFO] Creation of configuration .generated/{self.name}/{self.name}.config..."
        )
        with open(f".generated/{self.name}/{self.name}.config", "w") as file:
            file.write(self.createConfigurationCode())
        print("[GDOPT - INFO] Configuration done.")

        self.execute()

    def execute(self, args=""):
        argList = args.split() if args else []
        print(f"[GDOPT - INFO] Executing...")
        runResult = subprocess.run([f"./.generated/{self.name}/{self.name}"] + argList)
        print(
            f"\n[GDOPT - INFO] Exit: {backendReturnCode(runResult.returncode).replace('_', ' ')}.\n"
        )

        self.initAnalysis()

    def initialStatesCode(self):
        print("[GDOPT - INFO] Solving IVP for initial state guesses...")
        solveStart = timer.time()
        timeVals, stateVals = self.solveDynamic()
        print(
            f"[GDOPT - TIMING] Solving the IVP took {round(timer.time() - solveStart, 4)} seconds."
        )

        print(
            f"[GDOPT - INFO] Writing guesses to {self._initialStatesPath + '/initialValues.csv'}..."
        )
        with open(self._initialStatesPath + "/initialValues.csv", "w") as file:
            for i in range(len(timeVals)):
                row = (
                    []
                )  # could add timeColumn for debugging by adding: row = [str(timeVals[i])] -> change JUMP1 as well
                for dim in range(len(self.xVars)):
                    row.append(str(stateVals[dim][i]))

                file.write(",".join(row) + "\n")
        print("[GDOPT - INFO] Initial guesses done.")

    def createConfigurationCode(self):
        # generates the code for the .config file

        OUTPUT = "[standard model parameters]\n"
        OUTPUT += f"FINAL_TIME {self._tf}\n"
        OUTPUT += f"INTERVALS {self._steps}\n"
        OUTPUT += f"RADAU_INTEGRATOR {self._rksteps}\n"
        OUTPUT += f"LINEAR_SOLVER {self._linearSolver.name}\n"
        OUTPUT += f"INIT_VARS {self._initVars.name}\n"
        OUTPUT += f"TOLERANCE {self._tolerance}\n"
        OUTPUT += f"MAX_ITERATIONS {self._maxIterations}\n"
        OUTPUT += f"MESH_ALGORITHM {self._meshAlgorithm.name}\n"
        OUTPUT += f"MESH_ITERATIONS {self._meshIterations}\n"
        OUTPUT += f"REFINEMENT_METHOD {self._refinementMethod.name}\n"
        OUTPUT += f"USER_SCALING {'true' if self._userScaling else 'false'}\n"

        OUTPUT += "\n[constant derivatives]\n"
        OUTPUT += f"LINEAR_OBJECTIVE {'true' if self._linearObjective else 'false'}\n"
        OUTPUT += f"QUADRATIC_OBJECTIVE_LINEAR_CONSTRAINTS {'true' if ((self._quadraticObjective or self._linearObjective) and self._linearConstraints) else 'false'}\n"
        OUTPUT += (
            f"LINEAR_CONSTRAINTS {'true' if self._linearConstraints else 'false'}\n"
        )

        OUTPUT += "\n[optionals: ipopt flags]\n"
        if self._kktMuGlobalization is not None:
            OUTPUT += f"KKT_ERROR_MU_GLOBALIZATION {'true' if self._kktMuGlobalization else 'false'}\n"
        if self._ipoptPrintLevel is not None:
            OUTPUT += f"IPOPT_PRINT_LEVEL {self._ipoptPrintLevel}\n"
        if self._muInit is not None:
            OUTPUT += f"MU_INIT {self._muInit}\n"
        if self._muStrategy is not None:
            OUTPUT += f"MU_STRATEGY {self._muStrategy}\n"
        if self._muInitRefinement is not None:
            OUTPUT += f"MU_INIT_REFINEMENT {self._muInitRefinement}\n"
        if self._muStrategyRefinement is not None:
            OUTPUT += f"MU_STRATEGY_REFINEMENT {self._muStrategyRefinement}\n"

        OUTPUT += "\n[optionals: output]\n"
        if self._outputFilePath:
            OUTPUT += f'EXPORT_OPTIMUM_PATH "{self._outputFilePath}"\n'
        if self._exportHessianPath:
            OUTPUT += f'EXPORT_HESSIAN_PATH "{self._exportHessianPath}"\n'
        if self._exportJacobianPath:
            OUTPUT += f'EXPORT_JACOBIAN_PATH "{self._exportJacobianPath}"\n'
        if self._initVars == InitVars.SOLVE and self._initialStatesPath:
            OUTPUT += f'INITIAL_STATES_PATH "{self._initialStatesPath}"\n'

        OUTPUT += "\n[optionals: mesh refinement]\n"
        if self._meshSigma:
            OUTPUT += f"SIGMA {self._meshSigma}\n"
        if self._meshLevel:
            OUTPUT += f"LEVEL {self._meshLevel}\n"
        if self._meshCTol:
            OUTPUT += f"C_TOL {self._meshCTol}\n"
        if self._detectStatesAndControl is not None:
            OUTPUT += f"STATE_AND_CONTROL_DETECTION {'true' if self._detectStatesAndControl else 'false'}\n"
        if self._fullBisections is not None:
            OUTPUT += f"FULL_BISECTIONS {self._fullBisections}\n"

        OUTPUT += "\n[runtime parameters]\n"
        for rp in self.rpVars:
            info = varInfo[rp]
            OUTPUT += f'{str(info.symbol).upper().strip(" ")}_VALUE {info.value}\n'

        return OUTPUT

    def initVarNames(self):
        self.xVarNames = [
            info.symbol for info in varInfo.values() if isinstance(info, StateStruct)
        ]
        self.uVarNames = [
            info.symbol for info in varInfo.values() if isinstance(info, InputStruct)
        ]
        self.pVarNames = [
            info.symbol
            for info in varInfo.values()
            if isinstance(info, ParameterStruct)
        ]
        self.rpVarNames = [
            info.symbol
            for info in varInfo.values()
            if isinstance(info, RuntimeParameterStruct)
        ]

    def getResults(self, meshIteration=None):
        meshIteration = self.checkMeshIteration(meshIteration)

        if meshIteration not in self.resultHistory:
            try:
                results = pd.read_csv(
                    self._outputFilePath
                    + "/"
                    + self.name
                    + str(meshIteration)
                    + ".csv",
                    delimiter=",",
                )
            except:
                raise Exception(
                    "File does not exist or meshIteration out of range. Provide an exportOptimum path or set Model.maxMeshIteration to the maximum mesh iteration!"
                )

            # remove dummy column for purely parametric models
            if self._addedDummy:
                results = results.drop(columns=["x[0]"])

            alias = {variable.name: varInfo[variable].symbol for variable in varInfo}
            for col in results.columns:
                if col not in alias:
                    alias[col] = col
            results.rename(columns=alias, inplace=True)
            self.resultHistory[meshIteration] = results
        self.initVarNames()
        return self.resultHistory[meshIteration]

    def getAllResults(self):
        for it in range(self.modelInfo["maxMeshIteration"] + 1):
            self.getResults(meshIteration=it)
        return self.resultHistory

    def printResults(self, meshIteration=None):
        if meshIteration is None:
            meshIteration = self.modelInfo["maxMeshIteration"]
        meshIteration = self.checkMeshIteration(meshIteration)
        self.getResults(meshIteration)
        meshIteration = self.modelInfo["maxMeshIteration"]
        print("")
        self._printFull(self.resultHistory[meshIteration])

    def printResultParameters(self, meshIteration=None):
        meshIteration = self.checkMeshIteration(meshIteration)
        self.getResults(meshIteration)
        print("Optimal parameters:")
        for p, pValue in (
            self.resultHistory[meshIteration][self.pVarNames].iloc[0].items()
        ):
            print(f"{p} = {pValue}")
        print("")

    def _printFull(self, data):
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.width",
            2000,
            "display.max_colwidth",
            None,
        ):
            print(data)

    def checkMeshIteration(self, meshIteration):
        maxMeshIteration = self.modelInfo["maxMeshIteration"]
        if meshIteration is None:
            return maxMeshIteration
        if type(meshIteration) == int:
            if meshIteration > maxMeshIteration:
                print(
                    f"[GDOPT - ERROR] meshIteration too large. Setting meshIteration to maximum value of {maxMeshIteration}."
                )
                meshIteration = maxMeshIteration
        return meshIteration

    def exportToCombiTable(
        self,
        specificCols,
        path="",
        meshIteration=None,
        polySteps=40,
        linearControl=False,
    ):
        """
        Export to OpenModelica CombiTable format
        """

        print("[GDOPT - INFO] Exporting to CombiTable...")
        meshIteration = self.checkMeshIteration(meshIteration)
        self.getResults(meshIteration=meshIteration)

        if not path:
            path = (
                self._outputFilePath
                + "/"
                + self.name
                + "CombiTable"
                + str(meshIteration)
                + ".txt"
            )

        length = len(self.resultHistory[meshIteration]["time"])
        with open(path, "w") as f:
            # interpolate values according to collocation scheme here
            tvals = []
            vals = [[] for _ in range(len(specificCols))]
            intervals = [
                (i * self.rksteps, (i + 1) * self.rksteps)
                for i in range((length - 1) // self.rksteps)
            ]

            for idx, (start, end) in enumerate(intervals):
                t_interval = np.array(self.resultHistory[meshIteration]["time"])[
                    start : end + 1
                ]
                tDense = np.linspace(t_interval[0], t_interval[-1], polySteps)

                if idx == 0:
                    tvals.extend(tDense)
                else:
                    tvals.extend(tDense[1:])

                for j, col in enumerate(specificCols):
                    x_interval = np.array(self.resultHistory[meshIteration][col])[
                        start : end + 1
                    ]

                    # linear interpolation
                    if col in self.uVarNames and linearControl:
                        linInpVals = np.interp(tDense, t_interval, x_interval)
                        if idx == 0:
                            vals[j].extend(linInpVals)
                        else:
                            vals[j].extend(linInpVals[1:])

                    # piecewise polynomial
                    else:
                        polyCoeffs = np.polyfit(t_interval, x_interval, self.rksteps)
                        polyFunc = np.poly1d(polyCoeffs)

                        interpolated_values = polyFunc(tDense)
                        if idx == 0:
                            vals[j].extend(interpolated_values)
                        else:
                            vals[j].extend(interpolated_values[1:])

            f.write(f"#1\ndouble K({len(vals[0])}, {1 + len(specificCols)})\n")

            for idx in range(len(tvals)):
                line = str(tvals[idx]) + " "
                for j, col in enumerate(specificCols):
                    line += str(vals[j][idx]) + " "
                f.write(line[:-1] + "\n")

        print("[GDOPT - INFO] CombiTable export successful.")

    def setPlotDefaults(self):
        plt.rcParams.update(
            {
                "font.serif": ["Times New Roman"],
                "axes.labelsize": 14,
                "axes.titlesize": 16,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
                "legend.frameon": True,
                "legend.loc": "best",
                "grid.alpha": 0.3,
                "grid.linestyle": "--",
                "grid.linewidth": 0.5,
                "figure.figsize": (12, 8),
                "axes.titlepad": 20,
                "axes.labelpad": 10,
            }
        )

    # TODO: add plotting features for path constraints

    def plot(self, specifCols=None, meshIteration=None, interval=None, dots=Dots.OFF):
        self.initVarNames()
        self.plotGeneral(
            meshIteration=meshIteration,
            interval=interval,
            dots=dots,
            specifCols=specifCols,
        )

    def plotStates(self, meshIteration=None, interval=None, dots=Dots.OFF):
        self.initVarNames()
        self.plotGeneral(
            meshIteration=meshIteration,
            interval=interval,
            dots=dots,
            specifCols=self.xVarNames,
        )

    def plotInputs(self, meshIteration=None, interval=None, dots=Dots.OFF):
        self.initVarNames()
        self.plotGeneral(
            meshIteration=meshIteration,
            interval=interval,
            dots=dots,
            specifCols=self.uVarNames,
        )

    def parametricPlot(
        self, varX, varY, meshIteration=None, interval=None, dots=Dots.OFF
    ):
        self.initVarNames()
        if interval is None:
            interval = [0, self._tf]
        self._parametricPlot(
            varX=varX,
            varY=varY,
            meshIteration=meshIteration,
            interval=interval,
            dots=dots,
        )

    def plotInputsAndRefinement(
        self,
        meshIteration=None,
        interval=None,
        markerSize=30,
        dotsMesh=Dots.BASE,
        dotsGraph=Dots.BASE,
        epsilon=1e-14,
    ):
        self.initVarNames()
        self.plotVarsAndRefinement(
            meshIteration=meshIteration,
            interval=interval,
            specifCols=self.uVarNames,
            markerSize=markerSize,
            dotsMesh=dotsMesh,
            dotsGraph=dotsGraph,
            epsilon=epsilon,
        )

    def plotVarsAndRefinement(
        self,
        meshIteration=None,
        interval=None,
        specifCols=None,
        markerSize=30,
        dotsMesh=Dots.BASE,
        dotsGraph=Dots.OFF,
        epsilon=1e-14,
    ):
        from matplotlib.ticker import MaxNLocator

        if interval is None:
            interval = [0, self.tf]

        # Generate mesh refinement and general plots
        figMesh, axMesh = self._plotMeshRefinement(
            interval=interval, markerSize=markerSize, dots=dotsMesh, epsilon=epsilon
        )
        figGraph, axsGraph = self._plotGeneral(
            meshIteration=meshIteration,
            interval=interval,
            specifCols=specifCols,
            dots=dotsGraph,
        )

        # Close the generated figures to avoid duplicate display
        plt.close(figGraph)
        plt.close(figMesh)

        # Create new subplots with an extra axis for mesh refinement
        fig, axs = plt.subplots(len(axsGraph) + 1, 1, figsize=(12, 10), sharex=True)

        # Plot data for intermediate axes
        for i in range(len(axsGraph)):
            axs[i].plot(
                axsGraph[i].lines[0].get_xdata(), axsGraph[i].lines[0].get_ydata()
            )
            axs[i].set_ylabel(axsGraph[i].get_ylabel())
            axs[i].set_xlim(interval)

            # Hide x-ticks and labels for intermediate axes
            axs[i].tick_params(axis="x", which="both", bottom=False, labelbottom=False)

            # Add scatter points if collections are present
            if axsGraph[i].collections:
                for coll in axsGraph[i].collections:
                    offsets = coll.get_offsets()
                    axs[i].scatter(
                        offsets[:, 0],
                        offsets[:, 1],
                        color="red",
                        s=30,
                        edgecolor="black",
                        alpha=0.8,
                        zorder=5,
                    )

        # Plot mesh refinement data on the last axis
        if axMesh.collections:
            for coll in axMesh.collections:
                offsets = coll.get_offsets()
                axs[-1].scatter(
                    offsets[:, 0],
                    offsets[:, 1],
                    color="red",
                    s=markerSize,
                    edgecolor="black",
                    alpha=0.8,
                )
            axs[-1].set_ylabel(axMesh.get_ylabel())
            axs[-1].set_xlim(interval)
            axs[-1].yaxis.set_major_locator(MaxNLocator(integer=True))

        # Allow Matplotlib's default x-ticks on the last axis
        axs[-1].set_xlabel("Time")

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.subplots_adjust(left=0.075, right=0.95, top=0.925, bottom=0.075, hspace=0.1)
        plt.show()

    def plotMeshRefinement(
        self, interval=None, markerSize=30, dots=Dots.BASE, epsilon=1e-14
    ):
        fig, ax = self._plotMeshRefinement(
            interval=interval, markerSize=markerSize, dots=dots, epsilon=epsilon
        )
        ax.set_xlabel("Time")
        ax.set_title("Inserted Mesh Points Over Time")
        plt.tight_layout()
        plt.subplots_adjust(left=0.075, right=0.95, top=0.925, bottom=0.075, hspace=0.1)
        plt.show()

    def plotGeneral(
        self, meshIteration=None, interval=None, specifCols=None, dots=Dots.OFF
    ):
        fig, axs = self._plotGeneral(
            meshIteration=meshIteration,
            interval=interval,
            specifCols=specifCols,
            dots=dots,
        )
        axs[-1].set_xlabel("Time")
        axs[0].set_title(f"Optimal Solution: {self.name}")
        plt.tight_layout()
        plt.subplots_adjust(left=0.075, right=0.95, top=0.925, bottom=0.075, hspace=0.1)
        plt.show()

    def _plotGeneral(
        self, meshIteration=None, interval=None, specifCols=None, dots=Dots.OFF
    ):
        self.setPlotDefaults()

        meshIteration = self.checkMeshIteration(meshIteration)
        if interval is None:
            interval = [0, self._tf]
        self.getResults(meshIteration=meshIteration)

        columns_to_plot = specifCols or self.resultHistory[meshIteration].columns[1:]

        fig, axs = plt.subplots(
            len(columns_to_plot), 1, figsize=(12, 8 * len(columns_to_plot)), sharex=True
        )

        if len(columns_to_plot) == 1:
            axs = [axs]

        for idx, column in enumerate(columns_to_plot):
            ax = axs[idx]
            ax.plot(
                self.resultHistory[meshIteration]["time"],
                self.resultHistory[meshIteration][column],
                label=column,
                linewidth=2,
                linestyle="-",
                color="steelblue",
            )
            self._applyDots(ax, dots, meshIteration, column)
            ax.set_ylabel(column)
            ax.set_xlim(interval)
            ax.legend(frameon=True, loc="best")

        return fig, axs

    def _parametricPlot(
        self, varX, varY, meshIteration=None, interval=None, dots=Dots.OFF
    ):
        self.setPlotDefaults()

        meshIteration = self.checkMeshIteration(meshIteration)
        if interval is None:
            interval = [0, self._tf]

        self.getResults(meshIteration=meshIteration)

        x_data = self.resultHistory[meshIteration][varInfo[varX].symbol]
        y_data = self.resultHistory[meshIteration][varInfo[varY].symbol]
        time_data = self.resultHistory[meshIteration]["time"]
        time_filtered = (time_data >= interval[0]) & (time_data <= interval[1])

        plt.plot(x_data[time_filtered], y_data[time_filtered])

        if dots != Dots.OFF:
            self._applyDotsParametric(
                x_data[time_filtered], y_data[time_filtered], dots
            )

        plt.title(f"{varInfo[varX].symbol} vs {varInfo[varY].symbol}")
        plt.xlabel(varInfo[varX].symbol)
        plt.ylabel(varInfo[varY].symbol)
        plt.tight_layout()
        plt.show()

    # helper function to apply dots in regular plots
    def _applyDots(self, ax, dots, meshIteration, column):
        if dots == Dots.ALL:
            ax.scatter(
                self.resultHistory[meshIteration]["time"],
                self.resultHistory[meshIteration][column],
                color="red",
                s=30,
                edgecolor="black",
                alpha=0.8,
                zorder=5,
            )
        elif dots == Dots.BASE:
            ax.scatter(
                [
                    x
                    for i, x in enumerate(self.resultHistory[meshIteration]["time"])
                    if i % self._rksteps == 0
                ],
                [
                    x
                    for i, x in enumerate(self.resultHistory[meshIteration][column])
                    if i % self._rksteps == 0
                ],
                color="red",
                s=30,
                edgecolor="black",
                alpha=0.8,
                zorder=5,
            )

    # helper function to apply dots in parametric plots
    def _applyDotsParametric(self, x_data, y_data, dots):
        if dots == Dots.ALL:
            plt.scatter(
                x_data,
                y_data,
                color="red",
                s=30,
                edgecolor="black",
                alpha=0.8,
                zorder=5,
            )
        elif dots == Dots.BASE:
            plt.scatter(
                [x for i, x in enumerate(x_data) if i % self._rksteps == 0],
                [y for i, y in enumerate(y_data) if i % self._rksteps == 0],
                color="red",
                s=30,
                edgecolor="black",
                alpha=0.8,
                zorder=5,
            )

    # private mesh refinement plot function
    def _plotMeshRefinement(
        self, ax=None, interval=None, markerSize=30, dots=Dots.BASE, epsilon=1e-14
    ):
        self.setPlotDefaults()
        from matplotlib.ticker import MaxNLocator

        if interval is None:
            interval = [0, self._tf]

        prev_points = np.array([])

        fig, ax = plt.subplots(figsize=(12, 8)) if ax is None else (None, ax)

        for m in range(self.modelInfo["maxMeshIteration"] + 1):
            self.getResults(m)
            arr = self.resultHistory[m]["time"].to_numpy()[:: self._getModulo(dots)]

            if m == 0:
                ax.scatter(
                    arr,
                    np.ones_like(arr) * m,
                    color="red",
                    s=markerSize,
                    edgecolor="black",
                    alpha=0.8,
                    zorder=5,
                )
            else:
                new_points = np.array(
                    [
                        pt
                        for pt in arr
                        if prev_points.size == 0
                        or not np.any(np.abs(prev_points - pt) < epsilon)
                    ]
                )
                ax.scatter(
                    new_points,
                    np.ones_like(new_points) * m,
                    color="red",
                    s=markerSize,
                    edgecolor="black",
                    alpha=0.8,
                    zorder=5,
                )

            prev_points = arr

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(interval)
        ax.set_xlabel("Time")
        ax.set_ylabel("Iteration")
        ax.set_title("Inserted Mesh Points Over Time")
        return fig, ax

    def _getModulo(self, dots):
        return 1 if dots == Dots.ALL else self._rksteps

    def plotSparseMatrix(self, matrixType):
        from matplotlib.patches import Rectangle
        import scipy.sparse as sp

        if matrixType == MatrixType.JACOBIAN:
            file_path = self._exportJacobianPath + f"/{self.name}_jacobian.csv"
        elif matrixType == MatrixType.HESSIAN:
            file_path = self._exportHessianPath + f"/{self.name}_hessian.csv"
        else:
            raise InvalidMatrix(
                "Plotting is only possible for matrixTypes JACOBIAN or HESSIAN."
            )

        with open(file_path, "r") as f:
            lines = f.readlines()

        dim_row, dim_col = map(int, lines[1].strip().split(","))

        rows, cols = [], []
        for line in lines[3:]:
            row, col = map(int, line.strip().split(","))
            rows.append(row)
            cols.append(col)

        data = np.ones(len(rows))
        m = sp.coo_matrix((data, (rows, cols)), shape=(dim_row, dim_col))

        fig, ax = plt.subplots()

        for x, y, data in zip(m.col, m.row, m.data):
            ax.add_patch(
                Rectangle(
                    xy=(x, y),
                    width=1,
                    height=1,
                    edgecolor="black",
                    facecolor="blue",
                    alpha=0.6,
                )
            )

        ax.set_xlim(0, m.shape[1])
        ax.set_ylim(0, m.shape[0])
        ax.invert_yaxis()

        if matrixType == MatrixType.JACOBIAN:
            ax.set_xlabel("Variables")
            ax.set_ylabel("Equations")
            ax.set_title("Jacobian Sparsity")
        elif matrixType == MatrixType.HESSIAN:
            ax.set_xlabel("Variables")
            ax.set_ylabel("Variables")
            ax.set_title("Hessian Sparsity")

        plt.tight_layout()
        plt.subplots_adjust(left=0.075, right=0.95, top=0.925, bottom=0.075, hspace=0.1)
        plt.show()


def HelloWorld():
    # MWE with u*(t) = 1 constant.

    print("[GDOPT - Hello World] Running the Hello World problem...")
    print("[GDOPT - Hello World] u*(t) = 1, x*(t) = t - t², obj* = 1.\n")

    hw = Model("Hello World")

    x = hw.addState(start=0)
    u = hw.addControl(
        lb=0, ub=2, guess=(-2 + 2 * sqrt(exp(1))) / (1 + (-1 + sqrt(exp(1))) * t)
    )

    hw.addDynamic(x, 2 * t - u)

    hw.addFinal(x, eq=0)

    hw.addLagrange(u**2, obj=Objective.MINIMIZE)

    print(hw)

    hw.hasLinearConstraints()
    hw.hasQuadraticObjective()

    # steps=1, rksteps=2 would be sufficient for the analytic solution!
    hw.solve(tf=1, steps=15, rksteps=3, flags={"tolerance": 1e-14})

    hw.plot(dots=Dots.ALL)

    print("[GDOPT - Hello World] Success!")
