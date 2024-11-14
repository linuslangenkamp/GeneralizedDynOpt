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

from enum import Enum


class InvalidModel(Exception):
    pass


class InvalidMatrix(Exception):
    pass


class InitVars(Enum):
    CONST = 1
    SOLVE = 2
    SOLVE_EXPLICIT = 3
    SOLVE_EXPLICIT_EULER = 4


class Objective(Enum):
    MINIMIZE = 1
    MAXIMIZE = 2
    MAX = MAXIMIZE
    MIN = MINIMIZE


class LinearSolver(Enum):
    MUMPS = 1
    MA27 = 2
    MA57 = 3
    MA77 = 4
    MA86 = 5
    MA97 = 6
    PARDISO = 7


class RefinementMethod(Enum):
    LINEAR_SPLINE = 1
    POLYNOMIAL = 2


class MeshAlgorithm(Enum):
    NONE = 1
    BASIC = 2
    L2_BOUNDARY_NORM = 3


class MatrixType(Enum):
    JACOBIAN = 1
    HESSIAN = 2


class Dots(Enum):
    OFF = 1
    ALL = 2
    BASE = 3


class IVPSolver(Enum):
    Radau = 1
    BDF = 2
    LSODA = 3
    RK45 = 4
    DOP853 = 5
    RK23 = 6
    RADAU = Radau


class MuStrategy(Enum):
    ADAPTIVE = 1
    MONOTONE = 2

    def __str__(self):
        if self == MuStrategy.ADAPTIVE:
            return "adaptive"
        elif self == MuStrategy.MONOTONE:
            return "monotone"
        else:
            raise ValueError("Mu strategy must be MONOTONE or ADAPTIVE")
