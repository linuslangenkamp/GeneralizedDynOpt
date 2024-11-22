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

from .variables import *
from symengine import (
    Symbol,
    cse,
    diff,
    ccode,
    symbols,
    Lambdify,
    exp,
    sin,
    cos,
    tan,
    log,
    sqrt,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    tanh,
    asinh,
    acosh,
    atanh,
    pi,
    Max,
    Min,
    Piecewise,
)

# global map varInfo: symbol -> info
varInfo = {}

# definition of the global time symbol
t = Symbol("t")
time = t
TIME_SYMBOL = t  # use this in code to recognize it as a symbol

# definition of global final time symbol, has macro FINAL_TIME as name -> will be substituted at compile time
tf = Symbol("FINAL_TIME")
finalTime = tf
FINAL_TIME_SYMBOL = tf  # use this in code to recognize it as a symbol


class Expression:
    simplification = False

    def __init__(self, expr, nominal=None):
        if Expression.simplification:
            self.expr = expr.simplify()
        else:
            self.expr = expr

        self.nominal = nominal
        self.adj = []
        try:
            for sym in expr.free_symbols:
                if sym in varInfo and not isinstance(
                    varInfo[sym], RuntimeParameterStruct
                ):
                    self.adj.append(sym)
            sort_symbols(self.adj, varInfo)
        except:
            self.adj = []

    def __str__(self):
        return str(self.expr)

    def __repr__(self):
        return str(self.expr)

    def codegen(self, name):

        subst0, exprEval = cseCustom([self.expr])
        cEval = toCode(exprEval[0])

        # Generate second-order derivatives only for lower triangular part
        cEvalDiff2 = [[], [], [], [], [], []]
        adjDiff_indices = [
            [],
            [],
            [],
            [],
            [],
            [],
        ]  # indXX, indUX, indUU, indPX, indPU, indPP
        allDiffs2 = []
        diffVars2 = []

        for i, var1 in enumerate(self.adj):
            for j, var2 in enumerate(self.adj):
                info1, info2 = varInfo[var1], varInfo[var2]
                if not (
                    (
                        isinstance(info1, StateStruct)
                        and isinstance(info2, StateStruct)
                        and i < j
                    )
                    or (
                        isinstance(info1, InputStruct)
                        and isinstance(info2, InputStruct)
                        and i < j
                    )
                    or (
                        isinstance(info1, ParameterStruct)
                        and isinstance(info2, ParameterStruct)
                        and i < j
                    )
                ):

                    der = diff(self.expr, var1)
                    if var2 in der.free_symbols:
                        der = diff(der, var2)
                        if der != 0:
                            allDiffs2.append(der)
                            diffVars2.append((var1, var2))

        subst2, substExpr2 = cseCustom(allDiffs2)

        for i, expr in enumerate(substExpr2):
            var1, var2 = diffVars2[i]
            info1, info2 = varInfo[var1], varInfo[var2]
            if isinstance(info1, StateStruct) and isinstance(info2, StateStruct):
                cEvalDiff2[0].append(toCode(expr))
                adjDiff_indices[0].append((info1.id, info2.id))  # indXX
            elif isinstance(info1, InputStruct) and isinstance(info2, StateStruct):
                cEvalDiff2[1].append(toCode(expr))
                adjDiff_indices[1].append((info1.id, info2.id))  # indUX
            elif isinstance(info1, InputStruct) and isinstance(info2, InputStruct):
                cEvalDiff2[2].append(toCode(expr))
                adjDiff_indices[2].append((info1.id, info2.id))  # indUU
            elif isinstance(info1, ParameterStruct) and isinstance(info2, StateStruct):
                cEvalDiff2[3].append(toCode(expr))
                adjDiff_indices[3].append((info1.id, info2.id))  # indPX
            elif isinstance(info1, ParameterStruct) and isinstance(info2, InputStruct):
                cEvalDiff2[4].append(toCode(expr))
                adjDiff_indices[4].append((info1.id, info2.id))  # indPU
            elif isinstance(info1, ParameterStruct) and isinstance(
                info2, ParameterStruct
            ):
                cEvalDiff2[5].append(toCode(expr))
                adjDiff_indices[5].append((info1.id, info2.id))  # indPP

        # first derivatives
        adj_indices = [[], [], []]  # indX, indU, indP
        cEvalDiff = [[], [], []]
        allDiffs = []
        diffVars = []

        for i, var in enumerate(self.adj):
            der = diff(self.expr, var)
            if der != 0:
                allDiffs.append(der)
                diffVars.append(var)

        subst1, substExpr = cseCustom(allDiffs)

        for v, expr in enumerate(substExpr):
            var = diffVars[v]
            info = varInfo[var]
            if isinstance(info, StateStruct):
                adj_indices[0].append(info.id)
                cEvalDiff[0].append(toCode(expr))
            elif isinstance(info, InputStruct):
                adj_indices[1].append(info.id)
                cEvalDiff[1].append(toCode(expr))
            elif isinstance(info, ParameterStruct):
                adj_indices[2].append(info.id)
                cEvalDiff[2].append(toCode(expr))

        adj = "{{{}, {}, {}}}".format(
            "{{{}}}".format(", ".join(map(str, adj_indices[0]))),
            "{{{}}}".format(", ".join(map(str, adj_indices[1]))),
            "{{{}}}".format(", ".join(map(str, adj_indices[2]))),
        )
        adjDiff = "{{{}, {}, {}, {}, {}, {}}}".format(
            "{{{}}}".format(", ".join(f"{{{i}, {j}}}" for i, j in adjDiff_indices[0])),
            "{{{}}}".format(", ".join(f"{{{i}, {j}}}" for i, j in adjDiff_indices[1])),
            "{{{}}}".format(", ".join(f"{{{i}, {j}}}" for i, j in adjDiff_indices[2])),
            "{{{}}}".format(", ".join(f"{{{i}, {j}}}" for i, j in adjDiff_indices[3])),
            "{{{}}}".format(", ".join(f"{{{i}, {j}}}" for i, j in adjDiff_indices[4])),
            "{{{}}}".format(", ".join(f"{{{i}, {j}}}" for i, j in adjDiff_indices[5])),
        )

        out = f"class {name} : public Expression {{\n"
        out += "public:\n"
        out += f"\tstatic std::unique_ptr<{name}> create() {{\n"
        out += f"\t\tAdjacency adj{adj};\n"
        out += f"\t\tAdjacencyDiff adjDiff{adjDiff};\n"
        out += f"\t\treturn std::unique_ptr<{name}>(new {name}(std::move(adj), std::move(adjDiff)));\n"
        out += "\t}\n\n"

        out += "\tdouble eval(const double *x, const double *u, const double *p, double t) override {\n"

        for s in subst0:
            out += "        const double " + toCode(s[0]) + " = " + toCode(s[1]) + ";\n"

        out += f"\t\treturn {cEval};\n"
        out += "\t}\n\n"

        out += f"\tstd::array<std::vector<double>, 3> evalDiff(const double *x, const double *u, const double *p, double t) override {{\n"

        for s in subst1:
            out += "        const double " + toCode(s[0]) + " = " + toCode(s[1]) + ";\n"

        out += "\t\treturn {std::vector<double>"
        out += "{{{}}}, ".format(", ".join(cEvalDiff[0]))
        out += "{{{}}}, ".format(", ".join(cEvalDiff[1]))
        out += "{{{}}}".format(", ".join(cEvalDiff[2]))
        out += "};\n\t}\n\n"

        out += f"\tstd::array<std::vector<double>, 6> evalDiff2(const double *x, const double *u, const double *p, double t) override {{\n"
        for s in subst2:
            out += "        const double " + toCode(s[0]) + " = " + toCode(s[1]) + ";\n"

        out += "\t\treturn {std::vector<double>"
        out += "{{{}}}, ".format(", ".join(cEvalDiff2[0]))
        out += "{{{}}}, ".format(", ".join(cEvalDiff2[1]))
        out += "{{{}}}, ".format(", ".join(cEvalDiff2[2]))
        out += "{{{}}}, ".format(", ".join(cEvalDiff2[3]))
        out += "{{{}}}, ".format(", ".join(cEvalDiff2[4]))
        out += "{{{}}}".format(", ".join(cEvalDiff2[5]))
        out += "};\n\t}\n"

        out += "private:\n"
        out += f"\t{name}(Adjacency adj, AdjacencyDiff adjDiff) : Expression(std::move(adj), std::move(adjDiff)) {{}}\n"
        out += "};\n\n\n"
        return out


class DynExpression(Expression):

    def __init__(self, diffVar, expr, nominal=None):
        super().__init__(expr, nominal=nominal)
        self.diffVar = diffVar

    def __str__(self):
        return f"{self.diffVar}' = {self.expr}"

    def __repr__(self):
        return f"{self.diffVar}' = {self.expr}"


class Constraint(Expression):

    def __init__(self, expr, lb=-float("inf"), ub=float("inf"), eq=None, nominal=None):
        super().__init__(expr, nominal=nominal)
        if eq is not None:
            self.lb = eq
            self.ub = eq
        else:
            self.lb = lb
            self.ub = ub

    def __str__(self):
        if self.lb == self.ub:
            return f"{self.expr} == {self.lb}"
        else:
            return f"{self.lb} <= {self.expr} <= {self.ub}"

    def __repr__(self):
        if self.lb == self.ub:
            return f"{self.expr} == {self.lb}"
        else:
            return f"{self.lb} <= {self.expr} <= {self.ub}"

    def codegen(self, name):

        subst0, exprEval = cseCustom([self.expr])
        cEval = toCode(exprEval[0])

        # Generate second-order derivatives only for lower triangular part
        cEvalDiff2 = [[], [], [], [], [], []]
        adjDiff_indices = [
            [],
            [],
            [],
            [],
            [],
            [],
        ]  # indXX, indUX, indUU, indPX, indPU, indPP
        allDiffs2 = []
        diffVars2 = []

        for i, var1 in enumerate(self.adj):
            for j, var2 in enumerate(self.adj):
                info1, info2 = varInfo[var1], varInfo[var2]
                if not (
                    (
                        isinstance(info1, StateStruct)
                        and isinstance(info2, StateStruct)
                        and i < j
                    )
                    or (
                        isinstance(info1, InputStruct)
                        and isinstance(info2, InputStruct)
                        and i < j
                    )
                    or (
                        isinstance(info1, ParameterStruct)
                        and isinstance(info2, ParameterStruct)
                        and i < j
                    )
                ):

                    der = diff(self.expr, var1)
                    if var2 in der.free_symbols:
                        der = diff(der, var2)
                        if der != 0:
                            allDiffs2.append(der)
                            diffVars2.append((var1, var2))

        subst2, substExpr2 = cseCustom(allDiffs2)

        for i, expr in enumerate(substExpr2):
            var1, var2 = diffVars2[i]
            info1, info2 = varInfo[var1], varInfo[var2]
            if isinstance(info1, StateStruct) and isinstance(info2, StateStruct):
                cEvalDiff2[0].append(toCode(expr))
                adjDiff_indices[0].append((info1.id, info2.id))  # indXX
            elif isinstance(info1, InputStruct) and isinstance(info2, StateStruct):
                cEvalDiff2[1].append(toCode(expr))
                adjDiff_indices[1].append((info1.id, info2.id))  # indUX
            elif isinstance(info1, InputStruct) and isinstance(info2, InputStruct):
                cEvalDiff2[2].append(toCode(expr))
                adjDiff_indices[2].append((info1.id, info2.id))  # indUU
            elif isinstance(info1, ParameterStruct) and isinstance(info2, StateStruct):
                cEvalDiff2[3].append(toCode(expr))
                adjDiff_indices[3].append((info1.id, info2.id))  # indPX
            elif isinstance(info1, ParameterStruct) and isinstance(info2, InputStruct):
                cEvalDiff2[4].append(toCode(expr))
                adjDiff_indices[4].append((info1.id, info2.id))  # indPU
            elif isinstance(info1, ParameterStruct) and isinstance(
                info2, ParameterStruct
            ):
                cEvalDiff2[5].append(toCode(expr))
                adjDiff_indices[5].append((info1.id, info2.id))  # indPP

        # first derivatives
        adj_indices = [[], [], []]  # indX, indU, indP
        cEvalDiff = [[], [], []]
        allDiffs = []
        diffVars = []

        for i, var in enumerate(self.adj):
            der = diff(self.expr, var)
            if der != 0:
                allDiffs.append(der)
                diffVars.append(var)

        subst1, substExpr = cseCustom(allDiffs)

        for v, expr in enumerate(substExpr):
            var = diffVars[v]
            info = varInfo[var]
            if isinstance(info, StateStruct):
                adj_indices[0].append(info.id)
                cEvalDiff[0].append(toCode(expr))
            elif isinstance(info, InputStruct):
                adj_indices[1].append(info.id)
                cEvalDiff[1].append(toCode(expr))
            elif isinstance(info, ParameterStruct):
                adj_indices[2].append(info.id)
                cEvalDiff[2].append(toCode(expr))

        adj = "{{{}, {}, {}}}".format(
            "{{{}}}".format(", ".join(map(str, adj_indices[0]))),
            "{{{}}}".format(", ".join(map(str, adj_indices[1]))),
            "{{{}}}".format(", ".join(map(str, adj_indices[2]))),
        )
        adjDiff = "{{{}, {}, {}, {}, {}, {}}}".format(
            "{{{}}}".format(", ".join(f"{{{i}, {j}}}" for i, j in adjDiff_indices[0])),
            "{{{}}}".format(", ".join(f"{{{i}, {j}}}" for i, j in adjDiff_indices[1])),
            "{{{}}}".format(", ".join(f"{{{i}, {j}}}" for i, j in adjDiff_indices[2])),
            "{{{}}}".format(", ".join(f"{{{i}, {j}}}" for i, j in adjDiff_indices[3])),
            "{{{}}}".format(", ".join(f"{{{i}, {j}}}" for i, j in adjDiff_indices[4])),
            "{{{}}}".format(", ".join(f"{{{i}, {j}}}" for i, j in adjDiff_indices[5])),
        )

        lb = self.lb if self.lb != -float("inf") else "MINUS_INFINITY"
        ub = self.ub if self.ub != float("inf") else "PLUS_INFINITY"

        out = f"class {name} : public Constraint {{\n"
        out += "public:\n"
        out += f"\tstatic std::unique_ptr<{name}> create() {{\n"
        out += f"\t\tAdjacency adj{adj};\n"
        out += f"\t\tAdjacencyDiff adjDiff{adjDiff};\n"
        out += f"\t\treturn std::unique_ptr<{name}>(new {name}(std::move(adj), std::move(adjDiff), {toCode(lb)}, {toCode(ub)}));\n"
        out += "\t}\n\n"

        out += "\tdouble eval(const double *x, const double *u, const double *p, double t) override {\n"

        for s in subst0:
            out += "        const double " + toCode(s[0]) + " = " + toCode(s[1]) + ";\n"

        out += f"\t\treturn {cEval};\n"
        out += "\t}\n\n"

        out += f"\tstd::array<std::vector<double>, 3> evalDiff(const double *x, const double *u, const double *p, double t) override {{\n"

        for s in subst1:
            out += "        const double " + toCode(s[0]) + " = " + toCode(s[1]) + ";\n"

        out += "\t\treturn {std::vector<double>"
        out += "{{{}}}, ".format(", ".join(cEvalDiff[0]))
        out += "{{{}}}, ".format(", ".join(cEvalDiff[1]))
        out += "{{{}}}".format(", ".join(cEvalDiff[2]))
        out += "};\n\t}\n\n"

        out += f"\tstd::array<std::vector<double>, 6> evalDiff2(const double *x, const double *u, const double *p, double t) override {{\n"
        for s in subst2:
            out += "        const double " + toCode(s[0]) + " = " + toCode(s[1]) + ";\n"

        out += "\t\treturn {std::vector<double>"
        out += "{{{}}}, ".format(", ".join(cEvalDiff2[0]))
        out += "{{{}}}, ".format(", ".join(cEvalDiff2[1]))
        out += "{{{}}}, ".format(", ".join(cEvalDiff2[2]))
        out += "{{{}}}, ".format(", ".join(cEvalDiff2[3]))
        out += "{{{}}}, ".format(", ".join(cEvalDiff2[4]))
        out += "{{{}}}".format(", ".join(cEvalDiff2[5]))
        out += "};\n\t}\n"

        out += "private:\n"
        out += f"\t{name}(Adjacency adj, AdjacencyDiff adjDiff, double lb, double ub) : Constraint(std::move(adj), std::move(adjDiff), lb, ub) {{}}\n"
        out += "};\n\n\n"
        return out


class ParametricConstraint(Expression):

    def __init__(self, expr, lb=-float("inf"), ub=float("inf"), eq=None, nominal=None):
        super().__init__(expr, nominal=nominal)
        if eq is not None:
            self.lb = eq
            self.ub = eq
        else:
            self.lb = lb
            self.ub = ub

    def __str__(self):
        if self.lb == self.ub:
            return f"{self.expr} == {self.lb}"
        else:
            return f"{self.lb} <= {self.expr} <= {self.ub}"

    def __repr__(self):
        if self.lb == self.ub:
            return f"{self.expr} == {self.lb}"
        else:
            return f"{self.lb} <= {self.expr} <= {self.ub}"

    def codegen(self, name):

        subst0, exprEval = cseCustom([self.expr])
        cEval = toCode(exprEval[0])

        # Generate second-order derivatives only for lower triangular part
        cEvalDiff2 = []
        adjDiff_indices = []  # indPP
        allDiffs2 = []

        for i, var1 in enumerate(self.adj):
            for j, var2 in enumerate(self.adj):
                info1, info2 = varInfo[var1], varInfo[var2]
                if (
                    isinstance(info1, ParameterStruct)
                    and isinstance(info2, ParameterStruct)
                    and i >= j
                ):
                    der = diff(self.expr, var1)
                    if var2 in der.free_symbols:
                        der = diff(der, var2)
                        if der != 0:
                            allDiffs2.append(der)
                            adjDiff_indices.append((info1.id, info2.id))  # indPP

        subst2, substExpr2 = cseCustom(allDiffs2)

        for expr in substExpr2:
            cEvalDiff2.append(toCode(expr))

        adj_indices = []  # indP
        cEvalDiff = []
        allDiffs = []

        for i, var in enumerate(self.adj):
            info = varInfo[var]
            if isinstance(info, ParameterStruct):
                der = diff(self.expr, var)
                if der != 0:
                    adj_indices.append(info.id)
                    allDiffs.append(der)

        subst1, substExpr = cseCustom(allDiffs)

        for expr in substExpr:
            cEvalDiff.append(toCode(expr))

        adj = f"{{{', '.join(map(str, adj_indices))}}}"
        adjDiff = "{{{}}}".format(
            ", ".join(
                "{{{}}}".format(", ".join(map(str, tpl))) for tpl in adjDiff_indices
            )
        )

        lb = self.lb if self.lb != -float("inf") else "MINUS_INFINITY"
        ub = self.ub if self.ub != float("inf") else "PLUS_INFINITY"

        out = f"class {name} : public ParamConstraint {{\n"
        out += "public:\n"
        out += f"\tstatic std::unique_ptr<{name}> create() {{\n"
        out += f"\t\tParamAdjacency adj{{{adj}}};\n"
        out += f"\t\tParamAdjacencyDiff adjDiff{{{adjDiff}}};\n"
        out += f"\t\treturn std::unique_ptr<{name}>(new {name}(std::move(adj), std::move(adjDiff), {toCode(lb)}, {toCode(ub)}));\n"
        out += "\t}\n\n"

        out += "\tdouble eval(const double* p) override {\n"

        for s in subst0:
            out += "        const double " + toCode(s[0]) + " = " + toCode(s[1]) + ";\n"

        out += f"\t\treturn {cEval};\n"
        out += "\t}\n\n"

        out += f"\tstd::vector<double> evalDiff(const double* p) override {{\n"

        for s in subst1:
            out += "        const double " + toCode(s[0]) + " = " + toCode(s[1]) + ";\n"

        out += "\t\treturn std::vector<double>"
        out += f"{{{', '.join(cEvalDiff)}}}"
        out += ";\n\t}\n\n"

        out += f"\tstd::vector<double> evalDiff2(const double* p) override {{\n"

        for s in subst2:
            out += "        const double " + toCode(s[0]) + " = " + toCode(s[1]) + ";\n"

        out += "\t\treturn std::vector<double>"
        out += "{{{}}}".format(", ".join(cEvalDiff2))
        out += ";\n\t}\n"

        out += "private:\n"
        out += f"\t{name}(ParamAdjacency adj, ParamAdjacencyDiff adjDiff, double lb, double ub) : ParamConstraint(std::move(adj), std::move(adjDiff), lb, ub) {{}}\n"
        out += "};\n\n\n"
        return out


def toCode(expr):
    code = ccode(expr)

    # for piecewise functions, since ccode(True) = True not true
    # pi := acos(-1) in symengine, we undo this since a defined M_PI exists
    code = (
        code.replace("True", "true")
        .replace("False", "false")
        .replace("acos(-1)", "M_PI")
    )
    return code


def cseCustom(expressions):
    if type(expressions) is not list:
        expressions = [expressions]

    replacements, reducedExprs = cse(expressions)
    return replacements, reducedExprs
