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


# TODO: refactor this, set id_counter to length of list<var> in model, e.g. id=len(self.xVars), ...


class VariableStruct:
    id_counter = 0

    def __init__(self, symbol, lb=-float("inf"), ub=float("inf"), nominal=None):
        self.symbol = symbol
        self.lb = lb
        self.ub = ub
        self.id = VariableStruct.id_counter
        self.nominal = nominal
        VariableStruct.id_counter += 1


class StateStruct(VariableStruct):
    id_counter = 0

    def __init__(
        self, start, symbol=None, lb=-float("inf"), ub=float("inf"), nominal=None
    ):
        super().__init__(symbol, lb, ub, nominal=nominal)
        self.start = start
        self.symbol = symbol if symbol is not None else f"x[{StateStruct.id_counter}]"
        self.id = StateStruct.id_counter
        StateStruct.id_counter += 1


class InputStruct(VariableStruct):
    id_counter = 0

    def __init__(
        self,
        symbol=None,
        lb=-float("inf"),
        ub=float("inf"),
        initialGuess=0,
        nominal=None,
    ):
        super().__init__(symbol, lb, ub, nominal=nominal)
        self.initialGuess = initialGuess
        self.symbol = symbol if symbol is not None else f"u[{InputStruct.id_counter}]"
        self.id = InputStruct.id_counter
        InputStruct.id_counter += 1


class ParameterStruct(VariableStruct):
    id_counter = 0

    def __init__(
        self,
        symbol=None,
        lb=-float("inf"),
        ub=float("inf"),
        initialGuess=0,
        nominal=None,
    ):
        super().__init__(symbol, lb, ub, nominal=nominal)
        self.initialGuess = initialGuess
        self.symbol = (
            symbol if symbol is not None else f"p[{ParameterStruct.id_counter}]"
        )
        self.id = ParameterStruct.id_counter
        ParameterStruct.id_counter += 1


class RuntimeParameterStruct(VariableStruct):
    id_counter = 0

    def __init__(self, default, symbol=None, lb=-float("inf"), ub=float("inf")):
        super().__init__(symbol, lb, ub)
        self.symbol = (
            symbol if symbol is not None else f"rp_{RuntimeParameterStruct.id_counter}"
        )
        self.id = RuntimeParameterStruct.id_counter
        self.value = default
        RuntimeParameterStruct.id_counter += 1


class_order = {"State": 0, "Input": 1, "Parameter": 2}


def get_sort_key(symbol, obj_map):
    obj = obj_map[symbol]
    class_name = obj.__class__.__name__.replace("Struct", "")
    order = class_order.get(class_name, float("inf"))
    return order, obj.id


def sort_symbols(symbols, obj_map):
    return symbols.sort(key=lambda s: get_sort_key(s, obj_map))
