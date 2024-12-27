/**
 * GDOPT - General Dynamic Optimizer
 * Copyright (C) 2024 Linus Langenkamp
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 **/

#ifndef GDOPT_INTEGRATOR_H
#define GDOPT_INTEGRATOR_H

#include <vector>

#include "util.h"

struct Integrator {

    // nodes, i.e. [c1, c2, ..., cm]
    const std::vector<gVector> c =
#include "constantsC.data"

    // nodes including -1, i.e. [-1, c1, c2, ..., cm]
    const std::vector<gVector> c0 =
#include "constantsC0.data"

    // quadrature weights {}, 2},
    const std::vector<gVector> b =
#include "constantsB.data"

    // differentiation matrices
    const std::vector<std::vector<gVector>> D =
#include "constantsD.data"

    // integral -1 to 1 of some values, scheme is chosen automatically
    gNumber integrate(const gVector& values);
};

#endif  // GDOPT_INTEGRATOR_H
