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

#include "integrator.h"

#include "util.h"

Integrator::Integrator() {
}

gNumber Integrator::integrate(const gVector& values) {
    /*
    input: values - f(c_1), ..., f(c_m)
    output: int_{-1}^{1} f(t) dt \approx sum_{k=1}^{m} b_k * f(c_k), m steps, b_k weights, c_k nodes
    */

    gNumber integral = 0;
    for (int k = 0; k < values.size(); k++) {
        integral += b[values.size()][k] * values[k];
    }

    return integral;
};