/**
 * GDOPT - General Dynamic Optimizer
 * Copyright (C) 2024  Linus Langenkamp
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

#include "gdop.h"

#include <algorithm>
#include <cassert>
#include <exception>

#include "gdop_impl.h"
#include "util.h"

bool GDOP::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag, IndexStyleEnum& index_style) {
    // #variables
    n = numberVars;

    // #constraints
    m = sz(problem->A) + sz(problem->R) + (sz(problem->F) + sz(problem->G)) * rk.steps * mesh.intervals;

    // #nnz in jacobian
    init_jac(nnz_jac_g);

    // #nnz in hessian + creation of sparsity maps and offsets
    init_h(nnz_h_lag);

    // index starts at 0
    index_style = TNLP::C_STYLE;
    return true;
}

void GDOP::init_jac(Index& nnz_jac_g) {
    // eval jacobian nnz
    nnz_jac_g = 0;

    // nnz jacobian: dynamics
    int nnzDynBlock = 0;
    int containedIndex = 0;  // target idx
    for (const auto& dyn : problem->F) {
        nnzDynBlock += sz(dyn->adj.indX) + sz(dyn->adj.indU) + sz(dyn->adj.indP);
        // check if the k-th component of x, that is always contained in this block equation,
        // is contained in grad f_k(.) as well => reduce block nnz by 1
        auto idx = std::find(dyn->adj.indX.begin(), dyn->adj.indX.end(), containedIndex);
        if (idx != dyn->adj.indX.end())
            nnzDynBlock -= 1;
        containedIndex++;
    }
    nnz_jac_g += mesh.intervals * rk.steps * nnzDynBlock;                  // nnz for f(v_{ij}) eval
    nnz_jac_g += mesh.intervals * rk.steps * rk.steps * (sz(problem->F));  // nnz for sum_k ~a_{jk} * x_{ik}
    nnz_jac_g += (mesh.intervals - 1) * rk.steps * sz(problem->F);         // nnz for sum_k ~a_{jk} * (-x_{i-1,m}), i>=1

    // nnz jac: path constraints
    int nnzPathBlock = 0;
    for (const auto& pathConstr : problem->G) {
        nnzPathBlock += sz(pathConstr->adj.indX) + sz(pathConstr->adj.indU) + sz(pathConstr->adj.indP);
    }
    nnz_jac_g += mesh.intervals * rk.steps * nnzPathBlock;  // RHS = nnz for g(v_{ij}) eval

    // nnz jac: final constraints
    for (const auto& finalConstr : problem->R) {
        nnz_jac_g += sz(finalConstr->adj.indX) + sz(finalConstr->adj.indU) + sz(finalConstr->adj.indP);
    }

    // nnz jac: algebraic parameter constraints
    for (const auto& paramConstr : problem->A) {
        nnz_jac_g += sz(paramConstr->adj.indP);
    }
}

void GDOP::init_h(Index& nnz_h_lag) {
    nnz_h_lag = 0;

    // dense local hessian adjacency structure, will be reduced to hashmap later
    std::vector<std::vector<int>> denseA(offXU), denseAt(offXU), denseB(offP), denseBt(offP), denseC(offP);

    // init local hessian adjacency
    for (int i = 0; i < offXU; i++) {
        denseA[i] = std::vector<int>(i + 1, 0);
        denseAt[i] = std::vector<int>(i + 1, 0);
    }
    for (int i = 0; i < offP; i++) {
        denseB[i] = std::vector<int>(offXU, 0);
        denseBt[i] = std::vector<int>(offXU, 0);
        denseC[i] = std::vector<int>(offP, 0);
    }

    // update lagrange, dynamics, path constraint hessian structure
    if (problem->L)
        updateDenseHessianLFG(*problem->L, denseA, denseAt, denseB, denseBt, denseC);

    for (const auto& f : problem->F) {
        updateDenseHessianLFG(*f, denseA, denseAt, denseB, denseBt, denseC);
    }

    for (const auto& g : problem->G) {
        updateDenseHessianLFG(*g, denseA, denseAt, denseB, denseBt, denseC);
    }

    // update mayer term, final constraint hessian structure
    if (problem->M)
        updateDenseHessianMR(*problem->M, denseAt, denseBt, denseC);

    for (const auto& r : problem->R) {
        updateDenseHessianMR(*r, denseAt, denseBt, denseC);
    }

    // update algebraic parameter constraint hessian structure
    for (const auto& a : problem->A) {
        updateDenseHessianA(*a, denseC);
    }

    // at this point all local hessian structures have been obtained
    nnz_h_lag = (nnzMatrix(denseA) + nnzMatrix(denseB)) * (mesh.intervals * rk.steps - 1) + nnzMatrix(denseAt) + nnzMatrix(denseBt) + nnzMatrix(denseC);

    createSparseHessian(denseA, denseAt, denseB, denseBt, denseC, nnz_h_lag);
}

void GDOP::createSparseHessian(std::vector<std::vector<int>>& denseA, std::vector<std::vector<int>>& denseAt, std::vector<std::vector<int>>& denseB,
                               std::vector<std::vector<int>>& denseBt, std::vector<std::vector<int>>& denseC, Index& nnz_h_lag) {
    /**
     idxCOO  :  equation index in COO format
     A       :  shift block 0,0 -> i,j: shift by lengthA * (i * rk.steps + j)
     At      :  is exact -> idxCOO = correct eq index
     B       :  is not exact, idxCOO = starting index for p-th parameter:  rowLengthBlockB[p] * (i * rk.steps + j)
     Bt      :  is exact -> idxCOO = correct eq index
     C       :  is exact -> idxCOO = correct eq index
    **/
    int idxCOO = 0;  // eq index
    for (int i = 0; i < offXU; i++) {
        for (int j = 0; j <= i; j++) {
            if (denseA[i][j] == 1) {
                hessianA.insert({{i, j}, idxCOO});  // idxCOO not exact; shift needed based on
                idxCOO++;
            }
        }
    }
    lengthA = idxCOO;
    idxCOO *= (mesh.intervals * rk.steps - 1);  // shifting At

    for (int i = 0; i < offXU; i++) {
        for (int j = 0; j <= i; j++) {
            if (denseAt[i][j] == 1) {
                hessianAt.insert({{i, j}, idxCOO});  // idxCOO exact; no shift needed later on
                idxCOO++;
            }
        }
    }

    for (int i = 0; i < offP; i++) {
        int nnzRowB = 0;
        for (int j = 0; j < offXU; j++) {
            if (denseB[i][j] == 1) {
                hessianB.insert({{i, j}, idxCOO});  // idxCOO not exact; shift needed based on row
                nnzRowB++;
                idxCOO++;
            }
        }
        rowLengthBlockB.push_back(nnzRowB);
        idxCOO += nnzRowB * (mesh.intervals * rk.steps - 2);  // shifting to Bt: nnzRowB * (mesh.intervals * rk.steps - 1) - nnzRowB

        for (int j = 0; j < offXU; j++) {
            if (denseBt[i][j] == 1) {
                hessianBt.insert({{i, j}, idxCOO});  // idxCOO exact; no shift needed later on
                idxCOO++;
            }
        }

        for (int j = 0; j < offP; j++) {
            if (denseC[i][j] == 1) {
                hessianC.insert({{i, j}, idxCOO});  // idxCOO exact; no shift needed later on
                idxCOO++;
            }
        }
    }
    assert(idxCOO == nnz_h_lag);
}

void GDOP::updateDenseHessianLFG(const Expression& expr, std::vector<std::vector<int>>& denseA, std::vector<std::vector<int>>& denseAt,
                                 std::vector<std::vector<int>>& denseB, std::vector<std::vector<int>>& denseBt, std::vector<std::vector<int>>& denseC) const {
    // update of hessian blocks for lagrange term, dynamic and path constraints
    for (auto const [x1, x2] : expr.adjDiff.indXX) {
        denseA[x1][x2] = 1;
        denseAt[x1][x2] = 1;
    }
    for (auto const [u, x] : expr.adjDiff.indUX) {
        denseA[offX + u][x] = 1;
        denseAt[offX + u][x] = 1;
    }
    for (auto const [u1, u2] : expr.adjDiff.indUU) {
        denseA[offX + u1][offX + u2] = 1;
        denseAt[offX + u1][offX + u2] = 1;
    }
    for (auto const [p, x] : expr.adjDiff.indPX) {
        denseB[p][x] = 1;
        denseBt[p][x] = 1;
    }
    for (auto const [p, u] : expr.adjDiff.indPU) {
        denseB[p][offX + u] = 1;
        denseBt[p][offX + u] = 1;
    }
    for (auto const [p1, p2] : expr.adjDiff.indPP) {
        denseC[p1][p2] = 1;
    }
}

void GDOP::updateDenseHessianMR(const Expression& expr, std::vector<std::vector<int>>& denseAt, std::vector<std::vector<int>>& denseBt,
                                std::vector<std::vector<int>>& denseC) const {
    // update of hessian blocks for mayer term and final constraints
    for (auto const [x1, x2] : expr.adjDiff.indXX) {
        denseAt[x1][x2] = 1;
    }
    for (auto const [u, x] : expr.adjDiff.indUX) {
        denseAt[offX + u][x] = 1;
    }
    for (auto const [u1, u2] : expr.adjDiff.indUU) {
        denseAt[offX + u1][offX + u2] = 1;
    }
    for (auto const [p, x] : expr.adjDiff.indPX) {
        denseBt[p][x] = 1;
    }
    for (auto const [p, u] : expr.adjDiff.indPU) {
        denseBt[p][offX + u] = 1;
    }
    for (auto const [p1, p2] : expr.adjDiff.indPP) {
        denseC[p1][p2] = 1;
    }
}

void GDOP::updateDenseHessianA(const ParamExpression& expr, std::vector<std::vector<int>>& denseC) const {
    // update of hessian Blocks for parametric constraints
    for (auto const [p1, p2] : expr.adjDiff.indPP) {
        denseC[p1][p2] = 1;
    }
}

bool GDOP::get_bounds_info(Index n, Number* x_l, Number* x_u, Index m, Number* g_l, Number* g_u) {
    assert(n == numberVars);
    assert(m == sz(problem->A) + sz(problem->R) + (sz(problem->F) + sz(problem->G)) * rk.steps * mesh.intervals);

    // var bounds
    for (int i = 0; i < mesh.intervals; i++) {
        for (int j = 0; j < rk.steps; j++) {
            const int xij = i * offXUBlock + j * offXU;         // index of 1st x var at collocation point (i,j)
            const int uij = i * offXUBlock + j * offXU + offX;  // index of 1st u var at collocation point (i,j)

            // state bounds
            for (int d = 0; d < problem->sizeX; d++) {
                x_l[xij + d] = problem->lbX[d];
                x_u[xij + d] = problem->ubX[d];
            }

            // control bounds
            for (int d = 0; d < problem->sizeU; d++) {
                x_l[uij + d] = problem->lbU[d];
                x_u[uij + d] = problem->ubU[d];
            }
        }
    }

    // parameter bounds
    for (int d = 0; d < problem->sizeP; d++) {
        x_l[offXUTotal + d] = problem->lbP[d];
        x_u[offXUTotal + d] = problem->ubP[d];
    }

    // equation bounds
    int eq = 0;
    for (int i = 0; i < mesh.intervals; i++) {
        for (int j = 0; j < rk.steps; j++) {
            // dynamic bound must equal 0
            for (int d = 0; d < sz(problem->F); d++) {
                g_l[eq] = 0.0;
                g_u[eq] = 0.0;
                eq++;
            }

            // path constraint bounds
            for (auto const& pathConstr : problem->G) {
                g_l[eq] = pathConstr->lb;
                g_u[eq] = pathConstr->ub;
                eq++;
            }
        }
    }

    // final constraint bounds
    for (auto const& finalConstr : problem->R) {
        g_l[eq] = finalConstr->lb;
        g_u[eq] = finalConstr->ub;
        eq++;
    }

    // parameter constraint bounds
    for (auto const& paramConstr : problem->A) {
        g_l[eq] = paramConstr->lb;
        g_u[eq] = paramConstr->ub;
        eq++;
    }
    assert(m == eq);
    return true;
}

bool GDOP::get_starting_point(Index n, bool init_x, Number* x, bool init_z, Number* z_L, Number* z_U, Index m, bool init_lambda, Number* lambda) {
    // TODO: implement different strategies:
    //  * genetic / bionic algorithm
    assert(n == numberVars);
    if (init_x) {
        std::vector<std::vector<double>> initialStates;
        double tijOld;

        switch (initVars) {
            case InitVars::CONST:
                // states will be constant globally
                for (int i = 0; i < mesh.intervals; i++) {
                    for (int j = 0; j < rk.steps; j++) {
                        const double tij = mesh.grid[i] + rk.c[j] * mesh.deltaT[i];
                        const int xij = i * offXUBlock + j * offXU;         // index of 1st x var at collocation point (i,j)
                        const int uij = i * offXUBlock + j * offXU + offX;  // index of 1st u var at collocation point (i,j)

                        for (int dimX = 0; dimX < problem->sizeX; dimX++) {
                            x[xij + dimX] = problem->x0[dimX];
                        }

                        std::vector<double> uGuess = problem->uInitialGuess(tij);
                        for (int dimU = 0; dimU < problem->sizeU; dimU++) {
                            x[uij + dimU] = uGuess[dimU];
                        }
                    }
                }
                for (int dimP = 0; dimP < problem->sizeP; dimP++) {
                    x[offXUTotal + dimP] = problem->pInitialGuess[dimP];
                }

                break;

            case InitVars::SOLVE:
                // currently reading in a solution from the frontend (solved by scipy with Radau5 or BDF)
                initialStates = readInitialValues(problem->initialStatesPath);

                for (int i = 0; i < mesh.intervals; i++) {
                    for (int j = 0; j < rk.steps; j++) {
                        const double tij = mesh.grid[i] + rk.c[j] * mesh.deltaT[i];
                        const int xij = i * offXUBlock + j * offXU;         // index of 1st x var at collocation point (i,j)
                        const int uij = i * offXUBlock + j * offXU + offX;  // index of 1st u var at collocation point (i,j)

                        for (int dimX = 0; dimX < problem->sizeX; dimX++) {
                            x[xij + dimX] = initialStates[rk.steps * i + j][dimX];  // JUMP1: add 1 to dimX if timeColumn is contained
                        }

                        std::vector<double> uGuess = problem->uInitialGuess(tij);
                        for (int dimU = 0; dimU < problem->sizeU; dimU++) {
                            x[uij + dimU] = uGuess[dimU];
                        }
                    }
                }

                for (int dimP = 0; dimP < problem->sizeP; dimP++) {
                    x[offXUTotal + dimP] = problem->pInitialGuess[dimP];
                }

                break;

            case InitVars::SOLVE_EXPLICIT_EULER:
                // explicit Euler for initial guess of states (on each sub-interval [t_ij, t_{i, j+1}])
                for (int dimP = 0; dimP < problem->sizeP; dimP++) {
                    x[offXUTotal + dimP] = problem->pInitialGuess[dimP];
                }

                tijOld = 0;

                for (int i = 0; i < mesh.intervals; i++) {
                    for (int j = 0; j < rk.steps; j++) {
                        const double tij = mesh.grid[i] + rk.c[j] * mesh.deltaT[i];
                        const double dt = tij - tijOld;
                        const int xij = i * offXUBlock + j * offXU;         // index of 1st x var at collocation point (i,j)
                        const int uij = i * offXUBlock + j * offXU + offX;  // index of 1st u var at collocation point (i,j)

                        if (i == 0 && j == 0) {
                            for (int dimX = 0; dimX < problem->sizeX; dimX++) {
                                x[xij + dimX] = problem->x0[dimX];
                            }
                        }
                        else {
                            for (int dimX = 0; dimX < problem->sizeX; dimX++) {
                                x[xij + dimX] = x[xij - offXU + dimX] + dt * problem->F[dimX]->eval(&x[xij - offXU], &x[uij - offXU], &x[offXUTotal], tijOld);
                            }
                        }

                        const std::vector<double> uGuess = problem->uInitialGuess(tij);
                        for (int dimU = 0; dimU < problem->sizeU; dimU++) {
                            x[uij + dimU] = uGuess[dimU];
                        }

                        tijOld = tij;
                    }
                }

                break;

            case InitVars::SOLVE_EXPLICIT:
                // classic Runge-Kutta scheme for initial guess of states (on each sub-interval [t_ij, t_{i, j+1}])
                for (int dimP = 0; dimP < problem->sizeP; dimP++) {
                    x[offXUTotal + dimP] = problem->pInitialGuess[dimP];
                }

                tijOld = 0;

                for (int i = 0; i < mesh.intervals; i++) {
                    for (int j = 0; j < rk.steps; j++) {
                        const double tij = mesh.grid[i] + rk.c[j] * mesh.deltaT[i];
                        const double dt = tij - tijOld;
                        const int xij = i * offXUBlock + j * offXU;         // index of 1st x var at collocation point (i,j)
                        const int uij = i * offXUBlock + j * offXU + offX;  // index of 1st u var at collocation point (i,j)

                        if (i == 0 && j == 0) {
                            for (int dimX = 0; dimX < problem->sizeX; dimX++) {
                                x[xij + dimX] = problem->x0[dimX];
                            }
                        }
                        else {
                            double k1X[problem->sizeX];
                            double k2X[problem->sizeX];
                            double k3X[problem->sizeX];
                            double k4X[problem->sizeX];

                            // k1
                            for (int dimX = 0; dimX < problem->sizeX; dimX++) {
                                k1X[dimX] = problem->F[dimX]->eval(&x[xij - offXU], &x[uij - offXU], &x[offXUTotal], tijOld);
                            }

                            const double t2 = tijOld + 0.5 * dt;
                            std::vector<double> uGuess2 = problem->uInitialGuess(t2);

                            std::vector<double> xTemp(&x[xij - offXU], &x[xij - offXU] + problem->sizeX);

                            // x_i + 0.5 * dt * k1
                            for (int dimX = 0; dimX < problem->sizeX; dimX++) {
                                xTemp[dimX] += 0.5 * dt * k1X[dimX];
                            }

                            // k2
                            for (int dimX = 0; dimX < problem->sizeX; dimX++) {
                                k2X[dimX] = problem->F[dimX]->eval(xTemp.data(), uGuess2.data(), &x[offXUTotal], t2);
                            }

                            std::vector<double> uGuess3 = problem->uInitialGuess(t2);

                            // x_i + 0.5 * dt * k2
                            for (int dimX = 0; dimX < problem->sizeX; dimX++) {
                                xTemp[dimX] = x[xij - offXU + dimX] + 0.5 * dt * k2X[dimX];
                            }

                            // k3
                            for (int dimX = 0; dimX < problem->sizeX; dimX++) {
                                k3X[dimX] = problem->F[dimX]->eval(xTemp.data(), uGuess3.data(), &x[offXUTotal], t2);
                            }

                            std::vector<double> uGuess4 = problem->uInitialGuess(tij);

                            // x_i + dt * k3
                            for (int dimX = 0; dimX < problem->sizeX; dimX++) {
                                xTemp[dimX] = x[xij - offXU + dimX] + dt * k3X[dimX];
                            }

                            // k4
                            for (int dimX = 0; dimX < problem->sizeX; dimX++) {
                                k4X[dimX] = problem->F[dimX]->eval(xTemp.data(), uGuess4.data(), &x[offXUTotal], tij);
                            }

                            // x_{i+1} = x_i + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
                            for (int dimX = 0; dimX < problem->sizeX; dimX++) {
                                x[xij + dimX] = x[xij - offXU + dimX] + dt / 6.0 * (k1X[dimX] + 2.0 * (k2X[dimX] + k3X[dimX]) + k4X[dimX]);
                            }
                        }

                        const std::vector<double> uGuess = problem->uInitialGuess(tij);
                        for (int dimU = 0; dimU < problem->sizeU; dimU++) {
                            x[uij + dimU] = uGuess[dimU];
                        }

                        tijOld = tij;
                    }
                }

                break;

            case InitVars::CALLBACK:
                // strict callback case, optimizer will set initVars to Callback, if a mesh refinement is executed
                for (int i = 0; i < n; i++) {
                    x[i] = xInitCallback[i];
                }

                break;
        }
    }
    return true;
}

bool GDOP::get_scaling_parameters(Number& obj_scaling, bool& use_x_scaling, Index n, Number* x_scaling, bool& use_g_scaling, Index m, Number* g_scaling) {
    use_x_scaling = true;
    use_g_scaling = true;

    // objective scaling
    obj_scaling = checkNominalValue(problem->nominalObjective);

    // variable scaling
    for (int i = 0; i < mesh.intervals; i++) {
        for (int j = 0; j < rk.steps; j++) {
            const int xij = i * offXUBlock + j * offXU;         // index of 1st x var at collocation point (i,j)
            const int uij = i * offXUBlock + j * offXU + offX;  // index of 1st u var at collocation point (i,j)
            for (int dimX = 0; dimX < problem->sizeX; dimX++) {
                x_scaling[xij + dimX] = checkNominalValue(problem->nominalsX[dimX]);
            }

            for (int dimU = 0; dimU < problem->sizeU; dimU++) {
                x_scaling[uij + dimU] = checkNominalValue(problem->nominalsU[dimU]);
            }
        }
    }
    for (int dimP = 0; dimP < problem->sizeP; dimP++) {
        x_scaling[offXUTotal + dimP] = checkNominalValue(problem->nominalsP[dimP]);
    }

    // constraint scaling
    int eq = 0;
    for (int i = 0; i < mesh.intervals; i++) {
        for (int j = 0; j < rk.steps; j++) {
            for (int d = 0; d < sz(problem->F); d++) {
                g_scaling[eq] = checkNominalValue(problem->nominalsF[d]);
                eq++;
            }

            for (int d = 0; d < sz(problem->G); d++) {
                g_scaling[eq] = checkNominalValue(problem->nominalsG[d]);
                eq++;
            }
        }
    }

    for (int d = 0; d < sz(problem->R); d++) {
        g_scaling[eq] = checkNominalValue(problem->nominalsR[d]);
        eq++;
    }

    for (int d = 0; d < sz(problem->A); d++) {
        g_scaling[eq] = checkNominalValue(problem->nominalsA[d]);
        eq++;
    }
    assert(m == eq);

    return true;
}

bool GDOP::eval_f(Index n, const Number* x, bool new_x, Number& obj_value) {
    double MAY = 0;
    double LAG = 0;

    // mayer term evaluation
    if (problem->M) {
        MAY = problem->M->eval(&x[offXUTotal - offXU], &x[offXUTotal - offU], &x[offXUTotal], mesh.tf);
    }

    // lagrange term evaluation
    if (problem->L) {
        for (int i = 0; i < mesh.intervals; i++) {
            for (int j = 0; j < rk.steps; j++) {
                const double tij = mesh.grid[i] + rk.c[j] * mesh.deltaT[i];
                const int xij = i * offXUBlock + j * offXU;         // index of 1st x var at collocation point (i,j)
                const int uij = i * offXUBlock + j * offXU + offX;  // index of 1st u var at collocation point (i,j)
                LAG += mesh.deltaT[i] * rk.b[j] * problem->L->eval(&x[xij], &x[uij], &x[offXUTotal], tij);
            }
        }
    }
    obj_value = MAY + LAG;
    return true;
}

bool GDOP::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f) {
    // initialize the dense gradient as 0 for all elements
    std::fill(grad_f, grad_f + numberVars, 0);

    // mayer term derivative
    if (problem->M) {
        const int xnm = offXUTotal - offXU;  // index of 1st x var at collocation point (n,m)
        const int unm = offXUTotal - offU;   // index of 1st u var at collocation point (n,m)
        const auto diffMAY = problem->M->evalDiff(&x[xnm], &x[unm], &x[offXUTotal], mesh.tf);

        for (int v = 0; v < sz(diffMAY[0]); v++) {
            grad_f[xnm + problem->M->adj.indX[v]] += diffMAY[0][v];
        }
        for (int v = 0; v < sz(diffMAY[1]); v++) {
            grad_f[unm + problem->M->adj.indU[v]] += diffMAY[1][v];
        }
        for (int v = 0; v < sz(diffMAY[2]); v++) {
            grad_f[offXUTotal + problem->M->adj.indP[v]] += diffMAY[2][v];
        }
    }

    // lagrange term derivative
    if (problem->L) {
        for (int i = 0; i < mesh.intervals; i++) {
            for (int j = 0; j < rk.steps; j++) {
                const int xij = i * offXUBlock + j * offXU;         // index of 1st x var at collocation point (i,j)
                const int uij = i * offXUBlock + j * offXU + offX;  // index of 1st u var at collocation point (i,j)
                const double tij = mesh.grid[i] + rk.c[j] * mesh.deltaT[i];
                const double quadCoeff = mesh.deltaT[i] * rk.b[j];
                const auto diffLAG = problem->L->evalDiff(&x[xij], &x[uij], &x[offXUTotal], tij);

                for (int v = 0; v < sz(diffLAG[0]); v++) {
                    grad_f[xij + problem->L->adj.indX[v]] += quadCoeff * diffLAG[0][v];
                }
                for (int v = 0; v < sz(diffLAG[1]); v++) {
                    grad_f[uij + problem->L->adj.indU[v]] += quadCoeff * diffLAG[1][v];
                }
                for (int v = 0; v < sz(diffLAG[2]); v++) {
                    grad_f[offXUTotal + problem->L->adj.indP[v]] += quadCoeff * diffLAG[2][v];
                }
            }
        }
    }
    return true;
}

bool GDOP::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g) {
    int eq = 0;
    for (int i = 0; i < mesh.intervals; i++) {
        for (int j = 0; j < rk.steps; j++) {
            const double tij = mesh.grid[i] + rk.c[j] * mesh.deltaT[i];
            const int xij = i * offXUBlock + j * offXU;         // first index (dim=0) of x_{ij}
            const int uij = i * offXUBlock + j * offXU + offX;  // first index (dim=0) of u_{ij}
            const int xi1_m = i * offXUBlock - offXU;           // first index (dim=0) of x_{i-1,m}; m=rk.steps

            // dynamic constraints
            // 0 = sum_k ~a_{jk} * (x_{ik} - x_{i-1,m)) - del t_i * f(x_{ij}, u_{ij}, p, t_{ij}), i=0 -> x_{i-1,m) = x0
            for (int d = 0; d < sz(problem->F); d++) {
                double rkLinearComb = 0;
                for (int k = 0; k < rk.steps; k++) {
                    const int xik = i * offXUBlock + k * offXU;  // first index (dim=0) of x_{ik}
                    if (i == 0) {
                        rkLinearComb += rk.derMat[j + 1][k + 1] * (x[xik + d] - problem->x0[d]);
                    }
                    else {
                        rkLinearComb += rk.derMat[j + 1][k + 1] * (x[xik + d] - x[xi1_m + d]);
                    }
                }
                g[eq] = rkLinearComb - mesh.deltaT[i] * problem->F[d]->eval(&x[xij], &x[uij], &x[offXUTotal], tij);
                eq++;
            }

            // path constraints: LB_g <= g(x_{ij}, u_{ij}, p, t_{ij}) <= UB_g
            for (auto const& pathConstr : problem->G) {
                g[eq] = pathConstr->eval(&x[xij], &x[uij], &x[offXUTotal], tij);
                eq++;
            }
        }
    }

    // final constraints: LB_r <= r(x_{nm}, u_{nm}, p, t_{nm}) <= UB_r
    const int xnm = offXUTotal - offXU;  // first index (dim=0) of x_{n,m}
    const int unm = offXUTotal - offU;   // first index (dim=0) of u_{n,m}
    for (auto const& finalConstr : problem->R) {
        g[eq] = finalConstr->eval(&x[xnm], &x[unm], &x[offXUTotal], mesh.tf);
        eq++;
    }

    // parameter constraints: LB_a <= a(p) <= UB_a
    for (auto const& paramConstr : problem->A) {
        g[eq] = paramConstr->eval(&x[offXUTotal]);
        eq++;
    }
    assert(m == eq);
    return true;
}

int GDOP::init_jac_sparsity(Index* iRow, Index* jCol) {
    // Just iterates over all blocks atm, no use of block struct
    int idxCOO = 0;
    int eq = 0;
    for (int i = 0; i < mesh.intervals; i++) {
        for (int j = 0; j < rk.steps; j++) {
            const int xij = i * offXUBlock + j * offXU;         // first index (dim=0) of x_{ij}
            const int uij = i * offXUBlock + j * offXU + offX;  // first index (dim=0) of u_{ij}
            const int xi1_m = i * offXUBlock - offXU;           // first index (dim=0) of x_{i-1,m}; m=rk.steps

            for (int d = 0; d < sz(problem->F); d++) {
                // sum_k ~a_{jk} * (-x_{i-1,m}), i>=1
                if (i > 0) {
                    iRow[idxCOO] = eq;
                    jCol[idxCOO] = xi1_m + d;
                    idxCOO++;
                }

                // sum_k ~a_{jk} * (x_{i,k})
                for (int k = 0; k < rk.steps; k++) {
                    const int xik = i * offXUBlock + k * offXU;
                    iRow[idxCOO] = eq;
                    jCol[idxCOO] = xik + d;
                    idxCOO++;
                }

                // f(v_{ij})_x: care handle duplicates if d is contained in indX!!
                for (int v : problem->F[d]->adj.indX) {
                    if (v != d) {
                        iRow[idxCOO] = eq;
                        jCol[idxCOO] = xij + v;
                        idxCOO++;
                    }
                }

                // f(v_{ij})_u
                for (int v : problem->F[d]->adj.indU) {
                    iRow[idxCOO] = eq;
                    jCol[idxCOO] = uij + v;
                    idxCOO++;
                }

                // f(v_{ij})_p
                for (int v : problem->F[d]->adj.indP) {
                    iRow[idxCOO] = eq;
                    jCol[idxCOO] = offXUTotal + v;
                    idxCOO++;
                }
                eq++;
            }

            for (const auto& constrG : problem->G) {
                // g(v_{ij})_x
                for (int v : constrG->adj.indX) {
                    iRow[idxCOO] = eq;
                    jCol[idxCOO] = xij + v;
                    idxCOO++;
                }

                // g(v_{ij})_u
                for (int v : constrG->adj.indU) {
                    iRow[idxCOO] = eq;
                    jCol[idxCOO] = uij + v;
                    idxCOO++;
                }

                // g(v_{ij})_p
                for (int v : constrG->adj.indP) {
                    iRow[idxCOO] = eq;
                    jCol[idxCOO] = offXUTotal + v;
                    idxCOO++;
                }
                eq++;
            }
        }
    }

    const int xnm = offXUTotal - offXU;
    const int unm = offXUTotal - offU;

    for (const auto& constrR : problem->R) {
        // r(v_{nm})_x
        for (int v : constrR->adj.indX) {
            iRow[idxCOO] = eq;
            jCol[idxCOO] = xnm + v;
            idxCOO++;
        }

        // r(v_{nm})_u
        for (int v : constrR->adj.indU) {
            iRow[idxCOO] = eq;
            jCol[idxCOO] = unm + v;
            idxCOO++;
        }

        // r(v_{nm})_p
        for (int v : constrR->adj.indP) {
            iRow[idxCOO] = eq;
            jCol[idxCOO] = offXUTotal + v;
            idxCOO++;
        }
        eq++;
    }

    for (const auto& constrA : problem->A) {
        // a_p
        for (int v : constrA->adj.indP) {
            iRow[idxCOO] = eq;
            jCol[idxCOO] = offXUTotal + v;
            idxCOO++;
        }
        eq++;
    }
    return eq;
}

void GDOP::get_jac_values(const Number* x, Number* values) {
    int containedIndex;
    int idxCOO = 0;
    for (int i = 0; i < mesh.intervals; i++) {
        for (int j = 0; j < rk.steps; j++) {
            const int xij = i * offXUBlock + j * offXU;         // first index (dim=0) of x_{ij}
            const int uij = i * offXUBlock + j * offXU + offX;  // first index (dim=0) of u_{ij}
            const double tij = mesh.grid[i] + rk.c[j] * mesh.deltaT[i];

            for (int d = 0; d < sz(problem->F); d++) {
                // sum_k ~a_{jk} * (-x_{i-1,m}), i>=1
                if (i > 0) {
                    values[idxCOO] = rk.derMat[j + 1][0];
                    idxCOO++;
                }

                // sum_k ~a_{jk} * (x_{i,k})
                for (int k = 0; k < rk.steps; k++) {
                    values[idxCOO] = rk.derMat[j + 1][k + 1];
                    if (k == j) {
                        containedIndex = idxCOO;
                    }
                    idxCOO++;
                }

                // eval grad f(v_{ij})
                auto const diffF = problem->F[d]->evalDiff(&x[xij], &x[uij], &x[offXUTotal], tij);

                // f(v_{ij})_x: care handle duplicates if d is contained in indX!!
                for (int v = 0; v < sz(problem->F[d]->adj.indX); v++) {
                    auto idx = problem->F[d]->adj.indX[v];
                    if (idx != d) {
                        values[idxCOO] = -mesh.deltaT[i] * diffF[0][v];
                        idxCOO++;
                    }
                    else {
                        values[containedIndex] -= mesh.deltaT[i] * diffF[0][v];
                    }
                }

                // f(v_{ij})_u
                for (int v = 0; v < sz(problem->F[d]->adj.indU); v++) {
                    values[idxCOO] = -mesh.deltaT[i] * diffF[1][v];
                    idxCOO++;
                }

                // f(v_{ij})_p
                for (int v = 0; v < sz(problem->F[d]->adj.indP); v++) {
                    values[idxCOO] = -mesh.deltaT[i] * diffF[2][v];
                    idxCOO++;
                }
            }

            for (const auto& constrG : problem->G) {
                // eval grad g(v_{ij})
                auto const diffG = constrG->evalDiff(&x[xij], &x[uij], &x[offXUTotal], tij);

                // g(v_{ij})_x
                for (int v = 0; v < sz(constrG->adj.indX); v++) {
                    values[idxCOO] = diffG[0][v];
                    idxCOO++;
                }

                // g(v_{ij})_u
                for (int v = 0; v < sz(constrG->adj.indU); v++) {
                    values[idxCOO] = diffG[1][v];
                    idxCOO++;
                }

                // g(v_{ij})_p
                for (int v = 0; v < sz(constrG->adj.indP); v++) {
                    values[idxCOO] = diffG[2][v];
                    idxCOO++;
                }
            }
        }
    }

    const int xnm = offXUTotal - offXU;
    const int unm = offXUTotal - offU;

    for (const auto& constrR : problem->R) {
        // eval grad r(v_{nm})
        auto const diffR = constrR->evalDiff(&x[xnm], &x[unm], &x[offXUTotal], mesh.tf);

        // r(v_{nm})_x
        for (int v = 0; v < sz(constrR->adj.indX); v++) {
            values[idxCOO] = diffR[0][v];
            idxCOO++;
        }

        // r(v_{nm})_u
        for (int v = 0; v < sz(constrR->adj.indU); v++) {
            values[idxCOO] = diffR[1][v];
            idxCOO++;
        }

        // r(v_{nm})_p
        for (int v = 0; v < sz(constrR->adj.indP); v++) {
            values[idxCOO] = diffR[2][v];
            idxCOO++;
        }
    }

    for (const auto& constrA : problem->A) {
        // eval grad a(p)
        auto const diffA = constrA->evalDiff(&x[offXUTotal]);

        // a_p
        for (int v = 0; v < sz(constrA->adj.indP); v++) {
            values[idxCOO] = diffA[v];
            idxCOO++;
        }
    }
}

bool GDOP::eval_jac_g(Index n, const Number* x, bool new_x, Index m, Index nele_jac, Index* iRow, Index* jCol, Number* values) {
    assert(n == numberVars);
    assert(m == sz(problem->A) + sz(problem->R) + (sz(problem->F) + sz(problem->G)) * rk.steps * mesh.intervals);
    if (values == nullptr) {
        int eq = init_jac_sparsity(iRow, jCol);
        if (exportJacobian) {
            exportSparsity(iRow, jCol, nele_jac, {m, n}, exportJacobianPath + "/" + problem->name + "_jacobian.csv");
        }
        assert(eq == m);
    }
    else {
        std::fill(values, values + nele_jac, 0);
        get_jac_values(x, values);
    }
    return true;
}

void GDOP::init_h_sparsity(Index* iRow, Index* jCol) {
    // A block forall i, j except very last interval (n, m)
    for (const auto& [vars, idxCOO] : hessianA) {
        auto const [v1, v2] = vars;
        for (int i = 0; i < mesh.intervals - 1; i++) {
            for (int j = 0; j < rk.steps; j++) {
                const int idxij = idxCOO + lengthA * (i * rk.steps + j);
                const int v1ij = i * offXUBlock + j * offXU + v1;
                const int v2ij = i * offXUBlock + j * offXU + v2;
                iRow[idxij] = v1ij;
                jCol[idxij] = v2ij;
            }
        }
        for (int j = 0; j < rk.steps - 1; j++) {
            const int idxij = idxCOO + lengthA * ((mesh.intervals - 1) * rk.steps + j);
            const int v1ij = (mesh.intervals - 1) * offXUBlock + j * offXU + v1;
            const int v2ij = (mesh.intervals - 1) * offXUBlock + j * offXU + v2;
            iRow[idxij] = v1ij;
            jCol[idxij] = v2ij;
        }
    }

    // At for very last interval, note that idxCOO is the correct array index by construction
    for (const auto& [vars, idxCOO] : hessianAt) {
        auto const [v1, v2] = vars;
        for (int j = 0; j < rk.steps; j++) {
            const int v1ij = (mesh.intervals - 1) * offXUBlock + j * offXU + v1;
            const int v2ij = (mesh.intervals - 1) * offXUBlock + j * offXU + v2;
            iRow[idxCOO] = v1ij;
            jCol[idxCOO] = v2ij;
        }
    }

    // B block forall i, j except very last interval (n, m)
    for (const auto& [vars, idxCOO] : hessianB) {
        auto const [p, v] = vars;
        for (int i = 0; i < mesh.intervals - 1; i++) {
            for (int j = 0; j < rk.steps; j++) {
                const int idxij = idxCOO + rowLengthBlockB[p] * (i * rk.steps + j);
                const int pshifted = offXUTotal + p;
                const int vij = i * offXUBlock + j * offXU + v;
                iRow[idxij] = pshifted;
                jCol[idxij] = vij;
            }
        }
        for (int j = 0; j < rk.steps - 1; j++) {
            const int idxij = idxCOO + rowLengthBlockB[p] * ((mesh.intervals - 1) * rk.steps + j);
            const int pshifted = offXUTotal + p;
            const int vij = (mesh.intervals - 1) * offXUBlock + j * offXU + v;
            iRow[idxij] = pshifted;
            jCol[idxij] = vij;
        }
    }

    // Bt for very last interval, note that idxCOO is the correct array index by construction
    for (const auto& [vars, idxCOO] : hessianBt) {
        auto const [p, v] = vars;
        for (int j = 0; j < rk.steps; j++) {
            const int pshifted = offXUTotal + p;
            const int vij = (mesh.intervals - 1) * offXUBlock + j * offXU + v;
            iRow[idxCOO] = pshifted;
            jCol[idxCOO] = vij;
        }
    }

    // C
    for (const auto& [vars, idxCOO] : hessianC) {
        auto const [p1, p2] = vars;
        const int p1shifted = offXUTotal + p1;
        const int p2shifted = offXUTotal + p2;
        iRow[idxCOO] = p1shifted;
        jCol[idxCOO] = p2shifted;
    }
}

void GDOP::evalHessianA_B(Number* values, const Number* x, Expression& expr, const double factor, const int xij, const int uij, const double tij, const int i,
                          const int j) {
    auto const diff2Expr = expr.evalDiff2(&x[xij], &x[uij], &x[offXUTotal], tij);

    // shifting needed for A and B COO indices!
    for (int k = 0; k < sz(expr.adjDiff.indXX); k++) {
        auto const vars = expr.adjDiff.indXX[k];
        auto const idxCOO = lengthA * (i * rk.steps + j) + hessianA[vars];
        values[idxCOO] += factor * diff2Expr[0][k];
    }
    for (int k = 0; k < sz(expr.adjDiff.indUX); k++) {
        auto const [uvar, xvar] = expr.adjDiff.indUX[k];
        auto const idxCOO = lengthA * (i * rk.steps + j) + hessianA[{offX + uvar, xvar}];
        values[idxCOO] += factor * diff2Expr[1][k];
    }
    for (int k = 0; k < sz(expr.adjDiff.indUU); k++) {
        auto const [uvar1, uvar2] = expr.adjDiff.indUU[k];
        auto const idxCOO = lengthA * (i * rk.steps + j) + hessianA[{offX + uvar1, offX + uvar2}];
        values[idxCOO] += factor * diff2Expr[2][k];
    }
    for (int k = 0; k < sz(expr.adjDiff.indPX); k++) {
        auto const [pvar, xvar] = expr.adjDiff.indPX[k];
        auto const idxCOO = rowLengthBlockB[pvar] * (i * rk.steps + j) + hessianB[{pvar, xvar}];
        values[idxCOO] += factor * diff2Expr[3][k];
    }
    for (int k = 0; k < sz(expr.adjDiff.indPU); k++) {
        auto const [pvar, uvar] = expr.adjDiff.indPU[k];
        auto const idxCOO = rowLengthBlockB[pvar] * (i * rk.steps + j) + hessianB[{pvar, uvar + offX}];
        values[idxCOO] += factor * diff2Expr[4][k];
    }
    for (int k = 0; k < sz(expr.adjDiff.indPP); k++) {
        auto const vars = expr.adjDiff.indPP[k];
        auto const idxCOO = hessianC[vars];
        values[idxCOO] += factor * diff2Expr[5][k];
    }
}

void GDOP::evalHessianAt_Bt(Number* values, const Number* x, Expression& expr, const double factor, const int xij, const int uij, const double tij) {
    auto const diff2Expr = expr.evalDiff2(&x[xij], &x[uij], &x[offXUTotal], tij);

    // no shifting needed At, Bt and C have exact COO indices!
    for (int k = 0; k < sz(expr.adjDiff.indXX); k++) {
        auto const vars = expr.adjDiff.indXX[k];
        values[hessianAt[vars]] += factor * diff2Expr[0][k];
    }
    for (int k = 0; k < sz(expr.adjDiff.indUX); k++) {
        auto const [uvar, xvar] = expr.adjDiff.indUX[k];
        values[hessianAt[{offX + uvar, xvar}]] += factor * diff2Expr[1][k];
    }
    for (int k = 0; k < sz(expr.adjDiff.indUU); k++) {
        auto const [uvar1, uvar2] = expr.adjDiff.indUU[k];
        values[hessianAt[{offX + uvar1, offX + uvar2}]] += factor * diff2Expr[2][k];
    }
    for (int k = 0; k < sz(expr.adjDiff.indPX); k++) {
        auto const vars = expr.adjDiff.indPX[k];
        values[hessianBt[vars]] += factor * diff2Expr[3][k];
    }
    for (int k = 0; k < sz(expr.adjDiff.indPU); k++) {
        auto const [pvar, uvar] = expr.adjDiff.indPU[k];
        values[hessianBt[{pvar, uvar + offX}]] += factor * diff2Expr[4][k];
    }
    for (int k = 0; k < sz(expr.adjDiff.indPP); k++) {
        auto const vars = expr.adjDiff.indPP[k];
        values[hessianC[vars]] += factor * diff2Expr[5][k];
    }
}

void GDOP::evalHessianC(Number* values, const Number* x, ParamExpression& expr, double factor) {
    auto const diff2Expr = expr.evalDiff2(&x[offXUTotal]);
    for (int k = 0; k < sz(expr.adjDiff.indPP); k++) {
        auto const vars = expr.adjDiff.indPP[k];
        values[hessianC[vars]] += factor * diff2Expr[k];
    }
}

int GDOP::get_h_values(const Number* x, Number* values, Number obj_factor, const Number* lambda) {
    int eq = 0;
    for (int i = 0; i < mesh.intervals - 1; i++) {
        for (int j = 0; j < rk.steps; j++) {
            const double tij = mesh.grid[i] + rk.c[j] * mesh.deltaT[i];
            const int xij = i * offXUBlock + j * offXU;         // index of 1st x var at collocation point (i,j)
            const int uij = i * offXUBlock + j * offXU + offX;  // index of 1st u var at collocation point (i,j)

            // eval hessian lagrange
            if (problem->L) {
                const double lFactor = obj_factor * mesh.deltaT[i] * rk.b[j];
                evalHessianA_B(values, x, *problem->L, lFactor, xij, uij, tij, i, j);
            }

            // eval hessian dynamics
            for (auto const& f : problem->F) {
                const double fFactor = -lambda[eq] * mesh.deltaT[i];
                evalHessianA_B(values, x, *f, fFactor, xij, uij, tij, i, j);
                eq++;
            }

            // eval hessian path constraints
            for (auto const& g : problem->G) {
                const double gFactor = lambda[eq];
                evalHessianA_B(values, x, *g, gFactor, xij, uij, tij, i, j);
                eq++;
            }
        }
    }

    // same for the last rk.steps-1 blocks excluding the very last block -> At handling
    for (int j = 0; j < rk.steps - 1; j++) {
        const double tij = mesh.grid[mesh.intervals - 1] + rk.c[j] * mesh.deltaT[mesh.intervals - 1];
        const int xij = (mesh.intervals - 1) * offXUBlock + j * offXU;         // index of 1st x var at collocation point (i,j)
        const int uij = (mesh.intervals - 1) * offXUBlock + j * offXU + offX;  // index of 1st u var at collocation point (i,j)

        // eval hessian lagrange
        if (problem->L) {
            const double lagrFactor = obj_factor * mesh.deltaT[(mesh.intervals - 1)] * rk.b[j];
            evalHessianA_B(values, x, *problem->L, lagrFactor, xij, uij, tij, (mesh.intervals - 1), j);
        }

        // eval hessian dynamics
        for (auto const& f : problem->F) {
            const double fFactor = lambda[eq] * (-mesh.deltaT[(mesh.intervals - 1)]);
            evalHessianA_B(values, x, *f, fFactor, xij, uij, tij, mesh.intervals - 1, j);
            eq++;
        }

        // eval hessian path constraints
        for (auto const& g : problem->G) {
            const double gFactor = lambda[eq];
            evalHessianA_B(values, x, *g, gFactor, xij, uij, tij, mesh.intervals - 1, j);
            eq++;
        }
    }

    // At, Bt: contains M, L, F, G, R
    const int xnm = (mesh.intervals - 1) * offXUBlock + (rk.steps - 1) * offXU;         // index of 1st x var at collocation point (n,m)
    const int unm = (mesh.intervals - 1) * offXUBlock + (rk.steps - 1) * offXU + offX;  // index of 1st u var at collocation point (n,m)

    // eval hessian mayer term
    if (problem->M) {
        const double mayFactor = obj_factor;
        evalHessianAt_Bt(values, x, *problem->M, mayFactor, xnm, unm, mesh.tf);
    }

    // eval hessian lagrange term
    if (problem->L) {
        const double lagrFactor = obj_factor * mesh.deltaT[(mesh.intervals - 1)] * rk.b[rk.steps - 1];
        evalHessianAt_Bt(values, x, *problem->L, lagrFactor, xnm, unm, mesh.tf);
    }

    // eval hessian dynamics
    for (auto const& f : problem->F) {
        const double fFactor = lambda[eq] * (-mesh.deltaT[(mesh.intervals - 1)]);
        evalHessianAt_Bt(values, x, *f, fFactor, xnm, unm, mesh.tf);
        eq++;
    }

    // eval hessian path constraints
    for (auto const& g : problem->G) {
        const double gFactor = lambda[eq];
        evalHessianAt_Bt(values, x, *g, gFactor, xnm, unm, mesh.tf);
        eq++;
    }

    // eval hessian final constraints
    for (auto const& r : problem->R) {
        const double rFactor = lambda[eq];
        evalHessianAt_Bt(values, x, *r, rFactor, xnm, unm, mesh.tf);
        eq++;
    }

    // eval hessian algebraic parametric constraints
    for (auto const& a : problem->A) {
        const double aFactor = lambda[eq];
        evalHessianC(values, x, *a, aFactor);
        eq++;
    }
    return eq;
}

bool GDOP::eval_h(Index n, const Number* x, bool new_x, Number obj_factor, Index m, const Number* lambda, bool new_lambda, Index nele_hess, Index* iRow,
                  Index* jCol, Number* values) {
    if (values == nullptr) {
        init_h_sparsity(iRow, jCol);
        if (exportHessian) {
            exportSparsity(iRow, jCol, nele_hess, {n, n}, exportHessianPath + "/" + problem->name + "_hessian.csv");
        }
    }
    else {
        std::fill(values, values + nele_hess, 0);
        int eq = get_h_values(x, values, obj_factor, lambda);
        assert(eq == m);  // same as before eq index over lambda must be equal to #constrs
    }
    return true;
}

void GDOP::finalize_solution(SolverReturn status, Index n, const Number* x, const Number* z_L, const Number* z_U, Index m, const Number* g,
                             const Number* lambda, Number obj_value, const IpoptData* ip_data, IpoptCalculatedQuantities* ip_cq) {
    optimum.assign(x, x + n);
}

void GDOP::exportOptimum(const std::string& filename) const {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    std::string vars;
    for (int vx = 0; vx < problem->sizeX; vx++) {
        vars += ",x[" + std::to_string(vx) + "]";
    }
    for (int vu = 0; vu < problem->sizeU; vu++) {
        vars += ",u[" + std::to_string(vu) + "]";
    }
    for (int vp = 0; vp < problem->sizeP; vp++) {
        vars += ",p[" + std::to_string(vp) + "]";
    }
    auto header = "time" + vars;
    outFile << header << "\n";

    // time = 0 -> interpolate the control backwards
    std::string values = "0";
    for (int vx = 0; vx < problem->sizeX; vx++) {
        values += "," + double2Str(problem->x0[vx], 16);
    }
    for (int vu = 0; vu < problem->sizeU; vu++) {
        std::vector<double> uValues = {};
        uValues.reserve(rk.steps);
        for (int j = 0; j < rk.steps; j++) {
            uValues.push_back(optimum[vu + offX + offXU * j]);
        }
        values += "," + double2Str(Integrator::evalLagrange(rk.c, uValues, 0.0), 16);
    }
    for (int vp = 0; vp < problem->sizeP; vp++) {
        values += "," + double2Str(optimum[vp + offXUTotal], 16);
    }
    outFile << values << "\n";

    for (int i = 0; i < mesh.intervals; i++) {
        for (int j = 0; j < rk.steps; j++) {
            values = double2Str(mesh.grid[i] + rk.c[j] * mesh.deltaT[i], 16);
            for (int vx = 0; vx < problem->sizeX; vx++) {
                values += "," + double2Str(optimum[vx + offXU * j + offXUBlock * i], 16);
            }
            for (int vu = 0; vu < problem->sizeU; vu++) {
                values += "," + double2Str(optimum[vu + offX + offXU * j + offXUBlock * i], 16);
            }
            for (int vp = 0; vp < problem->sizeP; vp++) {
                values += "," + double2Str(optimum[vp + offXUTotal], 16);
            }
            outFile << values << "\n";
        }
    }
}

GDOP::GDOP(const std::shared_ptr<const Problem>& problem, Mesh& mesh, Integrator& rk, InitVars initVars)
    : problem(problem), mesh(mesh), rk(rk), initVars(initVars) {
}

GDOP* createGDOP(const std::shared_ptr<const Problem>& problem, Mesh& mesh, Integrator& rk, InitVars initVars) {
    return new GDOP(problem, mesh, rk, initVars);
}
