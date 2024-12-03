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

#include "solver.h"

#include <chrono>
#include <numeric>

#include "IpIpoptApplication.hpp"
#include "IpIpoptData.hpp"
#include "IpSolveStatistics.hpp"
#include "config.h"
#include "gdop_impl.h"

/* BIG TODOS: TYPE | IMPORTANCE | EFFORT from 1 to 5
 * TODO:

    main todos:
    1 add, construct, test more mesh refinement algorithms!              5, 4
    2 OpenModelica interface                                             5, 5
    3 test framework for huge examples / industry relevant               3, 2
    4 check long double to double cast in evals, refactor to typedef     2, 2
      e.g. GNumber, which is long (double) -> cast to double for ipopt

    delayed:
    5 tf as free variable                                                2, 4.5
    6 vectorized equation, derivatives with local substitutions          2, 4
    -> define vec(f,g), vec(r), vec(a(p)) : faster compilation, since
       less classes are generated or does it not matter?
    7 better initial guess: evolutionary algorithms, high deg poly       2, 3
    -> use a bdf method and interpolate the discrete state values
    8 detection for nominal, linear, quadratic, const hessian            1, 2
    9 constructing a p / hp-method?                                      5, 5
    10 add initial constraints -> x0 is variable aswell                  4, 3
    11 add proper scaling implemented by hand                            5, 4

    -> new branch with 5, 9, 10, 11

    others:
    12 plotting features for path constraints, lagrange terms            1, 1
    13 splitting const jacobian equality / inequality                    1, 1
    14 use argc, argv                                                    1, 1
    15 better memory management, not always vector.push_back             2, 1
*/

struct SolverPrivate {
    SmartPtr<GDOP> gdop;
};

Solver::Solver(GDOP* gdop) {
    this->_priv = std::make_unique<SolverPrivate>();
    this->_priv->gdop = gdop;
}

Solver::~Solver() = default;

std::string getLinearSolverName(LinearSolver solver) {
    switch (solver) {
        case LinearSolver::MUMPS:
            return "MUMPS";
        case LinearSolver::MA27:
            return "MA27";
        case LinearSolver::MA57:
            return "MA57";
        case LinearSolver::MA77:
            return "MA77";
        case LinearSolver::MA86:
            return "MA86";
        case LinearSolver::MA97:
            return "MA97";
        case LinearSolver::PARDISO:
            return "pardisomkl";
        default:
            return "MUMPS";
    }
}

int Solver::solve() {
    // init solving, pre optimization
    initSolvingProcess();

    // create IPOPT application, set linear solver, tolerances, etc.
    SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
    setSolverFlags(*app);
    ApplicationReturnStatus status = app->Initialize();

    // initial optimization
    status = app->OptimizeTNLP(_priv->gdop);
    postOptimization(*app);

    while (meshIteration <= MESH_ITERATIONS) {
        // detect intervals that have to be refined
        auto markedIntervals = detect();
        if (sz(markedIntervals) == 0) {
            finalizeOptimization(*app);
            return status;
        }

        // interpolate x and u, update the mesh
        refine(markedIntervals);

        // small overhead because of the reinitializing of the GDOP, but thus all
        // variables are initialized correctly and invariant constants during each
        // optimization remain constant
        _priv->gdop = new GDOP(_priv->gdop->problem, _priv->gdop->mesh, _priv->gdop->rk, InitVars::CALLBACK);

        // set new starting values
        _priv->gdop->xInitCallback = cbValues;

        // update solver flags
        setSolverFlags(*app);

        // optimize again
        status = app->OptimizeTNLP(_priv->gdop);
        postOptimization(*app);
    }

    finalizeOptimization(*app);
    return status;
}

std::vector<int> Solver::detect() const {
    switch (MESH_ALGORITHM) {
        case MeshAlgorithm::NONE:
            return {};
        case MeshAlgorithm::BASIC:
            return basicStrategy();
        case MeshAlgorithm::L2_BOUNDARY_NORM:
            return L2BoundaryNorm();
        default:
            return {};
    }
}

std::vector<int> Solver::L2BoundaryNorm() const {
    /* L2_BOUNDARY_NORM implementation:
     * for every v = (x, u):
     * estimates the first and second diff on each interval and bisects, if the L2 norm is too large
     * calculates the P1 error between adjacent intervals, if too large -> bisect
     */

    // states and control, if STATE_AND_CONTROL_DETECTION = true
    // only control, if STATE_AND_CONTROL_DETECTION = false
    int vOffset = _priv->gdop->offX;
    int vLength = _priv->gdop->offU;
    if (STATE_AND_CONTROL_DETECTION) {
        vOffset = 0;
        vLength = _priv->gdop->offXU;
    }

    const double cDiff = 1;
    const double cDiff2 = cDiff / 2;

    std::set<int> markerSet;

    // init last derivatives v^(d)_{i-1,m} as 0
    std::vector<std::vector<double>> lastDiffs;
    lastDiffs.reserve(vLength);
    for (int v = 0; v < vLength; v++) {
        lastDiffs.push_back({0, 0});
    }

    // calculate max, min of v^(d)
    std::vector<double> maxV;
    std::vector<double> minV;
    maxV.reserve(vLength);
    minV.reserve(vLength);
    for (int i = 0; i < _priv->gdop->mesh.intervals; i++) {
        for (int j = 0; j < _priv->gdop->rk.steps; j++) {
            for (int v = 0; v < vLength; v++) {
                if (i == 0 && j == 0) {
                    maxV.push_back(_priv->gdop->optimum[v + vOffset]);
                    minV.push_back(_priv->gdop->optimum[v + vOffset]);
                }
                else {
                    if (_priv->gdop->optimum[v + vOffset + i * _priv->gdop->offXUBlock + j * _priv->gdop->offXU] > maxV[v]) {
                        maxV[v] = _priv->gdop->optimum[v + vOffset + i * _priv->gdop->offXUBlock + j * _priv->gdop->offXU];
                    }
                    else if (_priv->gdop->optimum[v + vOffset + i * _priv->gdop->offXUBlock + j * _priv->gdop->offXU] < minV[v]) {
                        minV[v] = _priv->gdop->optimum[v + vOffset + i * _priv->gdop->offXUBlock + j * _priv->gdop->offXU];
                    }
                }
            }
        }
    }
    std::vector<double> rangeV;
    rangeV.reserve(vLength);
    for (int v = 0; v < vLength; v++) {
        rangeV.push_back(maxV[v] - minV[v]);
    }

    std::vector<double> boundsDiff;
    std::vector<double> boundsDiff2;
    for (int v = 0; v < vLength; v++) {
        boundsDiff.push_back(cDiff * rangeV[v] / initialIntervals * pow(10, -LEVEL));
        boundsDiff2.push_back(cDiff2 * rangeV[v] / initialIntervals * pow(10, -LEVEL));
    }

    for (int i = 0; i < _priv->gdop->mesh.intervals; i++) {
        bool cornerTrigger = false;
        for (int v = 0; v < vLength; v++) {
            std::vector<double> vCoeffs;
            if (i == 0) {
                for (int j = 0; j < _priv->gdop->rk.steps; j++) {
                    vCoeffs.push_back(_priv->gdop->optimum[v + vOffset + i * _priv->gdop->offXUBlock + j * _priv->gdop->offXU]);
                }
                // for xvars can also use x0, but interpolation results in same value
                vCoeffs.insert(vCoeffs.begin(), Integrator::evalLagrange(_priv->gdop->rk.c, vCoeffs, 0.0));
            }
            else {
                for (int j = -1; j < _priv->gdop->rk.steps; j++) {
                    vCoeffs.push_back(_priv->gdop->optimum[v + vOffset + i * _priv->gdop->offXUBlock + j * _priv->gdop->offXU]);
                }
            }

            // values of the (1st, 2nd) diff of the interpolating polynomial at 0, c1, c2, ...
            std::vector<double> p_vDiff = _priv->gdop->rk.evalLagrangeDiff(vCoeffs);
            // TODO: maybe remove 2nd diff matrix evalLagrangeDiff2(vCoeffs) <=> evalLagrangeDiff(p_UDiff), since D^{1}^2 = D^2
            std::vector<double> p_vDiff2 = _priv->gdop->rk.evalLagrangeDiff2(vCoeffs);
            // squared values of the (1st, 2nd) diff of the interpolating polynomial at c1, c2, ...
            std::vector<double> sq_p_vDiff;
            std::vector<double> sq_p_vDiff2;
            for (int k = 1; k < sz(p_vDiff); k++) {
                sq_p_vDiff.push_back(p_vDiff[k] * p_vDiff[k]);
                sq_p_vDiff2.push_back(p_vDiff2[k] * p_vDiff2[k]);
            }

            // (int_0^1 (d^{1,2}/dt^{1,2} p_u(t))^2 dt)^0.5 - L2 norm of the (1st, 2nd) diff
            double L2Diff1 = std::sqrt(_priv->gdop->rk.integrate(sq_p_vDiff));
            double L2Diff2 = std::sqrt(_priv->gdop->rk.integrate(sq_p_vDiff2));
            if (i > 0) {
                // difference in derivatives from polynomial of adjacent intervals
                // must not exceed some eps using p1 (+1) error; basically
                // isclose(.) in numpy bib
                double p1ErrorDiff = std::abs(p_vDiff[0] - lastDiffs[v][0]) / (1 + std::min({std::abs(p_vDiff[0]), std::abs(lastDiffs[v][0])}));
                double p1ErrorDiff2 = std::abs(p_vDiff2[0] - lastDiffs[v][1]) / (1 + std::min({std::abs(p_vDiff2[0]), std::abs(lastDiffs[v][1])}));

                if (p1ErrorDiff > C_TOL || p1ErrorDiff2 > C_TOL) {
                    cornerTrigger = true;
                }
            }
            lastDiffs[v] = {p_vDiff[_priv->gdop->rk.steps], p_vDiff2[_priv->gdop->rk.steps]};

            // detection which intervals should be bisected
            if (L2Diff1 > boundsDiff[v] || L2Diff2 > boundsDiff2[v] || cornerTrigger) {
                // splitting the interval itself
                markerSet.insert(i);

                // on interval / L2 criterion -> forces adjacent intervals to be split as well
                if (L2Diff1 > boundsDiff[v] || L2Diff2 > boundsDiff2[v]) {
                    if (i >= 1) {
                        markerSet.insert(i - 1);
                    }
                    if (i <= _priv->gdop->mesh.intervals - 2) {
                        markerSet.insert(i + 1);
                    }
                }

                // corner criterion (only exists for i > 0) -> forces left adjacent
                // interval split
                if (cornerTrigger) {
                    markerSet.insert(i - 1);
                }

                // break; -> maybe has influence on lastDiffs, cause not properly evaluated
            }
        }
    }
    return {markerSet.begin(), markerSet.end()};
}

std::vector<int> Solver::basicStrategy() const {
    std::vector<int> markedIntervals = {};
    std::vector<std::vector<double>> absIntSum = {};
    std::vector<double> means;
    std::vector<double> stds;

    for (int u = 0; u < _priv->gdop->problem->sizeU; u++) {
        absIntSum.emplace_back();
        for (int i = 0; i < _priv->gdop->mesh.intervals; i++) {
            double sum = 0;
            for (int j = -1; j < _priv->gdop->rk.steps; j++) {
                if (!((i == 0 && j == -1) || (i == _priv->gdop->mesh.intervals - 1 && j == _priv->gdop->rk.steps - 1))) {
                    double u1 = _priv->gdop->optimum[u + _priv->gdop->offX + i * _priv->gdop->offXUBlock + (j + 1) * _priv->gdop->offXU];
                    double u2 = _priv->gdop->optimum[u + _priv->gdop->offX + i * _priv->gdop->offXUBlock + j * _priv->gdop->offXU];
                    if (u1 > u2) {
                        sum += u1 - u2;
                    }
                    else {
                        sum += u2 - u1;
                    }
                }
            }
            absIntSum[u].push_back(sum);
        }
        std::vector<double> absIntSumCopy = absIntSum[u];
        std::sort(absIntSumCopy.begin(), absIntSumCopy.end());
        std::vector<double> absSum95 = std::vector<double>(absIntSumCopy.begin(), absIntSumCopy.begin() + int(0.95 * sz(absIntSumCopy)));
        const double mean = calculateMean(absSum95);
        means.push_back(mean);
        stds.push_back(calculateStdDev(absSum95, mean));
    }

    for (int i = 0; i < _priv->gdop->mesh.intervals; i++) {
        bool containsInterval = false;
        for (int u = 0; u < _priv->gdop->problem->sizeU; u++) {
            if (absIntSum[u][i] > means[u] + SIGMA * stds[u] || containsInterval) {
                markedIntervals.push_back(i);
                containsInterval = true;
            }
        }
    }
    return markedIntervals;
}

void Solver::refine(std::vector<int>& markedIntervals) {
    switch (REFINEMENT_METHOD) {
        case RefinementMethod::LINEAR_SPLINE:
            refineLinear(markedIntervals);
            break;

        case RefinementMethod::POLYNOMIAL:
            refinePolynomial(markedIntervals);
            break;

        default:
            refineLinear(markedIntervals);
            break;
    }
}

void Solver::refineLinear(std::vector<int>& markedIntervals) {
    // does a linear spline on each marked interval -> new values of control, states
    const int oldIntervalLen = _priv->gdop->mesh.intervals;
    _priv->gdop->mesh.update(markedIntervals);  // create new mesh here
    int newOffXUTotal = (_priv->gdop->problem->sizeX + _priv->gdop->problem->sizeU) * _priv->gdop->rk.steps * _priv->gdop->mesh.intervals;
    int newNumberVars = newOffXUTotal + _priv->gdop->problem->sizeP;
    cbValues.resize(newNumberVars, 0.0);

    // interpolate all values on marked intervals
    int index = 0;
    for (int i = 0; i < oldIntervalLen; i++) {
        if (markedIntervals[index] == i && index < sz(markedIntervals)) {
            for (int v = 0; v < _priv->gdop->offXU; v++) {  // iterate over every var in {x, u} -> interpolate
                std::vector<double> localVars{};
                // i > 0 interval cases
                if (i > 0) {
                    for (int j = -1; j < _priv->gdop->rk.steps; j++) {
                        localVars.push_back(_priv->gdop->optimum[v + i * _priv->gdop->offXUBlock + j * _priv->gdop->offXU]);
                    }
                    auto const splineVals = _priv->gdop->rk.evalLinearSplineNewNodes(localVars);
                    for (int k = 0; k < sz(splineVals); k++) {
                        cbValues[v + (i + index) * _priv->gdop->offXUBlock + k * _priv->gdop->offXU] = splineVals[k];
                    }
                }
                else {
                    // 0-th interval cases
                    if (v < _priv->gdop->offX) {
                        localVars.push_back(_priv->gdop->problem->x0[v]);
                        for (int j = 0; j < _priv->gdop->rk.steps; j++) {
                            localVars.push_back(_priv->gdop->optimum[v + j * _priv->gdop->offXU]);
                        }
                        auto const splineVals = _priv->gdop->rk.evalLinearSplineNewNodes(localVars);
                        for (int k = 0; k < sz(splineVals); k++) {
                            cbValues[v + i * _priv->gdop->offXUBlock + k * _priv->gdop->offXU] = splineVals[k];
                        }
                    }
                    else {
                        // 0-th control -> interpolate to get the value at t=0, linear splines
                        for (int j = 0; j < _priv->gdop->rk.steps; j++) {
                            localVars.push_back(_priv->gdop->optimum[v + j * _priv->gdop->offXU]);
                        }
                        localVars.insert(localVars.begin(), Integrator::evalLagrange(_priv->gdop->rk.c, localVars, 0.0));
                        auto const splineVals = _priv->gdop->rk.evalInterpolationNewNodes(localVars);
                        for (int k = 0; k < sz(splineVals); k++) {
                            cbValues[v + i * _priv->gdop->offXUBlock + k * _priv->gdop->offXU] = splineVals[k];
                        }
                    }
                }
            }
            index++;  // go to next marked interval
        }
        // not marked interval: copy optimal values
        else {
            for (int v = 0; v < _priv->gdop->offXU; v++) {
                for (int k = 0; k < _priv->gdop->rk.steps; k++) {
                    cbValues[v + (i + index) * _priv->gdop->offXUBlock + k * _priv->gdop->offXU] =
                        _priv->gdop->optimum[v + i * _priv->gdop->offXUBlock + k * _priv->gdop->offXU];
                }
            }
        }
    }
    for (int p = 0; p < _priv->gdop->offP; p++) {
        cbValues[newOffXUTotal + p] = _priv->gdop->optimum[_priv->gdop->offXUTotal + p];
    }
}

void Solver::refinePolynomial(std::vector<int>& markedIntervals) {
    const int oldIntervalLen = _priv->gdop->mesh.intervals;
    _priv->gdop->mesh.update(markedIntervals);  // create new mesh here
    int newOffXUTotal = (_priv->gdop->problem->sizeX + _priv->gdop->problem->sizeU) * _priv->gdop->rk.steps * _priv->gdop->mesh.intervals;
    int newNumberVars = newOffXUTotal + _priv->gdop->problem->sizeP;
    cbValues.resize(newNumberVars, 0.0);

    // interpolate all values on marked intervals
    int index = 0;
    for (int i = 0; i < oldIntervalLen; i++) {
        if (markedIntervals[index] == i && index < sz(markedIntervals)) {
            for (int v = 0; v < _priv->gdop->offXU; v++) {  // iterate over every var in {x, u} -> interpolate
                std::vector<double> localVars = {};
                // i > 0 interval cases
                if (i > 0) {
                    for (int j = -1; j < _priv->gdop->rk.steps; j++) {
                        localVars.push_back(_priv->gdop->optimum[v + i * _priv->gdop->offXUBlock + j * _priv->gdop->offXU]);
                    }
                    auto const polyVals = _priv->gdop->rk.evalInterpolationNewNodes(localVars);
                    for (int k = 0; k < sz(polyVals); k++) {
                        cbValues[v + (i + index) * _priv->gdop->offXUBlock + k * _priv->gdop->offXU] = polyVals[k];
                    }
                }
                else {
                    // 0-th interval cases
                    if (v < _priv->gdop->offX) {
                        localVars.push_back(_priv->gdop->problem->x0[v]);
                        for (int j = 0; j < _priv->gdop->rk.steps; j++) {
                            localVars.push_back(_priv->gdop->optimum[v + j * _priv->gdop->offXU]);
                        }
                        auto const polyVals = _priv->gdop->rk.evalInterpolationNewNodes(localVars);
                        for (int k = 0; k < sz(polyVals); k++) {
                            cbValues[v + i * _priv->gdop->offXUBlock + k * _priv->gdop->offXU] = polyVals[k];
                        }
                    }
                    else {
                        // 0-th control -> interpolate with order one less (only
                        // not rk.steps points, not rk.steps + 1)
                        for (int j = 0; j < _priv->gdop->rk.steps; j++) {
                            localVars.push_back(_priv->gdop->optimum[v + j * _priv->gdop->offXU]);
                        }
                        auto const polyVals = _priv->gdop->rk.interpolateFirstControl(localVars);
                        for (int k = 0; k < sz(polyVals); k++) {
                            cbValues[v + i * _priv->gdop->offXUBlock + k * _priv->gdop->offXU] = polyVals[k];
                        }
                    }
                }
            }
            index++;  // go to next marked interval
        }
        // not marked interval: copy optimal values
        else {
            for (int v = 0; v < _priv->gdop->offXU; v++) {
                for (int k = 0; k < _priv->gdop->rk.steps; k++) {
                    cbValues[v + (i + index) * _priv->gdop->offXUBlock + k * _priv->gdop->offXU] =
                        _priv->gdop->optimum[v + i * _priv->gdop->offXUBlock + k * _priv->gdop->offXU];
                }
            }
        }
    }
    for (int p = 0; p < _priv->gdop->offP; p++) {
        cbValues[newOffXUTotal + p] = _priv->gdop->optimum[_priv->gdop->offXUTotal + p];
    }
}

void Solver::setExportSparsityPath() {
    // export paths
    if (EXPORT_HESSIAN_PATH != "") {
        _priv->gdop->exportHessianPath = EXPORT_HESSIAN_PATH;
        _priv->gdop->exportHessian = true;
    }

    if (EXPORT_JACOBIAN_PATH != "") {
        _priv->gdop->exportJacobianPath = EXPORT_JACOBIAN_PATH;
        _priv->gdop->exportJacobian = true;
    }
}

void Solver::initSolvingProcess() {
    // set all flags based on the global configuration, mandatory
    printASCIIArt();
    setExportSparsityPath();
    initialIntervals = _priv->gdop->mesh.intervals;
    solveStartTime = std::chrono::high_resolution_clock::now();
}

void Solver::postOptimization(IpoptApplication& app) {
    auto ioStart = std::chrono::high_resolution_clock::now();
    SmartPtr<const SolveStatistics> stats = app.Statistics();
    SmartPtr<IpoptData> data = app.IpoptDataObject();

    numberOfIntervalsHistory.push_back(_priv->gdop->mesh.intervals);
    ipoptObjectiveHistory.push_back(stats->FinalObjective());
    ipoptIterationHistory.push_back(stats->IterationCount());
    ipoptIterationTotalTime.push_back(stats->TotalWallclockTime());
    ipoptIterationFuncEvalTime.push_back(data->TimingStats().TotalFunctionEvaluationWallclockTime());
    ipoptIterationNonfuncEvalTime.push_back(stats->TotalWallclockTime() - data->TimingStats().TotalFunctionEvaluationWallclockTime());

    if (EXPORT_OPTIMUM_PATH != "") {
        _priv->gdop->exportOptimum(EXPORT_OPTIMUM_PATH + "/" + _priv->gdop->problem->name + std::to_string(meshIteration) + ".csv");
    }
    meshIteration++;
    timedeltaIO += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - ioStart);
}

void Solver::printMeshIterationHistory() {
    std::cout << "\n---------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "\nMesh refinement history (times in seconds):\n" << std::endl;
    std::cout << std::setw(5) << "iteration" << std::setw(17) << "objective" << std::setw(18) << "intervals" << std::setw(8) << "iters" << std::setw(13)
              << "ipopt time" << std::setw(15) << "nonfunc time" << std::setw(13) << "func time" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------" << std::endl;
    for (int it = 0; it < meshIteration; it++) {
        std::cout << std::setw(5) << it << std::setw(27) << double2Str(ipoptObjectiveHistory[it], 14) << std::setw(12)
                  << std::to_string(numberOfIntervalsHistory[it]) << std::setw(8) << std::to_string(ipoptIterationHistory[it]) << std::setw(13)
                  << printTime(ipoptIterationTotalTime[it]) << std::setw(15) << printTime(ipoptIterationNonfuncEvalTime[it]) << std::setw(13)
                  << printTime(ipoptIterationFuncEvalTime[it]) << std::endl;
    }
    std::cout << "---------------------------------------------------------------------------------------------" << std::endl;
    std::cout << std::setw(5) << " overall" << std::setw(24) << double2Str(ipoptObjectiveHistory[meshIteration - 1], 14) << std::setw(12)
              << std::to_string(numberOfIntervalsHistory[meshIteration - 1]) << std::setw(8)
              << std::to_string(std::reduce(ipoptIterationHistory.begin(), ipoptIterationHistory.end())) << std::setw(13) << printTime(ipoptTotalTime)
              << std::setw(15) << printTime(ipoptActualTime) << std::setw(13) << printTime(ipoptFuncTime) << std::endl;
}

void Solver::createModelInfo(IpoptApplication& app) const {
    // writes a tmp file that contains all relevant information of the model like
    // the last mesh iteration and so on contains the metadata of the model
    std::ofstream outFile("/tmp/modelinfo.txt");
    if (!outFile) {
        std::cerr << "Error opening file for writing: "
                  << "/tmp/modelinfo.txt" << std::endl;
        return;
    }

    SmartPtr<const SolveStatistics> stats = app.Statistics();

    // standard info
    outFile << std::scientific << std::setprecision(15) << "objective: " << stats->FinalObjective() << "\n";
    outFile << std::fixed << std::setprecision(15);
    outFile << "maxMeshIteration: " << meshIteration - 1 << "\n";
    outFile << "initialIntervals: " << initialIntervals << "\n";

    // history
    outFile << "objectiveHistory: " << vectorToScientificString(ipoptObjectiveHistory) << "\n";
    outFile << "intervalHistory: " << vectorToString(numberOfIntervalsHistory) << "\n";
    outFile << "ipoptIterationHistory: " << vectorToString(ipoptIterationHistory) << "\n";
    outFile << "ipoptTimeTotalHistory: " << vectorToString(ipoptIterationTotalTime) << "\n";
    outFile << "ipoptTimeNonfuncHistory: " << vectorToString(ipoptIterationNonfuncEvalTime) << "\n";
    outFile << "ipoptTimeFuncHistory: " << vectorToString(ipoptIterationFuncEvalTime) << "\n";

    // overall
    outFile << "ipoptIterationsOverall: " << std::reduce(ipoptIterationHistory.begin(), ipoptIterationHistory.end()) << "\n";
    outFile << "ipoptTimeTotal: " << ipoptTotalTime << "\n";
    outFile << "ipoptTimeNonfunc: " << ipoptActualTime << "\n";
    outFile << "ipoptTimeFunc: " << ipoptFuncTime << "\n";
    outFile << "gdoptTimeAlgorithms: " << solveActualTimeTaken.count() - ipoptTotalTime << "\n";
    outFile << "gdoptTimeIO: " << timedeltaIO.count() << "\n";
    outFile << "totalTime: " << solveTotalTimeTaken.count() << "\n";

    outFile.close();
}

void Solver::finalizeOptimization(IpoptApplication& app) {
    std::cout << "\n---------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "\nOutput for optimization of model: " << _priv->gdop->problem->name << std::endl;

    ipoptFuncTime = std::reduce(ipoptIterationFuncEvalTime.begin(), ipoptIterationFuncEvalTime.end());  // func evals in ipopt
    ipoptTotalTime = std::reduce(ipoptIterationTotalTime.begin(), ipoptIterationTotalTime.end());       // total time in ipopt
    ipoptActualTime = ipoptTotalTime - ipoptFuncTime;                                                   // total - func evals
    solveTotalTimeTaken = std::chrono::high_resolution_clock::now() - solveStartTime;                   // total time in framework
    solveActualTimeTaken = solveTotalTimeTaken - timedeltaIO;                                           // total time in framework w/o I/O

    printMeshIterationHistory();

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "\n---------------------------------------------------------------------------------------------\n" << std::endl;
    std::cout << "Framework component timings:\n" << std::endl;
    std::cout << "Time in IPOPT (w/o func evals): " << std::setw(10) << printTime(ipoptActualTime) << " seconds" << std::endl;
    std::cout << "Time in IPOPT func evals: " << std::setw(16) << printTime(ipoptFuncTime) << " seconds" << std::endl;
    std::cout << "Time in GDOPT algorithms: " << std::setw(16) << printTime(solveActualTimeTaken.count() - ipoptTotalTime) << " seconds" << std::endl;
    std::cout << "Time in GDOPT I/O: " << std::setw(23) << printTime(timedeltaIO.count()) << " seconds" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "Total time in Framework: " << std::setw(17) << printTime(solveTotalTimeTaken.count()) << " seconds" << std::endl;
    std::cout << std::fixed << std::setprecision(15);

    createModelInfo(app);
}

void Solver::setSolverFlags(IpoptApplication& app) {
    // numeric jacobian and hessian
    // app->Options()->SetStringValue("hessian_approximation", "limited-memory");
    // app->Options()->SetStringValue("jacobian_approximation", "finite-difference-values");

    // test derivatives
    // app->Options()->SetStringValue("derivative_test", "second-order");

    if (meshIteration == 0) {
        // flags for the initial optimization
        app.Options()->SetNumericValue("bound_push", 1e-2);
        app.Options()->SetNumericValue("bound_frac", 1e-2);
        app.Options()->SetStringValue("mu_strategy", MU_STRATEGY);
        app.Options()->SetNumericValue("mu_init", MU_INIT);
    }
    else {
        // flags for every following refined optimization
        app.Options()->SetNumericValue("bound_push", 1e-4);
        app.Options()->SetNumericValue("bound_frac", 1e-4);
        app.Options()->SetStringValue("mu_strategy", MU_STRATEGY_REFINEMENT);
        app.Options()->SetNumericValue("mu_init", MU_INIT_REFINEMENT);
    }

    // setting the standard flags
    setStandardSolverFlags(app);
}

void Solver::setStandardSolverFlags(IpoptApplication& app) {
    // these are flags are always set, no matter if a mesh refinement is currently executed

    // mu-update strategy
    // turns out kkt-error adaptive_mu works really well for the provided examples (excluding poorly conditioned)
    // can be turned off with a flag

    if (KKT_ERROR_MU_GLOBALIZATION) {
        app.Options()->SetStringValue("adaptive_mu_globalization", "kkt-error");
    }

    // iterations and tolereances
    app.Options()->SetNumericValue("tol", TOLERANCE);
    app.Options()->SetNumericValue("acceptable_tol", TOLERANCE * 1e3);
    app.Options()->SetIntegerValue("max_iter", MAX_ITERATIONS);

    // ipopt dump
    app.Options()->SetStringValue("timing_statistics", "yes");
    app.Options()->SetIntegerValue("print_level", IPOPT_PRINT_LEVEL);

    // linear solver
    auto const libHSLPath = getenv("LIB_HSL");
    auto const linearSolver = getLinearSolverName(LINEAR_SOLVER);
    if (libHSLPath != nullptr) {
        // HSL found -> set chosen solver
        app.Options()->SetStringValue("hsllib", libHSLPath);
        app.Options()->SetStringValue("linear_solver", linearSolver);
    }
    else if ((libHSLPath == nullptr and linearSolver != "MUMPS")) {
        // HSL not found but set -> set MUMPS as fallback
        std::cout << "\nEnvironment variable 'LIB_HSL' not found! Fallback to standard linear solver 'MUMPS'\n" << std::endl;
        app.Options()->SetStringValue("linear_solver", "MUMPS");
    }
    else {
        // set chosen solver
        app.Options()->SetStringValue("linear_solver", linearSolver);
    }

    // scaling
    if (USER_SCALING) {
        app.Options()->SetStringValue("nlp_scaling_method", "user-scaling");
    }
    else {
        app.Options()->SetStringValue("nlp_scaling_method", "gradient-based");
    }

    // constant derivatives reduce the number of function evals
    if (QUADRATIC_OBJECTIVE_LINEAR_CONSTRAINTS) {
        app.Options()->SetStringValue("hessian_constant", "yes");
    }
    if (LINEAR_OBJECTIVE) {
        app.Options()->SetStringValue("grad_f_constant", "yes");
    }
    if (LINEAR_CONSTRAINTS) {
        app.Options()->SetStringValue("jac_c_constant", "yes");
        app.Options()->SetStringValue("jac_d_constant", "yes");
    }
}

void Solver::printASCIIArt() const {
    // slant stars border width 90, h pad 1
    const std::string art = R"(
************************************************************************************
*    __________  ____  ____  ______            ______                           __ *
*   / ____/ __ \/ __ \/ __ \/_  __/           / ____/__  ____  ___  _________ _/ / *
*  / / __/ / / / / / / /_/ / / /   ______    / / __/ _ \/ __ \/ _ \/ ___/ __ `/ /  *
* / /_/ / /_/ / /_/ / ____/ / /   /_____/   / /_/ /  __/ / / /  __/ /  / /_/ / /   *
* \____/_____/\____/_/     /_/          _   \____/\___/_/ /_/\___/_/   \__,_/_/    *
*    / __ \__  ______  ____ _____ ___  (_)____                                     *
*   / / / / / / / __ \/ __ `/ __ `__ \/ / ___/                                     *
*  / /_/ / /_/ / / / / /_/ / / / / / / / /__                                       *
* /_____/\__, /_/ /_/\__,_/_/ /_/ /_/_/\___/                                       *
*    ___/____/   __  _           _                          ____   ___    ____     *
*   / __ \____  / /_(_)___ ___  (_)___  ___  _____  _   __ / __ \ |__ \  / __ \    *
*  / / / / __ \/ __/ / __ `__ \/ /_  / / _ \/ ___/ | | / // / / / __/ / / / / /    *
* / /_/ / /_/ / /_/ / / / / / / / / /_/  __/ /     | |/ // /_/ / / __/_/ /_/ /     *
* \____/ .___/\__/_/_/ /_/ /_/_/ /___/\___/_/      |___(_)____(_)____(_)____/      *
*     /_/                                                                          *
*                                                                                  *
* This is GDOPT - General Dynamic Optimizer v.0.2.0, a framework for solving       *
* "General Dynamic Optimization Problems" using local collocation methods, based   *
* on RadauIIA formulas, and adaptive mesh refinement techniques. GDOPT utilizes    *
* the capabilities of the nonlinear optimizer IPOPT for solving the resulting      *
* large-scale NLPs. For help, visit https://github.com/linuslangenkamp/GDOPT and   *
* have a look at the provided User's Guide.                                        *
*                                                                                  *
************************************************************************************
)";
    std::cout << art << "\n";
}
