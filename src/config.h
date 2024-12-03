#ifndef GDOPT_CONFIG_H
#define GDOPT_CONFIG_H

#include <optional>

#include "solver.h"

extern InitVars INIT_VARS;
extern IntegratorSteps RADAU_INTEGRATOR;
extern RefinementMethod REFINEMENT_METHOD;
extern LinearSolver LINEAR_SOLVER;
extern MeshAlgorithm MESH_ALGORITHM;

extern double TOLERANCE;
extern double FINAL_TIME;
extern int INTERVALS;
extern int MAX_ITERATIONS;
extern int MESH_ITERATIONS;
extern bool USER_SCALING;

extern int FULL_BISECTIONS;
extern double SIGMA;
extern double LEVEL;
extern double C_TOL;
extern bool STATE_AND_CONTROL_DETECTION;

extern int IPOPT_PRINT_LEVEL;
extern bool KKT_ERROR_MU_GLOBALIZATION;
extern double MU_INIT;
extern double MU_INIT_REFINEMENT;
extern std::string MU_STRATEGY;
extern std::string MU_STRATEGY_REFINEMENT;

extern bool LINEAR_OBJECTIVE;
extern bool LINEAR_CONSTRAINTS;
extern bool QUADRATIC_OBJECTIVE_LINEAR_CONSTRAINTS;

extern std::string EXPORT_OPTIMUM_PATH;
extern std::string EXPORT_HESSIAN_PATH;
extern std::string EXPORT_JACOBIAN_PATH;
extern std::string INITIAL_STATES_PATH;

std::unordered_map<std::string, std::string> readConfig(const std::string& filename);
void setGlobalStandardConfiguration(const std::unordered_map<std::string, std::string>& configMap);

#endif  // GDOPT_CONFIG_H
