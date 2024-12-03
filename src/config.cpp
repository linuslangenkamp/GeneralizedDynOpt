#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "solver.h"

// declaration of global variables

// parameters that always have to be given
InitVars INIT_VARS;
IntegratorSteps RADAU_INTEGRATOR;
double FINAL_TIME;
int INTERVALS;

// parameters with default values
RefinementMethod REFINEMENT_METHOD = RefinementMethod::LINEAR_SPLINE;
LinearSolver LINEAR_SOLVER = LinearSolver::MUMPS;
MeshAlgorithm MESH_ALGORITHM = MeshAlgorithm::L2_BOUNDARY_NORM;

double TOLERANCE = 1e-14;
int MAX_ITERATIONS = 5000;
int MESH_ITERATIONS = 0;
bool USER_SCALING = false;

// ipopt flags
int IPOPT_PRINT_LEVEL = 5;
bool KKT_ERROR_MU_GLOBALIZATION = true;
double MU_INIT = 0.1;                             // mu init, only has effect, if monotone is chosen - only first mesh iteration
double MU_INIT_REFINEMENT = 0.1;                  // mu init, only has effect, if monotone is chosen - all mesh iterations
std::string MU_STRATEGY = "adaptive";             // mu strategy - only first mesh iteration
std::string MU_STRATEGY_REFINEMENT = "adaptive";  // mu strategy - all mesh iterations

// mesh parameters
int FULL_BISECTIONS = 0;                   // value determines in  how many refinement iterations all intervals are bisected
double SIGMA = 2.5;                        // basicStrategy: std deviation sigma
double LEVEL = 0;                          // L2BN: L2 criterion factor, log scale, std range -2.5 - 2.5
double C_TOL = 0.1;                        // L2BN: corner criterion P1-error threshold, std range 0.05 - 0.5
bool STATE_AND_CONTROL_DETECTION = false;  // L2BN: true: detect unsmooth states and control vars, false: only control

// flags for constant derivatives
bool LINEAR_OBJECTIVE = false;                        // true if M and L are linear
bool LINEAR_CONSTRAINTS = false;                      // true if f, g, r and a are all linear
bool QUADRATIC_OBJECTIVE_LINEAR_CONSTRAINTS = false;  // true if f, g, r and a are all linear and M and L are at most quadratic

// pseudo optional, if null -> empty string ""
std::string EXPORT_OPTIMUM_PATH;
std::string EXPORT_HESSIAN_PATH;
std::string EXPORT_JACOBIAN_PATH;
std::string INITIAL_STATES_PATH;

const std::unordered_map<std::string, InitVars> initVarsMap = {{"CONST", InitVars::CONST},
                                                               {"SOLVE", InitVars::SOLVE},
                                                               {"SOLVE_EXPLICIT", InitVars::SOLVE_EXPLICIT},
                                                               {"SOLVE_EXPLICIT_EULER", InitVars::SOLVE_EXPLICIT_EULER},
                                                               {"CALLBACK", InitVars::CALLBACK}};

const std::unordered_map<std::string, LinearSolver> linearSolverMap = {
    {"MUMPS", LinearSolver::MUMPS}, {"MA27", LinearSolver::MA27}, {"MA57", LinearSolver::MA57},      {"MA77", LinearSolver::MA77},
    {"MA86", LinearSolver::MA86},   {"MA97", LinearSolver::MA97}, {"PARDISO", LinearSolver::PARDISO}};

const std::unordered_map<std::string, RefinementMethod> refinementMethodMap = {{"POLYNOMIAL", RefinementMethod::POLYNOMIAL},
                                                                               {"LINEAR_SPLINE", RefinementMethod::LINEAR_SPLINE}};

const std::unordered_map<std::string, MeshAlgorithm> meshAlgorithmMap = {
    {"NONE", MeshAlgorithm::NONE}, {"BASIC", MeshAlgorithm::BASIC}, {"L2_BOUNDARY_NORM", MeshAlgorithm::L2_BOUNDARY_NORM}};

InitVars stringToInitVars(const std::string &str) {
    auto it = initVarsMap.find(str);
    if (it != initVarsMap.end()) {
        return it->second;
    }
    else {
        throw std::invalid_argument("Invalid InitVars configuration argument: " + str);
    }
}

LinearSolver stringToLinearSolver(const std::string &str) {
    auto it = linearSolverMap.find(str);
    if (it != linearSolverMap.end()) {
        return it->second;
    }
    else {
        throw std::invalid_argument("Invalid LinearSolver configuration argument: " + str);
    }
}

RefinementMethod stringToRefinementMethod(const std::string &str) {
    auto it = refinementMethodMap.find(str);
    if (it != refinementMethodMap.end()) {
        return it->second;
    }
    else {
        throw std::invalid_argument("Invalid RefinementMethod configuration argument: " + str);
    }
}

MeshAlgorithm stringToMeshAlgorithm(const std::string &str) {
    auto it = meshAlgorithmMap.find(str);
    if (it != meshAlgorithmMap.end()) {
        return it->second;
    }
    else {
        throw std::invalid_argument("Invalid MeshAlgorithm configuration argument: " + str);
    }
}

std::string trim(const std::string &str) {
    size_t first = str.find_first_not_of(" \t\";");
    if (first == std::string::npos)
        return "";
    size_t last = str.find_last_not_of(" \t\";");
    return str.substr(first, last - first + 1);
}

std::unordered_map<std::string, std::string> readConfig(const std::string &filename) {
    std::unordered_map<std::string, std::string> configMap;
    std::ifstream infile(filename);
    std::string line;

    while (std::getline(infile, line)) {
        line = trim(line);

        // skip
        if (line.empty() || line[0] == '#' || line.front() == '[' && line.back() == ']')
            continue;

        std::istringstream iss(line);
        std::string key, value;

        if (std::getline(iss, key, ' ') && std::getline(iss, value)) {
            key = trim(key);
            value = trim(value);
            configMap[key] = value;
        }
    }

    return configMap;
}

void setGlobalStandardConfiguration(const std::unordered_map<std::string, std::string> &configMap) {
    // sets the global variables from the given configuration (excluding runtime parameters)

    // basic types
    INTERVALS = std::stoi(configMap.at("INTERVALS"));
    MESH_ITERATIONS = std::stoi(configMap.at("MESH_ITERATIONS"));
    MAX_ITERATIONS = std::stoi(configMap.at("MAX_ITERATIONS"));
    FINAL_TIME = std::stod(configMap.at("FINAL_TIME"));
    TOLERANCE = std::stod(configMap.at("TOLERANCE"));
    USER_SCALING = configMap.at("USER_SCALING") == "true";

    // enums
    INIT_VARS = stringToInitVars(configMap.at("INIT_VARS"));
    LINEAR_SOLVER = stringToLinearSolver(configMap.at("LINEAR_SOLVER"));
    REFINEMENT_METHOD = stringToRefinementMethod(configMap.at("REFINEMENT_METHOD"));
    MESH_ALGORITHM = stringToMeshAlgorithm(configMap.at("MESH_ALGORITHM"));
    RADAU_INTEGRATOR = (IntegratorSteps)std::stoi(configMap.at("RADAU_INTEGRATOR"));

    // constant derivatives
    LINEAR_OBJECTIVE = configMap.at("LINEAR_OBJECTIVE") == "true";
    LINEAR_CONSTRAINTS = configMap.at("LINEAR_CONSTRAINTS") == "true";
    QUADRATIC_OBJECTIVE_LINEAR_CONSTRAINTS = configMap.at("QUADRATIC_OBJECTIVE_LINEAR_CONSTRAINTS") == "true";

    // important ipopt flag, which is often benefitial, but terrible for poorly conditioned problems
    if ((configMap.find("KKT_ERROR_MU_GLOBALIZATION") != configMap.end())) {
        KKT_ERROR_MU_GLOBALIZATION = configMap.at("KKT_ERROR_MU_GLOBALIZATION") == "true";
    }

    // other ipopt flags
    if ((configMap.find("MU_INIT") != configMap.end())) {
        MU_INIT = std::stod(configMap.at("MU_INIT"));
    }
    if ((configMap.find("MU_INIT_REFINEMENT") != configMap.end())) {
        MU_INIT_REFINEMENT = std::stod(configMap.at("MU_INIT_REFINEMENT"));
    }
    if ((configMap.find("MU_STRATEGY") != configMap.end())) {
        MU_STRATEGY = configMap.at("MU_STRATEGY");
    }
    if ((configMap.find("MU_STRATEGY_REFINEMENT") != configMap.end())) {
        MU_STRATEGY_REFINEMENT = configMap.at("MU_STRATEGY_REFINEMENT");
    }

    // optional output and dump flags
    if ((configMap.find("IPOPT_PRINT_LEVEL") != configMap.end())) {
        IPOPT_PRINT_LEVEL = std::stoi(configMap.at("IPOPT_PRINT_LEVEL"));
    }
    if ((configMap.find("EXPORT_OPTIMUM_PATH") != configMap.end())) {
        EXPORT_OPTIMUM_PATH = configMap.at("EXPORT_OPTIMUM_PATH");
    }
    if ((configMap.find("EXPORT_HESSIAN_PATH") != configMap.end())) {
        EXPORT_HESSIAN_PATH = configMap.at("EXPORT_HESSIAN_PATH");
    }
    if ((configMap.find("EXPORT_JACOBIAN_PATH") != configMap.end())) {
        EXPORT_JACOBIAN_PATH = configMap.at("EXPORT_JACOBIAN_PATH");
    }
    if ((configMap.find("INITIAL_STATES_PATH") != configMap.end())) {
        INITIAL_STATES_PATH = configMap.at("INITIAL_STATES_PATH");
    }

    //  mesh flags
    if ((configMap.find("SIGMA") != configMap.end())) {
        SIGMA = std::stod(configMap.at("SIGMA"));
    }
    if ((configMap.find("LEVEL") != configMap.end())) {
        LEVEL = std::stod(configMap.at("LEVEL"));
    }
    if ((configMap.find("C_TOL") != configMap.end())) {
        C_TOL = std::stod(configMap.at("C_TOL"));
    }
    if ((configMap.find("STATE_AND_CONTROL_DETECTION") != configMap.end())) {
        STATE_AND_CONTROL_DETECTION = configMap.at("STATE_AND_CONTROL_DETECTION") == "true";
    }
    if ((configMap.find("FULL_BISECTIONS") != configMap.end())) {
        FULL_BISECTIONS = std::stoi(configMap.at("FULL_BISECTIONS"));
    }
}
