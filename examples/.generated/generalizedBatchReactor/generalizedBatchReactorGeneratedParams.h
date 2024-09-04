//defines

#define INIT_VARS InitVars::CONST
#define RADAU_INTEGRATOR IntegratorSteps::Steps3
#define INTERVALS 50
#define FINAL_TIME 1
#define LINEAR_SOLVER LinearSolver::MA57
#define MESH_ALGORITHM MeshAlgorithm::L2_BOUNDARY_NORM
#define MESH_ITERATIONS 5
#define TOLERANCE 1e-14
#define EXPORT_OPTIMUM_PATH "/tmp"

// values for runtime parameters
#define EXPONENT_ENERGY_VALUE 4
#define DEPLETION_COEFF_VALUE 35