from gdopt import *

model = Model("dieselMotor")

# model.setExpressionSimplification(True)

p_amb = 1.0111134146341463e005
T_amb = 2.9846362195121958e002
gamma_a = 1.3964088397790055e000
R_a = 2.8700000000000000e002
cp_a = 1.0110000000000000e003
cv_a = 7.2400000000000000e002
p_es = 1.0111134146341463e005
gamma_e = 1.2734225621414914e000
R_e = 2.8600000000000000e002
cp_e = 1.3320000000000000e003
Hlhv = 4.2900000000000000e007
AFs = 1.4570000000000000e001
gamma_cyl = 1.3500000000000001e000
T_im = 3.0061857317073162e002
x_r0 = 0
T_10 = 3.0064178823529403e002
n_cyl = 6
V_D = 1.2699999999999999e-002
r_c = 1.7300000000000001e001
A_wg_eff = 8.8357293382212933e-004
J_genset = 3.5000000000000000e000
d_pipe = 0.1
n_pipe = 2
l_pipe = 1
R_c = 4.0000000000000001e-002
V_is = 2.1302555405064521e-002
V_em = 2.0024706650635410e-002
R_t = 4.0000000000000001e-002
J_tc = 1.9777955929704147e-004
state_norm1 = 2.2000000000000000e002
state_norm2 = 2.0000000000000000e005
state_norm3 = 3.0000000000000000e005
state_norm4 = 1.0000000000000000e004
control_norm1 = 2.5000000000000000e002
control_norm2 = 1
control_norm3 = 2.5400000000000000e005
Psi_max = 1.4374756793329366e000
dot_m_c_corr_max = 5.2690636559024850e-001
eta_igch = 6.8768988621327665e-001
c_fr1 = 7.1957840228405290e-001
c_fr2 = -1.4144357053459333e-001
c_fr3 = 3.5904197283929118e-001
eta_sc = 1.0515746242284574e000
x_cv = 5.7966369236798054e-001
A_t_eff = 9.9877716035454514e-004
eta_c = 5.2227332577901808e-001
eta_t = 6.8522930965034778e-001
eta_vol = 8.9059680994120261e-001
w_fric = 2.4723010996875069e-005

w_ice = model.addState(
    start=2.4989941562646081e-01,
    lb=4 / state_norm1,
    ub=220 / state_norm1,
    symbol="w_ice",
)
p_im = model.addState(
    start=5.0614999999999999e-01,
    lb=0.8 * p_amb / state_norm2,
    ub=2 * p_amb / state_norm2,
    symbol="p_im",
)
p_em = model.addState(
    start=3.3926666666666666e-01,
    lb=p_amb / state_norm3,
    ub=3 * p_amb / state_norm3,
    symbol="p_em",
)
w_tc = model.addState(
    start=6.8099999999999994e-02,
    lb=300 / state_norm4,
    ub=10000 / state_norm4,
    symbol="w_tc",
)

u_f = model.addInput(
    lb=0, ub=250 / control_norm1, symbol="u_f", guess=25 / control_norm1
)
u_wg = model.addInput(lb=0, ub=1, symbol="u_wg", guess=0.1)

W_ICE = state_norm1 * w_ice
P_IM = state_norm2 * p_im
P_EM = state_norm3 * p_em
W_TC = state_norm4 * w_tc
U_F = control_norm1 * u_f

Pi_c = P_IM / p_amb
w_tc_corr = state_norm4 * w_tc / sqrt(T_amb / T_amb)
Pi_c_max = ((((w_tc_corr**2) * (R_c**2) * Psi_max) / (2 * cp_a * T_amb)) + 1) ** (
    gamma_a / (gamma_a - 1)
)
Cm_temp = 1 - ((Pi_c / Pi_c_max) ** 2)
dot_m_c = (dot_m_c_corr_max * sqrt(Cm_temp)) * (p_amb / p_amb) / sqrt(T_amb / T_amb)
P_c = dot_m_c * cp_a * T_amb * ((Pi_c ** ((gamma_a - 1) / gamma_a)) - 1) / eta_c

dot_m_ci = eta_vol * P_IM * W_ICE * V_D / (4 * pi * R_a * T_im)

dot_m_f = U_F * W_ICE * n_cyl * 1e-6 / (4 * pi)

eta_ig = eta_igch * (1 - (1 / (r_c ** (gamma_cyl - 1))))
T_pump = V_D * (P_EM - P_IM)
T_ig = n_cyl * Hlhv * eta_ig * u_f * control_norm1 * 1e-6
T_fric = (
    V_D
    * (10**5)
    * (
        c_fr1 * ((W_ICE * 60 / (2 * pi * 1000)) ** 2)
        + c_fr2 * (W_ICE * 60 / (2 * pi * 1000))
        + c_fr3
    )
)
T_ice = (T_ig - T_fric - T_pump) / (4 * pi)

Pi_e = P_EM / P_IM
q_in = dot_m_f * Hlhv / (dot_m_f + dot_m_ci)
x_p = 1 + q_in * x_cv / (cv_a * T_im * (r_c ** (gamma_a - 1)))
T_eo = (
    eta_sc
    * (Pi_e ** (1 - 1 / gamma_a))
    * (r_c ** (1 - gamma_a))
    * (x_p ** (1 / gamma_a - 1))
    * (q_in * ((1 - x_cv) / cp_a + x_cv / cv_a) + T_im * (r_c ** (gamma_a - 1)))
)

Pi_t = p_es / P_EM
Pi_ts_low = (2 / (gamma_e + 1)) ** (gamma_e / (gamma_e - 1))
Pi_ts = sqrt(Pi_t)
Psi_t = sqrt(
    (2 * gamma_e / (gamma_e - 1))
    * ((Pi_ts ** (2 / gamma_e)) - (Pi_ts ** ((gamma_e + 1) / gamma_e)))
)
dot_m_t = P_EM * Psi_t * A_t_eff / (sqrt(T_eo * R_e))

P_t = dot_m_t * cp_e * T_eo * eta_t * (1 - (Pi_t ** ((gamma_e - 1) / gamma_e)))

Pi_wg = p_amb / P_EM
Pi_wgs = Pi_wg
Psi_wg = sqrt(
    2
    * gamma_e
    / (gamma_e - 1)
    * ((Pi_wgs ** (2 / gamma_e)) - (Pi_wgs ** ((gamma_e + 1) / gamma_e)))
)
dot_m_wg = P_EM * Psi_wg * A_wg_eff * u_wg / (sqrt(T_eo * R_e))
P_ICE = T_ice * W_ICE / control_norm3

model.addDynamic(w_ice, 0.0012987012987013 * T_ice)
model.addDynamic(
    p_im, 20.2505119361145 * ((0.526906365590249 * sqrt(Cm_temp)) - dot_m_ci)
)
model.addDynamic(
    p_em, 0.0476078551344513 * (T_eo * (dot_m_ci + dot_m_f - dot_m_t - dot_m_wg))
)
model.addDynamic(
    w_tc,
    0.0001
    * (
        (P_t - P_c) / (0.000197779559297041 * W_TC)
        - 2.47230109968751e-005 * W_TC * W_TC
    ),
)

model.addMayer(
    (w_ice - 0.515309170685596) ** 2
    + (p_im - 0.547055854225991) ** 2
    + (p_em - 0.381048005791294) ** 2
    + (w_tc - 0.271443000537680) ** 2
)
model.addLagrange(dot_m_f)

model.generate()

model.optimize(
    steps=75,
    rksteps=5,
    tf=0.5,
    flags={
        "linearSolver": LinearSolver.MUMPS,
        "ivpSolver": IVPSolver.RADAU,
        "initVars": InitVars.SOLVE,
        "tolerance": 1e-14,
    },
    meshFlags={
        "algorithm": MeshAlgorithm.L2_BOUNDARY_NORM,
        "iterations": 5,
        "refinementMethod": RefinementMethod.LINEAR_SPLINE,
    },
)

model.plot(dots=Dots.ALL)
model.plotMeshRefinement()
model.plotInputsAndRefinement()
