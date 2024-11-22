from gdopt import *

model = Model("reusableLaunchVehicle")

# conversion to real units
cft2m = 0.3048  # feet to meters
cft2km = cft2m / 1000  # feet to kilometers
cslug2kg = 14.5939029  # slugs to kilograms

# constants
Re = 20902900 * cft2m
S = 2690 * cft2m**2
cl1 = -0.2070
cl2 = 1.6756
cd1 = 0.0785
cd2 = -0.3529
cd3 = 2.04
b1 = 0.07854
b2 = -0.061592
b3 = 0.00621408
H = 23800 * cft2m
al1 = -0.20704
al2 = 0.029244
rho0 = 0.002378 * cslug2kg / cft2m**3
mu = 1.4076539e16 * cft2m**3
mass = 6309.433 * cslug2kg

# initial values
alt0 = 260000 * cft2m
rad0 = alt0 + Re
lon0 = 0
lat0 = 0
speed0 = 25600 * cft2m
fpa0 = -1 * pi / 180
azi0 = 90 * pi / 180

# final values
altf = 80000 * cft2m
radf = altf + Re
speedf = 2500 * cft2m
fpaf = -5 * pi / 180

# nominal values
nomRad = (rad0 + Re) / 2
nomSpeed = 45010 / 2

rad = model.addState(start=rad0, lb=Re, ub=rad0, symbol="height", nominal=nomRad)
speed = model.addState(start=speed0, lb=10, ub=45000, symbol="speed", nominal=nomSpeed)
lon = model.addState(start=lon0, lb=-pi, ub=pi, symbol="longitude")
lat = model.addState(start=lat0, lb=-70 * pi / 180, ub=70 * pi / 180, symbol="latitude")
fpa = model.addState(
    start=fpa0, lb=-80 * pi / 180, ub=80 * pi / 180, symbol="flightpath"
)
azi = model.addState(start=azi0, lb=-pi, ub=pi, symbol="azimuth")

# good guess are: 17/180 * pi, -89/360 * pi
aoa = model.addControl(lb=-pi / 2, ub=pi / 2, guess=0, symbol="angleOfAttack")
bank = model.addControl(
    lb=-pi / 2, ub=1 * pi / 180, guess=-89 / 360 * pi, symbol="bankAngle"
)

altitude = rad - Re

CD = cd1 + cd2 * aoa + cd3 * aoa**2
rho = rho0 * exp(-altitude / H)
CL = cl1 + cl2 * aoa
gravity = mu / rad**2
dynamic_pressure = 0.5 * rho * speed**2
D = dynamic_pressure * S * CD / mass
L = dynamic_pressure * S * CL / mass

model.addF(rad, speed * sin(fpa), nominal=nomRad)
model.addF(speed, -D - gravity * sin(fpa), nominal=nomSpeed)
model.addF(lon, speed * sin(fpa) * sin(azi) / (rad * cos(lat)))
model.addF(lat, speed * cos(fpa) * cos(azi) / rad)
model.addF(fpa, (L * cos(bank) - cos(fpa) * (gravity - speed**2 / rad)) / speed)
model.addF(
    azi,
    (L * sin(bank) / cos(fpa) + speed**2 * cos(fpa) * sin(azi) * tan(lat) / rad)
    / speed,
)

model.addFinal(rad, eq=radf, nominal=nomRad)
model.addFinal(speed, eq=speedf, nominal=nomSpeed)
model.addFinal(fpa, eq=fpaf)

model.addMayer(-lat)

model.tolerance = 1e-12
model.meshIterations = 8
model.meshAlgorithm = MeshAlgorithm.L2_BOUNDARY_NORM
model.muStrategyRefinement = MuStrategy.MONOTONE
model.muInitRefinement = 1e-14

model.generate()

model.optimize(
    tf=2009.35,
    steps=10,
    rksteps=4,
)
model.plot(dots=Dots.BASE)
model.plotInputsAndRefinement()
