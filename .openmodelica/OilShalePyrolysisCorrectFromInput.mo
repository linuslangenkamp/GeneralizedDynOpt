model OilShalePyrolysisCorrectFromInput

  // States
  Real x1(start = 1) "kerogen";
  Real x2(start = 0) "pyrolytic bitumen";
  Real x3(start = 0) "oil & gas";
  Real x4(start = 0) "organic carbon";
  // Input from file
  // Reaction rate constants
  Real k1;
  Real k2;
  Real k3;
  Real k4;
  Real k5;

  Modelica.Blocks.Interfaces.RealOutput u;
  Modelica.Blocks.Sources.CombiTimeTable combiTimeTable(
    tableOnFile=true,
    tableName="K",
    fileName="C:/Users/Linus/Desktop/Studium/Master/Modelica/tableRefinement.txt",
    columns=1:2);
equation 
// Reaction rate expressions
  k1 = exp(8.86 - 20300 / 1.9872 / u);
  k2 = exp(24.25 - 37400 / 1.9872 / u);
  k3 = exp(23.67 - 33800 / 1.9872 / u);
  k4 = exp(18.75 - 28200 / 1.9872 / u);
  k5 = exp(20.70 - 31000 / 1.9872 / u);
// Dynamics
  der(x1) = (-k1 * x1) - (k3 + k4 + k5) * x1 * x2;
  der(x2) = k1 * x1 - k2 * x2 + k3 * x1 * x2;
  der(x3) = k2 * x2 + k4 * x1 * x2;
  der(x4) = k5 * x1 * x2;
  u = combiTimeTable.y[2];

// Connect the table output
end OilShalePyrolysisCorrectFromInput;
