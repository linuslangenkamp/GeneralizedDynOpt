model OilShalePyrolysisCorrectFromInputX2
  // 50 steps, Radau5, tolerance 1e-15, 

  // it 5 - x2(8) = 0.353651578177002 by GDOPT
  // it 5 provides: 0.3536515795701177
  
  // it 0 - x2(8) = 0.353650284877929 by GDOPT
  // it 0 provides: 0.3536502806226824 
                    
  // TODO: state difference
                    
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
  // LinearSegments, ContinuousDerivative, ConstantSegments, MonotoneContinuousDerivative1, MonotoneContinuousDerivative2, ModifiedContinuousDerivative
  Modelica.Blocks.Interfaces.RealOutput u "temperature";
  Modelica.Blocks.Interfaces.RealOutput x2_GDOPT "x2_GDOPT";
  
  Real deltaX2;
  Real deltaX2Rel;
  
  Modelica.Blocks.Sources.CombiTimeTable combiTimeTable(
    tableOnFile=true,
    smoothness = Modelica.Blocks.Types.Smoothness.MonotoneContinuousDerivative1,
    tableName="K",
    fileName="C:/Users/Linus/Desktop/Studium/Master/Modelica/tableRefinementU_X2.txt",
    columns=1:3);
  Real dummy;
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
  x2_GDOPT = combiTimeTable.y[3];
  
  deltaX2 = abs(x2_GDOPT - x2);
  deltaX2Rel = abs(x2_GDOPT - x2) / (max(x2_GDOPT, x2) + 1e-16);
  
  dummy = u * 10^(-7) / 8;
// Connect the table output
end OilShalePyrolysisCorrectFromInputX2;
