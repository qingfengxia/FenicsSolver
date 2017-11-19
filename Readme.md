# Multiphysics solver based on Fenics

by Qingfeng Xia, 2017

## License

LGPL licensed as [FreeCAD]<https://github.com/FreeCAD/FreeCAD> and [fenics-project]<https://fenicsproject.org/>

## Screenshot

![FenicsSolver as a CFD solver in CfdWorkbench of FreeCAD](FenicsSolver_FreeCAD.png?raw=true "FenicsSolver as a CFD solver in CfdWorkbench of FreeCAD")

## Description

A set of multi-physics FEM solvers based on Fenics with GUI support(via integration Fenics into FreeCAD FemWorkbench and CfdWorkbench), focusing on nonlinear problem and mutlti-solver coupling.

+ Solvers implemented:  scaler transport (heat transfer, mass transfer, electric potential, etc), Navier Stokes flow, linear elasticity. etc. 
+ Solvers under development: compressible flow, Maxwell electromagnetics.
+ Coupling to external solvers: turbulent flow will be implemented by coupled to external CFD solver, OpenFOAM.

## Installation

copy this folder to any place on the python search path (PYTHON_PATH), assuming fenics has been installed. 

```
git clone https://github.com/qingfengxia/fsolvers.git
```

installation via PIP will be implemented later once API is stable. 

## Testing

This package is under heavy refactoring, considered alpha.

Fenics version tested is on Ubuntu16.04, python 2.7 with/without FreeCAD 0.17 dev.

Run the python script files with "test_" suffix, which are gtest compatible. 

This package is python 2 and python 3 compatibe, but current FreeCAD supports only Python2.


## Roadmap and progress

### 1. Initial demonstration (Sep 2017)

A series of object oriented solvers: *ScalerEquationSolver*, *CoupledNavierStokesSolver* and *LinearElasticitySolver*, derived from *BaseSolver*, while a few other are under active development. 

Case setup: json file format is the text case setup file, mapping directly to and from python dict data structure.

### 2. FreeCAD GUI integration (Nov 2017)

2D and 3D xml mesh and case setup writing has been implemented within [CfdWorkbench](https://github.com/qingfengxia/Cfd), this feature has yet been push to FreeCAD master.
 
Meanwhile, FreeCAD developer *joha2* has added mesh export function in FemWorkbench, once the boundary mesh can be exported, case setup for fenics solver will be write in FreeCAD workbench.


### 3. Coupling of multiple solvers in series

Not yet designed.

### 4. Coupling with external solvers OpenFOAM

VTK could the data exchange format. 


## Acknowledgement

Thanks for my family members' (esp, Mrs J Wang) understanding and support, so I can work at home.




