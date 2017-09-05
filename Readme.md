# Multiphysics solver based on Fenics

by Qingfeng Xia, 2017

## Description

In order to bring GUI setup support (via integration Fenics into FreeCAD FemWorkbench and CfdWorkbench), a series of object oriented solvers are developed to solve heat transfer, mass transfer, Navier Stokes flow, linear elasticity, static electrical problem, etc. 

## License
LGPL licensed as [FreeCAD]<> and [fenics-project]<https://fenicsproject.org/>


## Installation

copy this folder to any place on the python search path, assuming fenics has been installed. 

```
git clone https://github.com/qingfengxia/fsolvers.git
```

## Testing

This package is under heavy refactoring, considered alpha.

Fenics version tested is on Ubuntu16.04, python 2.7 with FreeCAD dev.

Run the script files with "test_" suffix, which are gtest compatible. This package is python 2 and python 3, but current FreeCAD supports only Python2.


## Roadmap and progress

### 1. Initial demonstration

A series of object oriented solvers: *ScalerEquationSolver* and *LinearElasticitySolver*, derived from *BaseSolver*, while NavierStokesSolver is under testing. 

Case setup: json file format could be the text case setup file, mapping directly to and from python dict data structure.

### 2. FreeCAD GUI integration (late 2017)

3D xml mesh and boundary export has been implemented by Qingfeng Xia via Gmsh, this feature has yet been push to FreeCAD master. Meanwhile, FreeCAD developer *joha2* has added mesh export function in FemWorkbench, once the boundary mesh can be exported, case setup for fenics solver will be write in FreeCAD workbench.


### 3. Coupling of multiple solvers in series

Not yet designed.

### 4. Coupling with external solvers

VTK could the data exchange format. 


## Acknowledgement

Thanks for my family members' understanding and support, so I can work at home.




