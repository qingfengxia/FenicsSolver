# Multiphysics solver based on Fenics

by Qingfeng Xia, 2017

## License

LGPL licensed as [FreeCAD]<https://github.com/FreeCAD/FreeCAD> and [fenics-project]<https://fenicsproject.org/>

## Screenshot

![FenicsSolver as a CFD solver in CfdWorkbench of FreeCAD](FenicsSolver_FreeCAD.png?raw=true "FenicsSolver as a CFD solver in CfdWorkbench of FreeCAD")

## Description

A set of multi-physics FEM solvers based on Fenics with GUI support(via integration Fenics into FreeCAD FemWorkbench and CfdWorkbench), focusing on nonlinear problem and mutlti-solver coupling.It functions like COMSOL or Moose, but it is free and it is made of Python.

+ Solvers implemented:
  - ScalerTransport (heat transfer, mass transfer, electric potential, etc)
  - Navier Stokes incompressible laminar flow, 
  - linear elasticity, nonlinear (hyperelastic) elasticity, large deformation, plasticity

+ Solvers under development:
  - scaler transport using DG
  - viscoelastic
  - Navier Stokes compressible laminar flow, 
  - Maxwell electromagnetics
  - drift-diffusion (plasma and semiconductor)
  - wave propagation

+ coupling of above solvers
  - flow-structure interaction
  - thermal, chemical, electrical, structure process of Lithium battery (yet complete)

+ Coupling to external solvers: turbulent flow and multiphase flow will be implemented by coupled to external CFD solver, OpenFOAM.


## Installation

It is python2 and python3 compatible, just as fenics itself. For Fenics version 2017.2 on ubuntu, Python3 is recommended, since there is some binary string/unicode problem in Python 2. Fenics 2017.2 also remove VTK plotting, and most of plotting in FenicsSolver examples are ignored.

copy this folder to any place on the python search path (PYTHON_PATH), assuming fenics has been installed. 

```
git clone https://github.com/qingfengxia/FenicsSolver.git
```

installation via PIP will be implemented later once API is stable. 

## Testing

This package is under heavy refactoring, considered alpha.

This package is python 2 and python 3 compatibe, but current FreeCAD supports only Python2 for the time being.
Fenics version 2017.1 tested is on Ubuntu16.04, python 2.7 with/without FreeCAD 0.17 dev.
Fenics version 2017.2 tested is on Ubuntu16.04 and Python 3, without FreeCAD GUI. 

Run the python script files with "test_" suffix, which are gtest compatible. 


## Roadmap and progress

see also my presentation at Fenics 18: [Automated Mechanical Engineering Design using Open Source CAE Software Packages]("doc/Fenics18_Xia.pdf")

### 1. Initial demonstration (Sep 2017)

A series of object oriented solvers: *ScalerEquationSolver*, *CoupledNavierStokesSolver* and *LinearElasticitySolver*, derived from *BaseSolver*, while a few other are under active development. 

Case setup: json file format is the text case setup file, mapping directly to and from python dict data structure.

### 2. FreeCAD GUI integration (Nov 2017)

2D and 3D xml mesh and case setup writing has been implemented within [CfdWorkbench](https://github.com/qingfengxia/Cfd), this feature has yet been push to FreeCAD master.
 
Meanwhile, FreeCAD developer *joha2* has added mesh export function in FemWorkbench, once the boundary mesh can be exported, case setup for fenics solver will be write in FreeCAD workbench.


### 3. Coupling of multiple solvers (planned in late 2018)

Fluid-structure interaction coupling has a initial implementation in segregate coupling mode, see engine seal FSI simulation:
[2D FSI simulation of labyrinth seal]("doc/fsi_velmag.pdf")

Tight coupling of all physical fields is under design, target on Lithium ion Battery thermal modelling.

### 4. Coupling with external solvers OpenFOAM

VTK is the data exchange format for one-way coupling from OpenFOAM to FenicsSolver, from Foam to VTK and VTK to Foam (mesh and internal field data files)
Two-way coupling should be implemented with the multiphysics coupling library [preCICE](https://github.com/precice/precice)

a video/presentation of my 13th OpenFOAM workshop presentation can be found here:
[Coupling OpenFOAM with FeniCS for multiphysis simulation](https://www.iesensor.com/blog/2018/06/25/coupling-openfoam-with-fenics-for-multiphysis-simulation-openfoam-workshop-13-presentation/)


## Acknowledgement

Thanks for my family members' (esp, Mrs J Wang) understanding and support, so I can work at home.




