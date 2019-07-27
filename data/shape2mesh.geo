// geo file for meshing with GMSH meshing software created by FreeCAD

// open brep geometry
Merge "./Box_Geometry.brep";

// group data 
Physical Surface("FluidBoundary") = {5};
Physical Surface("FluidBoundary001") = {6};

Physical Volume("Domain") = {1};

// no boundary layer settings for this mesh
// Characteristic Length
// min, max Characteristic Length
Mesh.CharacteristicLengthMax = 1.0;
Mesh.CharacteristicLengthMin = 0.0;

// optimize the mesh
Mesh.Optimize = 0;
Mesh.OptimizeNetgen = 0;
Mesh.HighOrderOptimize = 0;  // for more HighOrderOptimize parameter check http://gmsh.info/doc/texinfo/gmsh.html

// mesh order
Mesh.ElementOrder = 1;

// mesh algorithm, only a few algorithms are usable with 3D boundary layer generation
// 2D mesh algorithm (1=MeshAdapt, 2=Automatic, 5=Delaunay, 6=Frontal, 7=BAMG, 8=DelQuad)
Mesh.Algorithm = 8;
// 3D mesh algorithm (1=Delaunay, 2=New Delaunay, 4=Frontal, 5=Frontal Delaunay, 6=Frontal Hex, 7=MMG3D, 9=R-tree)
Mesh.Algorithm3D = 1;

// meshing
Geometry.Tolerance = 1e-06; // set gemetrical tolerance (also used for merging nodes)
Mesh  3;
Coherence Mesh; // Remove duplicate vertices

// output format 1=msh, 2=unv, 10=automatic, 27=stl, 32=cgns, 33=med, 39=inp, 40=ply2
Mesh.Format = 1;
Save "./mesh.msh";


//////////////////////////////////////////////////////////////////////
// GMSH documentation:
// http://gmsh.info/doc/texinfo/gmsh.html#Mesh
//
// We do not check if something went wrong, like negative jacobians etc. You can run GMSH manually yourself: 
//
// to see full GMSH log, run in bash:
// gmsh - /tmp/shape2mesh.geo
//
// to run GMSH and keep file in GMSH GUI (with log), run in bash:
// gmsh /tmp/shape2mesh.geo
