**upgrade from Fenics 2017.1 to 2019**


<https://fenicsproject.org/docs/dolfin/dev/python/ChangeLog.html>

To enjoy

+ python2 is gone in Fenics 2018.1
 but it should be easy to upgrade, tool `2to3`
`from __future__ import print_function, division`

+ SWIG wrapping is replaced by pybind11 in 2018.1

+ dolfin.dolfin_version() is gone, using dolfin.__version__
the output is the same, string of `2019.1.0` 
`ver = [int(s) for s in dolfin.__version__.split('.')]`

+ renaming in 2018.1
> Rename mpi_comm_world() to MPI.comm_world.
> Removed from fenics import *, use from dolfin import *
> Rename ERROR, CRITICAL etc. to LogLevel.ERROR, LogLevel.CRITICAL.

```python
try:
    mpi_comm_world_size = MPI.size(mpi_comm_world())
except:
    mpi_comm_world_size = MPI.size(MPI.comm_world)
if mpi_comm_world_size >1:
    self.parallel = True
```

##

+ VTK plot is gone, using matplotlib

```python
if int(ver[0]) <= 2017 and int(ver[1])<2:
    using_matplotlib = False  # using VTK
else:
    using_matplotlib = True

if not using_matplotlib:
    interactive()  
else:
    import matplotlib.pyplot as plt
    plt.show()
```

+ FacetFunction is gone

`boundary_facets = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)`


+ UserExpression instead of Expression
```python
if ver[0]<2018:
    UserExpression = Expression
```


+ Function.vector().get_local()
This function is available in 2017.1, just replace `vector().array()`


+ "Remove UnitQuadMesh and UnitHexMesh. Now use UnitSquareMesh and UnitCubeMesh with cell type qualifiers."