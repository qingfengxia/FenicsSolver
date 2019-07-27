from dolfin import *
import os.path
from subprocess import Popen, PIPE, check_output
 

fname="mesh"

"""
# -3 can update gmsh2d.msh file, but without gmsh will not write msh file, why?
cmd = 'gmsh - {}.geo'.format(fname)
print check_output([cmd], shell=True)  # run in shell mode in case you are not run in terminal
 
cmd = 'dolfin-convert -i gmsh {}.msh {}.xml'.format(fname, fname)
print check_output([cmd], shell=True)
"""

mesh = Mesh(fname+".xml")
if os.path.exists( fname+"_physical_region.xml"):
    subdomains = MeshFunction("size_t", mesh, fname+"_physical_region.xml")
    plot(subdomains)
if os.path.exists( fname+"_facet_region.xml"):
    boundaries = MeshFunction("size_t", mesh, fname+"_facet_region.xml")
    plot(boundaries)
 
plot(mesh)
interactive()
