# -*- coding: utf-8 -*-
# ***************************************************************************
# *                                                                         *
# *   Copyright (c) 2017 - Qingfeng Xia <qingfeng.xia iesensor.com>         *
# *                                                                         *
# *   This program is free software; you can redistribute it and/or modify  *
# *   it under the terms of the GNU Lesser General Public License (LGPL)    *
# *   as published by the Free Software Foundation; either version 2 of     *
# *   the License, or (at your option) any later version.                   *
# *   for detail see the LICENCE text file.                                 *
# *                                                                         *
# *   This program is distributed in the hope that it will be useful,       *
# *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
# *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
# *   GNU Library General Public License for more details.                  *
# *                                                                         *
# *   You should have received a copy of the GNU Library General Public     *
# *   License along with this program; if not, write to the Free Software   *
# *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  *
# *   USA                                                                   *
# *                                                                         *
# ***************************************************************************


from __future__ import print_function, division

import os.path

from dolfin import *
import numpy as np

def convert_to_hdf5_mesh_file(filename):
    filename_base = filename[:-4]
    mesh = Mesh(filename)
    hdf = HDF5File(mesh.mpi_comm(), filename_base + ".h5", "w")
    hdf.write(mesh, "/mesh")
    subdomain_file = filename_base + "_physical_region.xml"
    if os.path.exists(subdomain_file):
        subdomains = MeshFunction("size_t", mesh, subdomain_file)
        hdf.write(subdomains, "/subdomains")
    bmeshfile =filename_base + "_facet_region.xml"
    if os.path.exists(bmeshfile):
        boundaries = MeshFunction("size_t", mesh, bmeshfile)
        hdf.write(boundaries, "/boundaries")
    return filename_base + ".h5"

def image_to_dolfin_function(image, roi):
    """ image: 2D numpy image (maybe come from experiment), each pixel is a cell center value
    roi: region of interest, tuple/list of rect:  (xmin, ymin, xmax, ymax)
    return: dolfin CG1 function, the caller will project this onto a different mesh and solve PDE
    """
    nx,ny = image.shape
    nx -= 1
    ny -= 1
    xmin, ymin, xmax, ymax = roi
    mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), nx, ny)  # quad cell will be even better 
    print(mesh.num_vertices(), nx*ny)
    #assert sqrt(mesh.num_vertices()) == nx*ny

    x = mesh.coordinates().reshape((-1, mesh.geometry().dim()))
    hx = (xmax - xmin)/nx
    hy = (ymax - ymin)/ny

    ii, jj = (x[:, 0] - xmin)/hx, (x[:, 1]-ymin)/hy   # scaled to unit rect
    ii = np.array(ii, dtype=int)
    jj = np.array(jj, dtype=int)
    image_values = image[ii, jj]

    V = FunctionSpace(mesh, 'CG', 1)
    image_f = Function(V)

    # Values will be dof ordered, work only on serial?
    d2v = dof_to_vertex_map(V)
    image_values = image_values[d2v]
    image_f.vector()[:] = image_values
    
    '''
    # Get back to 'image'
    v2d = vertex_to_dof_map(V)
    new_image = np.zeros_like(image)
    values = image_f.vector().get_local()[v2d]
    for (i, j, v) in zip(ii, jj, values): new_image[i, j] = v
    '''

    return image_f
