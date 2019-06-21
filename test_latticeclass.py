# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

__all__ = ['Qbasis','Qcrystal']
from matplotlib.pyplot import *
import warnings
import types

try:
    import builtins
except:
    import __builtin__ as builtins

# import math functions from numpy.math: required for td string evaluation
from numpy import (arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh,
                   ceil, copysign, cos, cosh, degrees, e, exp, expm1, fabs,
                   floor, fmod, frexp, hypot, isinf, isnan, ldexp, log, log10,
                   log1p, modf, pi, radians, sin, sinh, sqrt, tan, tanh, trunc)
from scipy.sparse import (_sparsetools, isspmatrix, isspmatrix_csr,
                          csr_matrix, coo_matrix, csc_matrix, dia_matrix)
from qutip.fastsparse import fast_csr_matrix, fast_identity
from qutip.qobj import Qobj
from qutip.qobj import isherm
import numpy as np
from scipy.sparse.linalg import eigs

from latticeclass import *
from numpy.testing import assert_
from qutip import *



def test_hamiltonian():
    eps0 = 0.0; eps1 = 0.0;#assert_hermicity(q_x, True)

    t = -1.0; tp = -1.0;
    x_vec = 1;
    pos0 = [0.0]; pos1 = [0.5];

############ method 1: Adding Two orbitals at once ############################
    dimensions=1
    number_of_orbitals=2
    onsite_energy_array = [eps0, eps1]
    position_array = [pos0, pos1]
###  [...[orbital_m,orbital_n,hopping_mn],[orbital_p,orbital_q,hopping_pq]...]
    intra_hopping_array=[[0,1,t]]

    F1 = Qbasis(onsite_energy_array, position_array ,intra_hopping_array, dimensions, number_of_orbitals)
    c0 = basis(2,0)
    c1 = basis(2,1)
    super_unit_cell = t*tensor( c0*c0.dag(), sigmax() ) + t*tensor( c1*c1.dag(), sigmax() ) + tp* ( tensor( c1*c0.dag(), (sigmax()+ 1j*sigmay())/2 ) + tensor( c1*c0.dag(), (sigmax()+ 1j*sigmay())/2 ).dag() ) 

    F1.input_super_unit_cell(super_unit_cell)

#F1.display_model()
############ method 1: Adding Two orbitals at once ############################
#dimensions=1
#number_of_orbitals=1
#onsite_energy_array0 = [eps0]
#position_array0 = [pos0]
#intra_hopping_array0=[]

#F2 = Qbasis(onsite_energy_array0, position_array0 ,intra_hopping_array0, dimensions, number_of_orbitals)
#F2.display_model()

#F2.Add_orbital(eps1, pos1)
#intra_hopping = [0,1,t,[1]]
#F2.Add_intra_hopping(intra_hopping)
#F2.display_model()
###  [...[orbital_m,orbital_n,hopping_mn],[orbital_p,orbital_q,hopping_pq]...]

#F1.Unit_Hamiltonian()

#F2.Total_f()
#F1.basis_Hamiltonian()


    basis_vector_array = [[1]]
#basis_vector_array = [[]]
    inter_hopping_array=[[0,1,tp]]
    F1crys = Qcrystal(F1,inter_hopping_array,basis_vector_array)
    F1crys.display_model()
#F1crys.dispersion()

# must choose kpoints to be an odd integer
    F1crys.dispersion(kpoints = 101, k_start = -2*pi, k_end = 2*pi, to_display = 1)


    (Hamt,vals,vecs) = F1crys.form_specified_unit_cell(n_units = 2,PBC=0, eig_spectra = 1, eig_vectors = 1 )
#print(Hamt)

    Hamt_C = Qobj( [[ 0., -1.,  0.,  0.],
                    [-1.,  0., -1.,  0.],
                    [ 0., -1.,  0., -1.],
                    [ 0.,  0., -1.,  0.]] )

#    assert_== Hamt_C
    assert_(Hamt == 2*Hamt_C)


