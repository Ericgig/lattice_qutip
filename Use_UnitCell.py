#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:51:37 2019

@author: ssaumya7
"""
from qutip import *
from numpy import *
import numpy as np
#from qutip.latticeclass import UnitCell

############################## SSH model ######################################
########-C=C=C-C=C-C=C=C-C=C-C=C=C-C=C-C=C=C-C=C-C=C=C-C=C-C=C=C-C=C###########

eps0 = 0.0; eps1 = 0.0;
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
print(Hamt)




