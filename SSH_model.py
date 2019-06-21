#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 22:54:04 2019

@author: ssaumya7
"""
from qutip import *
from numpy import *
from matplotlib.pyplot import *
import numpy as np
############################## SSH model ######################################
########-C=C=C-C=C-C=C=C-C=C-C=C=C-C=C-C=C=C-C=C-C=C=C-C=C-C=C=C-C=C###########

eps0 = 0.0; eps1 = 0.0;
t = -0.5; tp = -1.0;
x_vec = 1;
pos0 = [0.0]; pos1 = [0.5];

c0 = basis(2,0)
c1 = basis(2,1)
super_unit_cell = t*tensor( c0*c0.dag(), sigmax() ) + t*tensor( c1*c1.dag(), sigmax() ) + tp* ( tensor( c1*c0.dag(), (sigmax()+ 1j*sigmay())/2 ) + tensor( c1*c0.dag(), (sigmax()+ 1j*sigmay())/2 ).dag() ) 

F1 = Qbasis()
F1.input_super_unit_cell(super_unit_cell)
#F1.display_model()


F1crys = Qcrystal(F1)
F1crys.input_super_unit_cell(super_unit_cell)
F1crys.dispersion(kpoints = 101, k_start = -2*pi, k_end = 2*pi, to_display = 1)
(Hamt,vals,vecs) = F1crys.form_specified_unit_cell(n_units = 50,PBC=0, eig_spectra = 1, eig_vectors = 1 )

xA = np.arange(0,100,1)

fig, ax = subplots();ax.plot(xA, vecs[:,49]);
ax.set_ylabel('Amplitude');
ax.set_xlabel('position');
show(fig);
fig.savefig('./Edge_states.pdf')
 


