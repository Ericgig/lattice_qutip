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

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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


class Qbasis(Qobj):
    """
    A subclass of qutip.qobj that stores all the attributes of a unit cell.
    """
    def __init__(self, onsite_energy_array = [], position_array = [],
                 intra_hopping_array=[], dimensions=None, number_of_orbitals=None):

        self._dimensions = dimensions                         # the number of unit vectors for 
                                                              # representing the orbitals in the
                                                              # unit cell
        self._number_of_orbitals = number_of_orbitals

        self._onsite_energy_array = onsite_energy_array
        self._position_array = position_array
        self._intra_hopping_array = intra_hopping_array
        self._number_of_intra_hopping = np.shape(intra_hopping_array)[0]
        self._checks()
        
    def Add_orbital(self, onsite_energy, position):
        self._onsite_energy_array.append(onsite_energy)
        self._position_array.append(position)
        self._number_of_orbitals = self._number_of_orbitals + 1
        self._checks()

    def Add_intra_hopping(self, intra_hopping):
        self._intra_hopping_array.append(intra_hopping)
        self._checks()

    def _checks(self):
        A = 1
        return A        

    def basis_Hamiltonian(self):
        data = np.zeros( (self._number_of_orbitals,self._number_of_orbitals),dtype=complex)
        for i in range(self._number_of_orbitals):
            data[i,i] = self._onsite_energy_array[i]

        
        for i in range(np.shape(self._intra_hopping_array)[0]):
            data[self._intra_hopping_array[i][0],self._intra_hopping_array[i][1]] = self._intra_hopping_array[i][2]
            data[self._intra_hopping_array[i][1],self._intra_hopping_array[i][0]] = np.conj(self._intra_hopping_array[i][2] )     

        csr_data = csr_matrix(data, dtype=complex)      
        basis_Hamiltonian = Qobj(csr_data)        
    
#        return basis_Hamiltonian
        return csr_data
        
    
    def input_super_unit_cell(self, super_unit_cell):
        print(super_unit_cell)

        # to be generalized
        dimensions=1    
        number_of_orbitals=2

        if (super_unit_cell[0,0] != super_unit_cell[2,2]) :
            raise Exception('Inconsistent on-site energies for the first atom in the two unit cells!')        
        
        if (super_unit_cell[1,1] != super_unit_cell[3,3]) :
            raise Exception('Inconsistent on-site energies for the second atom in the two unit cells!')        

        if (super_unit_cell[0,1] != super_unit_cell[2,3]) :
            raise Exception('Inconsistent inter hopping terms for the two unit cells!')        

        if ( not isherm(super_unit_cell) ) :
            raise Exception('super_unit_cell needs to be Hermitian.')        
 
        
        eps0 = super_unit_cell[0,0];           eps1 = super_unit_cell[1,1]
        onsite_energy_array = [eps0, eps1]

        t = super_unit_cell[0,1]; tp = super_unit_cell[1,2];
        
        pos0 = 0; pos1 = 0.5;
        position_array = [pos0, pos1]

        intra_hopping_array=[[0,1,t]]
    
        self._dimensions = dimensions
        self._number_of_orbitals = number_of_orbitals

        self._onsite_energy_array = onsite_energy_array
        self._position_array = position_array
        self._intra_hopping_array = intra_hopping_array
        self._number_of_intra_hopping = np.shape(intra_hopping_array)[0]
        self._checks()    
    
    
    def display_model(self):
        print("In the unit cell:  ")
        print("Number of orbitals: ", self._number_of_orbitals)
        print("On-site energies of the orbitals:  ", self._onsite_energy_array)
        print("Vectors representing the positions of the orbitals:  ", self._position_array)
        print("The hoppings within the unit cell: ", self._intra_hopping_array)
        print("The minimum dimension: ", self._dimensions)
    
    
    


class Qcrystal(Qobj):
    """
    A subclass of Qbasis that contains all the information of a Bravais lattice vectors
    and inter-hopping between unit cells to define the entire crystal
    """
    def __init__(self, Qbasis, inter_hopping_array=[], basis_vector_array = [],
        periodic_dimensions = None, width_1 = None, width_2 = None ):

        self._Qbasis = Qbasis
        self._basis_vector_array = basis_vector_array
        self._inter_hopping_array = inter_hopping_array
        self._width_1 = width_1       # number of unit cells in aperiodic dimension 1
        self._width_2 = width_2       # number of unit cells in aperiodic dimension 2        
        self._periodic_dimensions = periodic_dimensions



    def input_super_unit_cell(self, super_unit_cell):
        print(super_unit_cell)

        dimensions=1
        number_of_orbitals=2

        eps0 = super_unit_cell[0,0];           eps1 = super_unit_cell[1,1]
        onsite_energy_array = [eps0, eps1]

        t = super_unit_cell[0,1]; tp = super_unit_cell[1,2];
        
        pos0 = 0; pos1 = 0.5;
        position_array = [pos0, pos1]
        intra_hopping_array=[[0,1,t]]
        basis_vector_array = [[1]]
        #basis_vector_array = [[]]
        inter_hopping_array=[[0,1,tp]]

    
        self._Qbasis._dimensions = dimensions
        self._Qbasis._number_of_orbitals = number_of_orbitals

        self._Qbasis._onsite_energy_array = onsite_energy_array
        self._Qbasis._position_array = position_array
        self._Qbasis._intra_hopping_array = intra_hopping_array
        self._Qbasis._number_of_intra_hopping = np.shape(intra_hopping_array)[0]
        self._Qbasis._checks()    
        self._basis_vector_array = basis_vector_array
        self._inter_hopping_array = inter_hopping_array

    def Add_basis_vector(self, basis_vector):
        self._basis_vector_array.append(basis_vector)
        self._checks()


    def _checks(self, other, op):
        if (basis_vector_array == [[]]) :
            raise Exception('basis_vector_array can not be null!')

        
        if periodic_dimensions == None:
            self._periodic_dimensions = np.shape(self._basis_vector_array)[1]           
        else:
            self._periodic_dimensions = periodic_dimensions

        A = 1
        return A

    def display_model(self):
        print("In the unit cell:  ")
        print("Number of orbitals: ", self._Qbasis._number_of_orbitals)
        print("On-site energies of the orbitals:  ", self._Qbasis._onsite_energy_array)
        print("Vectors representing the positions of the orbitals:  ", self._Qbasis._position_array)
        print("The hoppings within the unit cell: ", self._Qbasis._intra_hopping_array)
        print("The minimum dimension: ", self._Qbasis._dimensions)

        print("The basis vector array: ", self._basis_vector_array)
        print("The inter hopping array: ",self._inter_hopping_array)
        print("The periodic dimensions: ", self._periodic_dimensions)
        if self._width_1 != None:
            print("width_1: ", self._width_1)
        if self._width_2 != None:
            print("width_2: ", self._width_2)
    
    
    def Add_inter_hopping(self, inter_hopping):
        self._inter_hopping_array.append(inter_hopping)
        self._checks()



    def _diag_a_matrix(self,H_k, calc_evecs = False):
        if np.max(H_k-H_k.T.conj())>1.0E-20:
            raise Exception("\n\nThe Hamiltonian matrix is not hermitian?!")
        #solve matrix
        if calc_evecs == False: # calculate only the eigenvalues
            vals=np.linalg.eigvalsh(H_k.todense())
            # sort eigenvalues and convert to real numbers
#            eval=_nicefy_eig(eval)
            return np.array(vals,dtype=float)
        else: # find eigenvalues and eigenvectors
            (vals, vecs)=np.linalg.eigh(H_k.todense())
            # transpose matrix eig since otherwise it is confusing
            # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
            vecs=vecs.T        
#            (eval,eig)=_nicefy_eig(eval,eig)
            return (vals,vecs)
        


    def dispersion(self, to_display = 0 , kdim1 = [] , kdim2 = []):

                
        if (  self._periodic_dimensions == 1  ) :
            k_start = kdim1[0]; k_end = kdim1[1]; kpoints = kdim1[2]
            (kxA,val_ks) = self._dispersion_1d(kpoints, k_start, k_end, to_display)        
            return (kxA,val_ks)


        if (  self._periodic_dimensions == 2  ) :
            k1_start = kdim1[0]; k1_end = kdim1[1]; k1points = kdim1[2]
            k2_start = kdim2[0]; k2_end = kdim2[1]; k2points = kdim2[2]                        
            (kxA,kyA,val_ks) = self._dispersion_2d(k1points, k1_start, k1_end, k2points, k2_start, k2_end, to_display)        
            return (kxA,kyA,val_ks)
    
        
        
        
        
    def _dispersion_1d(self, kpoints = 51, k_start = -pi, k_end = pi, to_display = 0):
        
        if (kpoints % 2 == 0 or (not isinstance(kpoints, int))  ) :
            raise Exception("\n\nPlease choose an odd integer for kpoints!")        
        
        Odat = np.zeros( (self._Qbasis._number_of_orbitals,self._Qbasis._number_of_orbitals),dtype=complex)
        val_ks=np.zeros((self._Qbasis._number_of_orbitals,kpoints),dtype=float)
        kxA = np.zeros((kpoints,1),dtype=float)
        G0_H = self._Qbasis.basis_Hamiltonian()
        
        k_1 = (kpoints-1)
        for ks in range(kpoints):
            kx = k_start + (ks*(k_end-k_start)/k_1)
            kxA[ks,0] = kx
            for i in range(  len(self._inter_hopping_array)  ):
                Odat[self._inter_hopping_array[i][0],self._inter_hopping_array[i][1]] = self._inter_hopping_array[i][2] * exp(complex(0,kx))
                Odat[self._inter_hopping_array[i][1],self._inter_hopping_array[i][0]] = np.conj(self._inter_hopping_array[i][2] )* exp(complex(0,-kx))     

            Of_d = csr_matrix(Odat, dtype=complex)
            H_k = G0_H+Of_d

#            (vals, vecs) = self._diag_a_matrix(H_k, calc_evecs = True)
            vals = self._diag_a_matrix(H_k, calc_evecs = False)                
                
            val_ks[:,ks] = vals[:]

        if (to_display == 1) :
            fig, ax = subplots()
            ax.plot(kxA/pi, val_ks[0,:]);
            ax.plot(kxA/pi, val_ks[1,:]);        
            ax.set_ylabel('Energy');
            ax.set_xlabel('k_x(pi)');
            show(fig)
            fig.savefig('./Dispersion.pdf')

        return (kxA,val_ks)
    
    
        
    def _dispersion_2d(self, k1points=51, k1_start=-pi, k1_end=pi, k2points=51, k2_start=-pi, k2_end=pi, to_display=0):
        
        if (k1points % 2 == 0 or (not isinstance(k1points, int))  ) :
            raise Exception("\n\nPlease choose an odd integer for k1points!")        
        
        if (k2points % 2 == 0 or (not isinstance(k2points, int))  ) :
            raise Exception("\n\nPlease choose an odd integer for k2points!")        
        
        
        

        val_ks=np.zeros((self._Qbasis._number_of_orbitals,k1points,k2points),dtype=float)
        kxA = np.zeros((k1points,1),dtype=float)
        kyA = np.zeros((k2points,1),dtype=float)
        
        G0_H = self._Qbasis.basis_Hamiltonian()
        print(G0_H)
        
        
        k_1 = (k1points-1);  k_2 = (k2points-1); 
        for ks in range(k1points):
            kx = k1_start + (ks*(k1_end-k1_start)/k_1)
            for kt in range(k2points):            
                ky = k2_start + (kt*(k2_end-k2_start)/k_2)
            
#                print(kx,ky)            
                Odat = np.zeros( (self._Qbasis._number_of_orbitals,self._Qbasis._number_of_orbitals),dtype=complex)
                kxA[ks,0] = kx;  kyA[kt,0] = ky
                for i in range(  len(self._inter_hopping_array)  ):                    
                    kx_C = self._inter_hopping_array[i][3][0]*self._basis_vector_array[0][0]  +  self._inter_hopping_array[i][3][1]*self._basis_vector_array[1][0]
                    ky_C = self._inter_hopping_array[i][3][0]*self._basis_vector_array[0][1]  +  self._inter_hopping_array[i][3][1]*self._basis_vector_array[1][1]                

                    Odat[self._inter_hopping_array[i][0],self._inter_hopping_array[i][1]] = Odat[self._inter_hopping_array[i][0],self._inter_hopping_array[i][1]] + self._inter_hopping_array[i][2] * exp(complex(0, (kx* kx_C + ky* ky_C)       ))
                    Odat[self._inter_hopping_array[i][1],self._inter_hopping_array[i][0]] = Odat[self._inter_hopping_array[i][1],self._inter_hopping_array[i][0]] + np.conj(self._inter_hopping_array[i][2] )* exp(complex(0, -(kx* kx_C + ky* ky_C)    ))     




                Of_d = csr_matrix(Odat, dtype=complex)
                H_k = G0_H+Of_d

#               (vals, vecs) = self._diag_a_matrix(H_k, calc_evecs = True)
                vals = self._diag_a_matrix(H_k, calc_evecs = False)                
                
                val_ks[:,ks,kt] = vals[:]



        if (to_display == 1) :
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
#            x = y = np.arange(-3.0, 3.0, 0.05)
#            X, Y = np.meshgrid(x, y)
#            zs = np.array(fun(np.ravel(X), np.ravel(Y)))
#            Z = zs.reshape(X.shape)

            A_kX, A_kY = np.meshgrid(kxA, kyA)
            ax.plot_surface(A_kX, A_kY, val_ks[0,:,:])
            ax.plot_surface(A_kX, A_kY, val_ks[1,:,:])
            ax.set_xlabel('kx')
            ax.set_ylabel('ky')
            ax.set_zlabel('E(kx,ky)')

            plt.show()




        return (kxA,kyA,val_ks)
        
    
    

    def form_specified_unit_cell(self, n_units = 1, PBC = 0, eig_spectra = 0, eig_vectors = 0 ):

        if ( (PBC != 0 ) and (PBC != 1)  ) :
            raise Exception("\n\nPBC can only be 0 or 1")        

        
        if ( (eig_spectra == 0 ) and (eig_vectors == 1)  ) :
            raise Exception("\n\nFor eig_vectors = 1, you must choose eig_spectra = 1")        
         
            
        H_base = self._Qbasis.basis_Hamiltonian().todense()
#        return Unit_Hamiltonian0
        T_coup = np.zeros( (self._Qbasis._number_of_orbitals,self._Qbasis._number_of_orbitals),dtype=complex)


        for i in range(   len(self._inter_hopping_array)   ):
            if (self._inter_hopping_array[i][0] < self._inter_hopping_array[i][1] ):
                T_coup[self._inter_hopping_array[i][1],self._inter_hopping_array[i][0]] = self._inter_hopping_array[i][2] 
            else:                
                T_coup[self._inter_hopping_array[i][0],self._inter_hopping_array[i][1]] = np.conj(self._inter_hopping_array[i][2] )

        T_coup_dag = np.conj( T_coup.T )


        if (PBC == 1):
            Row_0 = np.hstack((  H_base, T_coup , np.zeros( (self._Qbasis._number_of_orbitals,(n_units - i-3)*2 ),dtype=complex), T_coup_dag  ))  
        else:                    
            Row_0 = np.hstack((  H_base, T_coup , np.zeros( (self._Qbasis._number_of_orbitals,(n_units - i-2)*2 ),dtype=complex)  ))      
           
        Hamt = Row_0

        for i in range(1,n_units):

            if (i == (n_units-1) ):
                if (PBC == 1):
                    Row_i = np.hstack(( T_coup, np.zeros( (self._Qbasis._number_of_orbitals,(i-2)*2 ),dtype=complex) , T_coup_dag, H_base ))  
                else:                    
                    Row_i = np.hstack(( np.zeros( (self._Qbasis._number_of_orbitals,(i-1)*2 ),dtype=complex) , T_coup_dag, H_base ))  
  
            else:
                Row_i = np.hstack((  np.zeros( (self._Qbasis._number_of_orbitals,(i-1)*2 ),dtype=complex) , T_coup_dag, H_base, T_coup , np.zeros( (self._Qbasis._number_of_orbitals,(n_units - i-2)*2 ),dtype=complex)  )) 
                

            Hamt = np.vstack(( Hamt, Row_i ))                




        if ( eig_spectra == 1 and  eig_vectors == 0 ):
            vals=np.linalg.eigvalsh(Hamt)


        if ( eig_spectra == 1 and  eig_vectors == 1 ):
            (vals, vecs)=np.linalg.eigh(Hamt)

        k1_start = -pi; k1_end = pi; kpoints1 = 51;
        kdim1 = [k1_start,k1_end,kpoints1]
        
        to_display = 1;
        (kxA,val_ks) = self.dispersion(to_display, kdim1 )
    
        ind_e1 = np.arange(0, 1, 2/2/n_units)
        ind_e2 = np.arange(-1, 0, 2/2/n_units)
        ind_e = np.hstack((ind_e1, ind_e2))
    
    
    
        fig, ax = subplots()
        ax.plot(kxA/pi, val_ks[0,:]);
        ax.plot(kxA/pi, val_ks[1,:]);
        if (eig_spectra == 1):
#            ax.plot(ind_e, vals);  
            ax.scatter(ind_e, vals);
        ax.set_ylabel('Energy');
        ax.set_xlabel('k_x(pi)');
        show(fig)
        fig.savefig('./comparison.pdf')
            
        
        
        
        
    
        if ( eig_spectra == 1 and  eig_vectors == 0 ):
            return (Qobj(Hamt),vals)

        elif ( eig_spectra == 1 and  eig_vectors == 1 ):
            return (Qobj(Hamt),vals,vecs)
        
        else:
            return Qobj(Hamt)
    
    
#        Row0 = np.hstack((H_base,T_coup))

#        print(Row0)
#        self._width_1 = n*2




    def dispersion_on_path(self, other):
        print("dem eem")


