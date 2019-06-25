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
    A subclass of qutip.qobj that stores all the attributes of the basis of a
    crystal.

    Example
    -------
    >>> from qutip.latticeclass import *
    >>> eps0 = 0.0; eps1 = 0.0; t = -1.0; 
    >>> pos0 = [0.0,0,0]; pos1 = [2/3,-1/3];  
    >>> dimensions=2; number_of_orbitals=2; onsite_energy_array = [eps0, eps1]
    >>> position_array = [pos0, pos1]; intra_hopping_array=[[0,1,t]]
    >>> F1 = Qbasis(onsite_energy_array, position_array ,intra_hopping_array, dimensions, number_of_orbitals)
    >>> F1.display_model();
    
    Parameters
    ----------
    dimensions: int
        The dimensionality of the basis.
        default: None
        
    number_of_orbitals: int 
        The number of orbitals in the basis.
        default: None
        
    onsite_energy_array: numpy list, dtype = complex
        List of complex numbers quantifying the onsite energies of the orbitals. 
        default: []
        
    position_array: numpy list, dtype = complex
        List of complex numbers quantifying the positions of the orbitals. 
        default: []
        
    intra_hopping_array: numpy list of tuples: (int,int,complex)
        All the intra hoppings in the format [(int, int, complex) ]
        default: []

    number_of_intra_hopping: int
        The number of intra hoppping tuples in the intra_hopping_array.
        default: 0
        
    is_complete: bool
        Indicates if the Qbasis is correctly and completely defined.
        default: False

    Attributes
    ----------
    dimensions: int
        The dimensionality of the basis.
        default: None
        
    number_of_orbitals: int 
        The number of orbitals in the basis.
        default: None
        
    onsite_energy_array: numpy list, dtype = complex
        List of complex numbers quantifying the onsite energies of the orbitals. 
        default: []
        
    position_array: numpy list, dtype = complex
        List of complex numbers quantifying the positions of the orbitals. 
        default: []
        
    intra_hopping_array: numpy list of tuples: (int,int,complex)
        All the intra hoppings in the format [(int, int, complex) ]
        default: []

    number_of_intra_hopping: int
        The number of intra hoppping tuples in the intra_hopping_array.
        default: 0
        
    is_complete: bool
        Indicates if the Qbasis is correctly and completely defined.
        default: False    
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
        self._is_complete = self._checks(check_if_complete = True)
        
    def Add_orbital(self, onsite_energies, positions):
        """
        Adds a/multiple orbitals to the Qbasis. The onsite energy and position
        of the orbital is recorded.
        
        Parameters
        ==========
        onsite_energies : numpy list 
            The list of onsite energies of the orbitals to be added.
            
        positions: numpy list
            The list of positions of the orbitals to be added.
        """            
        self._onsite_energy_array.append(onsite_energies)
        self._position_array.append(positions)
        self._number_of_orbitals = self._number_of_orbitals + len(onsite_energy)
        self._checks()

    def Add_intra_hopping(self, intra_hopping):
        """
        Adds an intra hopping energy term between two orbitals already in the 
        Qbasis.
        
        Parameters
        ==========
        intra_hopping : numpy list of tuples
            The list in the format: [tuple0, tuple1, ...]
            Each tuple in the format: (orbital_A, orbital_B, intra_hopping_AB)            
        """                  
        self._intra_hopping_array.append(intra_hopping)
        self._number_of_intra_hopping = np.shape(self._intra_hopping_array)[0]        
        self._checks()

    def _checks(self,check_if_complete = False):
        """
        A checking code that confirms all the entries in the Qbasis is 
        consistent (code to be written).  The codenames indicate the last 
        action, that we would verify the correctness of. If it is called with
        check_if_complete == True, it returns if the Qbasis instance is 
        complete or not.
        """                 
        A = True
        return A        

    def basis_Hamiltonian(self, As_csr = False):
        """Returns the Hamiltonian for the basis (with no other basis units or 
        basis hopping considered)

		.. math::
			\begin{equation}\label{eq:basis_Hamiltonian}
            \begin{bmatrix}
            \epsilon_0  & t^i_{0,1} & t^i_{0,2} &  ..& t^i_{0,n}  \\
            t^i_{1,0} & \epsilon_1  & t^i_{1,2} &  ..& t^i_{1,n}  \\
            t^i_{2,0} & t^i_{2,1}   & \epsilon_2 & ..& t^i_{2,n}  \\
            ...... &   ......... &  ......... & ..& :  \\
            t^i_{n,0} & t^i_{n,1} & t^i_{n,1} & ..& \epsilon_n
            \end{bmatrix}
            \end{equation}

		**Note:** \epsilon_i is the onsite energy of orbital_i and t^i_{a,b}
        is the hopping between orbital_a and orbital_b. 
        
        Returns
        -------
        basis_Hamiltonian : :class:`qutip.Qobj`
            The Hamiltonian matrix of the basis unit cell, formed in the 
            creation-annihilation operator basis.
        """           
        data = np.zeros( (self._number_of_orbitals,self._number_of_orbitals),dtype=complex)
        for i in range(self._number_of_orbitals):
            data[i,i] = self._onsite_energy_array[i]

        
        for i in range(np.shape(self._intra_hopping_array)[0]):
            data[self._intra_hopping_array[i][0],self._intra_hopping_array[i][1]] = self._intra_hopping_array[i][2]
            data[self._intra_hopping_array[i][1],self._intra_hopping_array[i][0]] = np.conj(self._intra_hopping_array[i][2] )     

        csr_data = csr_matrix(data, dtype=complex)      
        basis_Hamiltonian = Qobj(csr_data)        
    
        if As_csr == False:
            return basis_Hamiltonian
        else:
            return csr_data
        
    
    def input_super_unit_cell(self, super_unit_cell):
        """A covenient way(alternate of __init__) of completely specifyig an 
        instance of Qbasis with a 'super_unit_cell'! So far, this alternate 
        method only works for one dimension. A 1 is returned if the Qbasis is
        completely defined, or 0 otherwise. For 1d, the super_unit_cell is 
        a qutip.Qobj that is the Hamiltonian of two unit cells. This way Qbasis
        gets the information of a unit cell as well as the inter hopping terms
        between unit cells. The Qcrystal.input_super_unit_cell() can completely
        define a Qcrystal without defining an instance of Qbasis first.

        Returns
        -------
        comp : int
            comp = 1 indicates completion of the definition of the instance of
            Qbasis.
        """           
        print('the input super_unit_cell in input_super_unit_cell(): ',super_unit_cell)

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
        
        comp = 1        #the Qbasis is complete
#        return comp    #whether the Qbasis is complete
    
    
    def display_model(self):
        """Displays all the values of the parameters of the instance of the 
        Qbasis.
        """           
        print("In the unit cell:  ")
        print("Number of orbitals: ", self._number_of_orbitals)
        print("On-site energies of the orbitals:  ", self._onsite_energy_array)
        print("Vectors representing the positions of the orbitals:  ", self._position_array)
        print("The hoppings within the unit cell: ", self._intra_hopping_array)
        print("The minimum dimension: ", self._dimensions)
    
    
    
    
class Qcrystal(Qobj):
    """
    A subclass of Qbasis that contains the complete definition of a particular 
    Bravais lattice. Translation vectors and inter-hopping between unit cells 
    define the entire crystal.
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
        """A covenient way(alternate of __init__) of completely specifyig an 
        instance of Qcrystal with a 'super_unit_cell'! So far, this alternate 
        method only works for one dimension. A 1 is returned if the Qbasis is
        completely defined, or 0 otherwise. For 1d, the super_unit_cell is 
        a qutip.Qobj that is the Hamiltonian of two unit cells. This way Qbasis
        gets the information of a unit cell as well as the inter hopping terms
        between unit cells. 
        
        Returns
        -------
        comp : int
            comp = 1 indicates completion of the definition of the instance of
            Qbasis.
        """  
        dimensions=1
        number_of_orbitals=2

        eps0 = super_unit_cell[0,0];           eps1 = super_unit_cell[1,1]
        onsite_energy_array = [eps0, eps1]

        t = super_unit_cell[0,1]; tp = super_unit_cell[1,2];
        
        pos0 = 0; pos1 = 0.5;
        position_array = [pos0, pos1]
        intra_hopping_array=[(0,1,t)]
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
        self._periodic_dimensions = np.shape(self._basis_vector_array)[1] 


    def Add_basis_vector(self, basis_vector):
        """Adds basis vectors to the existing self._basis_vector_array.        
        """          
        self._basis_vector_array.append(basis_vector)
        self._checks()

    def _checks(self,check_if_complete = False):
        """
        A checking code that confirms all the entries in the Qcrystal is 
        consistent (code to be written).  The codenames indicate the last 
        action, that we would verify the correctness of. If it is called with
        check_if_complete == True, it returns if the Qbasis instance is 
        complete or not.

        Returns
        -------
        A : bool
            A = True indicates correct,consistent entries/completion of the 
            definition of the instance of Qbasis.        
        """                         
        
        if (basis_vector_array == [[]]) :
            raise Exception('basis_vector_array can not be null!')
        
        if periodic_dimensions == None:
            self._periodic_dimensions = np.shape(self._basis_vector_array)[1]           
        else:
            self._periodic_dimensions = periodic_dimensions        
        
        A = True
        return A    

    def display_model(self):
        """Displays all the values of the parameters of the instance of the 
        Qbasis.
        """             
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
        """
        Adds an inter hopping energy term between two orbitals in two different 
        basis unit cells.
        
        Parameters
        ==========
        intra_hopping : numpy list of tuples
            Each tuple in the format:
            (orbital_A_in_base_unit_cell, orbital_B_in_neighbor_unit_cell, inter_hopping_AB, vector_from_cell_A_to_B)            
            vector_from_cell_A_to_B is a tuple of the coefficients of primitive translation vectors in self._basis_vector_array
        """          
        self._inter_hopping_array.append(inter_hopping)
        self._checks()

    def _EigVector_At_K(self, k_vector_array = []):
        """
        Calculates the entire set of eigen-vectors at k-vectors(d-dimensional) specified at
        the k_vector_array inputs. d is the dimensionality of Qcrystal.
        
        Parameters
        ==========
        k_vector_array : numpy array
            The array of k-vectors of the k-space given as input.
            
        Returns
        -------
        vecs_ks : numpy ndarray
            vecs_ks[kvector,:,:] = [band index,band eigen-vectors]                        
        """            
        Odat = np.zeros( (self._Qbasis._number_of_orbitals,self._Qbasis._number_of_orbitals),dtype=complex)
        G0_H = self._Qbasis.basis_Hamiltonian(As_csr = True)
        
        kxA = k_vector_array[0]; kyA = k_vector_array[1]
        k1points = np.shape(kxA)[0];  k2points = np.shape(kyA)[1] 
        vecs_ks=np.zeros((k1points,k2points,self._Qbasis._number_of_orbitals,self._Qbasis._number_of_orbitals),dtype=complex)
        
        for ks in range(k1points):
            for kt in range(k2points):            

                kx = kxA[ks,kt];  ky = kyA[ks,kt];
                Odat = np.zeros( (self._Qbasis._number_of_orbitals,self._Qbasis._number_of_orbitals),dtype=complex)

                for i in range(  len(self._inter_hopping_array)  ):                    
                    kx_C = self._inter_hopping_array[i][3][0]*self._basis_vector_array[0][0]  +  self._inter_hopping_array[i][3][1]*self._basis_vector_array[1][0]
                    ky_C = self._inter_hopping_array[i][3][0]*self._basis_vector_array[0][1]  +  self._inter_hopping_array[i][3][1]*self._basis_vector_array[1][1]                

                    Odat[self._inter_hopping_array[i][0],self._inter_hopping_array[i][1]] = Odat[self._inter_hopping_array[i][0],self._inter_hopping_array[i][1]] + self._inter_hopping_array[i][2] * exp(complex(0, (kx* kx_C + ky* ky_C)       ))
                    Odat[self._inter_hopping_array[i][1],self._inter_hopping_array[i][0]] = Odat[self._inter_hopping_array[i][1],self._inter_hopping_array[i][0]] + np.conj(self._inter_hopping_array[i][2] )* exp(complex(0, -(kx* kx_C + ky* ky_C)    ))     

                Of_d = csr_matrix(Odat, dtype=complex)
                H_k = G0_H+Of_d

                (vals, vecs) = self._diag_a_matrix(H_k, calc_evecs = True)
                vecs_ks[ks,kt,:,:] = vecs[:,:]

        return vecs_ks

    def _diag_a_matrix(self,H_k, calc_evecs = False):
        """
        Returns eigen-values and/or eigen-vectors of an input matrix.
        
        Parameters
        ==========
        H_k : numpy matrix
            The matrix to be diagonalized
            
        Returns
        -------
        vecs : numpy ndarray
            vecs[:,:] = [band index,:] = eigen-vector of band_index
        vals : numpy ndarray
            The diagonalized matrix                        
        """         
        
        if np.max(H_k-H_k.T.conj())>1.0E-20:
            raise Exception("\n\nThe Hamiltonian matrix is not hermitian?!")

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


    def dispersion(self, to_display = 0 , kdim1 = [] , kdim2 = [], kdim3 = []):
        """
        Calculates the entire set of eigen-values at the entire k-space formed 
        by the inputs kdim1, kdim2, ....
        
        Parameters
        ==========
        to_display : int
            Plots the dispersion relation if set to 1.        
        
        kdimi : numpy array
            In the format: [starting k, ending k, the number of sampling k points]
            
        Returns
        -------
        kxA : numpy array
            array of k-points on the first k-dimension at which bands were calculated.

        kyA : numpy array
            array of k-points on the second k-dimension at which bands were calculated.
            
        val_ks : numpy ndarray
            vecs_ks[band index] = band eigen-value                        
        """ 
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
        """
        Calculates the dispersion for a 1-dimensional crystal
        """                 
        if (kpoints % 2 == 0 or (not isinstance(kpoints, int))  ) :
            raise Exception("\n\nPlease choose an odd integer for kpoints!")        
        
        Odat = np.zeros( (self._Qbasis._number_of_orbitals,self._Qbasis._number_of_orbitals),dtype=complex)
        val_ks=np.zeros((self._Qbasis._number_of_orbitals,kpoints),dtype=float)
        kxA = np.zeros((kpoints,1),dtype=float)
        G0_H = self._Qbasis.basis_Hamiltonian(As_csr = True)
        
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
        """
        Calculates the dispersion for a 2-dimensional crystal
        """                  
        if (k1points % 2 == 0 or (not isinstance(k1points, int))  ) :
            raise Exception("\n\nPlease choose an odd integer for k1points!")        
        
        if (k2points % 2 == 0 or (not isinstance(k2points, int))  ) :
            raise Exception("\n\nPlease choose an odd integer for k2points!")        
  
        val_ks=np.zeros((self._Qbasis._number_of_orbitals,k1points,k2points),dtype=float)
        kxA = np.zeros((k1points,1),dtype=float)
        kyA = np.zeros((k2points,1),dtype=float)
        
        G0_H = self._Qbasis.basis_Hamiltonian(As_csr = True)       
        
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
            fig.savefig('./Dispersion.png')

        return (kxA,kyA,val_ks)
    

    def space_Hamiltonian(self, n_units = 1, PBC = 0, eig_spectra = 0, eig_vectors = 0 ):
        """
        Forms the real space Hamiltonian of specified length and dimensionality
        and periodic boundary condition.
        
        Parameters
        ==========
        n_units : int
            The length of the crystal in units of unit cell lengths.        
        
        PBC : int
            Periodic boundary codition enforced in the direction if set to 1.
            
        Returns
        -------
        Hamt : :class:`qutip.Qobj`
            The Hamiltonian matrix of specified size and boundary conditions, 
            formed in the creation-annihilation operator basis.
            
        vals : numpy array
            Diagonalized Hamt, returned if eig_spectra == 1
            
        vecs : numpy array
            vecs[band index] = band eigen-value
            vecs is returned if eig_vectors == 1                        
        """ 
        if ( (PBC != 0 ) and (PBC != 1)  ) :
            raise Exception("\n\nPBC can only be 0 or 1")        

        if ( (eig_spectra == 0 ) and (eig_vectors == 1)  ) :
            raise Exception("\n\nFor eig_vectors = 1, you must choose eig_spectra = 1")        
            
        H_base = self._Qbasis.basis_Hamiltonian(As_csr = True).todense()
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
    
    
    def dispersion_on_path(self, kpath, inter_pts):
        # to be written
        print("Bands on a path")


    def _one_flux_plane(self, wfs2d):
        "Compute fluxes on a two-dimensional plane of states."
        # size of the mesh
        nk0=wfs2d.shape[0]
        nk1=wfs2d.shape[1]
        # number of bands (will compute flux of all bands taken together)
        nbnd=wfs2d.shape[2]
    
        # here store flux through each plaquette of the mesh
        all_phases=np.zeros((nk0-1,nk1-1),dtype=float)
    
        # go over all plaquettes
        for i in range(nk0-1):
            for j in range(nk1-1):
                # generate a small loop made out of four pieces
                wf_use=[]
                wf_use.append(wfs2d[i,j])
                wf_use.append(wfs2d[i+1,j])
                wf_use.append(wfs2d[i+1,j+1])
                wf_use.append(wfs2d[i,j+1])
                wf_use.append(wfs2d[i,j])
                wf_use=np.array(wf_use,dtype=complex)
                # calculate phase around one plaquette
                all_phases[i,j]=self._one_berry_loop(wf_use)

        return all_phases



    def _one_berry_loop(self, wf,berry_evals=False):
        """Do one Berry phase calculation (also returns a product of M
        matrices).  Always returns numbers between -pi and pi.  wf has
        format [kpnt,band,orbital,spin] and kpnt has to be one dimensional.
        Assumes that first and last k-point are the same. Therefore if
        there are n wavefunctions in total, will calculate phase along n-1
        links only!  If berry_evals is True then will compute phases for
        individual states, these corresponds to 1d hybrid Wannier
        function centers. Otherwise just return one number, Berry phase."""
        # number of occupied states
        nocc=wf.shape[1]
        # temporary matrices
        prd=np.identity(nocc,dtype=complex)
        ovr=np.zeros([nocc,nocc],dtype=complex)
        # go over all pairs of k-points, assuming that last point is overcounted!
        for i in range(wf.shape[0]-1):
            # generate overlap matrix, go over all bands
            for j in range(nocc):
                for k in range(nocc):
                    wf1 = wf[i,j,:]
                    wf2 = wf[i+1,k,:]
                    ovr[j,k] = np.dot(wf1.flatten().conjugate(),wf2.flatten() )                    

                # only find Berry phase
            if berry_evals==False:
                # multiply overlap matrices
                prd=np.dot(prd,ovr)
            # also find phases of individual eigenvalues
            else:
                # cleanup matrices with SVD then take product
                matU,sing,matV=np.linalg.svd(ovr)
                prd=np.dot(prd,np.dot(matU,matV))
            # calculate Berry phase
        if berry_evals==False:
            det=np.linalg.det(prd)
            pha=(-1.0)*np.angle(det)
            return pha
        # calculate phases of all eigenvalues
        else:
            evals=np.linalg.eigvals(prd)
            eval_pha=(-1.0)*np.angle(evals)
            # sort these numbers as well
            eval_pha=np.sort(eval_pha)
            return eval_pha



