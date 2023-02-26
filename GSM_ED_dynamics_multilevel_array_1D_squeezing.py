
#import libraries

import scipy
from scipy.integrate import solve_ivp
import numpy as np
import cmath as cm
import h5py
from numpy.linalg import multi_dot
from scipy.linalg import logm
from scipy.special import factorial
from scipy.special import *
from scipy import sparse
from scipy.sparse import csr_matrix
from numpy.linalg import eig
from scipy.linalg import eig as sceig
import math
import time

#from math import comb
from sympy.physics.quantum.cg import CG
from sympy import S
import collections

import numpy.polynomial.polynomial as poly

import h5py


# some definitions (do not change!)

e0 = np.array([0, 0, 1])
ex = np.array([1, 0, 0])
ey = np.array([0, 1, 0])
eplus = -(ex + 1j*ey)/np.sqrt(2)
eminus = (ex - 1j*ey)/np.sqrt(2)
single_decay = 1 # single atom decay rate


import sys
argv=sys.argv

if len(argv) < 2:
    #Default
    run_id=1
else:
    try:
        Natoms = int(argv[2])
        det_ind_input = int(argv[1])
        #det_val_input = float(argv[3])
        tfin_input = float(argv[3])
        ratio_id = int(argv[4])

    except:
        print ("Input error", flush=True)
        run_id=1

direc = '/data/rey/saag4275/data_files/'   # directory for saving data




# parameter setting box (change change)


ratio = np.round(0.1*ratio_id,2) # distance between atoms in units of lambda, transition/incident wavelength
r_axis = np.array([1, 0, 0]) # orientation of the distance between atoms, 3 vector
r_axis = r_axis/np.linalg.norm(r_axis)
eL = np.array([0, 0, 1]) # polarisation of laser, can be expressed in terms of the vectors defined above
fg = 1/2 # ang momentum of ground state manifold
fe = 1/2 # ang momentum of excited state manifold
rabi = 0.1*single_decay # laser intensity
detuning_list = np.linspace(-40,40,81) #np.array([det_val_input*single_decay]) # detuning of laser from transition
del_ze = 0.0 # magnetic field, i.e., Zeeman splitting of excited state manifold
del_zg = 0.0 # magnetic field, i.e., Zeeman splitting of ground state manifold

IC_chosen = 'equal_gs'

num_pts_dr = int(5*1e2)

t_initial_dr = 0.0
t_final_dr = tfin_input #1e3 #10*int(1.0/rabi**2)  #6*1e4#0.5*1e3#
t_range_dr = [t_initial_dr, t_final_dr]
t_vals_dr = np.concatenate((np.arange(0,10,1)/100.0,np.logspace(np.log10(.1),np.log10(t_final_dr),num_pts_dr-10)))#np.linspace(t_initial_dr, t_final_dr,int(1e3))  # int(6*1e4))#int(0.5*1e3))np.concatenate((np.linspace(t_initial_dr, t_mid_dr-10, 1000),np.linspace(t_mid_dr, t_final_dr, 1000)))
t_vals_dr[-1] = t_final_dr


e0_desired = eL

if IC_chosen!='':
    add_txt_in_params = '_'+IC_chosen + '_IC'
else:
    add_txt_in_params = ''

turn_off = [] 

for item in turn_off:
    add_txt_in_params += '_no_' + item




# more definitions and functions (do not change!)

wavelength = 1 # wavelength of incident laser
k0 = 2*np.pi/wavelength
kvec = k0*np.array([0, 1, 0]) # k vector of incident laser


rvecall = np.zeros((Natoms, 3)) # position of each atom
for ind in range(1, Natoms): # positions of atoms if there are more than 1
    rvecall[ind] = r_axis*ratio*ind
    
def rotation_matrix_a_to_b(va, vb): #only works upto 1e15-ish precision
    ua = va/np.linalg.norm(va)
    ub = vb/np.linalg.norm(vb)
    if np.dot(ua, ub) == 1:
        return np.identity(3)
    elif np.dot(ua, ub) == -1: #changing z->-z changes y->-y, thus preserving x->x, which is the array direction (doesn't really matter though!)
        return -np.identity(3)
    uv = np.cross(ua,ub)
    c = np.dot(ua,ub)
    v_mat = np.zeros((3,3))
    ux = np.array([1,0,0])
    uy = np.array([0,1,0])
    uz = np.array([0,0,1])
    v_mat[:,0] = np.cross(uv, ux)
    v_mat[:,1] = np.cross(uv, uy)
    v_mat[:,2] = np.cross(uv, uz)
    matrix = np.identity(3) + v_mat + (v_mat@v_mat)*1.0/(1.0+c)
    return matrix

 
if np.abs(np.conj(e0)@e0_desired) < 1.0:
    rmat = rotation_matrix_a_to_b(e0,e0_desired)
    eplus = rmat@eplus
    eminus = rmat@eminus
    ex = rmat@ex
    ey = rmat@ey
    e0 = e0_desired

print('kL = '+str(kvec/np.linalg.norm(kvec)), flush=True)
print('e0 = '+str(e0), flush=True)
print('ex = '+str(ex), flush=True)
print('ey = '+str(ey), flush=True)

HSsize = int(2*fg + 1 + 2*fe + 1) # Hilbert space size of each atom
HSsize_tot = int(HSsize**Natoms) # size of total Hilbert space


adde = fe
addg = fg

# polarisation basis vectors
evec = {0: e0, 1:eplus, -1: eminus}
evec = collections.defaultdict(lambda : [0,0,0], evec) 
   
def sort_lists_simultaneously_cols(a, b): #a -list to be sorted, b - 2d array whose columns are to be sorted according to indices of a
    inds = a.argsort()
    sortedb = b[:,inds]
    return sortedb



# function for calculating partial trace of a system of N atoms, in which N-1 atoms are traced over

def f_partial_trace(rho_input, N_input, nstates_input, not_traced_input):
    shape_tuple = []
    for i in range(0, 2*N_input):
        shape_tuple.append(nstates_input)
    rho_proc = np.reshape(rho_input, shape_tuple)
    RDM = rho_proc
    n_remain = N_input
    del_n = 0
    for iN in range(0, N_input):
        if iN != not_traced_input:
            ax1 = iN - del_n
            ax2 = iN - del_n + n_remain
            RDM = np.trace(RDM, axis1 = ax1, axis2 = ax2)
            n_remain -= 1
            del_n += 1
            
    new_shape = [nstates_input, nstates_input]
            
    return np.reshape(RDM, new_shape)

def f_partial_trace_arbitrary_atoms(rho_input, N_input, nstates_input, traced_input): #traced_input is a list of atom indices to be traced over
    shape_tuple = []
    for i in range(0, 2*N_input):
        shape_tuple.append(nstates_input)
    rho_proc = np.reshape(rho_input, shape_tuple)
    RDM = rho_proc
    n_remain = N_input
    del_n = 0
    for iN in range(0, N_input):
        if iN in traced_input:
            ax1 = iN - del_n
            ax2 = iN - del_n + n_remain
            RDM = np.trace(RDM, axis1 = ax1, axis2 = ax2)
            n_remain -= 1
            del_n += 1
    N_left = N_input - len(traced_input)
    new_shape = [nstates_input**N_left, nstates_input**N_left] # N_left in power respresents the atoms that did not get traced over
            
    return np.reshape(RDM, new_shape)

def f_partial_trace_except_two_atoms(rho_input, N_input, nstates_input, not_traced_input_1, not_traced_input_2):
    shape_tuple = [] # not_traced_input_i = atoms not traced over
    for i in range(0, 2*N_input): # N_input = no of atoms
        shape_tuple.append(nstates_input) # nstates_input = no of states in single atom Hilbert space
    rho_proc = np.reshape(rho_input, shape_tuple) # rho_proc ~ |i><j| \otimes |k><l| for two atoms ~ rho[i,k,j,l] (i, k - row indices, j, l - col indices)
    RDM = rho_proc 
    n_remain = N_input
    del_n = 0
    for iN in range(0, N_input):
        if ((iN != not_traced_input_1) and (iN != not_traced_input_2)):
            ax1 = iN - del_n 
            ax2 = iN - del_n + n_remain
            RDM = np.trace(RDM, axis1 = ax1, axis2 = ax2) # traces over rows and cols corresponding to all atoms except the one corresponding to not_traced_input
            n_remain -= 1
            del_n += 1
    new_shape = [nstates_input**2, nstates_input**2] # 2 in power respresents the two atoms that did not get traced over
            
    return np.reshape(RDM, new_shape)


def f_concurrence_pure(rho_input, N_input, nstates_input, not_traced_input):
    red_rho = f_partial_trace(rho_input, N_input, nstates_input, not_traced_input)
    conc = np.sqrt((1 - np.trace(red_rho@red_rho))*nstates_input/(nstates_input-1))
    return conc

def f_partial_transpose(rho_input, N_input, nstates_input, atom_index): #rho_in = full density matrix, atom_index = subdivision index 
    shape_original = np.shape(rho_input) #nstates_input lies in 0...N-1
    shape_tuple = []
    for i in range(0, 2*N_input): # N_input = no of atoms
        shape_tuple.append(nstates_input) # nstates_input = no of states in single atom Hilbert space
    rho_proc = np.reshape(rho_input, shape_tuple) # rho_proc ~ |i><j| \otimes |k><l| for two atoms ~ rho[i,k,j,l] (i, k - row indices, j, l - col indices)
    if N_input == 2:
        if atom_index == 0:
            rho_transposed = np.einsum('ikjl->jkil', rho_proc)
            return np.reshape(rho_transposed, shape_original)
        elif atom_index == 1:
            rho_transposed = np.einsum('ikjl->iljk', rho_proc)
            return np.reshape(rho_transposed, shape_original)
        else:
            print('Incorrect atom index', flush=True)
            return 0
    else:
        
        axis_list = np.arange(0, 2*N_input)
        ax_row = atom_index
        ax_col = atom_index + N_input
        axis_list[ax_row] = ax_col
        axis_list[ax_col] = ax_row
        rho_transposed = np.transpose(rho_proc, axis_list)
        return np.reshape(rho_transposed, shape_original)
    
def f_partial_transpose_half_array(rho_input, N_input, nstates_input):
    if N_input%2==0: # even N
        trans_upto_index = int(Natoms/2.0)-1
    else: # odd N
        trans_upto_index = int((Natoms-1)/2.0)-1
    rho_trans_temp = rho_input
    for j in range(0, trans_upto_index+1):
        rho_trans_temp = f_partial_transpose(rho_trans_temp, N_input, nstates_input, j)
    return rho_trans_temp
    
def f_negativity(rho_input): # from wikipedia definition of negativity (quantum mechanics)
    # rho_input is partial transposed DM, numpy array
    temp_eigs = np.linalg.eigvals(rho_input)
    temp_neg = np.sum(np.abs(temp_eigs) - temp_eigs)/2.0
    return temp_neg



# levels
deg_e = int(2*fe + 1)
deg_g = int(2*fg + 1)

if (deg_e == 1 and deg_g == 1):
    qmax = 0
else:
    qmax = 1



# dictionaries




# Clebsch Gordan coeff
cnq = {}
arrcnq = np.zeros((deg_g, 2*qmax+1), complex)
if (deg_e == 1 and deg_g ==1):
    cnq[0, 0] = 1
    arrcnq[0, 0] =  1
else:
    for i in range(0, deg_g):
        mg = i-fg
        for q in range(-qmax, qmax+1):
            if np.abs(mg + q) <= fe:
                cnq[mg, q] =  np.float(CG(S(fg), S(mg), S(qmax), S(q), S(fe), S(mg+q)).doit())
                arrcnq[i, q+qmax] = cnq[mg, q]
cnq = collections.defaultdict(lambda : 0, cnq) 

# Dipole moment

dsph = {}
if (deg_e == 1 and deg_g ==1):
    dsph[0, 0] = np.conjugate(evec[0])
else:
    for i in range(0, deg_e):
        me = i-fe
        for j in range(0, deg_g):
            mg = j-fg
            dsph[me, mg] = (np.conjugate(evec[me-mg])*cnq[mg, me-mg])

dsph = collections.defaultdict(lambda : np.array([0,0,0]), dsph) 

#Rabi frequency for each atom

omega_atom = np.zeros((Natoms, deg_e, deg_g), complex)
for n in range(0, Natoms):
    for i in range(0, deg_e):
        me = i-fe
        for j in range(0, deg_g):
            mg = j-fg
            omega_atom[n, i, j] = (rabi*np.dot(dsph[me, mg],eL)*np.exp(1j*np.dot(kvec, rvecall[n]))) 
            


# normalise vector
def hat_op(v):
    return (v/np.linalg.norm(v))

# Green's function
def funcG(r):
    tempcoef = 3*single_decay/4.0
    temp1 = (np.identity(3) - np.outer(hat_op(r), hat_op(r)))*np.exp(1j*k0*np.linalg.norm(r))/(k0*np.linalg.norm(r)) 
    temp2 = (np.identity(3) - 3*np.outer(hat_op(r), hat_op(r)))*((1j*np.exp(1j*k0*np.linalg.norm(r))/(k0*np.linalg.norm(r))**2) - np.exp(1j*k0*np.linalg.norm(r))/(k0*np.linalg.norm(r))**3)
    return (tempcoef*(temp1 + temp2))

def funcGij(i, j):
    return (funcG(rvecall[i] - rvecall[j]))


fac_plus_int = 1.0
fac_minus_int = 1.0
fac_pi_int = 1.0
fac_inc = 1.0
fac_coh = 1.0
if turn_off!=[]:
    for item in turn_off:
        if item == 'sigma_plus':
            fac_plus_int = 0
        elif item == 'sigma_minus':
            fac_minus_int = 0
        elif item == 'pi':
            fac_pi_int = 0
        elif item == 'incoherent':
            fac_inc = 0
        elif item == 'coherent':
            fac_coh = 0
fac_list_int = np.array([fac_minus_int, fac_pi_int, fac_plus_int]) # corresp to q = -1, 0, 1

taD = time.time()


dictRij = {}
dictIij = {}
dictGij = {}
dictGijtilde = {}
for i in range(0, Natoms):
    for j in range(0, Natoms):
        for q1 in range(-1,2):
            for q2 in range(-1,2):
                if i!=j:
                    dictRij[i, j, q1, q2] = fac_coh*fac_list_int[q1+1]*fac_list_int[q2+1]*np.conjugate(evec[q1])@np.real(funcGij(i, j))@evec[q2]
                    dictIij[i, j, q1, q2] = fac_inc*fac_list_int[q1+1]*fac_list_int[q2+1]*np.conjugate(evec[q1])@np.imag(funcGij(i, j))@evec[q2]
                else:
                    dictRij[i, i, q1, q2] = 0
                    dictIij[i, i, q1, q2] = (single_decay/2.0)*np.dot(np.conjugate(evec[q1]),evec[q2])

                dictGij[i, j, q1, q2] = dictRij[i, j, q1, q2] + 1j*dictIij[i, j, q1, q2]
                dictGijtilde[i, j, q1, q2] = dictRij[i, j, q1, q2] - 1j*dictIij[i, j, q1, q2]
    
dictRij = collections.defaultdict(lambda : 0, dictRij) 
dictIij = collections.defaultdict(lambda : 0, dictIij) 


tbD = time.time()
print("time to assign Rij, Iij dict: "+str(tbD-taD), flush=True)

taG = time.time()

arrGij = np.zeros((Natoms, Natoms, deg_e, deg_g, deg_e, deg_g), complex)
arrGijtilde = np.zeros((Natoms, Natoms, deg_e, deg_g, deg_e, deg_g), complex)
arrIij = np.zeros((Natoms, Natoms, deg_e, deg_g, deg_e, deg_g), complex)
for i in range(0, Natoms):
    for j in range(0, Natoms):
        for ima in range(0, deg_e):
            ma = ima - fe
            for ina in range(0, deg_g):
                na = ina - fg
                for imb in range(0, deg_e):
                    mb = imb - fe
                    for inb in range(0, deg_g):
                        nb = inb - fg
                        arrGij[i, j, ima, ina, imb, inb] = dictGij[i, j, ma-na, mb-nb]*cnq[na, ma-na]*cnq[nb, mb-nb]
                        arrGijtilde[i, j, ima, ina, imb, inb] = dictGijtilde[i, j, ma-na, mb-nb]*cnq[na, ma-na]*cnq[nb, mb-nb]
                        arrIij[i, j, ima, ina, imb, inb] = dictIij[i, j, ma-na, mb-nb]*cnq[na, ma-na]*cnq[nb, mb-nb]
                        
tbG = time.time()
print("time to assign Gij matrix: "+str(tbG-taG), flush=True)



# plot properties

levels = int(deg_e + deg_g)
rdir = '1D array'

rdir += '('
rdir_fig = '_r_'+str(np.round(ratio, 2)).replace('.',',')+'_along_'
dirs = ['x','y','z']
temp_add = 0
for i in range(0,3):
    if r_axis[i]!=0:
        if temp_add == 0:
            rdir += dirs[i]
            rdir_fig += dirs[i]
        else:
            rdir += ' + '+dirs[i]
            rdir_fig += '_and_'+ dirs[i]
        temp_add += 1
rdir += ')'

eLdir = '('
eLdir_fig = '_eL_along_'
dirs = ['x','y','z']
temp_add = 0
for i in range(0,3):
    if eL[i]!=0:
        if temp_add == 0:
            eLdir += dirs[i]
            eLdir_fig += dirs[i]
        else:
            eLdir += ' + '+dirs[i]
            eLdir_fig += '_and_'+ dirs[i]
        temp_add += 1
eLdir += ')'

kLdir = '('
kLdir_fig = '_k_along_'
dirs = ['x','y','z']
temp_add = 0
for i in range(0,3):
    if kvec[i]!=0:
        if temp_add == 0:
            kLdir += dirs[i]
            kLdir_fig += dirs[i]
        else:
            kLdir += ' + '+dirs[i]
            kLdir_fig += '_and_'+ dirs[i]
        temp_add += 1
kLdir += ')'


rabi_add = '_rabi_'+str((rabi)).replace('.',',')
det_fig = '_det_'+str((detuning_list[det_ind_input])).replace('.',',')


h5_title = str(levels)+'_level_'+str(Natoms)+'_atoms'+rdir_fig+kLdir_fig+eLdir_fig+rabi_add+det_fig+add_txt_in_params+'.h5'



print('rvecall:', flush=True)
print(rvecall)

# defining ops for ED

HSsize = int(2*fg + 1 + 2*fe + 1) # Hilbert space size of each atom
HSsize_tot = int(HSsize**(Natoms)) # size of cluster Hilbert space

adde = fe
addg = fg

# polarisation basis vectors
evec = {0: e0, 1:eplus, -1: eminus}
evec = collections.defaultdict(lambda : [0,0,0], evec) 


def dsph(me, mg):
    return (np.conjugate(evec[me-mg])*cnq[mg, me-mg])
   
def sort_lists_simultaneously_cols(a, b): #a -list to be sorted, b - 2d array whose columns are to be sorted according to indices of a
    inds = a.argsort()
    sortedb = b[:,inds]
    return sortedb
    
def commutator(A, B):
    return (np.dot(A,B)-np.dot(B,A))

def anticommutator(A, B):
    return (np.dot(A,B)+np.dot(B,A))

def ketGn(mg):
    temp = np.zeros(deg_g)
    temp[int(mg + addg)] = 1
    return temp

def f_omega_atom(k, me, mg):
    return (rabi*np.dot(dsph(me, mg),eL)*np.exp(1j*np.dot(kvec, rvecall[k])))    


def sparse_trace(A):
    return A.diagonal().sum()




indgg = int(deg_g*deg_g)
indee = int(deg_e*deg_e)
indeg = int(deg_e*deg_g)
total_num = indgg+indee+indeg
# gs states of each atom

gs_states = np.zeros(deg_g)
for i in range(0, deg_g):
    gs_states[i] = i-fg
    
# es states of each atom

es_states = np.zeros(deg_e)
for i in range(0, deg_e):
    es_states[i] = i-fe


######################################################################################################


# dictionary of states in the cut-off Hilbert space

dict_mb_states = {}
dict_mb_states_rev = {}

GS_mfld_size = deg_g**Natoms
ES_mfld_size = Natoms*deg_e*(deg_g**(Natoms-1))
HS_size_tot_reduced = GS_mfld_size + ES_mfld_size
basis_states = np.identity(HS_size_tot_reduced)

# a dictionary assigning an integer to each single particle state

dict_single_particle_states = {}
dict_single_particle_states_rev = {}
ind = 0
for ig in range(0, deg_g):
    dict_single_particle_states['g', ig-fg] = ind 
    dict_single_particle_states_rev[ind] = ['g', ig-fg]
    ind += 1
for ie in range(0, deg_e):
    dict_single_particle_states['e', ie-fe] = ind 
    dict_single_particle_states_rev[ind] = ['e', ie-fe]
    ind += 1

# GS manifold many body states: g- = 0, g+ = 1

# the index of each state is the decimal int equivalent of the binary number string represented using |g1 g2 ... gN> where gi's are 0's and 1's
# to convert a binary number string x = '010100' to its decimal integer counterpart, use int(x, 2) 
# to convert a decimal int number a = 5 to its binary string counterpart, use np.binary_repr(a, width=w) where w is the number of digits desired, which is the number of atoms in the system in our case
# the index also represents the basis state in the code, i.e., index = 0 means that the state is represented as [1,0,0...,0] and index = 1 means that the state is [0,1,0,0,...,0] and so on.

# single_atom_rep_mb_states is a list of arrays with 0's, 1's, 2's ... in place of g-'s, g+'s, e-'s, ... it's basically the array conversion of the binary string of 0's and 1's for ground states. This will be useful for later when we want to check which gs each atom is in and if two states have an atom in same or different gs, for assigning matrix elements etc
# the order of states in single_atom_rep_mb_states is same as the order created by binary<->int conversion of states for GS mfld

single_atom_rep_mb_states = []

for i in range(0, GS_mfld_size):
    temp_state = i
    temp_binary_rep = np.binary_repr(temp_state, width=Natoms)
    temp_state_array = np.array([int(s) for s in temp_binary_rep])
    single_atom_rep_mb_states.append(temp_state_array)
     
# ES manifold many body states: 2, 3, 4, ... -> -fe, -fe+1, -fe+2, ....
# we'll first assign an excited state value 2 or 3 or 4 or ..
# then we'll assign atom number iN which is in excited state and has the state value assigned above
# then we'll create all possible combinations of rest of the ground state atoms using binary strings to represent them
# we'll have three labels for each excited state: 
# 1. magnetic quantum number(excited state value), 
# 2. atom number in excited state, 
# 3. decimal integer equivalent of the binary string representing ground state config of rest of the atoms
# these labels will together correspond to a single integer number
# this integer number is the order of a given excited state among all basis states

ind = GS_mfld_size # indices representing single excitation mfld states
for ie in range(0, deg_e):
    temp_state_int_index = dict_single_particle_states['e', ie-fe]
    for iN in range(0, Natoms):
        for ibin in range(0, deg_g**(Natoms-1)): # index corresp to gs atoms config
            dict_mb_states['e', iN, ie-fe, ibin] = ind
            dict_mb_states_rev[ind] = ['e', iN, ie-fe, ibin]
            if Natoms>1:
                temp_binary_rep_gs = np.binary_repr(ibin, width=Natoms-1)
                temp_binary_rep = temp_binary_rep_gs[:iN] + str(temp_state_int_index) + temp_binary_rep_gs[iN:]
                temp_state_array = np.array([int(s) for s in temp_binary_rep])
                single_atom_rep_mb_states.append(temp_state_array)
            else:
                temp_state_array = np.array([temp_state_int_index])
                single_atom_rep_mb_states.append(temp_state_array)
                                            
            ind += 1
        
        
# create matrix reps of single particle ops

# sigma_emem -- single atom operator

def sigma_emem(k, em1, em2): # k = 0, 1, .., Natoms-1; em = -fe..fe, gn = -fg..fg

    mat_sigma_k_em_em = sparse.lil_matrix(np.zeros((HS_size_tot_reduced, HS_size_tot_reduced), complex))
    # <e | sigma_k_em_em | e> type matrix elements

    for irow in range(GS_mfld_size, HS_size_tot_reduced):

        row_state = single_atom_rep_mb_states[irow]
        row_state_details = dict_mb_states_rev[irow]
        row_excited_atom = row_state_details[1]
        row_excited_atom_state = row_state_details[2]
        if ((row_excited_atom!=k) or (row_excited_atom_state!=em1)):
            continue
        
        for icol in range(GS_mfld_size, HS_size_tot_reduced):

            col_state = single_atom_rep_mb_states[icol]
            
            truth_sum = np.sum(np.array([int(s) for s in col_state==row_state]))
            if truth_sum < Natoms - 1: # if atoms, apart from e-e pair are in different states
                continue
            col_state_details = dict_mb_states_rev[icol]
            col_excited_atom = col_state_details[1]
            col_excited_atom_state = col_state_details[2]
            if ((col_excited_atom!=k) or (col_excited_atom_state!=em2)):
                continue
            elif ((em1==em2) and (truth_sum < Natoms)):
                continue
            mat_sigma_k_em_em[irow,icol] = 1.0
    return sparse.csr_matrix(mat_sigma_k_em_em)

# sigma_gngn -- single atom operator

def sigma_gngn(k, gn1, gn2): # k = 0, 1, .., Natoms-1; em = -fe..fe, gn = -fg..fg

    mat_sigma_k_gn_gn = sparse.lil_matrix(np.zeros((HS_size_tot_reduced, HS_size_tot_reduced), complex))
    
    for irow in range(0, GS_mfld_size):

        row_state = single_atom_rep_mb_states[irow]
        row_state_vals = np.array([dict_single_particle_states_rev[s][1] for s in row_state])
        if row_state_vals[k]!=gn1:
            continue
        
        for icol in range(0, GS_mfld_size):

            col_state = single_atom_rep_mb_states[icol]
            col_state_vals = np.array([dict_single_particle_states_rev[s][1] for s in col_state])
            if col_state_vals[k]!=gn2:
                continue
            
            truth_sum = np.sum(np.array([int(s) for s in col_state==row_state]))
            if  (truth_sum < Natoms - 1): # if atoms, apart from g-g pair are in different states
                continue
            elif ((gn1==gn2) and (truth_sum < Natoms)):
                continue
                
            mat_sigma_k_gn_gn[irow,icol] = 1.0
    
    # <e | sigma_k_gn_gn | e> type matrix elements

    for irow in range(GS_mfld_size, HS_size_tot_reduced):

        row_state = single_atom_rep_mb_states[irow]
        row_state_gs_atom_val = dict_single_particle_states_rev[row_state[k]][1]
        row_state_details = dict_mb_states_rev[irow]
        row_excited_atom = row_state_details[1]
        row_excited_atom_state = row_state_details[2]
        if ((row_excited_atom==k) or (row_state_gs_atom_val!=gn1)):
            continue
        
        for icol in range(GS_mfld_size, HS_size_tot_reduced):

            col_state = single_atom_rep_mb_states[icol]
            col_state_gs_atom_val = dict_single_particle_states_rev[col_state[k]][1]
            truth_sum = np.sum(np.array([int(s) for s in col_state==row_state]))
            if truth_sum < Natoms - 1: # if atoms, apart from e-e pair are in different states
                continue
            col_state_details = dict_mb_states_rev[icol]
            col_excited_atom = col_state_details[1]
            col_excited_atom_state = col_state_details[2]
            if ((gn1==gn2) and (truth_sum < Natoms)):
                continue
            elif ((col_excited_atom==k) or (col_state_gs_atom_val!=gn2)):
                continue
            elif col_excited_atom_state!=row_excited_atom_state:
                continue
            elif col_excited_atom!=row_excited_atom:
                continue
    
            mat_sigma_k_gn_gn[irow,icol] = 1.0
    return sparse.csr_matrix(mat_sigma_k_gn_gn)


# sigma_k_em_gn -- single atom operator

def sigma_emgn(k, em, gn): # k = 0, 1, .., Natoms-1; em = -fe..fe, gn = -fg..fg

    mat_sigma_k_em_gn = sparse.lil_matrix(np.zeros((HS_size_tot_reduced, HS_size_tot_reduced), complex))
    # <e | sigma_k_em_gn | g> type matrix elements

    for irow in range(GS_mfld_size, HS_size_tot_reduced):

        row_state = single_atom_rep_mb_states[irow]
        row_state_details = dict_mb_states_rev[irow]
        row_excited_atom = row_state_details[1]
        row_excited_atom_state = row_state_details[2]
        if ((row_excited_atom!=k) or (row_excited_atom_state!=em)):
            continue
        
        for icol in range(0, GS_mfld_size):

            col_state = single_atom_rep_mb_states[icol]
            truth_sum = np.sum(np.array([int(s) for s in col_state==row_state]))
            if truth_sum < Natoms - 1: # if atoms, apart from e-g pair are in different states
                continue
            col_atom_gs_state_index = col_state[row_excited_atom]
            col_atom_gs_state = dict_single_particle_states_rev[col_atom_gs_state_index][1]
            if (col_atom_gs_state!=gn):
                continue
            mat_sigma_k_em_gn[irow,icol] = 1.0
    return sparse.csr_matrix(mat_sigma_k_em_gn)

def sigma_gnem(k, gn, em): 
    return (sparse.csr_matrix.getH(sigma_emgn(k, em, gn)))


def funcL(A):
    return sparse.kron(A, np.identity(HS_size_tot_reduced))

def funcR(A):
    return sparse.kron(np.identity(HS_size_tot_reduced), np.transpose(A))

def funcL_GS(A):
    return sparse.kron(A, np.identity(GS_mfld_size))

def funcR_GS(A):
    return sparse.kron(np.identity(GS_mfld_size), np.transpose(A))
        
# diagonalising Lindbladian

# first construct array

L_size = Natoms*deg_e*deg_g



# (ki, n1, m1) = R, (kj, n2, m2) = C; R -> row index, C -> column index

dict_old_jump_ops_indices = {}
r = 0

Mij_mat = np.zeros((L_size, L_size), complex)
ind1 = 0
for ki in range(0, Natoms):
    for n1 in range(0, deg_g):
        for m1 in range(0, deg_e):
            qi = (m1-fe)-(n1-fg)
            cnqi = cnq[n1-fg, qi]
            ind2 = 0
            for kj in range(0, Natoms):
                for n2 in range(0, deg_g):
                    for m2 in range(0, deg_e):
                        qj = (m2-fe)-(n2-fg)
                        cnqj = cnq[n2-fg, qj]
                        Mij_mat[ind1, ind2] = -cnqi*cnqj*dictIij[ki,kj,qi,qj]
                        ind2 += 1
            if r == 0:
                dict_old_jump_ops_indices[ind1] = [ki,n1-fg,m1-fe]
            ind1 += 1

Mij_mat_eigvals, Mij_mat_eigvecs = np.linalg.eigh(Mij_mat) 
# using eigh instead of eig to get a unitary matrix of eigvecs (modal matrix which is used to diagonalise a matrix)
# with eig I was getting a non unitary modal matrix and that was giving different diagonalised operators on the left and the right, i.e., L_dag and L weren't corresponding to each other
# this only works because Mij_mat is symmetric (or Hermitian)
# a modal matrix is generally only unitary when all the eigenvectors are orthogonal to each other
# Mij_mat has repeated eigenvalues though so it is not necessary for a matrix of degenerate eigenvalues to have
# linearly independent eigenvectors, and hence it may not be diagonalizable. However, a symmetric matrix is always
# diagonalizable and so it works.
Mij_diag_U = Mij_mat_eigvecs
Mij_diag_U_inv = np.linalg.inv(Mij_diag_U)
diagonalised_Mij = np.round(Mij_diag_U_inv@Mij_mat@Mij_diag_U, 15)
    
# create matrix reps of operators we'll use

# single atom ground state Hamiltonian: H_g

mat_Hg_GSM = sparse.lil_matrix(np.zeros((GS_mfld_size, GS_mfld_size), complex))
mat_Hg_ESM = sparse.lil_matrix(np.zeros((HS_size_tot_reduced, HS_size_tot_reduced), complex))

for irow in range(0, GS_mfld_size):
    row_state = single_atom_rep_mb_states[irow]
    row_state_vals = np.array([dict_single_particle_states_rev[s][1] for s in row_state])
    #row_state = row_state - fg*np.ones(len(row_state))
    mat_Hg_GSM[irow,irow] = np.sum(row_state_vals)

for irow in range(GS_mfld_size, HS_size_tot_reduced):
    row_state = single_atom_rep_mb_states[irow]
    row_state_vals = np.array([dict_single_particle_states_rev[s][1] for s in row_state])
    #row_state = row_state - fg*np.ones(len(row_state))
    row_state_details = dict_mb_states_rev[irow]
    row_excited_atom = row_state_details[1]
    mat_Hg_ESM[irow,irow] = np.sum(row_state_vals[:row_excited_atom]) + np.sum(row_state_vals[row_excited_atom+1:])
    
mat_Hg_GSM = sparse.csr_matrix(mat_Hg_GSM*del_zg)
mat_Hg_ESM = sparse.csr_matrix(mat_Hg_ESM*del_zg)


### STUFF THAT DEPENDS ON DETUNING and r


mat_H_Rabi_plus = sparse.lil_matrix(np.zeros((HS_size_tot_reduced, HS_size_tot_reduced), complex))
mat_H_Rabi_minus = sparse.lil_matrix(np.zeros((HS_size_tot_reduced, HS_size_tot_reduced), complex))

for irow in range(GS_mfld_size, HS_size_tot_reduced):

    row_state = single_atom_rep_mb_states[irow]
    row_state_details = dict_mb_states_rev[irow]
    row_excited_atom = row_state_details[1]
    row_excited_atom_state = row_state_details[2]

    for icol in range(0, GS_mfld_size):

        col_state = single_atom_rep_mb_states[icol]
        truth_sum = np.sum(np.array([int(s) for s in col_state==row_state]))
        if truth_sum < Natoms - 1: # if atoms, apart from e-g pair are in different states
            continue
        col_atom_gs_state_index = col_state[row_excited_atom]
        col_atom_gs_state = dict_single_particle_states_rev[col_atom_gs_state_index][1]
        # <e | H | g> type matrix elements
        mat_H_Rabi_plus[irow,icol] = -f_omega_atom(row_excited_atom, row_excited_atom_state, col_atom_gs_state)
        # <g | H | e> type matrix elements
        mat_H_Rabi_minus[icol,irow] = np.conj(mat_H_Rabi_plus[irow,icol])

mat_H_Rabi_plus = sparse.csr_matrix(mat_H_Rabi_plus)
mat_H_Rabi_minus = sparse.csr_matrix(mat_H_Rabi_minus)

mat_Hint_coh = sparse.lil_matrix(np.zeros((HS_size_tot_reduced, HS_size_tot_reduced), complex))
for irow in range(GS_mfld_size, HS_size_tot_reduced):

    row_state = single_atom_rep_mb_states[irow]
    row_state_details = dict_mb_states_rev[irow]
    row_excited_atom = row_state_details[1]
    row_excited_atom_state = row_state_details[2]

    for icol in range(GS_mfld_size, HS_size_tot_reduced):

        col_state = single_atom_rep_mb_states[icol]
        col_state_details = dict_mb_states_rev[icol]
        col_excited_atom = col_state_details[1]
        col_excited_atom_state = col_state_details[2]

        truth_sum = np.sum(np.array([int(s) for s in col_state==row_state]))

        if ((row_excited_atom!=col_excited_atom) and (truth_sum < Natoms - 2)): # if atoms apart from the two e-g pair atoms are in different states
            continue
        if (row_excited_atom==col_excited_atom): # a single atom cannot go from an excited state to an excited state during interaction
            continue
        row_atom_gs_state_index = row_state[col_excited_atom]
        row_gs_atom_state = dict_single_particle_states_rev[row_atom_gs_state_index][1]

        col_atom_gs_state_index = col_state[row_excited_atom]
        col_gs_atom_state = dict_single_particle_states_rev[col_atom_gs_state_index][1]

        q1 = row_excited_atom_state - col_gs_atom_state
        q2 = col_excited_atom_state - row_gs_atom_state
        cnq1 = cnq[col_gs_atom_state, q1]
        cnq2 = np.conj(cnq[row_gs_atom_state, q2])
        mat_Hint_coh[irow,icol] = -cnq1*cnq2*dictRij[row_excited_atom, col_excited_atom, q1, q2]

mat_Hint_incoh = sparse.lil_matrix(np.zeros((HS_size_tot_reduced, HS_size_tot_reduced), complex))
for irow in range(GS_mfld_size, HS_size_tot_reduced):

    row_state = single_atom_rep_mb_states[irow]
    row_state_details = dict_mb_states_rev[irow]
    row_excited_atom = row_state_details[1]
    row_excited_atom_state = row_state_details[2]

    for icol in range(GS_mfld_size, HS_size_tot_reduced):

        col_state = single_atom_rep_mb_states[icol]
        col_state_details = dict_mb_states_rev[icol]
        col_excited_atom = col_state_details[1]
        col_excited_atom_state = col_state_details[2]

        truth_sum = np.sum(np.array([int(s) for s in col_state==row_state]))

        if ((row_excited_atom!=col_excited_atom) and (truth_sum < Natoms - 2)): # if atoms apart from the two e-g pair atoms are in different states
            continue
        if (row_excited_atom==col_excited_atom): # single atom decay case
            if (row_excited_atom_state!=col_excited_atom_state):
                continue
            elif ((row_excited_atom_state==col_excited_atom_state) and (truth_sum < Natoms)): # if atoms are in different states
                continue
            elif (row_excited_atom_state==col_excited_atom_state):
                temp_elem = 0+0*1j
                for ign in range(0, deg_g):
                    gn = ign - fg
                    q1 = row_excited_atom_state-gn
                    q2 = col_excited_atom_state-gn
                    cnq1 = cnq[gn, q1]
                    cnq2 = cnq[gn, q2]
                    temp_elem += -cnq1*cnq2*dictIij[row_excited_atom, col_excited_atom, q1, q2]
                mat_Hint_incoh[irow,icol] = temp_elem
                continue
        row_atom_gs_state_index = row_state[col_excited_atom]
        row_gs_atom_state = dict_single_particle_states_rev[row_atom_gs_state_index][1]

        col_atom_gs_state_index = col_state[row_excited_atom]
        col_gs_atom_state = dict_single_particle_states_rev[col_atom_gs_state_index][1]

        q1 = row_excited_atom_state - col_gs_atom_state
        q2 = col_excited_atom_state - row_gs_atom_state
        cnq1 = cnq[col_gs_atom_state, q1]
        cnq2 = np.conj(cnq[row_gs_atom_state, q2])
        mat_Hint_incoh[irow,icol] = -cnq1*cnq2*dictIij[row_excited_atom, col_excited_atom, q1, q2]

# getting master equation

# single atom excited state Hamiltonian: H_e
mat_He = sparse.lil_matrix(np.zeros((HS_size_tot_reduced, HS_size_tot_reduced), complex))
for irow in range(GS_mfld_size, HS_size_tot_reduced):
    row_state_details = dict_mb_states_rev[irow]
    row_excited_atom_state = row_state_details[2]
    mat_He[irow,irow] = (-detuning_list[det_ind_input] + del_ze*row_excited_atom_state)
mat_He = sparse.csr_matrix(mat_He)


mat_H_NH = sparse.lil_matrix(np.zeros((HS_size_tot_reduced, HS_size_tot_reduced), complex))
mat_H_NH_inv = sparse.lil_matrix(np.zeros((HS_size_tot_reduced, HS_size_tot_reduced), complex))

mat_H_NH = sparse.csc_matrix(mat_Hg_ESM + mat_He + mat_Hint_coh + 1j*mat_Hint_incoh)
mat_H_NH_inv = sparse.lil_matrix(np.zeros((HS_size_tot_reduced, HS_size_tot_reduced), complex))
mat_H_NH_inv[GS_mfld_size:, GS_mfld_size:] = sparse.linalg.inv(mat_H_NH[GS_mfld_size:, GS_mfld_size:])

mat_H_NH = sparse.csr_matrix(mat_H_NH)  
mat_H_NH_inv = sparse.csr_matrix(mat_H_NH_inv)   

mat_H_eff_GSM = (-0.5*sparse.csr_matrix.dot(sparse.csr_matrix.dot(mat_H_Rabi_minus,(mat_H_NH_inv + sparse.csr_matrix.getH(mat_H_NH_inv))),mat_H_Rabi_plus))[:GS_mfld_size, :GS_mfld_size] + mat_Hg_GSM

def f_L_eff_knm(k, gn, em):
    temp = sparse.csr_matrix.dot(sparse.csr_matrix.dot(sigma_gnem(k, gn, em),mat_H_NH_inv),mat_H_Rabi_plus)
    return temp[:GS_mfld_size, :GS_mfld_size]

def f_L_eff_knm_dagger(k, gn, em):
    return sparse.csr_matrix.getH(f_L_eff_knm(k, gn, em))

def f_diag_L_eff_knm(ind_alpha):
    temp_L = sparse.csr_matrix(np.zeros((GS_mfld_size, GS_mfld_size), complex))
    for indK in range(0, L_size):
        ki, ni, mi = dict_old_jump_ops_indices[indK]
        temp_L += Mij_diag_U_inv[ind_alpha, indK]*f_L_eff_knm(ki, ni, mi)
    return temp_L

def f_diag_L_eff_knm_dagger(ind_alpha):
    temp_L = sparse.csr_matrix(np.zeros((GS_mfld_size, GS_mfld_size), complex))
    for indK in range(0, L_size):
        ki, ni, mi = dict_old_jump_ops_indices[indK]
        temp_L += Mij_diag_U[indK, ind_alpha]*f_L_eff_knm_dagger(ki, ni, mi)
    return temp_L


# eff Lindbladian terms

# Anti-commutator term

mat_L_1 = sparse.csr_matrix(np.zeros((GS_mfld_size**2, GS_mfld_size**2), complex))
mat_Heff_Incoh = sparse.csr_matrix(np.zeros((GS_mfld_size, GS_mfld_size), complex))

for iK in range(0, L_size):
    mat_L_1 += Mij_mat_eigvals[iK]*funcL_GS(sparse.csr_matrix.dot(f_diag_L_eff_knm_dagger(iK),f_diag_L_eff_knm(iK)))
    mat_L_1 +=  Mij_mat_eigvals[iK]*funcR_GS(sparse.csr_matrix.dot(f_diag_L_eff_knm_dagger(iK),f_diag_L_eff_knm(iK)))
    mat_Heff_Incoh +=  Mij_mat_eigvals[iK]*sparse.csr_matrix.dot(f_diag_L_eff_knm_dagger(iK),f_diag_L_eff_knm(iK))


# Recycling term

mat_L_rec = sparse.csr_matrix(np.zeros((GS_mfld_size**2, GS_mfld_size**2), complex))
for iK in range(0, L_size):
    mat_L_rec += -2*Mij_mat_eigvals[iK]*sparse.csr_matrix.dot(funcL_GS(f_diag_L_eff_knm(iK)),funcR_GS(f_diag_L_eff_knm_dagger(iK)))

rho_GS_dot = -1j*(funcL_GS(mat_H_eff_GSM) - funcR_GS(mat_H_eff_GSM)) + mat_L_1 + mat_L_rec

    
def funcOp(kinput, A):
    k = kinput+1
    if (k> Natoms or k<1):
        return "error"
    elif k == 1:
        temp = csr_matrix(A)
        for i in range(1, Natoms):
            temp = csr_matrix(sparse.kron(temp, np.identity(deg_g)))
        return temp
    else:
        temp = csr_matrix(np.identity(deg_g))
        for i in range(2, k):
            temp = csr_matrix(sparse.kron(temp, np.identity(deg_g)))
        temp = csr_matrix(sparse.kron(temp, A))
        for i in range(k, Natoms):
            temp = csr_matrix(sparse.kron(temp, np.identity(deg_g)))
        return temp

    
# define the Pauli matrices


sigma_z = np.array([[1,0],[0,-1]])
sigma_plus = np.array([[0,1],[0,0]])
sigma_minus = np.array([[0,0],[1,0]])

# function for dynamics

def f_rho_dot_vec(t, rho_vec):
    return (sparse.csr_matrix.dot(rho_GS_dot,rho_vec))

######################################################################################################




# initial condition

if IC_chosen == 'single_gs' or IC_chosen == '':
    mg = -fg
    temp = ketGn(mg)
    for n in range(1, Natoms):
        temp = np.array(np.kron(temp,ketGn(mg)), complex)
elif IC_chosen == 'equal_gs':
    temp_single = 0 + 0*1j
    for img in range(0, deg_g):
        temp_single += ketGn(img-fg) #(ketGn(-fg) + ketGn(fg))/np.sqrt(2)
    temp_single = temp_single/np.sqrt(deg_g)
    temp = temp_single
    for n in range(1, Natoms):
        temp = np.array(np.kron(temp,temp_single), complex)
initial_state = temp + 0*1j
initial_rho = np.outer(initial_state, np.conjugate(initial_state))
initial_sig_vec = initial_rho.flatten()

print('trace of initial DM = '+str(np.trace(initial_rho)))



#  driven evolutiion from a ground state superposition to get to the steady state
ta1 = time.time()
sol = solve_ivp(f_rho_dot_vec, t_range_dr, initial_sig_vec, method='RK45', t_eval=t_vals_dr, dense_output=False, events=None, atol = 10**(-10), rtol = 10**(-5))
tb1 = time.time()
runtime1 = tb1-ta1


print('Time to run dynamics: '+str(runtime1), flush=True)


mixedness_dr = np.zeros(len(t_vals_dr))
trace_dr = np.zeros(len(t_vals_dr))
negativity_mid_dr = np.zeros(len(t_vals_dr))
total_Sx_dr = np.zeros((len(t_vals_dr)))
total_Sy_dr = np.zeros((len(t_vals_dr)))
total_Sz_dr = np.zeros((len(t_vals_dr)))
total_Sz_Sz_dr = np.zeros((len(t_vals_dr)))
total_Sz_Sx_dr = np.zeros((len(t_vals_dr)))
total_Sz_Sy_dr = np.zeros((len(t_vals_dr)))
total_Sx_Sx_dr = np.zeros((len(t_vals_dr)))
total_Sx_Sy_dr = np.zeros((len(t_vals_dr)))
total_Sy_Sy_dr = np.zeros((len(t_vals_dr)))


mid_index = int(Natoms/2.0)

total_Sp_Op = 0
total_Sz_Op = 0
total_Sx_Op = 0
total_Sy_Op = 0
for k in range(0, Natoms):
    total_Sp_Op += sigma_gngn(k, 1/2, -1/2)[:GS_mfld_size, :GS_mfld_size]
    total_Sz_Op += (sigma_gngn(k, 1/2, 1/2) - sigma_gngn(k, -1/2, -1/2))[:GS_mfld_size, :GS_mfld_size]
    total_Sx_Op += (sigma_gngn(k, 1/2, -1/2) + sigma_gngn(k, -1/2, 1/2))[:GS_mfld_size, :GS_mfld_size]
    total_Sy_Op += (-1j * (sigma_gngn(k, 1/2, -1/2) - sigma_gngn(k, -1/2, 1/2)))[:GS_mfld_size, :GS_mfld_size]

total_Sz_Sz_Op = sparse.csr_matrix.dot(total_Sz_Op, total_Sz_Op)
total_Sz_Sx_Op = sparse.csr_matrix.dot(total_Sz_Op, total_Sx_Op)
total_Sz_Sy_Op = sparse.csr_matrix.dot(total_Sz_Op, total_Sy_Op)
total_Sx_Sx_Op = sparse.csr_matrix.dot(total_Sx_Op, total_Sx_Op)
total_Sx_Sy_Op = sparse.csr_matrix.dot(total_Sx_Op, total_Sy_Op)
total_Sy_Sy_Op = sparse.csr_matrix.dot(total_Sy_Op, total_Sy_Op)

for i in range(0, len(sol.t)):
    index = 0
    rho_sol_dr = (np.reshape(sol.y[:,i],(GS_mfld_size,GS_mfld_size)))

    rho_trans = f_partial_transpose(rho_sol_dr, Natoms, deg_g, mid_index)
    temp_negativity = f_negativity(rho_trans)
    negativity_mid_dr[i] = temp_negativity.real

    rho_sol_dr = csr_matrix(rho_sol_dr)

    trace_dr[i] = sparse_trace(rho_sol_dr)
    mixedness_dr[i] = sparse_trace(sparse.csr_matrix.dot(rho_sol_dr, rho_sol_dr))
    
        
    total_Sp_dr = sparse_trace(sparse.csr_matrix.dot((rho_sol_dr), total_Sp_Op))
    total_Sx_dr[i] = 2 * np.real(total_Sp_dr)
    total_Sy_dr[i] = 2 * np.imag(total_Sp_dr)

    total_Sz_Sz_dr[i] = (sparse_trace(sparse.csr_matrix.dot((rho_sol_dr), total_Sz_Sz_Op))).real
    total_Sz_Sx_dr[i] = (sparse_trace(sparse.csr_matrix.dot((rho_sol_dr), total_Sz_Sx_Op))).real
    total_Sz_Sy_dr[i] = (sparse_trace(sparse.csr_matrix.dot((rho_sol_dr), total_Sz_Sy_Op))).real
    total_Sx_Sx_dr[i] = (sparse_trace(sparse.csr_matrix.dot((rho_sol_dr), total_Sx_Sx_Op))).real
    total_Sx_Sy_dr[i] = (sparse_trace(sparse.csr_matrix.dot((rho_sol_dr), total_Sx_Sy_Op))).real
    total_Sy_Sy_dr[i] = (sparse_trace(sparse.csr_matrix.dot((rho_sol_dr), total_Sy_Sy_Op))).real


C_matrix = np.array([[total_Sx_Sx_dr, total_Sx_Sy_dr, total_Sz_Sx_dr], [total_Sx_Sy_dr, total_Sy_Sy_dr, total_Sz_Sy_dr], [total_Sz_Sx_dr, total_Sz_Sy_dr, total_Sz_Sz_dr]])            

gamma_matrix = np.array([[total_Sx_Sx_dr - total_Sx_dr**2, total_Sx_Sy_dr - total_Sx_dr*total_Sy_dr, total_Sz_Sx_dr - total_Sx_dr*total_Sz_dr], 
    [total_Sx_Sy_dr - total_Sx_dr*total_Sy_dr, total_Sy_Sy_dr - total_Sy_dr**2, total_Sz_Sy_dr - total_Sz_dr*total_Sy_dr], 
    [total_Sz_Sx_dr - total_Sx_dr*total_Sz_dr, total_Sz_Sy_dr - total_Sz_dr*total_Sy_dr, total_Sz_Sz_dr - total_Sz_dr*total_Sz_dr]])            

X_matrix = (Natoms - 1)*gamma_matrix + C_matrix

spin_inequal_param_list_30a = np.zeros(len(t_vals_dr)) # if any of these are < 0, then that inequality is violated
spin_inequal_param_list_30b = np.zeros(len(t_vals_dr))
spin_inequal_param_list_30c = np.zeros(len(t_vals_dr))
spin_inequal_param_list_30d = np.zeros(len(t_vals_dr))

for i in range(0, len(t_vals_dr)):
    eigvals, eigvecs = scipy.linalg.eigh(X_matrix[:,:,i])
    
    spin_inequal_param_list_30a[i] = ((Natoms+2)*Natoms/4.0)/np.trace(C_matrix[:,:,i])
    spin_inequal_param_list_30b[i] = np.trace(gamma_matrix[:,:,i])/(Natoms/2.0)
    spin_inequal_param_list_30c[i] = np.min(eigvals)/(np.trace(C_matrix[:,:,i]) - Natoms/2.0)
    spin_inequal_param_list_30d[i] = ((Natoms-1)*np.trace(gamma_matrix[:,:,i]) - (Natoms-2)*Natoms/4.0)/np.max(eigvals)


hf = h5py.File(direc+'Data_GSM_ED_dynamics_to_equil_'+h5_title, 'w')

hf.create_dataset('t_vals_dr', data=t_vals_dr, compression="gzip", compression_opts=9)
hf.create_dataset('trace_dr', data=trace_dr, compression="gzip", compression_opts=9)
hf.create_dataset('mixedness_dr', data=mixedness_dr, compression="gzip", compression_opts=9)

hf.create_dataset('negativity_mid_dr', data=negativity_mid_dr, compression="gzip", compression_opts=9)

hf.create_dataset('spin_inequal_param_list_30a', data=spin_inequal_param_list_30a, compression="gzip", compression_opts=9)
hf.create_dataset('spin_inequal_param_list_30b', data=spin_inequal_param_list_30b, compression="gzip", compression_opts=9)
hf.create_dataset('spin_inequal_param_list_30c', data=spin_inequal_param_list_30c, compression="gzip", compression_opts=9)
hf.create_dataset('spin_inequal_param_list_30d', data=spin_inequal_param_list_30d, compression="gzip", compression_opts=9)
hf.close()

print('Filename: Data_GSM_ED_dynamics_to_equil_'  + h5_title, flush=True)


print('Data saved. Run done. Yay!!!', flush=True)