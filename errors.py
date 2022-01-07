import numpy as np
from error_support import *

L = 48
a_inv = 1.73 # GeV
Lambda = 0.35 # GeV

# from https://doi.org/10.1016/S0550-3213(01)00121-3
mu = 0.77
L1, L2, L3, L4 = 0.53/1000, 0.71/1000, -2.72/1000, 0
L5, L6, L7, L8 = 0.91/1000, 0, -0.32/1000, 0.62/1000
Lscat = 2*L1 +2*L2 + L3 - 2*L4 - 0.5*L5 + 2*L6 + L8
m_eta = 0.547862/1.73 #in lattice units
F0 = 0.0871

# from http://dx.doi.org/10.1103/PhysRevD.96.034510 
c1 = -2.837297
c2 = 6.375183 
c3 = -8.311951 

# from PDG
m_p_pm, m_p_0 = 0.13957061, 0.1349770
m_p_pdg = (2/3)*m_p_pm + (1/3)*m_p_0
m_k_pdg = 0.493677
pdg = {'m_p':m_p_pdg/1.73, 'm_k':m_k_pdg/1.73}

def load_errors(lat12, lat32, **kwargs):
    errors = {}
    errors_pc = {}
#=============================================================================
# cutoff effects

    cutoff = (Lambda/a_inv)**2
    errors.update({'cutoff effect':{'I=1/2':cutoff, 'I=3/2':cutoff}})
    errors_pc.update({'cutoff effect':{'I=1/2':100*round(cutoff,2), 
                                       'I=3/2':100*round(cutoff,2)}})

#=============================================================================
# finite volume effect

    vol12, vol32 = np.exp(-lat12['m_p']*L), np.exp(-lat32['m_p']*L)
    errors.update({'finite volume':{'I=1/2':vol12, 'I=3/2':vol32}})
    errors_pc.update({'finite volume':{'I=1/2':100*round(vol12,2), 'I=3/2':100*round(vol32,2)}})

#==============================================================================
# chiral perturbation theory calculations for error from unphysical quark masses


    a012_lat, a032_lat = a012(**lat12), a032(**lat32)
    a012_pdg, a032_pdg = a012(**pdg), a032(**pdg)

    errors.update({'quark masses':{'I=1/2':np.abs(a012_lat-a012_pdg),
                              'I=3/2':np.abs(a032_lat-a032_pdg)}})
    errors_pc.update({'quark masses':{'I=1/2':err_pc(a012_lat,a012_pdg),
                              'I=3/2':err_pc(a032_lat,a032_pdg)}}) 

#=============================================================
# error from truncating Luscher's formula for scattering length

    a0_12_order_L5, a0_32_order_L5 = a0_O5(**lat12), a0_O5(**lat32)
    a0_12_order_L6, a0_32_order_L6 = a0_O6(**lat12), a0_O6(**lat32)

    errors.update({'O(L^5) truncation':{'I=1/2':np.abs(a0_12_order_L5-a0_12_order_L6),
                                        'I=3/2':np.abs(a0_32_order_L5-a0_32_order_L6)}})
    errors_pc.update({'O(L^5) truncation':{'I=1/2':err_pc(a0_12_order_L5,a0_12_order_L6),
                                        'I=3/2':err_pc(a0_32_order_L5,a0_32_order_L6)}})

#================================================================
# error from fit systematics

    fit_sys12 = lat12['a0_sys']
    fit_val12 = lat12['a0']

    fit_sys32 = lat32['a0_sys']
    fit_val32 = lat32['a0']

    errors.update({'fit systematics':{'I=1/2':fit_sys12, 'I=3/2':fit_sys32}})
    errors_pc.update({'fit systematics':{'I=1/2':err_pc(fit_val12,fit_val12+fit_sys12),
                                         'I=3/2':err_pc(fit_val32,fit_val32+fit_sys32)}})

    return errors, errors_pc


























