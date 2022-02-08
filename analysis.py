'''
Author: Rajnandini Mukherjee

Primary analysis code which runs loops of fit ranges of the C_Kpi correlator,
in a combined fit with pion and kaon two-point functions, and other correlation
functions with information on around-the-world (ATW) matrix elements, to single
out the value of Delta E_Kpi and hence the scattering length. The data for point
and smeared sources is also combined for one big global fit in each of the 
isospin channels.'''


data_dir = 'correlators/'

import numpy as np
import matplotlib.pyplot as plt
from plot_settings import plotparams
plt.rcParams.update(plotparams)

T, K = 96, 100 # lattice time extent, number of bootstrap samples

from fit_routine import *
from scipy.linalg import block_diag
from numpy.linalg import svd, inv

def svd_model(cov, cuts=1, **kwargs):
    ''' models the covariance matrix by removing smallest
    singular values of the inverse matrix'''

    u, s, vt = svd(cov)
    s_inv = 1/s
    for i in range(cuts):
        s_inv[np.argmax(s_inv)] = 0
    L = vt.T@np.diag(s_inv**0.5)
    return L

def cov_block_diag(obj):
    '''gives block diagonal form to the covarianc matrix'''

    N = len(obj.corrs)
    covs = np.empty(N,dtype=object)
    for n in range(N):
        (s,e,t) = obj.corrs[n].interval
        covs[n] = obj.corrs[n].COV[s:e+1:t, s:e+1:t]

    return block_diag(*covs)

def KKpipi_ansatz(params, t, **kwargs):
    c0, temp = params
    temp = 1
    return c0

def piKpiK_ansatz(params, t, **kwargs):
    c0, temp = params
    temp = 1
    return c0

def CKpi_ansatz(params, t, **kwargs):
    A_CKpi, m_p, m_k, DE, c0_KKpipi, c0_piKpiK = params
    EKpi = m_p + m_k + DE
    denom = cosh([1,m_p],t,T=T)*cosh([1,m_k],t,T=T)
    interesting = A_CKpi*cosh([1,EKpi],t,T=T)/denom
    RTW_KKpipi = (c0_KKpipi**2)*np.exp(-m_p*t -m_k*(T-t))/denom
    RTW_piKpiK = (c0_piKpiK**2)*np.exp(-m_k*t -m_p*(T-t))/denom

    return interesting + RTW_KKpipi + RTW_piKpiK

L = 48
c1 = -2.837297
c2 = 6.375183

def scat_length(params, **kwargs):
    A_p, m_p = params[:2]
    A_k, A_k_sm, m_k = params[2:5]
    c0_KKpipi, c0_KKpipi_sm = params[5:7]
    c0_piKpiK, c0_piKpiK_sm = params[7:9]
    A_CKpi, A_CKpi_sm, DE = params[9:12]

    k0 = DE
    k1 = 2*np.pi*(m_p+m_k)/(m_p*m_k*(L**3))
    k2 = k1*c1/L
    k3 = k1*c2/(L**2)
    roots = np.roots([k3,k2,k1,k0])
    a = np.real(roots[np.isreal(roots)][0])
    return a*m_p

def alt_scat_length(params, **kwargs):
    A_p, m_p = params[:2]
    A_k, A_k_sm, m_k = params[2:5]
    c0_KKpipi, c0_KKpipi_sm = params[5:7]
    c0_piKpiK, c0_piKpiK_sm = params[7:9]
    A_CKpi, A_CKpi_sm, DE = params[9:12]

    mu = m_p*m_k/(m_p+m_k)
    k0 = DE
    k1 = 2*np.pi/(mu*(L**3))
    k2 = k1*c1/L
    k3 = k1*c2/(L**2)
    roots = np.roots([k3,k2,k1,k0])
    a = np.real(roots[np.isreal(roots)][0])
    return a*mu

def pt_sm_combined(params, t, **kwargs):
    A_p, m_p = params[:2]
    A_k, A_k_sm, m_k = params[2:5]
    c0_KKpipi, c0_KKpipi_sm = params[5:7]
    c0_piKpiK, c0_piKpiK_sm = params[7:9]
    A_CKpi, A_CKpi_sm, DE = params[9:12]

    pion_part = cosh([A_p,m_p],pion.x,T=T)
    kaon_part = cosh([A_k,m_k],kaon.x,T=T)
    kaon_sm_part = cosh([A_k_sm, m_k], kaon_sm.x, T=T)

    I_idx = int(kwargs['I']-0.5)
    KKpipi_part = KKpipi_ansatz([c0_KKpipi,0],ratios[0+I_idx,2].x)
    piKpiK_part = piKpiK_ansatz([c0_piKpiK,0],ratios[2+I_idx,2].x)
    KKpipi_sm_part = KKpipi_ansatz([c0_KKpipi_sm,A_KKpipi_sm,m_p],ratios_sm[0+I_idx,2].x)
    piKpiK_sm_part = piKpiK_ansatz([c0_piKpiK_sm,A_piKpiK_sm,m_p],ratios_sm[2+I_idx,2].x)

    CKpi_part = CKpi_ansatz([A_CKpi, m_p, m_k, DE, c0_KKpipi, c0_piKpiK], 
                            KpiI12_ratio.x if I_idx==0 else KpiI32_ratio.x)
    CKpi_sm_part = CKpi_ansatz([A_CKpi_sm, m_p, m_k, DE, c0_KKpipi_sm, c0_piKpiK_sm], t)

    return np.concatenate((pion_part, kaon_part, kaon_sm_part, KKpipi_part, 
                           piKpiK_part, KKpipi_sm_part, piKpiK_sm_part,
                           CKpi_part, CKpi_sm_part))

from correlation_functions import *
pion, kaon, ratios, KpiI12_ratio, KpiI32_ratio = get_correlators(data_dir, False) 
pion2, kaon_sm, ratios_sm, KpiI12_sm_ratio, KpiI32_sm_ratio = get_correlators(data_dir, True)

guess = [2e+4, 0.08, 1e+3, 1e+2, 0.28, 1.33, 1.33, 0.75, 0.75, 1, 1, 0.001]

# modified hyperweights for combined fits
hyperweights = {'pvalue_cost':0.5,
                'fit_stbl_cost':1,
                'err_cost':1,
                'val_stbl_cost':1}

pt_sm_corrI12 = stat_object([pion, kaon, kaon_sm, ratios[0,2], ratios[2,2],
                          ratios_sm[0,2], ratios_sm[2,2], KpiI12_ratio,
                          KpiI12_sm_ratio], object_type='combined', K=K,
                          name='pt_sm_corrI12')
pt_sm_corrI12.fit((0,pt_sm_corrI12.T-1,1), pt_sm_combined, guess, index=8, 
                  COV_model=cov_block_diag, 
                    I=0.5)
pt_sm_corrI12.autofit(range(8,18), range(5,15), pt_sm_combined, guess,
                  COV_model=cov_block_diag, hyperweights=hyperweights, 
                  param_names=['A_p', 'm_p', 'A_k', 'A_k_sm', 'm_k',
                  'c0_KKpipi', 'c0_KKpipi_sm', 'c0_piKpiK', 'c0_piKpiK_sm',
                  'A_CKpi', 'A_CKpi_sm', 'DE12'], I=0.5,
                    index=8, pfliter=True, calc_func=[scat_length, alt_scat_length],
                    calc_func_names=['m_p_a0_I12','mu_a0_I12'])
import pprint as pp
pp.pprint(pt_sm_corrI12.fit_dict)

pt_sm_corrI32 = stat_object([pion, kaon, kaon_sm, ratios[1,2], ratios[3,2],
                          ratios_sm[1,2], ratios_sm[3,2], KpiI32_ratio,
                          KpiI32_sm_ratio], object_type='combined', K=K,
                          name='pt_sm_corrI32')
pt_sm_corrI32.fit((0,pt_sm_corrI32.T-1,1), pt_sm_combined, guess, index=8, 
                  COV_model=cov_block_diag, 
                    I=1.5)
pt_sm_corrI32.autofit(range(5,15), range(5,15), pt_sm_combined, guess,
                  COV_model=cov_block_diag, hyperweights=hyperweights, 
                  param_names=['A_p', 'm_p', 'A_k', 'A_k_sm', 'm_k',
                  'c0_KKpipi', 'c0_KKpipi_sm', 'c0_piKpiK', 'c0_piKpiK_sm',
                  'A_CKpi', 'A_CKpi_sm', 'DE32'], I=1.5,
                    index=8, pfliter=True, calc_func=[scat_length, alt_scat_length],
                    calc_func_names=['m_p_a0_I32','mu_a0_I32'])

pp.pprint(pt_sm_corrI32.fit_dict)

fit_intervals = {corr.name:corr.interval for corr in pt_sm_corrI12.corrs}
fit_intervals.update({corr.name:corr.interval for corr in pt_sm_corrI32.corrs})
pickle.dump(fit_intervals,open('pickles/fit_intervals.p','wb'))

df12, dict12 = pt_sm_corrI12.autofit_df, pt_sm_corrI12.autofit_dict
df32, dict32 = pt_sm_corrI32.autofit_df, pt_sm_corrI32.autofit_dict
pickle.dump([df12, df32], open('pickles/pt_sm_dfs.p','wb'))
pickle.dump([dict12, dict32], open('pickles/pt_sm_dicts.p','wb'))

pt_sm_corrI12.autofit_plot(int_skip=2, plot_params=[14,15], savefig=True, plothist=True, 
                        hist_deltas=range(8,15), hist_t_min=8)
plt.close('all')
pt_sm_corrI32.autofit_plot(int_skip=2, plot_params=[14,15], savefig=True, plothist=True, 
                        hist_deltas=range(8,15), hist_t_min=8)
plt.close('all')
