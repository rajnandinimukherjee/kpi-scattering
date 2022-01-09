'''
Author: Rajnandini Mukherjee

This script studies the effect of accounting for the
around-the-world terms in the spectral decomposition of 
the C_Kpi correlator and how it effects the determination
of the value of the Delta E_Kpi and hence the scattering
length.'''


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
    c0, A, m_p = params
    if 'm_pion' in kwargs.keys():
        m_p = kwargs['m_pion']
    return c0 + A*np.exp(2*m_p*t)

def piKpiK_ansatz(params, t, **kwargs):
    c0, A, m_p = params
    if 'm_pion' in kwargs.keys():
        m_p = kwargs['m_pion']
    return c0 + A*np.exp(-2*m_p*t)

def CKpi_ansatz(params, t, ATW=True, **kwargs):
    A_CKpi, m_p, m_k, DE, c0_KKpipi, c0_piKpiK = params
    EKpi = m_p + m_k + DE
    denom = cosh([1,m_p],t,T=T)*cosh([1,m_k],t,T=T)
    interesting = A_CKpi*cosh([1,EKpi],t,T=T)/denom
    ATW_KKpipi = ATW*c0_KKpipi*np.exp(-m_p*t -m_k*(T-t))/denom
    ATW_piKpiK = ATW*c0_piKpiK*np.exp(-m_k*t -m_p*(T-t))/denom

    return interesting + ATW_KKpipi + ATW_piKpiK

L = 48
c1 = -2.837297
c2 = 6.375183

def scat_length(params, **kwargs):
    A_p, m_p = params[:2]
    A_k, A_k_sm, m_k = params[2:5]
    A_KKpipi, A_KKpipi_sm, c0_KKpipi, c0_KKpipi_sm = params[5:9]
    A_piKpiK, A_piKpiK_sm, c0_piKpiK, c0_piKpiK_sm = params[9:13]
    A_CKpi, A_CKpi_sm, DE = params[13:16]

    k0 = DE
    k1 = 2*np.pi*(m_p+m_k)/(m_p*m_k*(L**3))
    k2 = k1*c1/L
    k3 = k1*c2/(L**2)
    roots = np.roots([k3,k2,k1,k0])
    a = np.real(roots[np.isreal(roots)][0])
    return a*m_p

def combined_ansatz(params, t, **kwargs):

    A_p, m_p, A_k, m_k = params[:4]
    A_KKpipi, c0_KKpipi = params[4:6]
    A_piKpiK, c0_piKpiK = params[6:8]
    A_CKpi, DE = params[8:]

    pion_part = cosh([A_p,m_p],pion.x,T=T)
    kaon_part = cosh([A_k,m_k],kaon.x,T=T)

    I_idx = int(kwargs['I']-0.5)
    KKpipi_part = KKpipi_ansatz([c0_KKpipi,A_KKpipi,m_p],ratios[0+I_idx,2].x)
    piKpiK_part = piKpiK_ansatz([c0_piKpiK,A_piKpiK,m_p],ratios[2+I_idx,2].x)

    CKpi_part = CKpi_ansatz([A_CKpi, m_p, m_k, DE, c0_KKpipi, c0_piKpiK],
                            KpiI12_ratio.x if I_idx==0 else KpiI32_ratio.x)

    return np.concatenate((pion_part, kaon_part, KKpipi_part, piKpiK_part,
                           CKpi_part), axis=0)

def pt_sm_combined(params, t, **kwargs):
    A_p, m_p = params[:2]
    A_k, A_k_sm, m_k = params[2:5]
    A_KKpipi, A_KKpipi_sm, c0_KKpipi, c0_KKpipi_sm = params[5:9]
    A_piKpiK, A_piKpiK_sm, c0_piKpiK, c0_piKpiK_sm = params[9:13]
    A_CKpi, A_CKpi_sm, DE = params[13:16]

    pion_part = cosh([A_p,m_p],pion.x,T=T)
    kaon_part = cosh([A_k,m_k],kaon.x,T=T)
    kaon_sm_part = cosh([A_k_sm, m_k], kaon_sm.x, T=T)

    I_idx = int(kwargs['I']-0.5)
    KKpipi_part = KKpipi_ansatz([c0_KKpipi,A_KKpipi,m_p],ratios[0+I_idx,2].x)
    piKpiK_part = piKpiK_ansatz([c0_piKpiK,A_piKpiK,m_p],ratios[2+I_idx,2].x)
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

fit_intervals = pickle.load(open('pickles/fit_intervals.p','rb'))
for corr in [pion, kaon, KpiI12_ratio, KpiI32_ratio,
            kaon_sm, KpiI12_sm_ratio, KpiI32_sm_ratio]:
    (s,e,t) = fit_intervals[corr.name]
    corr.interval = (s,e,t)
    corr.x = np.arange(s,e+1,t)

guess = [2e+4, 0.08, 1e+3, 1e+2, 0.28, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.001]

pt_sm_corrI12 = stat_object([pion, kaon, kaon_sm, ratios[0,2], ratios[2,2],
                          ratios_sm[0,2], ratios_sm[2,2], KpiI12_ratio,
                          KpiI12_sm_ratio], object_type='combined', K=K,
                          name='pt_sm_corrI12')
pt_sm_corrI12.fit((0,pt_sm_corrI12.T-1,1), pt_sm_combined, guess, index=8, 
                  COV_model=cov_block_diag, I=0.5,
                  calc_func=[scat_length], calc_func_names=['a0_I12'], 
                  param_names=['A_p', 'm_p', 'A_k', 'A_k_sm', 'm_k',
                  'A_KKpipi', 'A_KKpipi_sm', 'c0_KKpipi', 'c0_KKpipi_sm',
                  'A_piKpiK', 'A_piKpiK_sm', 'c0_piKpiK', 'c0_piKpiK_sm',
                  'A_CKpi', 'A_CKpi_sm', 'DE12'], plot=False)
pt_sm_corrI32 = stat_object([pion, kaon, kaon_sm, ratios[1,2], ratios[3,2],
                          ratios_sm[1,2], ratios_sm[3,2], KpiI32_ratio,
                          KpiI32_sm_ratio], object_type='combined', K=K,
                          name='pt_sm_corrI32')
pt_sm_corrI32.fit((0,pt_sm_corrI32.T-1,1), pt_sm_combined, guess, index=8, 
                  COV_model=cov_block_diag, I=1.5,
                  calc_func=[scat_length], calc_func_names=['a0_I12'], 
                  param_names=['A_p', 'm_p', 'A_k', 'A_k_sm', 'm_k',
                  'A_KKpipi', 'A_KKpipi_sm', 'c0_KKpipi', 'c0_KKpipi_sm',
                  'A_piKpiK', 'A_piKpiK_sm', 'c0_piKpiK', 'c0_piKpiK_sm',
                  'A_CKpi', 'A_CKpi_sm', 'DE12'], plot=False)

[pt_sm_corrI12.autofit_df, pt_sm_corrI32.autofit_df] = pickle.load(open('pickles/pt_sm_dfs.p','rb'))
[pt_sm_corrI12.autofit_dict, pt_sm_corrI32.autofit_dict] = pickle.load(open('pickles/pt_sm_dicts.p','rb'))

def sep_ATW(params, t):
    [m_p, m_k, c0_KKpipi, c0_piKpiK, A_CKpi, DE] = params 
    denom = cosh([1,m_p],t,T=T)*cosh([1,m_k],t,T=T)
    interesting = cosh([A_CKpi,m_p+m_k+DE],t,T=T)/denom
    ATW_KKpipi = c0_KKpipi*np.exp(-m_p*t -m_k*(T-t))/denom
    ATW_piKpiK = c0_piKpiK*np.exp(-m_k*t -m_p*(T-t))/denom
    ATW = ATW_KKpipi+ATW_piKpiK
    return [interesting, ATW]

def CKpi_ansatz_sep(params, t, ATW=True, **kwargs):
    A_CKpi, DE = params
    if kwargs['I']==0.5:
        [A_CKpi, m_p, m_k, c0_KKpipi, c0_piKpiK] = pt_sm_corrI12.params[[14,1,4,8,12]]
    else:
        [A_CKpi, m_p, m_k, c0_KKpipi, c0_piKpiK] = pt_sm_corrI32.params[[14,1,4,8,12]]

    if not ATW:
        c0_KKpipi, c0_piKpiK = 0, 0

    interesting, ATW = sep_ATW([m_p, m_k, c0_KKpipi, c0_piKpiK, A_CKpi, DE], t)
    return interesting+ATW
    

def ATW_study(I, **kwargs):
    if I==0.5:
        glob, corr = pt_sm_corrI12, KpiI12_sm_ratio
    else:
        glob, corr = pt_sm_corrI32, KpiI32_sm_ratio
    
    param_slice = [1,4,8,12,14,15]
    params = glob.params[param_slice]
    params_dist = glob.params_dist[:,param_slice]
    avg_cosh, avg_ATW = sep_ATW(params,np.arange(T))
    errs = np.array([sep_ATW(params_dist[k,:],np.arange(T))
                    for k in range(K)])
    err_cosh = np.array([st_dev(errs[:,0,t],mean=avg_cosh[t]) for t in range(T)])
    err_ATW = np.array([st_dev(errs[:,1,t],mean=avg_ATW[t]) for t in range(T)])

    plt.figure()
    plt.errorbar(np.arange(T), avg_cosh, yerr=err_cosh, 
                 linestyle='None', fmt='o', label='cosh')
    plt.errorbar(np.arange(T), avg_ATW, yerr=err_ATW, 
                 linestyle='None', fmt='o', label='ATW')
    plt.xlabel('$t$')
    title = 'I=1/2' if I==0.5 else 'I=3/2'
    plt.title(title)
    plt.legend()
    iso = '12' if I==0.5 else '32'
    plt.savefig('plots/cosh_ATW_ratio_I'+iso+'.pdf')

    corr.fit(corr.interval, CKpi_ansatz_sep, params[-2:], ATW=False)
    params[-2:]=corr.params
    avg_no_ATW_cosh, temp = sep_ATW(params,np.arange(T))
    params_dist[:,-2:] = corr.params_dist
    errs_no_ATW = np.array([sep_ATW(params_dist[k,:],np.arange(T))
                    for k in range(K)])
    #err_no_ATW_cosh = np.array([st_dev(errs_no_ATW[:,0,t],
    #                            mean=avg_no_ATW_cosh[t]) for t in range(T)])
    ratio_ATW = avg_no_ATW_cosh/avg_cosh
    ratio_err = np.array([st_dev(errs_no_ATW[:,0,t]/errs[:,0,t],mean=ratio_ATW[t])
                        for t in range(T)])

    plt.figure()
    plt.errorbar(np.arange(T), ratio_ATW, yerr=ratio_err, capsize=2, fmt='o',
                 linestyle='None', label='no_ATW/with_ATW')
    plt.xlabel('$t$')
    plt.text(30,ratio_ATW[0],s=f'slope={ratio_ATW[15]-ratio_ATW[14]}') 
    plt.legend()
    plt.title(title)
    plt.savefig('plots/no_ATW_vs_with_ATW_I'+iso+'.pdf')

    return ratio_ATW, ratio_err
    

rat12, err12 = ATW_study(I=0.5)
rat32, err32 = ATW_study(I=1.5)
