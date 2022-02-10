'''
Author: Rajnandini Mukherjee

Uses the raw data on correlators generated from load_data.py to create
all correlation functions as class objects of stat_object (calculates
various statistical information on correlator data), and then fits them
over various fit ranges (autofit) to find a 'best fit' and saves in
best_fits_sm(True/False).p file for speedy import later without needing
to find best fit over and over again. See correlation_functions.py for eg.

'''

data_dir = 'correlators/'

import numpy as np
import matplotlib.pyplot as plt
import pickle
from load_data import load_Kpi_data
from fit_routine import *
T = 96

from scipy.linalg import block_diag
def cov_block_diag(obj):
    N = len(obj.corrs)
    covs = np.empty(N,dtype=object)
    for n in range(N):
        (s,e,t) = obj.corrs[n].interval
        covs[n] = obj.corrs[n].COV[s:e+1:t, s:e+1:t]

    return block_diag(*covs)

smeared = bool(int(input('smeared(0/1):')))
K=100
import pdb
#def get_autofit_file(smeared, K=100, **kwargs):

def del_t_binning(data, delta=0, binsize=96, **kwargs): 
    T, cfgs = data.shape[:2]
    T_bin = int(T/binsize)
    binned_data = np.zeros(shape=(T,cfgs*T_bin))
    data = np.roll(data,-delta,axis=0)

    for del_t in range(T):
        for c in range(cfgs):
            binned_data[del_t,c*T_bin:(c+1)*T_bin] = np.array([np.mean(data[(t*binsize):((t+1)*binsize),c,del_t],axis=0) for t in range(T_bin)])
    
    return binned_data

all_data = load_Kpi_data(data_dir, smeared)
[pion_data, kaon_data, C_data, R_data, KKpipiC_data, piKpiKC_data, KKpipiR_data, piKpiKR_data] = all_data

#==========================================================================
# pion and kaon two-point functions
cfgs = pion_data.shape[1]
delta = 1

pion = stat_object(del_t_binning(pion_data), fold=True, K=K, name='pion')
pion.autofit(range(5,15), range(5,15), cosh, [2e+4, 0.08], 
            thin_list=[1,2], param_names=['A_p','m_p'])

kaon = stat_object(del_t_binning(kaon_data,delta=1), fold=True, K=K, name='kaon')
kaon.autofit(range(5,20), range(5,15), cosh, [1e+3, 0.28], 
            thin_list=[1,2], param_names=['A_k','m_k'])

m_pion, m_kaon = pion.params[1], kaon.params[1]

best_fits = {pion.name:pion.interval, kaon.name:kaon.interval}

#============================================================================
# KKpipi and piKpiK ratios

Delta = [15,20,25,30,40]

KKpipiD_data = np.zeros(shape=(len(Delta),T,cfgs,T))
piKpiKD_data = np.zeros(shape=(len(Delta),T,cfgs,T)) 
KKpipiD = np.zeros(shape=(len(Delta),T,cfgs)) 
piKpiKD = np.zeros(shape=(len(Delta),T,cfgs))

for Del in Delta:
    for t_src in range(T):
        for t in range(T):
            KKpipiD_data[Delta.index(Del),t_src,:,t] = pion_data[t_src,:,t]*kaon_data[(t+t_src+delta)%T,:,(Del-t-delta)%T]
            piKpiKD_data[Delta.index(Del),t_src,:,t] = pion_data[(t+t_src)%T,:,(Del-t)%T]*kaon_data[t_src,:,(t+delta)%T]
    KKpipiD[Delta.index(Del),:,:] = del_t_binning(KKpipiD_data[Delta.index(Del),:,:,:])
    piKpiKD[Delta.index(Del),:,:] = del_t_binning(piKpiKD_data[Delta.index(Del),:,:,:])

KKpipiC = np.array([del_t_binning(KKpipiC_data[i,:,:,:]) for i in range(len(Delta))])
KKpipiR = np.array([del_t_binning(KKpipiR_data[i,:,:,:]) for i in range(len(Delta))])
piKpiKC = np.array([del_t_binning(piKpiKC_data[i,:,:,:]) for i in range(len(Delta))])
piKpiKR = np.array([del_t_binning(piKpiKR_data[i,:,:,:]) for i in range(len(Delta))])

KKpipi32 = np.array([stat_object(KKpipiD[i,:,:]-KKpipiC[i,:,:], K=K) for i in range(len(Delta))], dtype=object)
KKpipi12 = np.array([stat_object(KKpipiD[i,:,:]+0.5*KKpipiC[i,:,:]-1.5*KKpipiR[i,:,:], K=K) for i in range(len(Delta))], dtype=object)
piKpiK32 = np.array([stat_object(piKpiKD[i,:,:]-piKpiKC[i,:,:], K=K) for i in range(len(Delta))], dtype=object)
piKpiK12 = np.array([stat_object(piKpiKD[i,:,:]+0.5*piKpiKC[i,:,:]-1.5*piKpiKR[i,:,:], K=K) for i in range(len(Delta))], dtype=object)

ATW_corrs = np.array([KKpipi12, KKpipi32, piKpiK12, piKpiK32], dtype=object)

def KKpipi_ansatz(params, t, **kwargs):
    c0, temp = params
    temp = 1
    return np.zeros(t.shape)+c0 

def piKpiK_ansatz(params, t, **kwargs):
    c0, temp = params
    temp = 1
    return np.zeros(t.shape)+c0 

names, I = ['KKpipi', 'piKpiK'], ['12','32']
ansatz_list = [KKpipi_ansatz, piKpiK_ansatz]
ratios = np.empty(shape=(4,len(Delta)),dtype=object)

for i in range(len(Delta)):

    p1_data = pion_data
    k1_data = np.zeros(shape=(T,cfgs,T))
    p2_data = np.zeros(shape=(T,cfgs,T))
    k2_data = kaon_data
    for t_src in range(T):
        for t in range(T):
            k1_data[t_src,:,t] = kaon_data[t_src,:,(Delta[i]-t)%T]
            p2_data[t_src,:,t] = pion_data[t_src,:,(Delta[i]-t)%T]

    p1 = stat_object(del_t_binning(p1_data), K=K)
    k1 = stat_object(del_t_binning(k1_data), K=K)
    p2 = stat_object(del_t_binning(p2_data), K=K)
    k2 = stat_object(del_t_binning(k2_data), K=K)

    denoms = np.array([p1.samples*k1.samples, p2.samples*k2.samples])
    pk_avgs = np.array([p1.data_avg*k1.data_avg, p2.data_avg*k2.data_avg])

    for j in range(4):
        ratio_name = names[j//2]+I[j%2]+'DEL'+str(Delta[i])
        ratios[j,i] = stat_object(ATW_corrs[j][i].samples/denoms[j//2],K=K,
                data_avg=ATW_corrs[j][i].data_avg/pk_avgs[j//2],name=ratio_name)
        ratios[j,i].autofit(range(4,int(Delta[i]/2)-1), range(6,14),
                            ansatz_list[j//2], [1,0], thin_list=[1,2],
                            limit=Delta[i],param_names=['c0_'+ratios[j,i].name, 'temp'],
                            int_skip=2, correlated=True)
        #print(f'{ratios[j,i].name} fit dict:\n{ratios[j,i].fit_dict}')
        best_fits.update({ratios[j,i].name:ratios[j,i].interval})

#==========================================================================
# C_Kpi ratios for both isospin channels

delta = 1
D_data = pion_data*np.roll(kaon_data,-delta,axis=0)
D = del_t_binning(D_data)
C, R = del_t_binning(C_data), del_t_binning(R_data)

KpiI12 = stat_object(D+0.5*C-1.5*R, K=K)
KpiI12_ratio = stat_object(KpiI12.org_samples/(pion.org_samples*kaon.org_samples),
        data_avg=KpiI12.org_data_avg/(pion.org_data_avg*kaon.org_data_avg),
        K=K, name='KpiI12_ratio', I=0.5)

KpiI32 = stat_object(D-C, K=K)
KpiI32_ratio = stat_object(KpiI32.org_samples/(pion.org_samples*kaon.org_samples),
        data_avg=KpiI32.org_data_avg/(pion.org_data_avg*kaon.org_data_avg),
        K=K, name='KpiI32_ratio', I=1.5)

for corr in [KpiI12_ratio, KpiI32_ratio]: # choosing random guess intervals for ratio fits
    corr.interval = (10,17,1)
    corr.x = np.arange(10,17+1,1)
    corr.correlated = True

def CKpi_ansatz(params, t, **kwargs):
    A_Ckpi, m_p, m_k, DE, c0_KKpipi, c0_piKpiK = params
    EKpi = m_p + m_k + DE
    denom = cosh([1,m_p],t,T=T)*cosh([1,m_k],t,T=T)
    interesting = A_Ckpi*cosh([1,EKpi],t,T=T)/denom
    c0_KKpipi = c0_KKpipi*np.exp(m_k*delta)
    c0_piKpiK = c0_piKpiK*np.exp(-m_k*delta)
    RTW_KKpipi = (c0_KKpipi**2)*np.exp(-m_p*t -m_k*(T-t))/denom
    RTW_piKpiK = (c0_piKpiK**2)*np.exp(-m_k*t -m_p*(T-t))/denom

    return interesting + RTW_KKpipi + RTW_piKpiK

def combined_ansatz(params, t, **kwargs):
    A_p, m_p, A_k, m_k = params[:4]
    c0_KKpipi, c0_piKpiK = params[4:6]
    A_CKpi, DE = params[6:]

    pion_part = cosh([A_p,m_p],pion.x,T=T)
    kaon_part = cosh([A_k,m_k],kaon.x,T=T)

    I_idx = int(kwargs['I']-0.5)
    KKpipi_part = KKpipi_ansatz([c0_KKpipi,0],ratios[0+I_idx,3].x)
    piKpiK_part = piKpiK_ansatz([c0_piKpiK,0],ratios[2+I_idx,3].x)

    CKpi_part = CKpi_ansatz([A_CKpi, m_p, m_k, DE, c0_KKpipi, c0_piKpiK], t)

    return np.concatenate((pion_part, kaon_part, KKpipi_part, piKpiK_part,
                           CKpi_part), axis=0)

guess = [2e+4, 0.08, 1e+3, 0.28, 1.3, 1.3, 1, 0.001]

CMBI12 = stat_object([pion,kaon,ratios[0,3],ratios[2,3],KpiI12_ratio], K=K,
                    name='CMBI12', object_type='combined')
CMBI12.fit((0,CMBI12.T-1,1), combined_ansatz, guess, COV_model=cov_block_diag, 
            I=0.5, index=4)
CMBI12.autofit(range(5,15), range(5,15), combined_ansatz, guess,
               index=4, I=0.5, COV_model=cov_block_diag,
               param_names=['A_p','m_p','A_k','m_k',
               'c0_KKpipi', 'c0_piKpiK', 'A_CKpi', 
               'DE12'])

CMBI32 = stat_object([pion,kaon,ratios[1,3],ratios[3,3],KpiI32_ratio], K=K,
                    name='CMBI32', object_type='combined')
CMBI32.fit((0,CMBI32.T-1,1), combined_ansatz, guess, COV_model=cov_block_diag,
            I=1.5, index=4)
CMBI32.autofit(range(5,15), range(5,15), combined_ansatz, guess,
               index=4, I=1.5, COV_model=cov_block_diag,
               param_names=['A_p','m_p','A_k','m_k',
               'c0_KKpipi', 'c0_piKpiK', 'A_CKpi', 
               'DE32'])

best_fits.update({'KpiI12_ratio':CMBI12.corrs[4].interval,
                  'KpiI32_ratio':CMBI32.corrs[4].interval})

pickle.dump(best_fits, open('pickles/best_fits_sm'+str(smeared)+'.p','wb'))
#return [pion, kaon, ratios, KpiI12_ratio, KpiI32_ratio], best_fits


