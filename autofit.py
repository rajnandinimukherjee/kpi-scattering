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

import pdb
def get_autofit_file(smeared, K=100, **kwargs):
    all_data = load_Kpi_data(data_dir, smeared, **kwargs)
    [pion_data, kaon_data, C_data, R_data, KKpipiC_data, piKpiKC_data, KKpipiR_data, piKpiKR_data] = all_data

    #==========================================================================
    # pion and kaon two-point functions
    cfgs = pion_data.shape[1]
    delta = 1

    pion = stat_object(del_t_binning(pion_data), fold=True, K=K, name='pion')
    pion.autofit(range(8,20), range(8,20), cosh, [2e+4, 0.08], 
                thin_list=[1,2], plot=True, savefig=True, 
                param_names=['A_p','m_p'], int_skip=2, pfilter=True)

    kaon = stat_object(del_t_binning(kaon_data,delta=1), fold=True, K=K, name='kaon')
    kaon.autofit(range(8,20), range(8,20), cosh, [2e+4, 0.08], 
                thin_list=[1,2], plot=True, savefig=True, 
                param_names=['A_k','m_k'], int_skip=2, pfilter=True)

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

    names, I = ['KKpipi', 'piKpiK'], ['12','32']
    ansatz_list = [KKpipi_ansatz, piKpiK_ansatz]
    ratios = np.empty(shape=(4,len(Delta)),dtype=object)

    for i in range(len(Delta)):

        p1_data = pion_data
        k1_data = np.zeros(shape=(T,cfgs,T))
        p2_data = np.zeros(shape=(T,cfgs,T))
        k2_data = np.zeros(shape=(T,cfgs,T))
        for t_src in range(T):
            for t in range(T):
                k1_data[t_src,:,t] = kaon_data[(t+t_src+delta)%T,:,(Delta[i]-t-delta)%T]
                p2_data[t_src,:,t] = pion_data[(t+t_src)%T,:,(Delta[i]-t)%T]
                k2_data[t_src,:,t] = kaon_data[t_src,:,(t+delta)%T]

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
                                ansatz_list[j//2], [1,1,0.8], thin_list=[1,2],
                                limit=Delta[i], #plot=True, savefig=True,
                                param_names=['A_'+ratios[j,i].name,
                                             'c0_'+ratios[j,i].name],
                                int_skip=2, m_p=m_pion, pfilter=True)
            best_fits.update({ratios[j,i].name:ratios[j,i].interval})

    c0 = np.array([[ratios[i,j].params[1] for j in range(len(Delta))] for i in range(4)])
    #==========================================================================
    # C_Kpi ratios for both isospin channels

    delta = 1
    D_data = pion_data*np.roll(kaon_data,-delta,axis=0)
    D = del_t_binning(D_data)
    C, R = del_t_binning(C_data), del_t_binning(R_data)
    
    T_arr = np.arange(T)
    def ATW_t_dep(c_KKpipi, c_piKpiK, t, m_p, m_k):
        numerator = c_KKpipi*np.exp(-m_k*t-m_p*(T-t))+c_piKpiK*np.exp(-m_k*t-m_p*(T-t))
        denominator = cosh([1,m_p],t)*cosh([1,m_k],t)
        return numerator/denominator

    def ATW(DEL, I, **kwargs):
        d = Delta.index(DEL) 
        I = int(I-0.5)
        avg_data = ATW_t_dep(c0[0+I,d], c0[2+I,d], np.arange(T), m_pion, m_kaon) 
        sampled_data = np.array([ATW_t_dep(ratios[0+I,d].params_dist[:,1], 
                                           ratios[2+I,d].params_dist[:,1], t,
                                           pion.params_dist[:,1], kaon.params_dist[:,1])
                                for t in range(T)])
        ATW = stat_object(sampled_data, K=K, data_avg=avg_data)
        return ATW

    ATW_12, ATW_32 = ATW(25,1/2), ATW(25,3/2)

    def ratio_ansatz(params, t, **kwargs):
        numerator = cosh([params[0],params[1]+m_pion+m_kaon],t)
        denominator = cosh([1,m_pion],t)*cosh([1,m_kaon],t)
        return numerator/denominator

    KpiI12 = stat_object(D+0.5*C-1.5*R, K=K)
    KpiI12_ratio = stat_object(KpiI12.org_samples/(pion.org_samples*kaon.org_samples),
            data_avg=KpiI12.org_data_avg/(pion.org_data_avg*kaon.org_data_avg),
            K=K, name='KpiI12_ratio', I=0.5)
    KpiI12_ratio_IB = stat_object((KpiI12_ratio.samples-ATW_12.samples), fold=True,
                            K=K, data_avg=(KpiI12_ratio.data_avg-ATW_12.data_avg))
    KpiI12_ratio_IB.autofit(range(8,20), range(4,15), ratio_ansatz, [1, 0.001],
                            thin_list=[1,2], pfliter=True, #plot=True, savefig=True,
                            param_names=['A_Kpi12','DE12'], int_skip=2)

    KpiI32 = stat_object(D-C, K=K)
    KpiI32_ratio = stat_object(KpiI32.org_samples/(pion.org_samples*kaon.org_samples),
            data_avg=KpiI32.org_data_avg/(pion.org_data_avg*kaon.org_data_avg),
            K=K, name='KpiI32_ratio', I=1.5)
    KpiI32_ratio_IB = stat_object((KpiI32_ratio.samples-ATW_32.samples), fold=True,
                            K=K, data_avg=(KpiI32_ratio.data_avg-ATW_32.data_avg))
    KpiI32_ratio_IB.autofit(range(8,20), range(4,15), ratio_ansatz, [1, 0.001],
                            thin_list=[1,2], pfilter=True, #plot=True, savefig=True,
                            param_names=['A_Kpi32','DE32'], int_skip=2)

    best_fits.update({'KpiI12_ratio':KpiI12_ratio_IB.interval,
                      'KpiI32_ratio':KpiI32_ratio_IB.interval})

    pickle.dump(best_fits, open('pickles/best_fits_sm'+str(smeared)+'.p','wb'))
    return best_fits


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

def del_t_binning(data, delta=0, binsize=96, **kwargs): 
    T, cfgs = data.shape[:2]
    T_bin = int(T/binsize)
    binned_data = np.zeros(shape=(T,cfgs*T_bin))
    data = np.roll(data,-delta,axis=0)

    for del_t in range(T):
        for c in range(cfgs):
            binned_data[del_t,c*T_bin:(c+1)*T_bin] = np.array([np.mean(data[(t*binsize):((t+1)*binsize),c,del_t],axis=0) for t in range(T_bin)])
    
    return binned_data


best_fits_smFalse = get_autofit_file(smeared=False)
best_fits_smTrue = get_autofit_file(smeared=True)

