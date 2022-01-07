import numpy as np
import pickle

#smeared = bool(int(input('smearing(0/1): ')))
T, seed = 96, 1

import glob
from os.path import basename, splitext
def cfgs_list(direc):
    cfgs_list = np.array([],dtype=object)
    for fn in glob.glob(direc+'*'):
        bn = basename(fn)
        cfsplit = (splitext(bn)[0]).rsplit('_')
        cfnumber = len(cfsplit) -1
        cfpart = cfsplit[cfnumber]
        cfgs_list = np.append(cfgs_list,cfpart)

    return np.array(sorted(cfgs_list, key=float))

import collections
def TC(files):
    # returns a dictionary of configurations available for each time source from a list of files
    TC = collections.defaultdict(list)
    for name in files:
        [t_src, cfnm] = name.rsplit('.')
        TC[t_src].append(cfnm)
    TC = collections.OrderedDict(sorted(TC.items()))
    for k in TC.keys():
        TC[k].sort()
    return TC

import h5py
def import_data(TC_dict, direc, prefix):
    # reads the dictionary of relevant configurations and imports files corresponding to them
    data = np.array([[h5py.File(direc+str(t_src)+'.'+cf+'.h5','r')[prefix+str(t_src)+'/correlator']['re'] for cf in TC_dict[str(t_src)]] for t_src in range(T)])
    return data

#===========================================================================================
# importing pion, kaon data

import time
def load_Kpi_data(dir_in_use, smeared, **kwargs):
    start = time.time()
    print('Loading Kpi data from '+dir_in_use+' for smearing='+str(smeared)+'...')

    pion_dir = dir_in_use+'corr_pion_t0/g5g5_llll_1111_mom1_0_0_0_mom2_0_0_0/g5a_0_g5b_dt_'
    # loading the unsmeared kaon to make sure to use the same gauge configs as in the unsmeared case (seems like there is more smeared data)
    kaon_unsmeared_dir = dir_in_use+'corr_kaon_t0/g5g5_ls_11_11_mom1_0_0_0_mom2_0_0_0/g5a_0_g5b_dt_'
    if smeared:
        kaon_dir = dir_in_use+'corr_kaon_t0/g5g5_ls_11_11_mom1_0_0_0_mom2_0_0_0_smearLSl_pt_smearLSs_gauss35_smearSLl_pt_smearSLs_gauss35/g5a_0_g5b_dt_'
        C_dir = dir_in_use+'corr_C_para_delta1_t0/g5g5g5g5_mom1_0_0_0_mom2_0_0_0_smearl_pt_smears_gauss35/g5a_0_g5b_1_g5c_1_g5d_dt_'
        R_dir = dir_in_use+'corr_R_para_delta1_t0/g5g5g5g5_mom1_0_0_0_mom2_0_0_0_smearl_pt_smears_gauss35/g5a_0_g5b_0_g5c_1_g5d_dt_'
    else:
        kaon_dir = dir_in_use+'corr_kaon_t0/g5g5_ls_11_11_mom1_0_0_0_mom2_0_0_0/g5a_0_g5b_dt_'
        C_dir = dir_in_use+'corr_C_para_delta1_t0/g5g5g5g5_mom1_0_0_0_mom2_0_0_0/g5a_0_g5b_1_g5c_1_g5d_dt_'
        R_dir = dir_in_use+'/corr_R_para_delta1_t0/g5g5g5g5_mom1_0_0_0_mom2_0_0_0/g5a_0_g5b_0_g5c_1_g5d_dt_'

    pk_prefix = 'g5a_0_g5b_dt_'
    C_prefix = 'g5a_0_g5b_1_g5c_1_g5d_dt_'
    R_prefix = 'g5a_0_g5b_0_g5c_1_g5d_dt_'


    pion_files = cfgs_list(pion_dir)
    kaon_unsmeared_files = cfgs_list(kaon_unsmeared_dir)
    kaon_files = cfgs_list(kaon_dir)
    C_files = cfgs_list(C_dir)
    R_files = cfgs_list(R_dir)

    commons = np.array(list(set(pion_files) & set(kaon_unsmeared_files) & set(kaon_files) & set(C_files)& set(R_files)))

    TC_dict= TC(commons)

    pion_data = import_data(TC_dict, pion_dir, pk_prefix)
    kaon_data = import_data(TC_dict, kaon_dir, pk_prefix)
    C_data = import_data(TC_dict, C_dir, C_prefix)
    R_data = import_data(TC_dict, R_dir, R_prefix)

    cfgs = pion_data.shape[1]

#===============================================================================================
    # import KKpipi and piKpiK C, R data

    Delta = [15,20,25,30,40]

    KKpipiC_data = np.zeros(shape=(len(Delta),T,cfgs,T))
    piKpiKC_data = np.zeros(shape=(len(Delta),T,cfgs,T))
    KKpipiR_data = np.zeros(shape=(len(Delta),T,cfgs,T))
    piKpiKR_data = np.zeros(shape=(len(Delta),T,cfgs,T))

    for Del in Delta:
        if smeared:
            KKpipiC_dir = dir_in_use+'corr_KKpipi'+str(Del)+'C_delta1_t0/g5g5g5g5_mom1_0_0_0_mom2_0_0_0_smearLSl_pt_smearLSs_gauss35_smearSLl_pt_smearSLs_gauss35/g5a_0_g5b_'+str(Del)+'_g5c_1_g5d_dt_'
            KKpipiR_dir = dir_in_use+'corr_KKpipi'+str(Del)+'R_delta1_t0/g5g5g5g5_mom1_0_0_0_mom2_0_0_0_smearLSl_pt_smearLSs_gauss35_smearSLl_pt_smearSLs_gauss35/g5a_0_g5b_0_g5c_'+str(Del)+'_g5d_dt_'
            piKpiKC_dir = dir_in_use+'corr_piKpiK'+str(Del)+'C_delta1_t0/g5g5g5g5_mom1_0_0_0_mom2_0_0_0_smearLSl_pt_smearLSs_gauss35_smearSLl_pt_smearSLs_gauss35/g5a_0_g5b_1_g5c_0_g5d_dt_'
            piKpiKR_dir = dir_in_use+'corr_piKpiK'+str(Del)+'R_delta1_t0/g5g5g5g5_mom1_0_0_0_mom2_0_0_0_smearLSl_pt_smearLSs_gauss35_smearSLl_pt_smearSLs_gauss35/g5a_0_g5b_'+str(Del)+'_g5c_1_g5d_dt_'
        else:
            KKpipiC_dir = dir_in_use+'corr_KKpipi'+str(Del)+'C_delta1_t0/g5g5g5g5_mom1_0_0_0_mom2_0_0_0/g5a_0_g5b_'+str(Del)+'_g5c_1_g5d_dt_'
            KKpipiR_dir = dir_in_use+'corr_KKpipi'+str(Del)+'R_delta1_t0/g5g5g5g5_mom1_0_0_0_mom2_0_0_0/g5a_0_g5b_0_g5c_'+str(Del)+'_g5d_dt_'
            piKpiKC_dir = dir_in_use+'corr_piKpiK'+str(Del)+'C_delta1_t0/g5g5g5g5_mom1_0_0_0_mom2_0_0_0/g5a_0_g5b_1_g5c_0_g5d_dt_'
            piKpiKR_dir = dir_in_use+'corr_piKpiK'+str(Del)+'R_delta1_t0/g5g5g5g5_mom1_0_0_0_mom2_0_0_0/g5a_0_g5b_'+str(Del)+'_g5c_1_g5d_dt_'


        KKpipiC_prefix = 'g5a_0_g5b_'+str(Del)+'_g5c_1_g5d_dt_'
        KKpipiC_data[Delta.index(Del),:,:,:] = import_data(TC_dict,KKpipiC_dir,KKpipiC_prefix)

        KKpipiR_prefix = 'g5a_0_g5b_0_g5c_'+str(Del)+'_g5d_dt_'
        KKpipiR_data[Delta.index(Del),:,:,:] = import_data(TC_dict,KKpipiR_dir,KKpipiR_prefix)

        #=================================

        piKpiKC_prefix = 'g5a_0_g5b_1_g5c_0_g5d_dt_'
        piKpiKC_data[Delta.index(Del),:,:,:] = import_data(TC_dict,piKpiKC_dir,piKpiKC_prefix)

        piKpiKR_prefix = 'g5a_0_g5b_'+str(Del)+'_g5c_1_g5d_dt_'
        piKpiKR_data[Delta.index(Del),:,:,:] = import_data(TC_dict,piKpiKR_dir,piKpiKR_prefix)

    all_data = [pion_data, kaon_data, C_data, R_data, KKpipiC_data, piKpiKC_data, KKpipiR_data, piKpiKR_data]
    #pickle.dump(all_data, open('raw_data_sm'+str(smeared)+'.p','wb')) 
    end = time.time()
    print(f'Loading complete. Took {round(end-start,2)} secs.')
    return all_data
    
