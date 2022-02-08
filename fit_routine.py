import numpy as np
import matplotlib.pyplot as plt
from plot_settings import plotparams
plt.rcParams.update(plotparams)
import sys
import pandas as pd
from scipy.special import gammaincc
import pdb

def bootstrap(data, seed=1, K=1000, sigma=None, **kwargs):
    ''' bootstrap samples generator - if input data has same size as K,
    assumes it's already a bootstrap sample and does no further sampling '''
    
    [M, N] = data.shape

    if N==K: # goes off when data itself is bootstrap data
        samples = data
    else:
        np.random.seed(seed)
        slicing = np.random.randint(0, N, size=(N,K))
        samples = np.array([np.mean(data[m,:][slicing.T], axis=1)
                            for m in range(M)])

    return samples

def jackknife(data, **kwargs):
    '''jacknife samples generator - can be further modified for blocked
    methods - to be added later'''

    [M, N] = data.shape

    slicing = ~np.eye(N,dtype=bool)
    samples = np.array([np.mean((data[m,:]*slicing)[slicing].reshape(N,-1),
                        axis=1) for m in range(M)])

    return samples                

def COV(data, **kwargs):
    ''' covariance matrix calculator - accounts for cov matrices
        centered around sample avg vs data avg and accordingly normalises'''

    [M, N] = data.shape

    if 'center' in kwargs.keys():
        center = kwargs['center']
        norm = N
    else:
        center = np.mean(data, axis=1)
        norm = N-1 

    COV = np.array([[((data[m1,:]-center[m1]).dot(data[m2,:]-center[m2]))/norm
                    for m2 in range(M)] for m1 in range(M)])

    return COV

def cosh(params, t, T=0, **kwargs):
    ''' cosh ansatz, if T unspecified, then returns regular cosh function,
    other exp(-m*(T-t)). Value of T automatically set to lattice T, but can
    be overwritten using T_overwrite kwarg.'''

    if 'T_overwrite' in kwargs.keys():
        T = kwargs['T_overwrite']
    return params[0]*(np.exp(-params[1]*t) + np.exp(-params[1]*(T-t)))

from scipy.optimize import leastsq
def fit_func(corr, y, **kwargs):
    ''' fitting routine by least square minimisation of difference
    vector times covariance '''

    def diff(params):
        #if corr.dict['object_type']=='combined':
        #    n = corr.dict['index']
        #    t = corr.corrs[n].x
        #else:
        #    t = corr.x
        return y - corr.ansatz(params, corr.t, **corr.dict)

    try:
        if 'COV_inv_L' in corr.dict.keys():
            L = corr.dict['COV_inv_L'](corr.COV_trunc, **corr.dict)
        else:
            L_inv = np.linalg.cholesky(corr.COV_trunc)
            L = np.linalg.inv(L_inv)
        corr.COV_inv = L@L.T
        corr.cond = np.linalg.cond(corr.COV_inv)

        def LD(params):
            return L.dot(diff(params))

        res, ier = leastsq(LD, corr.guess, ftol=1e-10, gtol=1e-10)
        chi_sq = LD(res).dot(LD(res))

        corr.fit_success = corr.fit_success*True
        return res, chi_sq 

    except np.linalg.LinAlgError as err:
        if 'Matrix is not positive definite' in str(err):
            corr.fit_success = corr.fit_success*False
            return corr.guess, 100
        else:
            raise

class stat_object:
    def __init__(self, data, sampler=bootstrap, fold=False, name='',
                COV_data_centering=True, seed=1, object_type='single', **kwargs):

        self.dict = {}
        self.dict.update(kwargs)

        self.dict['object_type'] = object_type
        if object_type=='combined':
            self.corrs = data
            n = len(self.corrs)

            data = np.concatenate([c.samples[c.x,:] for c in self.corrs], axis=0)
            data_avg = np.concatenate([c.data_avg[c.x] for c in self.corrs], axis=0)
            self.dict['data_avg'] = data_avg

        self.org_data = data
        [self.T, self.cfgs] = data.shape
        self.dict['T'] = self.T

        if 'data_avg' in self.dict.keys():
            self.org_data_avg = self.dict['data_avg']
        else:
            self.org_data_avg = np.mean(self.org_data, axis=1)

        self.fold = fold
        if self.fold:
            self.data = 0.5*(self.org_data + np.roll(self.org_data[::-1,:],1,axis=0))
            self.data_avg = 0.5*(self.org_data_avg + np.roll(self.org_data_avg[::-1],1,axis=0)) 
        else:
            self.data = self.org_data
            self.data_avg = self.org_data_avg
        
        self.sampler = sampler
        self.seed = seed
        self.org_samples = sampler(self.org_data, seed, **self.dict)
        self.samples = sampler(self.data, seed, **self.dict)
        self.K = self.samples.shape[1]

        self.COV_data_centering = COV_data_centering
        if COV_data_centering:
            COV_center = self.data_avg
            self.COV  = COV(self.samples, center=COV_center)
        else:
            self.COV  = COV(self.samples)

        if object_type=='combined':
            COV_mask = np.concatenate([np.array([c.correlated]*len(c.x)) for c in self.corrs])
            COV_mask = np.outer(COV_mask, COV_mask)
            np.fill_diagonal(COV_mask, True)
            self.COV = self.COV*COV_mask

        self.input_err = (np.diag(self.COV))**0.5
        self.CORR = np.diag(1/self.input_err)@self.COV@np.diag(1/self.input_err)

        self.name = name

    def fit(self, interval, ansatz, guess, param_names=None, prnt=False, plot=False,
            correlated=True, full_fit=True, **kwargs):

        self.dict.update(kwargs)
        self.dict.update({'plot':plot, 'prnt':prnt})
        (s,e,t) = interval 
        self.interval = interval
        self.x = np.arange(s,e+1,t)
        y = self.data_avg[self.x]
        if self.dict['object_type']=='single':
            self.t = self.x
        elif self.dict['object_type']=='combined':
            n = self.dict['index']
            self.t = self.corrs[n].x

        self.guess = guess
        self.DOF = len(self.x) - np.count_nonzero(self.guess)

        self.correlated = correlated
        self.COV_trunc = self.COV[s:e+1:t, s:e+1:t]
        if self.dict['object_type']=='single':
            x = self.x
        elif self.dict['object_type']=='combined':
            x = np.concatenate([corr.x for corr in self.corrs])
        self.adjacency = np.array([[np.abs(i-j) for i in x] for j in x]) 

        if 'COV_model' in self.dict.keys():
            self.COV_trunc = self.dict['COV_model'](self)
        if not self.correlated:
            self.COV_trunc = np.diag(np.diag(self.COV_trunc))

        temp = np.diag(1/self.input_err[self.x])
        self.CORR_trunc = temp@self.COV_trunc@temp
        self.cond = np.linalg.cond(self.COV_trunc)

        self.fit_success = 1
        self.ansatz = ansatz
        self.dict['instance']='central'
        self.params, self.chi_sq = fit_func(self,y,**self.dict)
        self.fit_avg = self.ansatz(self.params, self.t, **self.dict)
        self.pvalue = gammaincc(self.DOF/2, self.chi_sq/2)

        if full_fit:
            if param_names==None:
                self.param_names = ['param_'+str(n) for n in range(len(guess))] 
            else:
                self.param_names = param_names

            params_dist = np.zeros(shape=(self.K,len(self.guess)))
            fit_dist = np.zeros(shape=(self.K,len(self.x)))
            for k in range(self.K):
                y_k = self.samples[self.x,k]
                self.dict.update({'instance':'sample','k':k})
                params_dist[k,:], chi_sq_k = fit_func(self, y_k, **self.dict)
                fit_dist[k,:] = self.ansatz(params_dist[k,:], self.t, **self.dict)
            self.params_err = np.array([st_dev(params_dist[:,i], mean=self.params[i])
                                        for i in range(len(self.guess))])
            #self.fit_err = np.array([st_dev(fit_dist[:,t], mean=self.fit_avg[t]) 
            #                         for t in range(len(self.x))])

        else:
            params_dist = np.zeros(shape=(self.K,len(self.guess)))
            for k in range(self.K):
                y_k = self.samples[self.x,k] 
                self.dict.update({'instance':'sample','k':k})
                params_dist[k,:], chi_sq_k = fit_func(self, y_k, **self.dict)
            self.params_err = np.array([st_dev(params_dist[:,i], mean=self.params[i])
                                        for i in range(len(self.guess))])
        self.fit_dict = {'int':interval, 
                         'params':self.params,
                         'params_err':self.params_err,
                         'X^2/DOF':self.chi_sq/self.DOF,
                         'pvalue':self.pvalue,
                         'cond':self.cond}
        if self.dict['object_type']=='combined' and 'index' in self.dict.keys():
            self.fit_dict['int'] = self.corrs[self.dict['index']].interval 
        self.params_dist = params_dist
        if 'calc_func' in self.dict.keys() and type(self.dict['calc_func'])==list:
            self.fit_dict['calc_func'] = []
            self.fit_dict['calc_func_err'] = []
            for func in self.dict['calc_func']:
                func_val = func(self.params, t=self.t)
                func_dist = np.array([func(self.params_dist[k,:], t=self.t)
                                      for k in range(self.K)])
                self.fit_dict['calc_func'].append(func_val)
                #pdb.set_trace()
                self.fit_dict['calc_func_err'].append(st_dev(func_dist,
                                            mean=func_val))
            n = len(self.dict['calc_func'])
            if 'calc_func_names' not in self.dict.keys():
                self.dict['calc_func_names'] = ['func'+str(f) for f in range(n)]
                                                  
        if full_fit and prnt:
            self.output()
        if full_fit and plot:
            self.plot(**self.dict)

    def autofit(self, t_min_range, fit_ints, ansatz, guess,
                t_max_range=None, thin_list=[1], **kwargs):
        
        self.dict.update(kwargs)
        self.guess = guess
        self.ansatz = ansatz
        self.autofit_dict = {}
        from progress.bar import Bar

        bar = Bar(self.name+' autofit:', max=len(t_min_range)*len(fit_ints)*len(thin_list))

        T = self.corrs[kwargs['index']].T if self.dict['object_type']=='combined' else self.T
        limit = kwargs['limit'] if 'limit' in kwargs.keys() else T

        for t in thin_list:
            for s in t_min_range:
                for i in fit_ints:
                    bar.next()
                    if s+i+1>=limit:
                        continue
                    self.dict['full_fit']=False
                    if self.dict['object_type']=='combined':
                        n = kwargs['index']
                        self.corrs[n].interval = (s,s+i,t)
                        self.corrs[n].x = np.arange(s,s+i+1,t)
                        temp_corr = stat_object(self.corrs,**self.dict)
                        temp_corr.fit((0,temp_corr.T-1,1),ansatz,guess,**self.dict)
                    else:
                        temp_corr = stat_object(self.data,fold=False,**self.dict)
                        temp_corr.fit((s,s+i+1,t),ansatz,guess,**self.dict) 

                    if temp_corr.fit_success and temp_corr.DOF!=0:
                        self.autofit_dict[str((s,i,t))] = temp_corr.fit_dict
                    del temp_corr
        bar.finish()

        self.autofit_df = pd.DataFrame(self.autofit_dict.values())
        self.autofit_dict = {'t_mins': t_min_range,
                             't_ints': fit_ints,
                             'thin_list': thin_list} 
        best_int = self.best_fit(**self.dict)

        self.dict['full_fit']=True
        if self.dict['object_type']=='combined':
            (s,e,t) = best_int
            self.corrs[n].x = np.arange(s,e+1,t)
            self.corrs[n].interval = (s,e,t)
            self.data = np.concatenate([c.samples[c.x,:] for c in self.corrs], axis=0)
            [self.T, self.cfgs] = self.data.shape
            self.dict['T'] = self.T
            self.data_avg = np.concatenate([c.data_avg[c.x] for c in self.corrs], axis=0)
            self.samples = self.sampler(self.data, self.seed, **self.dict)
            if self.COV_data_centering:
                COV_center = self.data_avg
                self.COV  = COV(self.samples, center=COV_center)
            else:
                self.COV  = COV(self.samples)
            COV_mask = np.concatenate([np.array([c.correlated]*len(c.x)) for c in self.corrs])
            COV_mask = np.outer(COV_mask, COV_mask)
            np.fill_diagonal(COV_mask, True)
            self.COV = self.COV*COV_mask
            self.input_err = (np.diag(self.COV))**0.5
            self.CORR = np.diag(1/self.input_err)@self.COV@np.diag(1/self.input_err)
            self.fit((0,self.T-1,1),ansatz,guess,**self.dict)
        else:
            self.fit(best_int,ansatz,guess,**self.dict)

        if self.dict['plot']:
            self.autofit_plot(**self.dict)


    def best_fit(self, imp_params='all', weights='equal',
                 hyperweights={'pvalue_cost':1,
                              'fit_stbl_cost':1,
                              'err_cost':1,
                              'val_stbl_cost':1}, **kwargs):
        if imp_params=='all':
            imp_params = range(len(self.guess))
        if weights=='equal':
            weights = [1,]*len(self.guess)

        df = self.autofit_df

        pvalue_cost =  np.array(normalize(np.abs(0.5-df.pvalue)))
        pvalue_cost += np.where(np.logical_and(df.pvalue>0.05, df.pvalue<0.95), 0, 100)

        prev_pvalue_cost = np.roll(pvalue_cost, 1, axis=0)/4.0
        next_pvalue_cost = np.roll(pvalue_cost, -1, axis=0)/4.0
        fit_stbl_cost = prev_pvalue_cost + next_pvalue_cost

        err_cost = np.zeros(len(pvalue_cost))
        val_stbl_cost = np.zeros(len(pvalue_cost)) 
        imp_params = list(np.nonzero(self.guess)[0])
        for p in imp_params:
            err_cost += weights[p]*normalize(df['params_err'].str[p]) 

            val = df['params'].str[p]
            std = df['params_err'].str[p]
            prev_val = np.roll(df['params'].str[p], 1, axis=0)
            next_val = np.roll(df['params'].str[p], -1, axis=0)
            val_diff = np.divide(np.abs(prev_val-val)+np.abs(next_val-val), std)
            
            val_stbl_cost += weights[p]*normalize(val_diff)
        
        self.best_fit_dict = {'pvalue_cost': pvalue_cost,
                              'fit_stbl_cost': fit_stbl_cost,
                              'err_cost': err_cost,
                              'val_stbl_cost': val_stbl_cost}

        self.autofit_df['cost'] = self.autofit_df.pvalue*0
        for c in ['pvalue_cost', 'fit_stbl_cost', 'err_cost', 'val_stbl_cost']:
            if np.isnan(self.best_fit_dict[c]).any():
                #print(c+str(' has nan'))
                continue
            self.autofit_df['cost'] += hyperweights[c]*self.best_fit_dict[c]

        best_int = df['int'][self.autofit_df['cost'].idxmin()]
        return best_int
    
    def autofit_plot(self, plot_params='all', int_skip=1, savefig=False,
                     pfilter=False, **kwargs):
        df = self.autofit_df
        if pfilter:
            df = df[df.pvalue>0.05][df.pvalue<0.95]
        thin = list(self.autofit_dict['thin_list'])
        n_t = len(thin)

        N = len(self.autofit_df.pvalue)/n_t
        if plot_params=='all':
            param_idx = np.arange(len(self.guess))
        else:
            param_idx = plot_params

        for p in param_idx:
            plt.figure()
            plt.suptitle(self.param_names[p])
            for t in range(n_t):
                plt.subplot(n_t,1,t+1)
                for idx, t_int in enumerate(self.autofit_dict['t_ints']):
                    if idx%int_skip==0:
                        x = df.int[df.int.str[1]-df.int.str[0]==t_int]
                        x = x[df.int.str[2]==thin[t]].str[0]
                        y = df.params[df.int.str[1]-df.int.str[0]==t_int]
                        y = y[df.int.str[2]==thin[t]].str[p]
                        yerr = df.params_err[df.int.str[1]-df.int.str[0]==t_int]
                        yerr = yerr[df.int.str[2]==thin[t]].str[p]

                        markers, caps, bars = plt.errorbar(x, y, yerr=yerr,
                                            capsize=4, label=str(t_int),
                                            linestyle='None', fmt='o')
                        [bar.set_alpha(0.5) for bar in bars]
                        [cap.set_alpha(0.5) for cap in caps]
                plt.legend(loc=1, bbox_to_anchor=(0.985,0.89),
                            bbox_transform=plt.gcf().transFigure)
            plt.xlabel('$t_{min}$')
            if savefig:
                plt.savefig('plots/'+self.name+'_'+self.param_names[p]+'.pdf')

        if 'calc_func' in self.dict.keys():
            n = len(self.dict['calc_func'])
            self.fit_dict['calc_func_sys_err'] = [0]*n
            for f in range(n):
                plt.figure()
                plt.suptitle(self.dict['calc_func_names'][f])
                for t in range(n_t):
                    plt.subplot(n_t,1,t+1)
                    for idx, t_int in enumerate(self.autofit_dict['t_ints']):
                        if idx%int_skip==0:
                            x = df.int[df.int.str[1]-df.int.str[0]==t_int]
                            x = x[df.int.str[2]==thin[t]].str[0]
                            y = df.calc_func[df.int.str[1]-df.int.str[0]==t_int]
                            y = y[df.int.str[2]==thin[t]].str[f]
                            yerr = df.calc_func_err[df.int.str[1]-df.int.str[0]==t_int]
                            yerr = yerr[df.int.str[2]==thin[t]].str[f]

                            markers, caps, bars = plt.errorbar(x, y, yerr=yerr,
                                                capsize=4, label=str(t_int),
                                                linestyle='None', fmt='o')
                            [bar.set_alpha(0.5) for bar in bars]
                            [cap.set_alpha(0.5) for cap in caps]
                    plt.legend(loc=1, bbox_to_anchor=(0.985,0.89),
                                bbox_transform=plt.gcf().transFigure)
                plt.xlabel('$t_{min}$')
                if savefig:
                    plt.savefig('plots/'+self.name+'_'+self.dict['calc_func_names'][f]+'.pdf')
                
                plt.figure()
                plt.suptitle(self.dict['calc_func_names'][f])
                for t in range(n_t):
                    plt.subplot(n_t,1,t+1)
                    if 'hist_deltas' in kwargs.keys():
                        deltas = kwargs['hist_deltas']
                    else:
                        deltas = list(self.autofit_dict['t_ints'])
                    df_t = df[df.int.str[2]==t+1]
                    if 'hist_t_min' in kwargs.keys():
                        tmin = kwargs['hist_t_min']
                        df_t = df_t[df_t.int.str[0]>=tmin]

                    stack = [np.array(df_t.calc_func[(df_t.int.str[1]-df_t.int.str[0])==diff].str[0])
                            for diff in deltas]
                    a, b, fig = plt.hist(stack, stacked=True, 
                                label=[str(i) for i in deltas])
                    spacing = b[1]-b[0]
                    err_abs = (b[-1]-b[0])/4
                    self.fit_dict['calc_func_sys_err'][f] = err_abs

                    val = float(self.fit_dict['calc_func'][f])
                    plt.axvline(val, c='k', label='value')

                    err = float(self.fit_dict['calc_func_err'][f])
                    y_lo, y_up = plt.ylim()
                    x = np.linspace(val-err, val+err, 10)
                    plt.fill_between(x, y_lo, y_up, alpha=0.2, color='k', label='error')
                    plt.ylim(top=y_up)
                    plt.ylim(bottom=y_lo)

                    plt.legend(title='$\delta t$', bbox_to_anchor=(1.05,1.0), loc='upper left')
                    plt.tight_layout()
                if savefig:
                    plt.savefig('plots/'+self.name+'_'+self.dict['calc_func_names'][f]+'_hist.pdf')

                
    def plot(self, datarange=None, savefig=False, **kwargs):
        x = self.x
        fig = plt.figure()
        if type(datarange) is not np.ndarray:
            (s,e,t) = self.interval
            if self.dict['object_type']=='combined':
                n = self.dict['index']
                ansatz_t = self.corrs[n].x
                data_x = np.arange(s,e+1)
            else:
                ansatz_t = np.arange(s,e+1)
                data_x = np.arange(s,e+1)

        else:
            data_x = datarange
            ansatz_t = datarange

        plt.plot(data_x, self.ansatz(self.params, ansatz_t, **self.dict))
        plt.errorbar(data_x, self.data_avg[data_x], yerr=self.input_err[data_x], fmt='o', capsize=4)
        #plt.errorbar(x, self.fit_avg, yerr=self.fit_err, fmt='o', capsize=4)
        plt.title(self.name+' fit')
        plt.xlabel('$t$')
        if savefig: 
            plt.savefig('plots/'+self.name+'_fit.pdf')

        if self.ansatz==cosh:
            self.m_eff = m_eff(self.data_avg)
            sample_wise_m_eff = np.array([m_eff(self.samples[:,k])
                                          for k in range(self.K)])
            self.m_eff_err = np.array([st_dev(sample_wise_m_eff[:,t],self.m_eff[t])
                                  for t in range(len(self.m_eff))])
            plt.figure()
            x = np.arange(len(self.m_eff))+1
            plt.errorbar(x, self.m_eff, yerr=self.m_eff_err, fmt='o', capsize=4)
            plt.fill_between(self.x,self.params[1]+self.params_err[1], 
                            self.params[1]-self.params_err[1], alpha=0.5)
            plt.title(self.name+' $m_{eff}$')

    
    def output(self):
        print(self.name+':')
        if hasattr(self, 'fit_success'):
            print('int:'+str(self.interval), 
                  'correlated:'+str(self.correlated),
                  'guess:'+str(self.guess))
            text = '{name}: {val}'
            for i in range(len(self.params)):
                print(text.format(name=self.param_names[i],
                    val=err_disp(self.params[i],self.params_err[i],**self.dict)))
            for key in ['X^2/DOF', 'pvalue', 'cond']:
                print(text.format(name=key, val=round(self.fit_dict[key],4)))


def m_eff(data, ansatz=cosh, **kwargs):
    if ansatz==cosh:
        m_eff = np.arccosh(0.5*(data[2:]+data[:-2])/data[1:-1])
    elif ansatz==exp:
        m_eff = np.abs(np.log(data[1:]/data[:-1]))
    return m_eff

def st_dev(data, mean=None, **kwargs):
    '''standard deviation function - finds stdev around data mean or mean
    provided as input'''

    n = len(data)
    if mean is None:
        mean = np.mean(data)
    return np.sqrt(((data-mean).dot(data-mean))/n)

def err_disp(num, err, n=2, **kwargs):
    ''' converts num and err into num(err) in scientific notation upto n digits
    in error, can be extended for accepting arrays of nums and errs as well, for
    now only one at a time'''

    err_dec_place = int(np.floor(np.log10(np.abs(err))))
    err_n_digits = int(err*10**(-(err_dec_place+1-n)))
    num_dec_place = int(np.floor(np.log10(np.abs(num))))
    if num_dec_place<err_dec_place:
        print('Error is larger than measurement')
        return 0
    else:
        num_sf = num*10**(-(num_dec_place))
        num_trunc = round(num_sf, num_dec_place-(err_dec_place+1-n))
        return str(num_trunc)+'('+str(err_n_digits)+')E%+d'%num_dec_place
def normalize(a):
    ''' a is an np.array or DataFrame column'''

    return (a - a.min())/(a.max() - a.min())
