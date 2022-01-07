# Kaon-pion scattering analysis code

This project allows a user to run through the entire Kpi analysis by themselves
and inspect all correlations functions and fit variations that ultimately lead to 
the calculation of the Kpi scattering lengths for the isospin channels I=1/2, 3/2.

It is encouraged to use an interactive python version, such as `iPython`, to run
these scripts and inspect the correlation functions as objects.

The most important results can be viewed right away by running `run_through.py`, 
however, one can go through the entire analysis step-by-step if they wish using
the other scripts in this folder. 

## Sec 1. OVERVIEW


* The directory [correlators](correlators/) contains all the data in form of HDF5 files.

* This data is loaded into numpy arrays using the [load_data.py](load_data.py) script. This script
returns the data for each correlation function in the format (T_src, config, Delta T).

* The [autofit.py](autofit.py) script takes this raw data and generates a class object which
computes and stores statistical information about the data as attributes. Then for
each type of correlation function, it performs several fits over various fit ranges
and determines a best fit interval and stores it in a dictionary. This dictionary is
then stored as a pickle file in [best_fits_sm(True/False).p](pickles/) corresponding to
use of point-like or smeared source data. 

* The script [correlation_functions.py](correlation_functions.py) also loads the raw data and performs single
fits for all correlation functions with the choice of fit interval available in 
best_fits_sm(True/False).p](pickles/). This skips the time taken in finding the best
fit ranges, and easily makes these correlation functions available for secondary
analysis as class objects which have already been fitted for primary observables
(for examples masses of mesons from their two-point functions etc.)

* The script [analysis.py](analysis.py) uses [correlation_functions.py](correlation_functions.py) to load all the prepped
correlation functions to perform combined fits of the point-like and smeared
data together over various fit ranges to find the best fit range (of the C_Kpi
correlation function - from where we wish to extract the Kpi composite energy
and hence compute the scattering length). It stores the best fit ranges chosen
into [fit_intervals.p](pickles/fit_intervals.p) and also saves the pandas DataFrames containing
information on all the fits attempted in [pt_sm_df.p](pickles/pt_sm_df.p). It also saves
some information on these DataFrames in dictionaries in [pt_sm_dicts.p](pickles/pt_sm_dicts.p).
This is so that this information can be reloaded easily later without having
to rerun the several fit routines. The plots are made available in the [plots](plots/)
folder by running this script.

* The script [run_through.py](run_through.py) uses all the pre-computed pickles to recreate the
class objects for the combined fits, which have the final information on the
computed scattering length, other observables, and errors. The combined fit
class objects are called `pt_sm_corrI12` and `pt_sm_corrI32`. Details on the various
objects available for closer inspection after running this script is described
in Section 2.

## Sec 2. CORRELATION FUNCTIONS

Correlators are stored as instances of the class `stat_object` (in [fit_routine.py](fit_routine.py))
and have several attributes that can be inspected. Refer to the following examples:

* Example 1: A single correlation function: Pion two-point function from data (with
100 bootstrap samples):
```
pion = fit_routine.stat_object(data=pion_data, sampler=bootstrap, K=100,
                               fold=True, object_type='single', name='pion')
pion.fit(interval=(12,24,2), ansatz=cosh, guess=[2e+4,0.08],
         param_names=['A_p','m_p'], plot=True)
```
One can then view the major fit results via the `fit_dict` attribute:
```
In [1]: pion.fit_dict
In [2]: 
```
The final values of the fit parameters are stored in `pion.params` with their errors
in `pion.params_err`. The bootstrap distribution of the paramters is available in 
`pion.params_dist`.

One can also compute some additional function of the fit parameters and its behavior 
over the bootstraps (useful for computing scattering length) by defining:
```
def func1(params, **kwargs):
    return params[0]+params[1]
   
pion.fit(interval=(12,24,2), ansatz=cosh, guess=[2e+4,0.08], param_names=['A_p','m_p'],
        calc_func=[func1], calc_func_names=['func1'])
```
Now the `fit_dict` contains information on this function
```
In [1]: pion.fit_dict
In [2]:
```

* Example 2: Combined fits: Combined pion and C_Kpi correlator:

```
combined_corr = fit_routine.stat_object(data=[pion, C_Kpi], sampler=bootstrap, K=100,
                                        object_type='combined', name='pion and C_Kpi')
combined_corr.fit(interval=(0,combined_corr.T-1,1), ansatz=combined_ansatz, index=1,
                  guess=[2e+4,0.08,1,0.001], param_names=['A_p', 'm_p', 'A_CKpi', 'E_Kpi'])
```
`index` specifies the correlator with the flexible fit range in the list of correlators. This
is useful with varying the global fit via the fit range of one of the many correlation functions.

## Sec 3. COMPUTING ERRORS

