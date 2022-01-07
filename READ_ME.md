Kaon-pion scattering analysis code
Rajnandini Mukherjee

This project allows a user to run through the entire Kpi analysis by themselves
and inspect all correlations functions and fit variations that ultimately lead to 
the calculation of the Kpi scattering lengths for the isospin channels I=1/2, 3/2.

The most important results can be viewed right away by running 'run_through.py', 
however, one can go through the entire analysis step-by-step if they wish using
the other scripts in this folder. 

#===============================================================================
Sec 1. OVERVIEW
#===============================================================================

*The directory 'correlators/' contains all the data in form of HDF5 files.

*This data is loaded into numpy arrays using the 'load_data.py' script. This script
returns the data for each correlation function in the format (T_src, config, Delta T).

*The 'autofit.py' script takes this raw data and generates a class object which
computes and stores statistical information about the data as attributes. Then for
each type of correlation function, it performs several fits over various fit ranges
and determines a best fit interval and stores it in a dictionary. This dictionary is
then stored as a pickle file in pickles/best_fits_sm(True/False).p correspoding to
use of point-like or smeared source data. 

*The script 'correlation_functions.py' also loads the raw data and performs single
fits for all correlation functions with the choice of fit interval available in 
'pickles/best_fits_sm(True/False).p'. This skips the time taken in finding the best
fit ranges, and easily makes these correlation functions available for secondary
analysis as class objects which have already been fitted for primary observables
(for examples masses of mesons from their two-point functions etc.)

*The script 'analysis.py' uses 'correlation_functions.py' to all the prepped
correlation functions to perform combined fits of the point-like and smeared
data together over various fit ranges to find the best fit range (of the C_Kpi
correlation function - from where we wish to extract the Kpi composite energy
and hence compute the scattering length). It stores the best fit ranges chosen
into 'pickles/fit_intervals.p' and also saves the pandas DataFrames containing
information on all the fits attempted in 'pickles/pt_sm_df.p'. It also saves
some information on these DataFrames in dictionaries in 'pickles/pt_sm_df.p'.
This is so that this information can be reloaded easily later without having
to rerun the several fit routines. The plots are made available in the 'plots/'
folder by running this script.

*The script 'run_through.py' uses all the pre-computed pickles to recreate the
class objects for the combined fits, which have the final information on the
computed scattering length, other observables, and errors. The combined fit
class objects are called 'pt_sm_corrI12' and 'pt_sm_corrI32'. One can view the
fit results in their 'fit_dict' attribute, and find information on all the 
various fit ranges in their 'autofit_df' DataFrames. Details on the various
objects available for closer inspection after running this script is described
in Section 2.

#----------------------------------------------------------------------------
The heirarchy of files is summarized below:

correlators/ -> load_data.py

load_data.py -> autofit.py

autofit.py -> best_fits_sm(True/False).p

load data.py,
best_fits_sm(True/False).p -> correlation_functions.py

correlation_functions.py -> analysis.py

analysis.py -> fit_intervals.p, pt_sm_dfs.p, pt_sm_dicts.p

correlation_functions.py,
pt_sm_dfs.p, pt_sm_dicts.p,
fit_intervals.p -> run_through.py


#===============================================================================
Sec 2. CORRELATION FUNCTIONS
#===============================================================================
Correlators are stored as instances of the class 'stat_object' (in 'fit_routine.py')
and have several attributes that can be inspected. Refer to the following examples:

*Example 1: A single correlation function: Pion
    















#===============================================================================
Sec 3. COMPUTING ERRORS
#===============================================================================

