import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
from plot_settings import plotparams
plt.rcParams.update(plotparams)

da12, da32 = pickle.load(open('pickles/double_fit_dicts.p','rb'))

def pf(d, **kwargs):
    d_fil = {}
    for key1 in d.keys():
        for key2, val in d[key1].items():
            if val['pvalue']>0.05:
                d_fil.update({key1:{key2:val}})
    return d_fil

#===variations with pt t_min===========================
def pt_t_min_var(I, pfilter=False, **kwargs):
    d = da12 if I==0.5 else da32
    if pfilter:
        d = pf(d)

    stack = []
    for t_min in range(8,15):
        t_stack = np.array([])
        for key1 in d.keys():
            for key2 in d[key1].keys():
                if int(re.findall("\d+",key1)[0])==t_min:
                    scat = d[key1][key2]['calc_func'][0]
                    t_stack = np.append(t_stack,scat)
        stack.append(t_stack)

    plt.figure()
    plt.hist(stack, stacked=True, 
             label=[str(i) for i in range(8, 15)])
    plt.legend(title="pt $t_{min}$", 
               bbox_to_anchor=(1.05, 1.0),
               loc="upper left")
    plt.tight_layout()
    a0_str = "$a_0^{I=1/2}$" if I==0.5 else "$a_0^{I=3/2}$"
    plt.xlabel(a0_str)
    plt.title("Variation with $t_{min}$ of pt-$C_{K\pi}$")

#===variation with pt delta_t============================
def pt_del_t_var(I, pfilter=False, **kwargs):
    d = da12 if I==0.5 else da32
    if pfilter:
        d = pf(d)

    stack = []
    for delta_t in range(5,12):
        t_stack = np.array([])
        for key1 in d.keys():
            for key2 in d[key1].keys():
                if int(re.findall("\d+",key1)[1])-int(re.findall("\d+",key1)[0])==delta_t:
                    scat = d[key1][key2]['calc_func'][0]
                    t_stack = np.append(t_stack,scat)
        stack.append(t_stack)

    plt.figure()
    plt.hist(stack, stacked=True, 
             label=[str(i) for i in range(5, 12)])
    plt.legend(title="pt $\delta t$", 
               bbox_to_anchor=(1.05, 1.0),
               loc="upper left")
    plt.tight_layout()
    a0_str = "$a_0^{I=1/2}$" if I==0.5 else "$a_0^{I=3/2}$"
    plt.xlabel(a0_str)
    plt.title("Variation with $\delta t$ of pt-$C_{K\pi}$")

#===variations with sm t_min===========================
def sm_t_min_var(I, pfilter=False, **kwargs):
    d = da12 if I==0.5 else da32
    if pfilter:
        d = pf(d)

    stack = []
    for t_min in range(5,12):
        t_stack = np.array([])
        for key1 in d.keys():
            for key2 in d[key1].keys():
                if int(re.findall("\d+",key2)[0])==t_min:
                    scat = d[key1][key2]['calc_func'][0]
                    t_stack = np.append(t_stack,scat)
        stack.append(t_stack)

    plt.figure()
    plt.hist(stack, stacked=True, 
             label=[str(i) for i in range(5, 12)])
    plt.legend(title="sm $t_{min}$", 
               bbox_to_anchor=(1.05, 1.0),
               loc="upper left")
    plt.tight_layout()
    a0_str = "$a_0^{I=1/2}$" if I==0.5 else "$a_0^{I=3/2}$"
    plt.xlabel(a0_str)
    plt.title("Variation with $t_{min}$ of sm-$C_{K\pi}$")

#===variation with sm delta_t============================
def sm_del_t_var(I, pfilter=False, **kwargs):
    d = da12 if I==0.5 else da32
    if pfilter:
        d = pf(d)

    stack = []
    for delta_t in range(5,12):
        t_stack = np.array([])
        for key1 in d.keys():
            for key2 in d[key1].keys():
                if int(re.findall("\d+",key2)[1])-int(re.findall("\d+",key1)[0])==delta_t:
                    scat = d[key1][key2]['calc_func'][0]
                    t_stack = np.append(t_stack,scat)
        stack.append(t_stack)

    plt.figure()
    plt.hist(stack, stacked=True, 
             label=[str(i) for i in range(5, 12)])
    plt.legend(title="sm $\delta t$", 
               bbox_to_anchor=(1.05, 1.0),
               loc="upper left")
    plt.tight_layout()
    a0_str = "$a_0^{I=1/2}$" if I==0.5 else "$a_0^{I=3/2}$"
    plt.xlabel(a0_str)
    plt.title("Variation with $\delta t$ of sm-$C_{K\pi}$")

