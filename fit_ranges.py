import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import re
from plot_settings import plotparams
plt.rcParams.update(plotparams)

dict12, dict32 = pickle.load(open('pickles/double_fit_dicts.p','rb'))

da12 = {}
da32 = {}
for k1 in dict12.keys():
    t_min_x, t_max_x, temp = [int(s) for s in re.findall("\d+",k1)]
    da12[f'({t_min_x}, {t_max_x})'] = {}
    da32[f'({t_min_x}, {t_max_x})'] = {}
    for k2 in dict12[k1].keys():
        t_min_y, t_max_y, temp = [int(s) for s in re.findall("\d+",k2)]
        da12[f'({t_min_x}, {t_max_x})'][f'({t_min_y}, {t_max_y})'] = dict12[k1][k2]
        da32[f'({t_min_x}, {t_max_x})'][f'({t_min_y}, {t_max_y})'] = dict32[k1][k2]

def pf(d, **kwargs):
    d_fil = {}
    for key1 in d.keys():
        for key2, val in d[key1].items():
            if val['pvalue']>0.01:
                d_fil.update({key1:{key2:val}})
    return d_fil

def minmax(d, key, func=0, **kwargs):
    if key=='calc_func' or key=='calc_func_err':
        vals = np.array([[d[k1][k2][key][func] for k2 in d[k1].keys()]
                        for k1 in d.keys()])
    else:
        vals = np.array([[d[k1][k2][key] for k2 in d[k1].keys()]
                        for k1 in d.keys()])

    return np.min(vals), np.max(vals)

#===variations with pt t_min===========================
def pt_t_min_var(I, pfilter=False, func=0, **kwargs):
    d = da12 if I==0.5 else da32
    if pfilter:
        d = pf(d)

    stack = []
    for t_min in range(8,15):
        t_stack = np.array([])
        for key1 in d.keys():
            for key2 in d[key1].keys():
                t_min_x, t_max_x, x = [int(s) for s in re.findall("\d+",key1)]
                t_min_y, t_max_y, y = [int(s) for s in re.findall("\d+",key2)]
                if t_min_x==t_min:
                    scat = d[key1][key2]['calc_func'][func]
                    t_stack = np.append(t_stack,scat)
        stack.append(t_stack)

    plt.figure()
    a, b, fig = plt.hist(stack, stacked=True, 
             label=[str(i) for i in range(8, 15)])
    err = np.abs(b[-1]-b[0])/4
    #print(f'sys err:{err}')
    plt.legend(title="pt $t_{min}$", 
               bbox_to_anchor=(1.05, 1.0),
               loc="upper left")
    plt.tight_layout()
    a0_str = "$a_0^{I=1/2}$" if I==0.5 else "$a_0^{I=3/2}$"
    plt.xlabel(a0_str)
    plt.title("Variation with $t_{min}$ of pt-$C_{K\pi}$")
    return err


#===variation with pt delta_t============================
def pt_del_t_var(I, pfilter=False, func=0, **kwargs):
    d = da12 if I==0.5 else da32
    if pfilter:
        d = pf(d)

    stack = []
    for delta_t in range(5,12):
        t_stack = np.array([])
        for key1 in d.keys():
            for key2 in d[key1].keys():
                if int(re.findall("\d+",key1)[1])-int(re.findall("\d+",key1)[0])==delta_t:
                    scat = d[key1][key2]['calc_func'][func]
                    t_stack = np.append(t_stack,scat)
        stack.append(t_stack)

    plt.figure()
    a, b, fig = plt.hist(stack, stacked=True, 
             label=[str(i) for i in range(8, 15)])
    err = np.abs(b[-1]-b[0])/4
    #print(f'sys err:{err}')
    plt.legend(title="pt $\delta t$", 
               bbox_to_anchor=(1.05, 1.0),
               loc="upper left")
    plt.tight_layout()
    a0_str = "$a_0^{I=1/2}$" if I==0.5 else "$a_0^{I=3/2}$"
    plt.xlabel(a0_str)
    plt.title("Variation with $\delta t$ of pt-$C_{K\pi}$")
    return err

#===variations with sm t_min===========================
def sm_t_min_var(I, pfilter=False, func=0, **kwargs):
    d = da12 if I==0.5 else da32
    if pfilter:
        d = pf(d)

    stack = []
    for t_min in range(8,15):
        t_stack = np.array([])
        for key1 in d.keys():
            for key2 in d[key1].keys():
                if int(re.findall("\d+",key2)[0])==t_min:
                    scat = d[key1][key2]['calc_func'][func]
                    t_stack = np.append(t_stack,scat)
        stack.append(t_stack)

    plt.figure()
    a, b, fig = plt.hist(stack, stacked=True, 
             label=[str(i) for i in range(8, 15)])
    err = np.abs(b[-1]-b[0])/4
    #print(f'sys err:{err}')
    plt.legend(title="sm $t_{min}$", 
               bbox_to_anchor=(1.05, 1.0),
               loc="upper left")
    plt.tight_layout()
    a0_str = "$a_0^{I=1/2}$" if I==0.5 else "$a_0^{I=3/2}$"
    plt.xlabel(a0_str)
    plt.title("Variation with $t_{min}$ of sm-$C_{K\pi}$")
    return err

#===variation with sm delta_t============================
def sm_del_t_var(I, pfilter=False, func=0, **kwargs):
    d = da12 if I==0.5 else da32
    if pfilter:
        d = pf(d)

    stack = []
    for delta_t in range(5,12):
        t_stack = np.array([])
        for key1 in d.keys():
            for key2 in d[key1].keys():
                if int(re.findall("\d+",key2)[1])-int(re.findall("\d+",key1)[0])==delta_t:
                    scat = d[key1][key2]['calc_func'][func]
                    t_stack = np.append(t_stack,scat)
        stack.append(t_stack)

    plt.figure()
    a, b, fig = plt.hist(stack, stacked=True, 
             label=[str(i) for i in range(8, 15)])
    err = np.abs(b[-1]-b[0])/4
    #print(f'sys err:{err}')
    plt.legend(title="sm $\delta t$", 
               bbox_to_anchor=(1.05, 1.0),
               loc="upper left")
    plt.tight_layout()
    a0_str = "$a_0^{I=1/2}$" if I==0.5 else "$a_0^{I=3/2}$"
    plt.xlabel(a0_str)
    plt.title("Variation with $\delta t$ of sm-$C_{K\pi}$")
    return err

#===heat maps==========================================
import seaborn as sns

def perc(val, err):
    return round(np.abs(err*100/val), 2)

def heatmap(I,func=0, cuts=False, **kwargs): 
    d = da12 if I==0.5 else da32
    if cuts:
        d = choices(I, **kwargs)
    label = '$a_0^{I=1/2}$' if I==0.5 else '$a_0^{I=3/2}$'

    k1 = [k for k in d.keys()]
    k2 = [k for k in d[list(d)[0]].keys()]
    scat_2D = np.array([[d[key1][key2]['calc_func'][func] for key1 in k1] for key2 in k2]) 
    errs_2D = np.array([[perc(d[key1][key2]['calc_func'][func],d[key1][key2]['calc_func_err'][func])
                    for key1 in k1] for key2 in k2])
    pval = np.array([[d[key1][key2]['pvalue']
                    for key1 in k1] for key2 in k2])
    

    fig, ax = plt.subplots()
    im = ax.imshow(scat_2D)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(label, rotation=90, va="bottom")

    ax.set_xticks(np.arange(len(k1)), labels=k1)
    ax.set_yticks(np.arange(len(k2)), labels=k2)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
                     rotation_mode="anchor")
    ax.set_xlabel('pt fit range')
    ax.set_ylabel('sm fit range')

    #for i in range(len(k1)):
    #    for j in range(len(k2)):
    #        text = ax.text(i, j, pval[i, j], fontsize=5,
    #                       ha="center", va="center", color="w")


def heatmap_subset(I, ax=None, func=0, corr='pt', case='indiv', cb=True,
                   fix='t_min', fix_val=9, plot_cbar=False, **kwargs):
    d = da12 if I==0.5 else da32
    #min_val, max_val = minmax(d, 'calc_func')
    min_val, max_val = minmax(d, 'pvalue')
    label = '$a_0^{I=1/2}$' if I==0.5 else '$a_0^{I=3/2}$'

    idx = 0 if fix=='t_min' else 1
    if corr=='pt':
        k1 = [k for k in d.keys() if int(re.findall("\d+",k)[idx])==fix_val]
        k2 = [k for k in d[list(d)[0]].keys()]
        #scat_2D = np.array([[d[key1][key2]['calc_func'][func] for key1 in k1]
        #                    for key2 in k2])
        scat_2D = np.array([[d[key1][key2]['pvalue'] for key1 in k1]
                            for key2 in k2])
    else:
        k1 = [k for k in d.keys()]
        k2 = [k for k in d[list(d)[0]].keys() if int(re.findall("\d+",k)[idx])==fix_val]
        #scat_2D = np.array([[d[key1][key2]['calc_func'][func] for key1 in k1]
        #                    for key2 in k2])
        scat_2D = np.array([[d[key1][key2]['pvalue'] for key1 in k1]
                            for key2 in k2])
    
    ax = ax or plt.gca()
    im = ax.imshow(scat_2D, vmin=min_val, vmax=max_val, aspect='auto')
    if cb:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(label, rotation=90, va="bottom")

    ax.set_xticks(np.arange(len(k1)), labels=k1, fontsize=8)
    ax.set_yticks(np.arange(len(k2)), labels=k2, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
                     rotation_mode="anchor")
    if case=='indiv':
        ax.set_xlabel('pt fit range')
        ax.set_ylabel('sm fit range')

    return im 

def heatmap_subplots(I, func=0, corr='pt', fix='t_min', **kwargs):
    d = da12 if I==0.5 else da32
    #min_val, max_val = minmax(d, 'calc_func')
    min_val, max_val = minmax(d, 'pvalue')
    label = '$a_0^{I=1/2}$' if I==0.5 else '$a_0^{I=3/2}$'

    idx = 0 if fix=='t_min' else 1
    if corr=='pt':
        fix_vals = list(set([int(re.findall("\d+",k)[idx]) for k in d.keys()]))
        fig, axes = plt.subplots(nrows = 1, ncols = len(fix_vals), sharey=True)
        for i, ax in enumerate(axes.flatten()):
            im = heatmap_subset(I, ax=ax, func=func, corr='pt', case='sub', 
                    fix=fix, fix_val=fix_vals[i], cb=False, **kwargs)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        im.set_clim(min_val, max_val)
        fig.text(0.04, 0.5, 'sm fit range', va='center', rotation='vertical')
        fig.text(0.5, 0.96, label+' with pt fit range variations', ha='center')
    else:
        fix_vals = list(set([int(re.findall("\d+",k)[idx]) for k in d[list(d)[0]].keys()]))
        fig, axes = plt.subplots(nrows = len(fix_vals), ncols = 1, sharex=True)
        for i, ax in enumerate(axes.flatten()):
            im = heatmap_subset(I, ax=ax, func=func, corr='sm', case='sub',
                    fix=fix, fix_val=fix_vals[i], cb=False, **kwargs)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        im.set_clim(min_val, max_val)
        fig.text(0.04, 0.5, label+' with sm fit range', va='center', rotation='vertical')
        fig.text(0.5, 0.96, 'pt fit range', ha='center')

    return fig

def make_all_subplots(I):
    d = da12 if I==0.5 else da32
    fig1 = heatmap_subplots(I, corr="pt", fix="t_min")
    fig2 = heatmap_subplots(I, corr="pt", fix="t_max")
    fig3 = heatmap_subplots(I, corr="sm", fix="t_min")
    fig4 = heatmap_subplots(I, corr="sm", fix="t_max")

    
def choices(I, pt_t_min=9, pt_t_max=18, 
               sm_t_min=8, sm_t_max=18, **kwargs):
    d = da12 if I==0.5 else da32
    new_d = {}
    for key1 in d.keys():
        min_x, max_x = [int(s) for s in re.findall("\d+",key1)]
        if min_x>=pt_t_min and max_x<=pt_t_max:
            new_d[key1] = {}
            for key2 in d[key1].keys():
                min_y, max_y = [int(s) for s in re.findall("\d+",key2)]
                if min_y>=sm_t_min and max_y<=sm_t_max:
                    new_d[key1][key2] = d[key1][key2]
    return new_d

def plot_linear(I, cuts=False, func=0,  **kwargs):

    d = da12 if I==0.5 else da32
    if cuts:
        d = choices(I, **kwargs)
    label = '$a_0^{I=1/2}$' if I==0.5 else '$a_0^{I=3/2}$'

    k1 = [k for k in d.keys()]
    k2 = [k for k in d[list(d)[0]].keys()]
    scat_2D = np.array([[d[key1][key2]['calc_func'][func] for key1 in k1] for key2 in k2]) 
    errs_2D = np.array([[perc(d[key1][key2]['calc_func'][func], 
            d[key1][key2]['calc_func_err'][func])
                    for key1 in k1] for key2 in k2])
    x = []
    y = []
    err = []
    for key1 in k1:
        for key2 in k2:
            x.append(key1+key2)
            y.append(d[key1][key2]['calc_func'][func])
            err.append(d[key1][key2]['calc_func_err'][func])

    plt.figure()
    plt.errorbar(x,y, yerr=err, capsize=4, fmt='o')
    plt.ylabel(label)
    plt.xticks(rotation=90, fontsize=8)

