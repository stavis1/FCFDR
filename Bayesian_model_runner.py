#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:49:38 2023

@author: 4vt
"""

import argparse
parser = argparse.ArgumentParser(
                    prog='Nonparametric Fold Change FDR',
                    description='Controls the false discovery rate of fold changes based on nonparametric simulations of the null distribution.')
parser.add_argument('-o', '--out', action = 'store', required = False, default = 'results.txt',
                    help='output filename, default is "results"')
parser.add_argument('-f', '--fdr', action = 'store', required = False, default = 0.01, type = float,
                    help='FDR cutoff, default is 0.01')
parser.add_argument('-r', '--ROPE', action = 'store', required = False, default = 1, type = float,
                    help='Region Of Pracical Equivalence in fold change units, default is 1.')
parser.add_argument('-x', '--excel', action = 'store', required = False, default = False,
                    help='excel file input, use either this or the tsv + experimental design input option')
parser.add_argument('-t', '--tsv', action = 'store', required = False, default = False,
                    help='tsv file input, use either this and the experimental design file or the excel input option')
parser.add_argument('-d', '--design', action = 'store', required = False,
                    help='experimental design file, only for use with the --tsv input mode')
parser.add_argument('-m', '--model', action = 'store', required = False, default = 'simpleAnalyteModel.stan',
                    help='.stan model file, default is to look for simpleAnalyteModel.stan in current directory')
args = parser.parse_args()

from collections import defaultdict
import os

from tqdm import tqdm
import pandas as pd
import numpy as np
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
from scipy.stats import zscore
import logging
import re


if args.excel:
    with open(args.excel, 'rb') as xl:
        data = pd.read_excel(xl, 'data')
        design = pd.read_excel(xl, 'design')
        comparisons = pd.read_excel(xl, 'comparisons')

elif args.tsv:
    data = pd.read_csv(args.tsv, sep='\t')
    with open(args.design, 'r') as txt:
        blocks = txt.read().split('#')[1:]
    design = [f.split('\t') for f in blocks[0].split('\n')[1:] if f]
    design = pd.DataFrame({'sample':[f[0] for f in design],
                           'condition':[f[1] for f in design]})
    comparisons = [f.split('\t') for f in blocks[1].split('\n')[1:] if f]
    comparisons = pd.DataFrame({'condition1':[f[0] for f in comparisons],
                                'condition2':[f[1] for f in comparisons]})

else:
    raise Exception('Use either the excel input option or the tsv + experimental design file input option')

#check that inputs are well formatted
all_comp_conds = set([c for col in ['condition1','condition2'] for c in comparisons[col]])
all_des_conds = set(design['condition'])
all_data_samps = set(data.columns[1:])
all_des_samps = set(design['sample'])
if not all(c in all_des_conds for c in all_comp_conds):
    raise Exception('There is a mismatch between condition names in the design and comparisons inputs')
if all_comp_conds != all_des_conds:
    print('Warning: not all conditions are participating in a comparison')
if not all(s in all_des_samps for s in all_data_samps):
    raise Exception('Not all samples are listed in the experimental design')
if not all(s in all_data_samps for s in all_des_samps):
    raise Exception('There are samples listed in the experimental design that are not present in the data')
if not len(set(data['analyte'])) == data.shape[0]:
    raise Exception('Analyte IDs must be unique in the dataset')

data.index = range(1,data.shape[0]+1)
cond_samp = defaultdict(lambda:[])
_=[cond_samp[c].append(s) for c,s in zip(design['condition'],design['sample'])]

def lz_norm(vals):
    vals = np.log(vals)
    gmean = np.nanmean(vals)
    gstd = np.nanstd(vals)
    vals = zscore(vals,axis = 0, nan_policy='omit')
    vals = (vals*gstd)+gmean
    vals = np.exp(vals)
    return vals

data.iloc[:,1:] = lz_norm(data.iloc[:,1:])

cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = False
cmdstanpy_logger.handlers = []
cmdstanpy_logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('/home/4vt/Documents/data/SLT09_FCFDR/data/all.log')
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                        "%H:%M:%S",))
cmdstanpy_logger.addHandler(handler)
model = CmdStanModel(stan_file = args.model)

label_map = {c+n:l+1 for l,c in enumerate('ABC') for n in '123'}
vals = np.asarray([v for c in data.columns[1:] for v in data[c]])
labels = [label_map[c] for c in data.columns[1:] for _ in range(data.shape[0])]

analytes = []
for _ in data.columns[1:]:
    analytes.extend(data.index)

stds = []
means = []
valstds = []
valmeans = []
for cond in 'ABC':
    condvals = data[[c for c in data.columns[1:] if c.startswith(cond)]]
    condmeans = np.nanmean(condvals, axis = 1)
    condstds = np.nanstd(condvals, axis = 1)
    means.append(condmeans)
    stds.append(condstds)
    valmeans.extend(list(condmeans)*condvals.shape[1])
    valstds.extend(list(condstds)*condvals.shape[1])

vals = (vals - valmeans)/valstds

data = {'N':len(vals),
        'N_analyte':data.shape[0],
        'N_cond':3,
        'N_comp':3,
        'y':vals,
        'label':labels,
        'analyte':analytes,
        'comparisons':[[1,2],[1,3],[2,3]],
        'means':means,  
        'stds':stds}

fit = model.sample(data=data,
                   show_console=True,
                   inits = 0,
                   iter_sampling=4000)

draws = fit.draws_pd()
summary = fit.summary()


#fuzzy caterpiller plots
fsize = 5
varcols = ['sigma_sigma', 'mu_sigma', 'nu','lnorm_mu','lnorm_sigma','alpha','beta']
varcols = [c for c in varcols if c in summary.index]
fig, axes = plt.subplots(nrows = len(varcols),sharex = True)
for i,ax in enumerate(axes):
    varcol = varcols[i]
    for chain in set(draws['chain__']):
        subset = draws[draws['chain__'] == chain]
        ax.plot(range(subset.shape[0]), subset[varcol], '-', linewidth = 1)
    ax.plot([0,subset.shape[0]],[np.mean(draws[varcol])]*2, '-k', linewidth = 0.5)
    ax.set_ylabel(varcol, rotation = 0, fontsize = fsize)
    yticks = [min(draws[varcol]),max(draws[varcol])]
    ax.set_yticks(yticks,[str(round(t,2)) for t in yticks], fontsize = fsize)
ax.set_xlim(0,subset.shape[0])
ax.set_xticks([])
plt.show()
plt.close('all')

#pairs plot
fig, axes = plt.subplots(nrows = len(varcols), ncols = len(varcols),
                       layout = 'constrained', figsize = (8,8))
for i,var1 in enumerate(varcols):
    for j,var2 in enumerate(varcols):
        ax = axes[i,j]
        if var1 != var2:
            ax.scatter(draws[var2],draws[var1], s = 1, c = 'k', marker = '.')
        else:
            ax.hist(draws[var1], bins = 100, color = 'k')
        if i == len(varcols)-1:
            ax.set_xlabel(var2)
        if j == 0:
            ax.set_ylabel(var1)

#QC variable dists
varclasses = ['mu','sigma','fold_changes']
good_rows = [any(r.startswith(v) for v in varclasses) for r in summary.index]
fig, axes = plt.subplots(nrows = 3, layout = 'constrained')
for ax,var in zip(axes,['N_Eff','R_hat']):
    bins = np.linspace(min(summary.loc[good_rows,var]),max(summary.loc[good_rows,var]),100)
    for varclass in varclasses:
        ax.hist(summary.loc[[v.startswith(varclass) for v in summary.index],var],
                bins = bins, alpha = 0.5, label = varclass)
    ax.set_xlim(min(summary.loc[good_rows,var]), max(summary.loc[good_rows,var]))
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(var)
    ax.legend(fontsize = fsize)
ax = axes[2]
var = 'MCSE'
bins = np.logspace(np.log10(min(summary.loc[good_rows,var])),
                   np.log10(max(summary.loc[good_rows,var])),
                   100)
for varclass in varclasses:
    ax.hist(summary.loc[[v.startswith(varclass) for v in summary.index],var],
            bins = bins, alpha = 0.5, label = varclass)
ax.set_xlim(min(summary.loc[good_rows,var]), max(summary.loc[good_rows,var]))
ax.set_xscale('log')
ax.set_yscale('log')
ax.yaxis.set_label_position("right")
ax.set_ylabel(var)
ax.legend(fontsize = fsize)
plt.show()
plt.close('all')

#variable dists
fc_rows = [v.startswith('fold_changes') for v in summary.index],
mean_rows = [v.startswith('mu') for v in summary.index]
std_rows = [v.startswith('sigma[') for v in summary.index]
fig, axes = plt.subplots(nrows = 6, layout = 'constrained', figsize = (6,8))
#fold changes
ax = axes[0]
fcs = summary.loc[fc_rows,'Mean'].to_numpy()
bins = np.logspace(np.log10(min(fcs)),np.log10(max(fcs)),100)
ax.hist(fcs, bins = bins, color = 'k')
ax.set_xscale('log')
ax.set_xlabel('Fold change')
ax.set_ylabel('Count')
#interval widths
ax = axes[1]
lb = summary.loc[fc_rows,'5%'].to_numpy()
ub = summary.loc[fc_rows,'95%'].to_numpy()
relwidth = (ub-lb)/fcs
ax.hist(relwidth, bins = 100, color = 'k')
ax.set_xlabel('Fold change posterior 90% interval relative width')
ax.set_ylabel('Count')    
#mu
ax = axes[2]
zmeans = summary.loc[mean_rows,'Mean']
ax.hist(zmeans, bins = 100, color = 'k')
ax.set_xlabel('Z-transformed means')
ax.set_ylabel('Count')
#mu interval widths
ax = axes[3]
lb = summary.loc[mean_rows,'5%'].to_numpy()
ub = summary.loc[mean_rows,'95%'].to_numpy()
relwidth = ub-lb
ax.hist(relwidth, bins = 100, color = 'k')
ax.set_xlabel('Z-transformed mean posterior 90% interval absolute width')
ax.set_ylabel('Count')
#mu
ax = axes[4]
zstds = summary.loc[std_rows,'Mean']
ax.hist(zstds, bins = 100, color = 'k')
ax.set_xlabel('Z-transformed standard deviations')
ax.set_ylabel('Count')
#mu interval widths
ax = axes[5]
lb = summary.loc[std_rows,'5%'].to_numpy()
ub = summary.loc[std_rows,'95%'].to_numpy()
relwidth = ub-lb
ax.hist(relwidth, bins = 100, color = 'k')
ax.set_xlabel('Z-transformed standard deviation posterior 90% interval absolute width')
ax.set_ylabel('Count')


plt.show()
plt.close('all')

#more variable plots    
newmus = sorted([i for i in summary.index if i.startswith('newmu')],
                key = lambda x: summary.loc[x,'Mean'])
fcs = sorted([i for i in summary.index if i.startswith('fold_change')],
             key = lambda x: summary.loc[x,'Mean']) 


oldmus = []
for newmu in newmus:
    i0 = int(re.search(r'\[(\d+),',newmu).group(1))-1
    i1 = int(re.search(r',(\d+)\]',newmu).group(1))-1
    oldmus.append(means[i0][i1])
est_means = [summary.loc[m,'Mean'] for m in newmus]

sigmas = sorted([i for i in summary.index if i.startswith('sigma[')], 
                key = lambda x: summary.loc[x,'Mean'])

sigmas_t = []
sigmas_bounds = []
oldstds = []
for sigma in sigmas:
    i0 = int(re.search(r'\[(\d+),',sigma).group(1))-1
    i1 = int(re.search(r',(\d+)\]',sigma).group(1))-1
    oldstds.append(stds[i0][i1])
    sigmas_t.append((summary.loc[sigma,['Mean']].item()*stds[i0][i1])+means[i0][i1])
    sigmas_bounds.append((summary.loc[sigma,['5%','95%']]*stds[i0][i1])+means[i0][i1])
sigmas_bounds = sorted(sigmas_bounds, key = lambda x: np.mean(x))

fig, axes = plt.subplots(figsize = (6,6), nrows = 2, ncols = 2, layout = 'constrained')
ax = axes[0,0]
for i,newmu in enumerate(newmus):
    ax.plot(summary.loc[newmu,['5%','95%']],[i]*2, '-k', linewidth = 0.5)
ax.set_xscale('log')
ax.set_xlabel('Posterior Mean')
ax.set_ylabel('Rank')
ax = axes[0,1]
for i,fc in enumerate(fcs):
    ax.plot(summary.loc[fc,['5%','95%']],[i]*2, '-k', linewidth = 0.5)
ax.set_xscale('log')
ax.set_xlabel('Posterior Fold change')
ax.set_ylabel('Rank')
ax = axes[1,0]
for i,std in enumerate(sigmas):
    ax.plot(summary.loc[std,['5%','95%']],[i]*2, '-k', linewidth = 0.5,zorder=1)
    ax.scatter(summary.loc[std,'Mean'],i,s=1,c='w',marker = '.',zorder=2)
# ax.set_xscale('log')
ax.set_xlabel('Posterior Z-standard deviation')
ax.set_ylabel('Rank')
ax = axes[1,1]
for i,std in enumerate(sigmas_bounds):
    ax.plot(std,[i]*2, '-k', linewidth = 0.5)
ax.set_xscale('log')
ax.set_xlabel('Posterior standard deviation')
ax.set_ylabel('Rank')
plt.show()
plt.close('all')

fig,axes = plt.subplots(nrows = 2, figsize = (4,8))
ax = axes[0]
ax.scatter(oldmus,est_means, s = 1, c = 'k', marker = '.')
lims = [np.min([oldmus,est_means]),np.max([oldmus,est_means])]
ax.plot(lims,lims,'-r',linewidth = 0.5)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Calculated mean')
ax.set_ylabel('Posterior mean')
ax = axes[1]
ax.scatter(oldstds,sigmas_t,
                     s = 1, c = 'k', marker = '.')
lims = [np.min([oldstds,sigmas_t]),np.max([oldstds,sigmas_t])]
ax.plot(lims,lims,'-r',linewidth = 0.5)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Calculated standard deviation')
ax.set_ylabel('Posterior standard deviation')
plt.show()
plt.close('all')    

print(fit.diagnose())

