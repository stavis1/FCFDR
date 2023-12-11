#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 18:09:47 2023

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
                    help='Region Of Pracical Equivalence in fold change units, default = 1.')
parser.add_argument('-x', '--excel', action = 'store', required = False, default = False,
                    help='excel file input, use either this or the tsv + experimental design input option')
parser.add_argument('-t', '--tsv', action = 'store', required = False, default = False,
                    help='tsv file input, use either this and the experimental design file or the excel input option')
parser.add_argument('-d', '--design', action = 'store', required = False,
                    help='experimental design file, only for use with the --tsv input mode')
args = parser.parse_args()

from collections import defaultdict
import os

from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde as kde
from scipy.stats import ttest_ind

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

data.index = data['analyte']

cond_samp = defaultdict(lambda:[])
_=[cond_samp[c].append(s) for c,s in zip(design['condition'],design['sample'])]

print('Calculating true log 2 fold changes')
true_ts = []
for cond1, cond2 in zip(comparisons['condition1'], comparisons['condition2']):
    tstats = ttest_ind(data[cond_samp[cond1]],data[cond_samp[cond2]], axis = 1, equal_var=False).statistic
    true_ts.extend([(tstat,True,a,cond1,cond2) for a, tstat in zip(data['analyte'],tstats)])
true_dist = [(i+1,*n) for i,n in enumerate(sorted(true_ts, key = lambda x: abs(x[0]), reverse = True))]

print('Generating sampling distributions')
ac_nulls = {}
for cond in all_comp_conds:
    print(f'Condition: {cond}')
    for analyte in tqdm(data['analyte']):
        ac_nulls[(analyte,cond)] = kde(data.loc[analyte,cond_samp[cond]])

print('Calculating null fold changes')
null_ts = []
for conds in zip(comparisons['condition1'], comparisons['condition2']):
    for cond in conds:
        n_samps = len(cond_samp[cond])
        means = np.mean(data[cond_samp[cond]], axis = 1)
        fc_offsets = (means*args.ROPE) - means
        tstats = ttest_ind([list(ac_nulls[(a,cond)].resample(n_samps)[0]) for a in data['analyte']],
                           [list(ac_nulls[(a,cond)].resample(n_samps)[0] + o )for a,o in zip(data['analyte'],fc_offsets)],
                           axis = 1, equal_var=False).statistic
        null_ts.extend([(tstat,True,a,cond1,cond2) for a, tstat in zip(data['analyte'],tstats)])

n_null = len(null_ts)
null_dist = [((i+1)/2,*n) for i,n in enumerate(sorted(null_ts, key = lambda x: abs(x[0]), reverse = True))]
dist = list(sorted(null_dist + true_dist, key = lambda x: abs(x[1])))

sig_idx = len(dist)
exp_null = 0
for i,tstat in enumerate(dist):
    if tstat[2] and exp_null/tstat[0] < args.fdr:
        sig_idx = i
        break
    if not tstat[2]:
        exp_null = tstat[0]

sig_ts = [(*t, i > sig_idx) for i,t in enumerate(dist) if t[2]]
significant = pd.DataFrame({'analyte':[t[3] for t in sig_ts],
                            'condition1':[t[4] for t in sig_ts],
                            'condition2':[t[5] for t in sig_ts],
                            'tstat':[t[1] for t in sig_ts],
                            'significant':[int(t[-1]) for t in sig_ts]})

def get_bounds(analyte, c1, c2):
    niter = 1000
    resamps1 = np.reshape(ac_nulls[(analyte,c1)].resample(niter*len(cond_samp[c1])), (niter,len(cond_samp[c1]))).astype(float)
    resamps2 = np.reshape(ac_nulls[(analyte,c2)].resample(niter*len(cond_samp[c2])), (niter,len(cond_samp[c2]))).astype(float)
    l2fc = np.log2(np.nanmean(resamps1, axis = 1)/np.nanmean(resamps2, axis = 1))
    l2fc = l2fc[np.isfinite(l2fc)]
    return np.quantile(l2fc, (0.1,0.9))

print('Calculating interval estimates')
bounds = [get_bounds(*i) for i in tqdm(zip(*[significant[c] for c in significant.columns[:3]]))]

significant['l2fc'] = [np.log2(np.mean(data.loc[a,cond_samp[c1]])/np.mean(data.loc[a,cond_samp[c2]])) for a,c1,c2 in zip(significant['analyte'],
                                                                                                                         significant['condition1'],
                                                                                                                         significant['condition2'])]
significant['lower_bound'] = [b[0] for b in bounds]
significant['upper_bound'] = [b[1] for b in bounds]
coherent = np.equal(np.sign(significant['upper_bound']),np.sign(significant['lower_bound']))
significant['significant'] = [s if c else 0 for s,c in zip(significant['significant'],coherent)]


if args.excel:
    with pd.ExcelWriter(f'{args.out}.xlsx') as xlsx:
        for cond1, cond2 in zip(comparisons['condition1'], comparisons['condition2']):
            tmpdf = significant[np.logical_and(significant['condition1'] == cond1,significant['condition2'] == cond2)]
            tmpdf.to_excel(xlsx,sheet_name = f'{cond1}-{cond2}', index = False)

else:
    os.mkdir(args.out)
    for cond1, cond2 in zip(comparisons['condition1'], comparisons['condition2']):
        significant[significant['condition1'] == cond1][significant['condition2'] == cond2].to_csv(f'{args.out}/{cond1}-{cond2}.tsv', 
                                                                                                   sep = '\t', 
                                                                                                   index = False)

