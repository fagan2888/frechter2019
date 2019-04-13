from os import path as op
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import scipy.optimize as opt
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.use("Agg") # So we don't actually need an X display to plot
from matplotlib import pyplot as plt
from matplotlib import cm

from copy import deepcopy


def divisive_normalize(X, n, a):
    return (X**n)/(X**n + a**n)

def firing_rate_distribution_entropy(x, num_bins = 21):
    pvals, _ = np.histogram(x, num_bins, density=True)
    return stats.entropy(pvals)

def get_divisive_normalization_parameters(x, n_range=(0.1,10), a_range=(1,100)):
    # Determines the divisive normalization parameters (n and a) that
    # maximize the entropy of the output distribution by grid search.
    obj = lambda na: -firing_rate_distribution_entropy(divisive_normalize(x, np.exp(na[0]), np.exp(na[1])))
    results = opt.brute(obj, [(np.log(n_range[0]), np.log(n_range[1])),(np.log(a_range[0]), np.log(a_range[1]))], finish=None, Ns=20)
    n, a = np.exp(results)
    xn = divisive_normalize(x,n,a)        
    return {"a":a,"n":n,"h_before":firing_rate_distribution_entropy(x), "h_after":firing_rate_distribution_entropy(xn), "xn":xn }

data_dir   = "./input_data"
output_dir = "./output"

print("Loading the input data.")
input_data = pickle.load(open(op.join(data_dir, "stats_by_class_subset.p"), "rb"))

odour_names   = input_data["odour_names"]
odour_groups  = input_data["odour_groups"]
unique_groups = list(set(odour_groups))

rownames_per_pop = input_data["rownames_per_pop"]
pops             = [pop for pop, names in rownames_per_pop.items()] # PN, O, L
data_orig        = input_data["stats_by_class_per_pop"]

print("Computing divisive normalization parameters.")
norm_params_df = pd.DataFrame(columns=('pop','iclass','n','a','h_before','h_after'))
curr_row = 0
data_norm = deepcopy(data_orig)
for pop in pops:
    for iclass, class_name in enumerate(rownames_per_pop[pop]):
        results = get_divisive_normalization_parameters(data_orig[pop][iclass,:], n_range=(1,2), a_range=(1,100))
        norm_params_df.loc[curr_row] = [pop, iclass, results["n"], results["a"], results["h_before"], results["h_after"]]
        data_norm[pop][iclass, :] = results["xn"]
        curr_row+=1

print("The first few parameters:")
print(norm_params_df.head())

print("Baseline subtracting the data.")
data = {"orig":data_orig, "norm":data_norm}
# Since the first of the 7 odour response bins is the baseline,
# we just kron this first bin out 7 times and subtract it from
# the raw data
data_bl_sub = {cond:{pop: data[cond][pop] - np.kron(data[cond][pop][:, range(0, data[cond][pop].shape[1],7)], np.ones((1,7))) for pop in pops} for cond in ["orig", "norm"]}

print("Performing PCA on the baseline subtracted data.")
pca = PCA(n_components=10, whiten=False)
data_pca = {cond:{pop: pca.fit(data_bl_sub[cond][pop].T).transform(data_bl_sub[cond][pop].T).T for pop in pops} for cond in ["orig", "norm"]}

odour_group_colors = cm.hsv(np.linspace(0,1,len(unique_groups)+1))
colors             = dict(zip(sorted(unique_groups), odour_group_colors))

def add_color_legend(xpos, ypos, dy = 0.05):
    xl = plt.xlim()
    yl = plt.ylim()
    x0 = xl[0] + xpos * np.diff(xl)
    y0 = yl[0] + ypos * np.diff(yl)
    dy = np.diff(yl) * dy
    for i, k in enumerate(sorted(list(colors.keys()))):
        plt.text(x0, y0-i * dy, k, color=colors[k])

which_pca_cmps = [1,2]
print("Plotting PCA component {} against {}.".format(*which_pca_cmps))
plt.figure(figsize=(8,5))
import pdb
for icond, cond in enumerate(["orig", "norm"]):
    for t, pop in enumerate(pops):
        Y = data_pca[cond][pop]        
        plt.subplot(2, 3, 3*icond + t + 1)
        for i, odour_group in enumerate(odour_groups):
            index = np.arange(7) + i*7
            col   = colors[odour_group]
            plt.plot(Y[which_pca_cmps[0],index], Y[which_pca_cmps[1],index], "-", color = col )
    
        plt.title(pop, fontsize=20)
        if pop == "PN":
            if cond == "orig":
                plt.ylabel("Raw",fontsize=20)
            else:
                plt.ylabel("Divisive\nNormalization",fontsize=20)                

        if pop =="O" and cond == "norm":
            add_color_legend(1.0,1.3,dy=0.15)
    
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)        
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.axis("tight")
    
plt.tight_layout(pad=0.0, w_pad=0, h_pad=0.0)
output_file = op.join(output_dir, "fig_pca_raw_normalized.pdf")
plt.savefig(output_file,bbox_inches="tight")
plt.close("all")
print("Wrote {}".format(output_file))
print("ALLDONE.")
