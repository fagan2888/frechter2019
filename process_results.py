import argparse
from os import path as op
parser = argparse.ArgumentParser()
parser.add_argument("res_root", help="Root folder containg the results.", type=str)
args = parser.parse_args()
res_root = args.res_root
if not op.isdir(res_root):
    raise ValueError("{} is not a directory.".format(res_root))

input_dir = "./input_data"
output_dir= "./output"

import os, sys
import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib import cm
from collections import namedtuple
import pdb

class TimedBlock:
    def __init__(self, name):
        self.name = name
        pass

    def __enter__(self):
        print("{}: Started {}.".format(datetime.now(), self.name))
        self.start_time = time.time()

    def __exit__(self, *args):
        print("{}: Finished {} in {:.2f} seconds.".format(datetime.now(), self.name, time.time() - self.start_time))


I = lambda x: x
ARR2STR = lambda x: ";".join(x)
keep_fields = {"y_true":I, "seed":I, "pool_baseline":I, "wnd_size":I, "pop":I, "shuffle":I, "label":I, "n_classes":I, "which_classes":ARR2STR, "which_cells":ARR2STR}
subdict = lambda d: {f:keep_fields[f](d[f]) for f in d if f in keep_fields}

with TimedBlock("Processing results from {}".format(res_root)):

    with TimedBlock("loading odour and group definitions"):
        btoc = pickle.load(open(op.join(input_dir, "btoc.p"), "rb"))
        btoc_groups = btoc["groups"]
        btoc_odours = btoc["odours"]
    
    df_file = op.join(res_root, "all_acc.df.p.bz2")
    if op.isfile(df_file):
        print("Found dataframe containing accuracy scores at {}".format(df_file))
        with TimedBlock("loading from {}".format(df_file)):
            df = pd.read_pickle(df_file)
    else:
        print("Did not find dataframe containing accuracy scores at {}".format(df_file))
        print("Creating.")
        records = []
        n_proc = 0
        with TimedBlock("reading accuracy files"):
            for root, dirs, files in os.walk(res_root):
                for f in files:
                    if f.endswith("_summary.p"):
                        with open(op.join(root,f), "rb") as in_file:
                            summary = pickle.load(in_file)
                            acc_file = "seed{}{}_acc.npy".format(summary["seed"], "shuf" if summary["shuffle"] else "orig")
                            acc_full_file = op.join(root, acc_file)
                            if not op.exists(op.join(root, acc_file)):
                                print("Could not find accuracy file {} for summary file {}, skipping.".format(acc_full_file, f))
                                continue
                            else:
                                acc = np.load(acc_full_file)
                                C_vals = summary["C_vals"]
                                if len(C_vals) != acc.shape[0]:
                                    raise ValueError("The first dimension of acc should have the same length {} as C_vals but has length {}.".format(len(C_vals), acc.shape[0]))
                                n_bins = acc.shape[1]
                                acc_cv_mean = np.mean(acc,axis=-1)
                                record_base = subdict(summary)
                                for ic, C in enumerate(C_vals):
                                    for ib in range(n_bins):
                                        record = dict(record_base)
                                        record["bin"] = ib
                                        record["C"] = C
                                        record["acc"] = acc_cv_mean[ic,ib]
                                        records.append(record)
                        n_proc += 1
                        if not (n_proc % 1000):
                            print("{} files processed.".format(n_proc))
            
        print("Read {} records.".format(len(records)))
        with TimedBlock("creating dataframe from records"):
            df = pd.DataFrame.from_records(records)                    
    
        df_file = op.join(res_root, "all_acc.df.p.bz2")
        with TimedBlock("saving dataframe to {}".format(df_file)):
            df.to_pickle(df_file, "bz2")
        print("Done.")
    

    # split the indices into two groups
    with TimedBlock("determining groups"):
        d = df.groupby(["pool_baseline", "wnd_size", "C","bin","pop","label","n_classes","shuffle"])
        k = d.groups.keys()
        
    with TimedBlock("computing halves"):
        halves = [[],[]]
        n_proc = 0
        for k in d.groups:
            ii = d.groups[k]
            l  = len(ii)
            if l < 2:
                print("Group {} had insufficient members.".format(k))
            if len(ii[:l//2]) != len(ii[l//2:]):
                print("Lengths don't match for key {}.".format(k))
        
            halves[0] += list(ii[:l//2])
            halves[1] += list(ii[l//2:])
    
            n_proc+=1        
            if not (n_proc % 10000):
                print("Processed {} groups.".format(n_proc))
    
    # Halve the data
    with TimedBlock("splitting dataframe into halves"):               
        df0 = df.drop(["which_cells", "which_classes"], axis=1).iloc[halves[0],:]
        df1 = df.drop(["which_cells", "which_classes"], axis=1).iloc[halves[1],:]

    print("Halve sizes: {}".format(len(df0), len(df1)))

    # Compute the best C values
    with TimedBlock("averaging performance across seeds"):
        m1 = df0.groupby(["pool_baseline", "wnd_size", "C", "bin", "pop", "label", "n_classes", "shuffle"], as_index = False).mean().drop(["seed"], axis=1)
    
    with TimedBlock("computing best C for each bin, population etc"):
        rows = []
        for a in m1.groupby(["pool_baseline", "wnd_size", "bin","pop", "label", "n_classes", "shuffle"]):
            dfa = a[1] # The first element of a is the index, the second is a dataframe
            rows.append(dfa.loc[dfa["acc"].idxmax()]) # get the row with the highest accuracy
        df_best_C = pd.DataFrame.from_records(rows)
    
    print(df_best_C.head())

    # subset the second have to use the best C values
    with TimedBlock("subset second half to use best C"):
        grouping = ["pool_baseline", "wnd_size", "pop", "label", "n_classes", "bin", "shuffle"] 
        g_best_C = df_best_C.groupby(grouping)
        subs = []
        for g in df1.groupby(grouping):  #Index into the half we want to subset
            # Get the C for this grouping
            C = df_best_C.iloc[g_best_C.groups[g[0]][0]]["C"] # g[0] is the group index.
            subs.append(g[1][g[1]["C"] == C]) # Subset according to C
        print("Appended {} groups.".format(len(subs)))
    
        with TimedBlock("concatenating dataframes"):
            df_sub = pd.concat(subs,axis=0)
            print("{} rows. ".format(len(df_sub)))

    # Each of the seeds should now map to one C value per bin.
    # We should then be able to plot a trace for each seed
    # We can then maximize over bins per each trace
    def subset_per_seed(pool_baseline, wnd_size, pop, label, n_classes, shuffle, field="acc"):
        # Returns a bins x (# seeds) matrix containing the values for each bin using the optimized value of C.
        dd = df_sub[(df_sub.pool_baseline == pool_baseline) & (df_sub.wnd_size == wnd_size) & (df_sub["pop"]==pop) & (df_sub.label==label) & (df_sub.n_classes==n_classes) & (df_sub.shuffle==shuffle)]    
        seeds = list(set(dd["seed"])) # Get the different seeds used for this config
        # The sort below ensures that we get the values in the right temporal sequence
        X = np.stack([dd[dd.seed == seed].sort_values("bin")[field].values for seed in seeds], axis = 1) 
        return X, seeds
    
    # ########  ##        #######  ######## 
    # ##     ## ##       ##     ##    ##    
    # ##     ## ##       ##     ##    ##    
    # ########  ##       ##     ##    ##    
    # ##        ##       ##     ##    ##    
    # ##        ##       ##     ##    ##    
    # ##        ########  #######     ##
    
    cell_colours = {"PN":"red", "L":"limegreen", "O":"dodgerblue"}
    cell_light_colours = {"PN":"mistyrose", "L":"honeydew", "O":"lightcyan"}
    classes_per_pop = pickle.load(open(op.join(input_dir, "classes_per_pop.p"), "rb"))

    print("Classes per population:")
    for pop, cl in classes_per_pop.items():
        print("{:>4}: {}".format(pop, cl))

    pop_sizes = {pop:len(cl) for pop, cl in classes_per_pop.items()} #"PN":21, "O":50, "L":16
    print("Number of classes / population:")
    for pop, sz in pop_sizes.items():
        print("{:>4}: {}".format(pop, sz))

    pop_names = {"PN":"PN", "O":"LHON", "L":"LHLN"}

    with TimedBlock("plot individual traces."):
        for pop in pop_sizes:
            for task in ["identity", "category"]:
                Xorig, seeds = subset_per_seed(0, 1, pop, task, pop_sizes[pop], False, "acc")
                Xshuf, seeds = subset_per_seed(0, 1, pop, task, pop_sizes[pop], True,  "acc")    
                plt.figure(figsize=(12,8))
                plt.plot(Xorig, color=cell_light_colours[pop])
                plt.plot(Xshuf, color="#AAAAAA")
                plt.plot(np.mean(Xorig,axis=1),color=cell_colours[pop],linewidth=2, label="original")
                plt.plot(np.mean(Xshuf,axis=1),"k",linewidth=2, label="shuffled")
                plt.xlabel("bin",fontsize=14)
                plt.ylabel("accuracy",fontsize=14)
                plt.ylim(0,1);
                plt.title("{} {} decoding accuracy using {} classes.".format(pop_names[pop], task, pop_sizes[pop]), fontsize=18)
                plt.legend()
                file_name = op.join(output_dir, "{}_{}_{}classes_timecourse.pdf".format(pop_names[pop], task, pop_sizes[pop]))
                plt.savefig(file_name, bbox_inches="tight")
                plt.close("all")
                print("Wrote {}".format(file_name))


    with TimedBlock("plot timecourse of C values for each task, population, and n classes."):
        for pop in pop_sizes:
            for task in ["identity", "category"]:
                plt.figure(figsize=(12,12))
                npops = sorted(list(set(df_sub[df_sub["pop"] == pop]["n_classes"].values)))
                cols  = [cm.jet(float(i)/(len(npops)-1)) for i in range(len(npops))]
                for shuf in [0,1]:
                    plt.subplot(2,1,shuf+1)                        
                    for ipop, npop in enumerate(npops):
                        X, _ = subset_per_seed(0, 1, pop, task, npop, shuf, "C")
                        # X will have the same value in each bin, because the best C value is being used.
                        # So the mean below is equivalent to extracting the first row
                        LX = np.mean(np.log10(X), axis = 1) 
                        plt.plot(LX, color=cols[ipop], linewidth=2, label="# classes = {}".format(npop) if not ipop else str(npop))                            
                    plt.xlabel("bin",fontsize=14)
                    plt.ylabel("log10(C)",fontsize=14)
                    plt.ylim(-10,2);
                    plt.title("{} {} best C value ({}).".format(pop_names[pop], task, "shuf" if shuf else "orig" ), fontsize=18)
                    if not shuf:
                        plt.legend()
                file_name = op.join(output_dir, "{}_{}_C_timecourse.pdf".format(pop_names[pop], task))
                plt.savefig(file_name, bbox_inches="tight")
                plt.close("all")
                print("Wrote {}".format(file_name))
        
    with TimedBlock("plot overlayed traces."):
        for task in ["identity", "category"]:
            plt.figure(figsize=(12,8))
            for pop in pop_sizes:
                Xorig, seeds = subset_per_seed(0, 1, pop, task, 10 + 0*pop_sizes[pop], False, "acc")
                Xshuf, seeds = subset_per_seed(0, 1, pop, task, 10 + 0*pop_sizes[pop], True,  "acc")    
                #plt.plot(Xorig, color=cell_light_colours[pop])
                plt.plot(Xshuf, color="#AAAAAA")
                plt.plot(np.mean(Xorig,axis=1),color=cell_colours[pop],linewidth=2, label="{}".format(pop))
                #plt.plot(np.mean(Xshuf,axis=1),color=cell_colours[pop],linewidth=2)
            plt.xlabel("bin",fontsize=14)
            plt.ylabel("accuracy",fontsize=14)
            plt.ylim(0,1);
            plt.title("{} decoding accuracy using 10 classes / type.".format(task), fontsize=18)
            plt.legend()
            file_name = op.join(output_dir, "overlayed_{}_timecourse.pdf".format(task))
            plt.savefig(file_name, bbox_inches="tight")
            plt.close("all")
            print("Wrote {}".format(file_name))

    with TimedBlock("plot performance"):
        g = {"identity":btoc_odours, "category":btoc_groups}
    
        pool_baseline = 0
        wnd_size = 1
        use_shuffle = True

        def _plot_performance(label, legend=False, ylim = [0,1], title = True):
            lines = []
            line_labels = []
            for pop in cell_colours:
                n_classes = sorted(list(set(df_sub[df_sub["pop"]==pop]["n_classes"].values)))
                data = [np.max(subset_per_seed(pool_baseline, wnd_size, pop, label, n, False)[0], axis=0) for n in n_classes]
                acc_mean = np.array([np.mean(d) for d in data])
                acc_std  = np.array([np.std(d)  for d in data])#/np.sqrt(len(data))
                lines.append(plt.plot(n_classes, acc_mean, "o-", color = cell_colours[pop], label=pop, linewidth=4, markersize=10)[0])
                line_labels.append(pop)
                plt.fill_between(n_classes, acc_mean - acc_std, acc_mean + acc_std, alpha=0.25, facecolor = cell_colours[pop], label=pop)
                
                if use_shuffle:
                    data = [np.max(subset_per_seed(pool_baseline, wnd_size, pop, label, n, True)[0], axis=0) for n in n_classes]
                    acc_mean = np.array([np.mean(d) for d in data])
                    acc_std  = np.array([np.std(d)  for d in data])#/np.sqrt(len(data))
                    lines.append(plt.plot(n_classes, acc_mean, "o-", color = cell_colours[pop], label=pop, linewidth=2, markersize=5)[0])
                    line_labels.append(pop + " (shuf)")
                    plt.fill_between(n_classes, acc_mean - acc_std, acc_mean + acc_std, alpha=0.25, facecolor = cell_colours[pop], label=pop)
                    
                plt.xlabel("# of classes", fontsize=12)
                plt.ylabel("accuracy",   fontsize=12)

            if ylim is not None:
                plt.ylim(ylim)

            if title:
                plt.title("Decoding " + label.upper(), fontsize=16)
            if legend:
                plt.legend(lines,line_labels, framealpha=0)
        
        plt.figure(figsize=(12,8))
        for ipanel,label in enumerate(["identity", "category"]):

            # The full performance plot
            plt.subplot(2,2,ipanel+1)
            _plot_performance(label, legend=(ipanel==0))

            # The same plot, but zoomed in showing results for # classes up to 10.
            plt.subplot(2,2,2+ipanel+1)
            _plot_performance(label, title = False, ylim = [0, 0.75] if label == "identity" else [0.2, 0.75])
            plt.gca().set_xlim(0,11)
            plt.gca().set_xticks([1,5,10])
            

            file_name = op.join(output_dir, "id_vs_cat{}.pdf".format("_w_shuff" if use_shuffle else ""))
        plt.tight_layout()
        plt.savefig(file_name, bbox_inches="tight")
        plt.close("all")
        
        print("Wrote {}".format(file_name))

print("ALLDONE.")
