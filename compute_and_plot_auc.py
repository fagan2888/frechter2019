from os import path as op
import pickle
import pandas as pd
import numpy as np
from sklearn import metrics as metrics
import aesthetics
from matplotlib import pyplot as plt
from scipy import stats
from importlib import reload

data_dir   = "./input_data"
output_dir = "./output"

# 1. LOAD THE DATA THAT WE NEED
data = pickle.load(open(op.join(data_dir, "stats_by_class_subset.p"), "rb"))

odour_names            = data["odour_names"]
odour_groups           = data["odour_groups"]
unique_groups          = list(set(odour_groups))

rownames_per_pop       = data["rownames_per_pop"]
pops                   = [pop for pop, names in rownames_per_pop.items()] # PN, O, L
stats_by_class_per_pop = data["stats_by_class_per_pop"]

# 2. SUMMARIZE THE ODOUR RESPONSES.
# The odours responses consist of average spike counts
# in 7 bins: 1 baseline and 6 after odour onset.
# We're going to summarize these 7 bins into single value,
# the maximum across all 7 bins.

def summarize_per_odor(X, summary_function):
    # Summarizes the 7 bins of data for each odour into
    # a single value according to summary_function.
    Xsum = np.zeros((X.shape[0],X.shape[1]//7))
    for i in range(Xsum.shape[0]):
        for j in range(Xsum.shape[1]):
            Xsum[i,j] = summary_function(X[i,range(j*7,(j+1)*7)])
    return Xsum
baseline_subtracted_max = lambda resp: np.max(resp - resp[0])
data_max = { pop : summarize_per_odor(stats_by_class_per_pop[pop], baseline_subtracted_max) for pop in pops}

# 3. COMPUTE THE AUC SCORES
# We're now going to compute the AUC scores measuring
# how well the odour responses of each class, as summarized
# in the previous step, can classify odour groups.
#
num_shuffles = 5
num_groups   = len(unique_groups)
auc_df = pd.DataFrame(columns=('pop', 'iclass','class','shuffled','shuffle_index','group','auc') )
curr_row = 0
np.random.seed(0)
for pop in pops:
    print("Computing AUCs for {:<2} classes...".format(pop))
    X = data_max[pop]
    for shuffle_index in range(num_shuffles+1):
        Xthis = 1. * X # Copy X
        shuffled = shuffle_index>0
        if shuffled is True:
            # Shuffle the responses of each class
            for row in range(Xthis.shape[0]):
                Xthis[row,:] = np.random.permutation(Xthis[row,:])
    
        for iclass, class_name in enumerate(rownames_per_pop[pop]):
            y_scores = Xthis[iclass,:]
            for k in range(num_groups):
                y_true = [g == unique_groups[k] for g in odour_groups]
                auc = metrics.roc_auc_score(y_true, y_scores, average='weighted')
                auc_df.loc[curr_row] = [pop, iclass, class_name, shuffled, shuffle_index, unique_groups[k], auc]
                curr_row+=1
output_file = op.join(output_dir, "auc_results_full.csv")
print("Writing AUC scores to {}.".format(output_file))
auc_df.to_csv(output_file, index=False)

# The AUC scores have now been computed for each class and each odour group.
# Since we don't know which group each class is a classifer for, we'll assume it's the
# group it classifies best, i.e. the one with the highest AUC score.
# In the next chunk of code, we'll measure the classification ability of each class
# by its best AUC score across odour groups. We can the compare the resulting maximum 
# AUC scores for the shuffled and unshuffled data to see which populations do better than
# chance at classifiying odour groups.

print("Computing the maximum AUC across odour groups for each population and class.")
# Compute the maximum auc for each cell and each shuffle index
auc_max = auc_df.groupby(by=["pop", "iclass", "shuffle_index"], as_index=False).max().drop(["group"],axis=1)
# Now compute the mean of the maximum over the shuffles and the unshuffled
auc_max = auc_max.groupby(by=["pop","iclass","shuffled"], as_index=False).mean().drop(["shuffle_index"],axis=1)

dist_original = {pop: auc_max.loc[(auc_max['shuffled']==False) & (auc_max['pop']==pop)]["auc"].tolist() for pop in pops}
dist_shuff    = {pop: auc_max.loc[(auc_max['shuffled']==True)  & (auc_max['pop']==pop)]["auc"].tolist() for pop in pops}

# 4. COMPUTE P_VALUES
# Next we'll compute p-values comparing the populations to
# shuffled versions of themselves, and to each other.
comps = [["PN","shuff"],["L","shuff"],["O","shuff"],["L","PN"],["O","L"],["O","PN"]]
pvals = [0 for c in comps]
print("P-values:")
for i in range(len(comps)):
    x = dist_original[comps[i][0]]
    if comps[i][1]=="shuff":
        y = dist_shuff[comps[i][0]]
    else:
        y = dist_original[comps[i][1]]
    pvals[i] = stats.mannwhitneyu(x,y,alternative="greater").pvalue
    print("{:<2} > {:<6}: {:1.3f} ({:1.3e})".format(comps[i][0],comps[i][1],pvals[i],pvals[i]))

# 5. PLOT THE RESULTS
# Finally, we're going to create a box and whisker plot
# summarizing the results
plt.style.use("ggplot")
plt.figure(figsize=(8,5))

centers = np.array([2,4,6])
offset = 0.3
bp_orig = plt.boxplot([dist_original[p] for p in pops], whis=[5,95], positions=centers-offset, notch=True, bootstrap=10000);
for i, pop in enumerate(pops):
    col = aesthetics.pop_colours[pop]
    bp_orig["boxes"][i].set(color=col, linewidth=2)
    bp_orig["medians"][i].set(color=col)
    bp_orig["fliers"][i].set(marker="o",markerfacecolor=col)

bp_shuf = plt.boxplot([dist_shuff[p] for p in pops], whis=[5,95],positions=centers+offset, notch=True, bootstrap=10000);
for i, pop in enumerate(pops):
    col = "lightgray"
    bp_shuf["boxes"][i].set(color=col, linewidth=2)
    bp_shuf["medians"][i].set(color=col)
    bp_shuf["fliers"][i].set(marker="o",markerfacecolor=col)
    
plt.gca().set_xticks(centers);
plt.gca().set_xticklabels(pops)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xlim([1,7])
plt.ylim([0.5,1.1]);
xl = plt.xlim()

# PLOT THE SIGNIFANCE BARS
# The next chunk of code is for plotting the bars
# indicating the significant differences with lines
# and the right number of stars.

centers   = dict(zip(pops, centers))
num_stars = [np.floor(-np.log10(p/5))-1 for p in pvals]
curr_y    = 1.05
dy        = 0.12/2

# The next few variables are for checking to see if there is
# some overlap happening when plotting the significance bars.
occupancy = np.zeros((100,))
min_occupancy_gap = 1
x2i = lambda x: int(np.round((x - xl[0])/np.diff(xl)*100)) # discretizes 'x' positions

for i, comp in enumerate(comps):
    if num_stars[i]>0:
        x1 = centers[comp[0]]-offset
        if comp[1]=="shuff":
            x2 = x1 + 2*offset
        else:
            x2 = centers[comp[1]]-offset
        xs = np.sort([x1,x2])
        x1 = xs[0]
        x2 = xs[1]
        smeared_occupancy = np.array([int(x) for x in np.convolve(occupancy,[1]*(2*min_occupancy_gap+1),'same')>0])        
        smeared_occupancy[range(x2i(x1),x2i(x2))]+=1
        if any(smeared_occupancy>1): # If two signifance bars would be overlapping or within the desired gap
            curr_y += dy   # move the current bar higher
            occupancy *= 0 # reset the occupancy
        else:
            occupancy[range(x2i(x1),x2i(x2))]+=1
            
        plt.plot([x1,x2],[curr_y, curr_y],"k")
        plt.plot([x1,x1],[curr_y, curr_y-dy/3],"k")
        plt.plot([x2,x2],[curr_y, curr_y-dy/3],"k")
        plt.text((x1+x2)/2,curr_y+dy/2,"*"*min(int(num_stars[i]),3),horizontalalignment="center",verticalalignment="top",fontsize=24)
plt.axis("tight")
plt.ylim([0.5,max(plt.ylim())])
plt.plot(np.array([1,1])*min(plt.xlim()),[0,1],"k")
plt.gca().set_yticks(np.arange(0.5,1.05,0.1));
plt.xlabel("Population")
plt.ylabel("Maximum AUC score")
plt.gca().yaxis.set_label_coords(-0.075,0.25/np.diff(plt.ylim()))
plt.tight_layout()
output_file = op.join(output_dir, "fig_auc_all.pdf")
plt.savefig(output_file, bbox_inches="tight")
print("Plotted AUC box plots in {}".format(output_file))
plt.close("all")
print("ALLDONE.")
