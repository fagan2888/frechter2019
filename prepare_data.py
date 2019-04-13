from os import path as op
import pandas
import pickle
import numpy as np
import rpy2.robjects as robjects

data_dir = "./input_data"

print("Loading stats_by_class_mat36.")
robjects.r['load'](op.join(data_dir, "stats_by_class_mat36_python.RData"))

# stats_by_class_mat36 is a matrix where each row
# contains the data for one class. The data itself
# consists of 7 binned mean spike counts for 36 odours.
# The next chunk of code extracts subsets of this matrix
# corresponding to the data from each population (PN, L, O)
# and the subset of odours that we use in the analysis.

# Get the class label for each row
class_per_row  = list(robjects.r["class_per_row"])

# groups: A 373 element vector indicating whether each cell is PN, L, or O.
group_per_cell = list(robjects.r["group_per_cell"])
pops           = set(group_per_cell) # PN, O, L

# classes: Same as groups, but indicating the class of the cell.
class_per_cell = list(robjects.r["class_per_cell"])

# Create a list of all the classes for each population
classes_per_pop  = {pop:[cl for i, cl in enumerate(class_per_cell) if group_per_cell[i] == pop]   for pop in pops}

# Determine the rows for each population 
rows_per_pop     = {pop:[ i for i, cl in enumerate(class_per_row) if cl in classes_per_pop[pop]] for pop in pops}
rownames_per_pop = {pop:[cl for i, cl in enumerate(class_per_row) if cl in classes_per_pop[pop]] for pop in pops}

# Get the list of odour names and groups and use only a prescribed subset.
odour_names  = list(robjects.r["odour_names"])
odour_groups = list(robjects.r["odour_groups"])
bad_groups   = ['Mix','Blank','min_acid']
# Get the indices of the valid odours. We'll need this to subset the data later
good_odour_inds = [i for i, g in enumerate(odour_groups) if g not in bad_groups]
# Keep all odours whose group isn't one of the bad groups
odour_names  =  [odour_names[i]  for i in good_odour_inds]
# Keep all the groups which are not one of the bad groups
odour_groups =  [odour_groups[i] for i in good_odour_inds]
odours = dict(zip(odour_names, odour_groups))

# Finally, extract the stats data...
stats_by_class = np.array(robjects.r["stats_by_class_mat36"])
# ... subset the rows for each population...
stats_by_class_per_pop = {pop:stats_by_class[rows_per_pop[pop], :] for pop in pops}
# ... and subset the columns to use the odours we're interested in.
stats_by_class_per_pop = {pop:X[:, [i for i in range(X.shape[1]) if i//7 in good_odour_inds]] for pop, X in stats_by_class_per_pop.items()}

# Save the data to file
data = {"odours":odours,
        "rownames_per_pop":rownames_per_pop,
        "stats_by_class_per_pop":stats_by_class_per_pop}
output_file = op.join(data_dir, "stats_by_class_subset.p")
print("Saving subseted stats_by_class data to {}".format(output_file))
pickle.dump(data, open(output_file, "wb"))
    

print("Loading cell class information.")
# groups: A 373 element vector indicating whether each cell is PN, L, or O.
group_per_cell = list(robjects.r["group_per_cell"])
# classes: Same as groups, but indicating the class of the cell.
class_per_cell = list(robjects.r["class_per_cell"])

classes_per_pop = {pop:list({cl for i, cl in enumerate(class_per_cell) if group_per_cell[i] == pop}) for pop in ["PN","O", "L"]}
for pop,cl in classes_per_pop.items():
    print("{:>4}: {:2d} classes: {}".format(pop,len(cl), "; ".join(cl)))

# Write the classes information to disk
output_file = op.join(data_dir, "classes_per_pop.p")
print("Writing classes per population to {}".format(output_file))
pickle.dump(classes_per_pop, open(output_file,  "wb"))

# Load the table of cells we're going to use
print("Loading the physplit table.")
db = pandas.read_csv(op.join(data_dir, "physplit.csv"))

print("Loading spike counts data.")
data_file  = op.join(data_dir, "spike_counts_per_trial.csv")
count_data = pandas.read_csv(data_file)
# The count data is for more cells and odours than we need.
# So subset the data for the odours we need...
count_data = count_data.loc[count_data['odour'].isin(odour_names)]

# ...and for the cells we need.
count_data = count_data.loc[count_data["cell"].isin(set(db["cell"].values))] 
cells = set(count_data["cell"])

print("Filtering out cells with incorrectly shaped data.")
required_shape  = (6120, 3) # 6120 = 30 odors * 51 bins * 4 trials
cell_validity   = {cell:count_data[count_data["cell"] == cell][["trial","bin","odour"]].values.shape == required_shape for cell in cells}
valid_cells   = {cell for cell,valid in cell_validity.items() if valid}
invalid_cells = cells - valid_cells
print("{} valid   cells.".format(len(valid_cells)))
print("{} invalid cell(s): {}".format(len(invalid_cells), invalid_cells))

print("Subsetting to valid cells.")
count_data_valid = count_data.loc[count_data["cell"].isin(valid_cells)]
class_information_present = all([len(x)>0 for x in db.loc[db["cell"].isin(valid_cells)]["class"].values])
if not class_information_present:
    raise Exception("Class information was not present for all cells.")

print("Reshaping spike counts data into array.")
# Sort the count table by the columns so that we can pull out the cell and odour names in the correct order,
# and so that the trials and bins are in the correct order.
counts      = count_data_valid.sort_values(by=["cell","odour", "trial", "bin"])["count"].values
btoc        = np.reshape(counts, (51, 4, len(odour_names), len(valid_cells)), order="F") # bins, trial, odors, cells
btoc_cells  = sorted(list(set(count_data_valid["cell"])))  
btoc_odours = sorted(list(set(count_data_valid["odour"])))
btoc_groups = [odour_groups[odour_names.index(odour)] for odour in btoc_odours]

# Write the btoc data to disk
output_file = op.join(data_dir,"btoc")
print("Writing spike counts data to {}.npy".format(output_file))
np.save(output_file, btoc)

output_file = op.join(data_dir, "btoc.p")
print("Writing spike count labels data to {}".format(output_file))
pickle.dump({"cells":btoc_cells, "groups":btoc_groups, "odours":btoc_odours}, open(output_file, "wb"))

# Write the list of valid cells to disk
output_file = op.join(data_dir, "db.csv")
print("Writing the database of valid cells to {}".format(output_file))
db_valid = db.loc[db["cell"].isin(valid_cells)]
db_valid.to_csv(output_file, index=False)

print("ALLDONE.")
