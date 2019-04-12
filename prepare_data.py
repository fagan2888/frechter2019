from os import path as op
import pandas
import pickle
import numpy as np
import rpy2.robjects as robjects

data_dir = "./input_data"

# LOAD DATA ABOUT ODOURS AND GROUPS
print("Loading odour names and groups.")
robjects.r['load'](op.join(data_dir, "stats_by_class_mat36_python.RData"))
odour_names  = list(robjects.r["odour_names"])
odour_groups = list(robjects.r["odour_groups"])
bad_groups      = ['Mix','Blank','min_acid']
good_odour_inds = [i for i, g in enumerate(odour_groups) if g not in bad_groups]
odour_names  = [odour_names[i]  for i in good_odour_inds]
odour_groups = [odour_groups[i] for i in good_odour_inds]

## LOAD THE SPIKE COUNTS DATA
print("Loading spike counts data.")
db = pandas.read_csv(op.join(data_dir, "physplit.csv"))
data_file = op.join(data_dir, "spike_counts_per_trial.csv")
count_data = pandas.read_csv(data_file)
# The count data is for more cells and odours than we need.
# So subset the data for the odours we need...
count_data = count_data.loc[count_data['odour'].isin(odour_names)]
# ...and for the cells we need.
count_data = count_data.loc[count_data["cell"].isin(set(db["cell"].values))] 
cells = set(count_data["cell"])

## FILTER OUT THE CELLS WITHOUT THE RIGHT AMOUNT OF DATA
print("Filtering out cells with incorrectly shaped data.")
required_shape = (6120, 3) # 6120 = 30 odors * 51 bins * 4 trials
cell_validity   = {cell:count_data[count_data["cell"] == cell][["trial","bin","odour"]].values.shape == required_shape for cell in cells}
valid_cells   = {cell for cell,valid in cell_validity.items() if valid}
invalid_cells = cells - valid_cells
print("{} valid cells".format(len(valid_cells)))
print("{} invalid cells: {}".format(len(invalid_cells), invalid_cells))

## SUBSET TO THE VALID CELLS
print("Subsetting to valid cells.")
count_data_valid = count_data.loc[count_data["cell"].isin(valid_cells)]
class_information_present = all([len(x)>0 for x in db.loc[db["cell"].isin(valid_cells)]["class"].values])
if not class_information_present:
    raise Exception("Class information was not present for all cells.")

## RESHAPE THE BINNED SPIKETIMES DATA INTO A BINS x TRIALS x ODOURS x CELLS (btoc) array
print("Reshaping spike counts data into array.")
counts     = count_data_valid.sort_values(by=["cell","odour", "trial", "bin"])["count"].values
btoc       = np.reshape(counts, (51, 4, len(odour_names), len(valid_cells)), order="F") # bins, trial, odors, cells
btoc_cells = sorted(list(set(count_data_valid["cell"])))
btoc_odours= sorted(list(set(count_data_valid["odour"])))
btoc_groups= [odour_groups[odour_names.index(n)] for n in btoc_odours]

# Write the btoc data to disk
print("Writing spike counts data to disk.")
np.save(op.join(data_dir,"btoc"), btoc)
pickle.dump(btoc_cells,  open(op.join(data_dir, "btoc_cells.p"),  "wb"))
pickle.dump(btoc_groups, open(op.join(data_dir, "btoc_groups.p"), "wb"))
pickle.dump(btoc_odours, open(op.join(data_dir, "btoc_odours.p"), "wb"))

# Write the list of valid cells to disk
print("Writing the database of valid cells to disk.")
db_valid = db.loc[db["cell"].isin(valid_cells)]
db_valid.to_csv(op.join(data_dir, "db.csv"), index=False)

print("ALLDONE")
