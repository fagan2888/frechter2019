from os import path as op
import pandas
import pickle
import numpy as np
import rpy2.robjects as robjects

data_dir = "./input_data"

print("Loading odour names and groups.")
robjects.r['load'](op.join(data_dir, "stats_by_class_mat36_python.RData"))
odour_names  = list(robjects.r["odour_names"])
odour_groups = list(robjects.r["odour_groups"])
bad_groups   = ['Mix','Blank','min_acid']
# Keep all odours whose group isn't one of the bad groups
odour_names  = [name  for i, name  in enumerate(odour_names) if odour_groups[i] not in bad_groups]
# Keep all the groups which are not one of the bad groups
odour_groups = [group for group in odour_groups if group not in bad_groups]

print("Loading cell class information.")
# groups: A 373 element vector indicating whether each cell is PN, L, or O.
group_per_cell = list(robjects.r["group_per_cell"])
# classes: Same as groups, but indicating the class of the cell.
class_per_cell = list(robjects.r["class_per_cell"])

classes = {pop:list({cl for i, cl in enumerate(class_per_cell) if group_per_cell[i] == pop}) for pop in ["PN","O", "L"]}
for pop,cl in classes.items():
    print("{:>4}: {:2d} classes: {}".format(pop,len(cl), "; ".join(cl)))

# Write the classes information to disk
print("Writing classes per population to disk.")
pickle.dump(classes,  open(op.join(data_dir, "classes.p"),  "wb"))

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
print("Writing spike counts data to disk.")
np.save(op.join(data_dir,"btoc"), btoc)
pickle.dump(btoc_cells,  open(op.join(data_dir, "btoc_cells.p"),  "wb"))
pickle.dump(btoc_groups, open(op.join(data_dir, "btoc_groups.p"), "wb"))
pickle.dump(btoc_odours, open(op.join(data_dir, "btoc_odours.p"), "wb"))

# Write the list of valid cells to disk
print("Writing the database of valid cells to disk.")
db_valid = db.loc[db["cell"].isin(valid_cells)]
db_valid.to_csv(op.join(data_dir, "db.csv"), index=False)

print("ALLDONE.")
