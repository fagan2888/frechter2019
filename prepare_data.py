from os import path as op
import pandas
import pickle
import numpy as np
import rpy2.robjects as robjects

data_dir = "./input_data"

print("Loading odour names and groups.")
robjects.r['load'](op.join(data_dir, "stats_by_class_mat36_python.RData"))
# classes is the set of classes used in stats_by_class_mat36 (the row names)
classes = list(robjects.r["classes"])
bucket_classes = ["30","62", "67", "80", "103", "105", "42", "66", "73"] 
# We're using a prespecified list of classes that already takes the bucket classes
# into account. So make sure no bucket classes show up here.
if any([r in bucket_classes for r in classes]):
    raise Exception("Some bucket classes were found!")
print("Verified that no bucket classes were found.")

# groups: A 373 element vector indicating whether each cell is PN, L, or O.
group_per_cell = list(robjects.r["group_per_cell"])
# classes: Same as groups, but indicating the class of the cell.
class_per_cell = list(robjects.r["class_per_cell"])

pn_classes = [cl for i, cl in enumerate(class_per_cell) if group_per_cell[i] == "PN"]
on_classes = [cl for i, cl in enumerate(class_per_cell) if group_per_cell[i] == "O"]
ln_classes = [cl for i, cl in enumerate(class_per_cell) if group_per_cell[i] == "L"]


ind_pn = [i for i, cl in enumerate(classes) if cl in pn_classes]
ind_on = [i for i, cl in enumerate(classes) if cl in on_classes]
ind_ln = [i for i, cl in enumerate(classes) if cl in ln_classes]

names_pn = [classes[i] for i in ind_pn]
names_on = [classes[i] for i in ind_on]
names_ln = [classes[i] for i in ind_ln]

stats_by_class = np.array(robjects.r["stats_by_class_mat36"])

Xpn = stats_by_class[ind_pn,:]
Xon = stats_by_class[ind_on,:]
Xln = stats_by_class[ind_ln,:]

odour_names  = list(robjects.r["odour_names"])
odour_groups = list(robjects.r["odour_groups"])
bad_groups   = ['Mix','Blank','min_acid']
good_odour_inds = [i for i, g in enumerate(odour_groups) if g not in bad_groups]
# Keep all odours whose group isn't one of the bad groups
odour_names  =  [odour_names[i]  for i in good_odour_inds]
# Keep all the groups which are not one of the bad groups
odour_groups =  [odour_groups[i] for i in good_odour_inds]
odours = dict(zip(odour_names, odour_groups))

Xpn = Xpn[:,[i for i in range(Xpn.shape[1]) if i//7 in good_odour_inds]]
Xon = Xon[:,[i for i in range(Xon.shape[1]) if i//7 in good_odour_inds]]
Xln = Xln[:,[i for i in range(Xln.shape[1]) if i//7 in good_odour_inds]]

pops                    = ["PN", "LN", "ON"]
class_names_per_pop     = {"PN":names_pn, "LN":names_ln,"ON":names_on}
stats_by_class_per_pop  = {"PN":Xpn, "ON":Xon, "LN":Xln}

print("Loading cell class information.")
# groups: A 373 element vector indicating whether each cell is PN, L, or O.
group_per_cell = list(robjects.r["group_per_cell"])
# classes: Same as groups, but indicating the class of the cell.
class_per_cell = list(robjects.r["class_per_cell"])

classes_per_pop = {pop:list({cl for i, cl in enumerate(class_per_cell) if group_per_cell[i] == pop}) for pop in ["PN","O", "L"]}
for pop,cl in classes_per_pop.items():
    print("{:>4}: {:2d} classes: {}".format(pop,len(cl), "; ".join(cl)))

# Write the classes information to disk
print("Writing classes per population to disk.")
pickle.dump(classes_per_pop,  open(op.join(data_dir, "classes_per_pop.p"),  "wb"))

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
