library(gphys)
library(physplitdata)
library(physplit.analysis)

output.folder = "./input_data"

## 1. Compute per-class statistics for each population and write to file.
summary_array=create_raw_summary_array()
clean_summary_array=createSummarySpikesArray(summary_array, NALimit = 25) # 440 x 36 x 7

valid.cells = c(
"nm20120216c0", "nm20120131c0", "nm20120321c0", "nm20120531c0",
"nm20120605c0", "nm20120613c0", "nm20120614c0", "nm20120614c1",
"nm20120618c0", "nm20120619c1", "nm20120626c1", "nm20120628c1",
"nm20120703c2", "nm20120704c1", "nm20120708c1", "nm20120710c0",
"nm20120712c1", "nm20120714c0", "nm20120714c4", "nm20120203c0",
"nm20120726c3", "nm20120730c0", "nm20120806c0", "nm20120811c1",
"nm20120813c0", "nm20120813c1", "nm20120814c3", "nm20120815c1",
"nm20120816c1", "nm20120828c0", "nm20120828c1", "nm20120830c0",
"nm20120830c1", "nm20120830c2", "nm20120906c0", "nm20120913c0",
"nm20120913c1", "nm20120914c0", "nm20120914c2", "nm20120921c0",
"nm20120922c1", "nm20120925c1", "nm20120926c0", "nm20121013c0",
"nm20121015c0", "nm20121016c0", "nm20121017c0", "nm20121018c0",
"nm20121024c0", "nm20121029c1", "nm20121101c0", "nm20121105c0",
"nm20121114c0", "nm20121115c1", "nm20121122c1", "nm20121127c0",
"nm20121127c1", "nm20121128c2", "nm20121128c3", "nm20121129c0",
"nm20121129c1", "nm20121130c0", "nm20121130c1", "nm20121204c2",
"nm20121206c0", "nm20121207c2", "nm20121212c0", "nm20121213c0",
"nm20121215c0", "nm20121217c0", "nm20121221c0", "nm20121225c0",
"nm20121225c2", "nm20121227c2", "nm20121227c3", "nm20121030c0",
"nm20121112c0", "nm20121229c0", "nm20130109c0", "nm20130110c1",
"nm20130118c0", "nm20130122c0", "nm20130123c1", "nm20130206c1",
"nm20130207c1", "nm20130208c1", "nm20130211c3", "nm20130213c0",
"nm20130214c1", "nm20130219c0", "nm20130221c0", "nm20130313c0",
"nm20130408c0", "nm20130313c1", "nm20130403c0", "nm20130411c2",
"nm20130416c0", "nm20130423c1", "nm20130424c0", "nm20130424c1",
"nm20130425c1", "nm20130430c2", "nm20130501c0", "nm20130501c2",
"nm20130516c0", "nm20130523c0", "nm20130524c0", "nm20130605c1",
"nm20130606c1", "nm20130617c1", "nm20130612c0", "nm20130612c1",
"nm20130617c3", "nm20130618c0", "nm20130619c0", "nm20130619c1",
"nm20130620c0", "nm20130620c1", "nm20130611c0", "nm20130624c1",
"nm20130625c0", "nm20130625c2", "nm20130702c0", "nm20130702c1",
"nm20130703c0", "nm20130703c1", "nm20130704c1", "nm20130704c2",
"nm20130709c0", "nm20130710c0", "nm20130710c3", "nm20130711c0",
"nm20130711c1", "nm20130711c3", "nm20130712c0", "nm20130712c1",
"nm20130716c2", "nm20130716c3", "nm20130717c0", "nm20130717c1",
"nm20130717c2", "nm20130724c0", "nm20130724c2", "nm20130726c0",
"nm20130726c1", "nm20130729c0", "nm20130731c1", "nm20130801c1",
"nm20130801c2", "nm20130807c0", "nm20130807c1", "nm20130808c0",
"nm20130808c1", "nm20130808c2", "nm20130809c0", "nm20130809c2",
"nm20130813c0", "nm20130814c0", "nm20130814c1", "nm20130815c1",
"nm20130816c0", "nm20130816c2", "nm20130819c0", "nm20130819c1",
"nm20130819c2", "nm20130820c0", "nm20130820c2", "nm20130820c3",
"nm20130821c0", "nm20130821c1", "nm20130821c2", "nm20130822c0",
"nm20130822c1", "nm20130826c0", "nm20130823c0", "nm20130827c0",
"nm20130827c1", "nm20130829c0", "nm20130829c1", "nm20130910c0",
"nm20130911c0", "nm20130911c2", "nm20130911c3", "nm20130912c0",
"nm20130912c1", "nm20130913c0", "nm20130913c1", "nm20130924c0",
"nm20130924c1", "nm20130925c0", "nm20130925c1", "nm20130925c3",
"nm20130926c0", "nm20130926c1", "nm20130927c0", "nm20130927c1",
"nm20131001c0", "nm20131001c1", "nm20131001c2", "nm20131002c0",
"nm20131002c2", "nm20131003c0", "nm20131003c1", "nm20131004c0",
"nm20131007c0", "nm20131007c1", "nm20131008c0", "nm20131008c1",
"nm20131009c0", "nm20131009c1", "nm20131009c4", "nm20131009c5",
"nm20131010c2", "nm20131011c0", "nm20131011c1", "nm20131014c0",
"nm20131014c1", "nm20131014c2", "nm20131014c3", "nm20131016c1",
"nm20131016c2", "nm20131018c0", "nm20131018c1", "nm20131018c2",
"nm20131022c0", "nm20131024c0", "nm20131024c1", "nm20131028c0",
"nm20131028c2", "nm20131030c0", "nm20131030c1", "nm20131031c1",
"nm20131106c1", "nm20131107c2", "nm20131108c0", "nm20131111c0",
"nm20131111c1", "nm20131112c1", "nm20131113c0", "nm20131113c1",
"nm20131113c2", "nm20131118c3", "nm20131118c1", "nm20131120c1",
"nm20131122c0", "nm20131122c1", "nm20131125c0", "nm20131126c1",
"nm20131126c3", "nm20131127c1", "nm20131204c0", "nm20131204c1",
"nm20131212c0", "nm20131212c1", "nm20140117c1", "nm20140117c2",
"nm20140122c0", "nm20140205c0", "nm20140205c2", "nm20140206c2",
"nm20140207c0", "nm20140210c2", "nm20140210c3", "nm20140211c0",
"nm20140211c1", "nm20140212c0", "nm20140212c1", "nm20140213c1",
"nm20140218c0", "nm20140327c1", "nm20140328c1", "nm20140401c0",
"nm20140402c0", "nm20140402c1", "nm20140403c0", "nm20140403c1",
"nm20140508c0", "nm20140514c1", "nm20140514c2", "nm20140515c0",
"nm20140515c1", "nm20140516c0", "nm20140528c0", "nm20140529c0",
"nm20140530c0", "nm20140612c0", "nm20140613c0", "nm20140616c0",
"nm20140618c0", "nm20140710c1", "nm20140714c1", "nm20140716c2",
"nm20140717c0", "nm20140717c1", "nm20140721c0", "nm20140721c1",
"nm20140721c3", "nm20140723c1", "nm20140723c2", "nm20140723c3",
"nm20140730c2", "nm20140821c2", "nm20140822c1", "nm20140822c4",
"nm20140828c1", "nm20140829c1", "nm20140901c0", "nm20140903c2",
"nm20140908c0", "nm20140909c0", "nm20140911c0", "nm20140911c1",
"nm20140912c1", "nm20141006c2", "nm20141008c3", "nm20141013c3",
"nm20141015c0", "nm20141016c0", "nm20141016c2", "nm20141022c0",
"nm20141022c1", "nm20141024c1", "nm20141027c0", "nm20141029c0",
"nm20141031c0", "nm20141031c1", "nm20141105c0", "nm20141105c1",
"nm20141107c0", "nm20141107c1", "nm20141110c0", "nm20141112c0",
"nm20141113c0", "nm20141117c2", "nm20141119c0", "nm20141126c1",
"nm20141126c2", "nm20141204c1", "nm20141208c2", "nm20141210c1",
"nm20141211c1", "nm20150115c2", "nm20150119c0", "nm20150126c0",
"nm20150204c0", "nm20150205c0", "nm20150302c0", "nm20150302c1",
"nm20151229c2", "nm20151231c1", "nm20160119c0", "nm20160120c0",
"nm20160121c0", "nm20160121c1", "nm20160125c0", "nm20160125c1",
"nm20160128c0", "nm20160129c0", "nm20160129c1", "nm20160201c0",
"nm20160201c1", "nm20160204c1", "nm20160208c0", "nm20160211c0",
"nm20160211c1", "nm20160215c0", "nm20160217c0", "nm20160217c1",
"nm20160218c0", "nm20160304c1", "nm20160304c3", "nm20160307c0",
"nm20160310c1")

# Create a subset of PhySplitDB containing only the cells above
physplit = subset(PhySplitDB, cell %in% valid.cells)

# Add a new column indicating the "correct" class label based on the cell type
physplit$new_class = with(physplit, ifelse(Group=="PN", Anatomy.type, FinalClass)) 

# Find all the cells of a given class by using their Igor file name,
classnames=unique(physplit$new_class)
cellsofclass=lapply(classnames, function(x) basename(physplit[physplit$new_class==x,"Igor.file"]))

# Make cellsofclass accessible by names as well.
# So e.g. cellsofclass[[73]] and cellsofclass$DC3 work
names(cellsofclass)=classnames 

statnames=c("baseline", "max1", "max2","max3","max4","max5","max6")
# This next bit is complicated.
# We have stats for each cell and each odour.
# We want to compute averages over the cells in each class.
# 1. lapply is being used, so the output is going to be  alist, one for each element of cellsofclass
# 2. The variable 'cellclass' contains the cells in the current class.
# 3. clean_summary_array[...] selects just the subset for those cells.
# 4. The outer apply runs along margin 2, which is the odours.
# 5. We thus will iterate over this dimension, slicing the clean_summary_array for all cells and that odour in turn.
# 6. This slice (which is two dimensional: # cells in class x 7) is passed to the inner apply.
# 7. This then iterates over the outer dimension (the stats), and computes the average over the first dimension (cells)
# 8. This yields a single value for each stat.
# 9. Bubbling out to the outer apply, we have 7 such summaries for each of the 36 odours.
# 10. Hence the output will be a list of such 36 x 7 element summaries.
statsbyclass = lapply(cellsofclass, function(cellclass)
  apply(clean_summary_array[rownames(clean_summary_array) %in% cellclass, , , drop = FALSE], 2, function(x)
    apply(x, 2, mean, na.rm = TRUE)))

# The next chunk of code restructures the data from a list into a matrix of the desired dimensions
require(abind)
t.stats_by_class_array = do.call(abind, c(statsbyclass, list(along=3))) # # Bind the list into a 3-dimensional array of size 7 x 36 x 73
stats_by_class_array = aperm(t.stats_by_class_array,c(3,2,1)) # Rearrange dims to 73 x 36 x 7
stats_by_class_array36=stats_by_class_array[,colnames(clean_summary_array),] #  Reorder the columns to match the order in clean_summary_array
stats_by_class_mat36=createSummarySpikesMat(stats_by_class_array36, NALimit=0) # Convert to a 2D matrix of size 73 x 252
attr(stats_by_class_mat36,'df')=physplit

odour_groups <-
  c(
    OilBl = "Blank",
    E2Hex = "aldehyde",
    GerAc = "ester",
    Prpyl = "ester",
    IPenA = "ester",
    Et3HB = "ester",
    Nonnl = "aldehyde",
    CiVAc = "ester",
    MetSl = "phenyl",
    HexAc = "ester",
    PeEtA = "phenyl",
    AceAc = "ester",
    EtHex = "ester",
    `2PnAc` = "phenyl",
    `5OdMx` = "Mix",
    BeZal = "aldehyde",
    bCitr = "alcohol",
    `1HxOl` = "alcohol",
    Frnsl = "alcohol",
    WatBl = "Blank",
    Cdvrn = "amine",
    Sprmn = "amine",
    Acoin = "alcohol",
    MtAct = "ester",
    AcAcd = "carboxyl",
    PrpnA = "carboxyl",
    BtrAc = "carboxyl",
    Amnia = "amine",
    Pyrdn = "amine",
    PAcHd = "phenyl",
    HCL36 = "min_acid",
    PAcAc = "phenyl",
    Vingr = "Mix",
    Geosn = "alcohol",
    VinGe = "Mix",
    PEtAm = "phenyl"
  )
odour_names    <- names(odour_groups)
all_classes    <- rownames(stats_by_class_mat36)
group_per_cell <- physplit$Group
class_per_cell <- physplit$new_class

output.file = paste(output.folder, "stats_by_class_mat36_python.RData", sep="/")
print(paste("Saving stats_by_class_mat36 and associated properties to", output.file))
save(all_classes, group_per_cell, class_per_cell, odour_names, odour_groups, stats_by_class_mat36, file=output.file)
print("Done.")

## 2. Write the physplit table to file
output.file = paste(output.folder, "physplit.csv", sep="/")
print(paste("Writing physplit data to", output.file))
out <- file(output.file, "w")
line  <- c("cell,class,group")
write(line, out, append=TRUE)
cell  <- physplit[["cell"]]
class <- physplit[["new_class"]]
group <- physplit[["Group"]]

for (i in 1:length(cell)){
  line  <- sprintf("%s,%s,%s",cell[[i]],class[[i]],group[[i]])
  write(line, out, append=TRUE)
  print(line)
}
close(out)
print("Done.")

## 3. Write the binned spike data to file
data <- physplitdata::smSpikes
cell_names <- attr(data, "names")
output.file = paste(output.folder, "spike_counts_per_trial.csv", sep="/") 
print(paste("Writing the binned spike data for all cells to", output.file))
out <- file(output.file, "w")
line = c("cell,odour,trial,bin,count")
write(line, out, append=TRUE)
for (cell in cell_names){
  print(cell)
  odours <- attr(data[[cell]], "names")
  for (odour in odours){
    counts <- data[[cell]][[odour]]$counts
    d <- dim(counts)
    for (trial in 1:d[[1]]){
      for (bin in 1:d[[2]]){
        line  <- sprintf("%s,%s,%d,%d,%d",cell,odour,trial,bin,counts[[trial,bin]])
        write(line, out, append=TRUE)
      }
    }
  }
}
close(out)

print("ALLDONE.")

