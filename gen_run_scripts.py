# The template contents of the slurm script for each run.
# The fields in all-caps (except SLURM) will be filled
# in for each job by the loop below.
template = """#!/bin/bash
# Simple SLURM sbatch example
#SBATCH --job-name=JOBNAME
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=compute

ml purge > /dev/null 2>&1
ml gnuplot
python -u ../single.py SEED POP NCLASS TASK POOLBASELINE WNDSIZE SHUFFLE WHICH_BINS
"""
import os.path as op
import json, pickle
from functools import reduce

print("Loading input data.")
input_dir       = op.join(op.dirname(op.abspath(__file__)), "input_data")
classes_per_pop = pickle.load(open(op.join(input_dir, "classes_per_pop.p"), "rb"))
pop_sizes       = {pop:len(classes) for pop, classes in classes_per_pop.items()}
print("Number of classes for each population: {}".format(pop_sizes))

print("Loading configuration file.")
with open("config.json", "rb") as in_file:
    config = json.load(in_file)

# The following are (validator, message) pairs.
# Each field in the configuration file must satisfy
# its own subset of these conditions. If it does not,
# an exception with the corresponding message is raised.
positive    = (lambda x: x > 0, "Must be positive.") # The test, and what to say if failed.
unique      = (lambda x: len(x) == len(set(x)), "Values must be unique.")
allin       =  lambda valid: (lambda x: all([iv in valid for iv in x]), "All values must be in {}.".format(valid))
allpass     = (lambda x: True, "All pass.")
allpositive = (lambda x: all([xi > 0 for xi in x]), "All must be positive.")

# The validators that are applied to each field.
validators = {
    "n_runs": [positive],
    "indiv":  [allpositive],
    "fracs":  [(lambda x: all([f>0 and f<=1 for f in x]), "All values must be in (0,1].")],
    "tasks":  [allin(["identity", "category"]), unique],
    "pops":   [allin(["O","L","PN"]), unique],
    "pool_baseline_vals": [allin([0,1]), unique],
    "wnd_size_vals":      [allpositive, unique],
    "shuffle_vals":       [allin([True,False]), unique],
    "which_bins":         [allpass]
}

# Now apply the validators of each field.
print("Checking configuration file.")
for field, value in config.items():
    print("{:>24}: {}".format(field, value))
    for vfun, errmsg in validators[field]:
        if not vfun(value):
            raise ValueError("Config field {} with values {} invalid: {}".format(field, value, errmsg)) 
    print("{:>24}".format("OK"))

# The configuration file was valid, so generate the jobs scripts.
print("Generating job scripts.")
seed = 0
for shuffle in config["shuffle_vals"]:
    for pool_baseline in config["pool_baseline_vals"]: 
        for wnd_size in config["wnd_size_vals"]: 
            for irun in range(config["n_runs"]):
                for d in config["tasks"]:
                    for p in config["pops"]:
                        counts = sorted(list(set(config["indiv"] + [int(f*pop_sizes[p]) for f in config["fracs"]])))
                        for c in counts:
                            job_file = "job{}.sh".format(seed)
                            job_name = "{}{}{}{}{}{}{}".format(
                                p[0], # population
                                c,    # n_classes
                                d[0], # task
                                "p" if pool_baseline else "",
                                wnd_size, 
                                "s" if shuffle else "o",
                                irun)
                            with open(job_file, "w") as f_out:
                                reps = [("JOBNAME", job_name),
                                        ("SEED", str(seed)),
                                        ("POP", p),
                                        ("NCLASS", str(c)),
                                        ("TASK", d),
                                        ("POOLBASELINE", "1" if pool_baseline else "0"),
                                        ("WNDSIZE", str(wnd_size)),
                                        ("SHUFFLE", "1" if shuffle else "0"),
                                        ("WHICH_BINS", '--bins "{}"'.format(" ".join([str(b) for b in config["which_bins"]])) if config["which_bins"] else "")]
                                content = reduce(lambda a, b: a.replace(b[0], b[1]), reps, template)
                                print(content)
                                f_out.write(content)
                                print("Wrote {}".format(job_file))
                            seed += 1
                            
print("ALLDONE.")
