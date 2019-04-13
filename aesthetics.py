import matplotlib as mpl

SMALL_SIZE  = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
PANEL_LABEL_SIZE = 20

mpl.rc('font',   size=SMALL_SIZE)          # controls default text sizes
mpl.rc('axes',   titlesize=SMALL_SIZE)     # fontsize of the axes title
mpl.rc('axes',   labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
mpl.rc('xtick',  labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('ytick',  labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

pop_colours       = {"PN":"red",       "L":"limegreen", "O":"dodgerblue"}
pop_light_colours = {"PN":"mistyrose", "L":"honeydew",  "O":"lightcyan"}

