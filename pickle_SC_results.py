import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

fs = 14
plt.rc('font', size=fs+2) #controls default text size
plt.rc('axes', titlesize=fs+3) #fontsize of the title
plt.rc('axes', labelsize=fs+3) #fontsize of the x and y labels
plt.rc('xtick', labelsize=fs) #fontsize of the x tick labels
plt.rc('ytick', labelsize=fs) #fontsize of the y tick labels
plt.rc('legend', fontsize=fs+2) #fontsize of the legend

save_plots = True
path_fig_base = '/home/reluctanthero/Code/osim-rl(1)/examples/figures/SC_figures'

folder_date = f'custom_muscle_subplots'
path_fig = path_fig_base + "/" + folder_date

# Make directory if path fig does not exist
if not os.path.isdir(path_fig):
    os.mkdir(path_fig)

file_pickle = '/home/reluctanthero/Code/osim-rl(1)/examples/figures/SC_figures/list_results.pkl'
with open(file_pickle, 'rb') as fp:
    # d_combined_states
    r = pickle.load(fp)

#print(r)
#print(r.keys())
r_Ia = np.array(r['r_Ia'])
r_mn = np.array(r['r_mn'])
#print(r_Ia)


x = list(range(r_Ia.shape[0]))
fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.plot(x, r_Ia)
ax.set_ylabel("y")
ax.grid()
ax.legend(["a","a","a","a","a","a"])
ax.set_xlabel("time[-]")
ax.set_title("r_Ia")

if save_plots:
    file_name_fig = f'r_Ia_plots.png'
    file_fig = path_fig + '/' + file_name_fig
    plt.savefig(file_fig, bbox_inches='tight')
    plt.close()
else:
    plt.show()

x = list(range(r_mn.shape[0]))
fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.plot(x, r_mn)
ax.set_ylabel("y")
ax.grid()
ax.legend(["a", "a", "a", "a", "a", "a"])
ax.set_xlabel("time[-]")
ax.set_title('r_mn')

if save_plots:
    file_name_fig = f'r_mn_plots.png'
    file_fig = path_fig + '/' + file_name_fig
    plt.savefig(file_fig, bbox_inches='tight')
    plt.close()
else:
    plt.show()