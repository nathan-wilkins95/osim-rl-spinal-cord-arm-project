import pickle
import matplotlib.pyplot as plt
import os

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

file_pickle = '/home/reluctanthero/Code/osim-rl(1)/examples/figures/SC_figures/d_combined_states.pkl'
with open(file_pickle, 'rb') as fp:
    # d_combined_states
    d = pickle.load(fp)

print(d.keys())

x = list(range(len(d[list(d.keys())[0]])))
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(18, 12))
ax[0].plot(x, d['TRIlat_activation'], label="TRI lat")
ax[0].plot(x, d['TRIlong_activation'], label="TRI long")
ax[0].plot(x, d['TRImed_activation'], label="TRI med")
ax[0].set_ylabel("N")
ax[0].grid()
ax[0].legend()
ax[1].plot(x, d['TRIlat_fiber_length'], label="TRI lat")
ax[1].plot(x, d['TRIlong_fiber_length'], label="TRI long")
ax[1].plot(x, d['TRImed_fiber_length'], label="TRI med")
ax[1].set_ylabel("m")
ax[1].grid()
ax[1].legend()
ax[2].plot(x, d['TRIlat_fiber_velocity'], label="TRI lat")
ax[2].plot(x, d['TRIlong_fiber_velocity'], label="TRI long")
ax[2].plot(x, d['TRImed_fiber_velocity'], label="TRI med")
ax[2].set_ylabel("m/s")
ax[2].set_xlabel("time[-]")
ax[2].grid()
ax[2].legend()

if save_plots:
    file_name_fig = f'Extensor_plots.png'
    file_fig = path_fig + '/' + file_name_fig
    plt.savefig(file_fig, bbox_inches='tight')
    plt.close()
else:
    plt.show()
