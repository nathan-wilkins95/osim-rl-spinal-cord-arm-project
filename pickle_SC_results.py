import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

fs = 14
plt.rc('font',   size=fs+2)
plt.rc('axes',   titlesize=fs+3)
plt.rc('axes',   labelsize=fs+3)
plt.rc('xtick',  labelsize=fs)
plt.rc('ytick',  labelsize=fs)
plt.rc('legend', fontsize=fs+2)

# FIX: relative paths so script runs on any machine
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
path_fig_base  = os.path.join(BASE_DIR, 'figures', 'SC_figures')
path_fig       = os.path.join(path_fig_base, 'custom_muscle_subplots')
os.makedirs(path_fig, exist_ok=True)

file_pickle = os.path.join(path_fig_base, 'list_results.pkl')
with open(file_pickle, 'rb') as fp:
    r = pickle.load(fp)

r_Ia = np.array(r['r_Ia'])
r_mn = np.array(r['r_mn'])

# FIX: replaced placeholder ["a","a",...] legend labels with real muscle names
muscle_names = ['BIClong', 'BICshort', 'BRA', 'TRIlong', 'TRIlat', 'TRImed']

x = list(range(r_Ia.shape[0]))
fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.plot(x, r_Ia)
ax.set_ylabel('norm. rate [-]')
ax.grid()
ax.legend(muscle_names)
ax.set_xlabel('time [-]')
ax.set_title('Ia afferent firing rates (r_Ia)')
plt.tight_layout()
plt.savefig(os.path.join(path_fig, 'r_Ia_plots.png'), bbox_inches='tight')
plt.close()

x = list(range(r_mn.shape[0]))
fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.plot(x, r_mn)
ax.set_ylabel('activation [-]')
ax.grid()
ax.legend(muscle_names)
ax.set_xlabel('time [-]')
ax.set_title('Motor neuron activations (r_mn)')
plt.tight_layout()
plt.savefig(os.path.join(path_fig, 'r_mn_plots.png'), bbox_inches='tight')
plt.close()
