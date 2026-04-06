import pickle
import matplotlib.pyplot as plt
import os

fs = 14
plt.rc('font',   size=fs+2)
plt.rc('axes',   titlesize=fs+3)
plt.rc('axes',   labelsize=fs+3)
plt.rc('xtick',  labelsize=fs)
plt.rc('ytick',  labelsize=fs)
plt.rc('legend', fontsize=fs+2)

# FIX: relative paths so script runs on any machine
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
path_fig_base  = os.path.join(BASE_DIR, 'figures', 'MS_figures')
path_fig       = os.path.join(path_fig_base, 'custom_muscle_subplots')
os.makedirs(path_fig, exist_ok=True)

file_pickle = os.path.join(path_fig_base, 'd_combined_states.pkl')
with open(file_pickle, 'rb') as fp:
    d = pickle.load(fp)

print(d.keys())

x = list(range(len(d[list(d.keys())[0]])))
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(18, 12))

ax[0].plot(x, d['TRIlat_activation'],  label='TRI lat')
ax[0].plot(x, d['TRIlong_activation'], label='TRI long')
ax[0].plot(x, d['TRImed_activation'],  label='TRI med')
ax[0].set_ylabel('[-]')   # FIX: activation is dimensionless, not Newtons
ax[0].grid()
ax[0].legend()

ax[1].plot(x, d['TRIlat_fiber_length'],  label='TRI lat')
ax[1].plot(x, d['TRIlong_fiber_length'], label='TRI long')
ax[1].plot(x, d['TRImed_fiber_length'],  label='TRI med')
ax[1].set_ylabel('m')
ax[1].grid()
ax[1].legend()

ax[2].plot(x, d['TRIlat_fiber_velocity'],  label='TRI lat')
ax[2].plot(x, d['TRIlong_fiber_velocity'], label='TRI long')
ax[2].plot(x, d['TRImed_fiber_velocity'],  label='TRI med')
ax[2].set_ylabel('m/s')
ax[2].set_xlabel('time [-]')
ax[2].grid()
ax[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(path_fig, 'Extensor_plots.png'), bbox_inches='tight')
plt.close()
