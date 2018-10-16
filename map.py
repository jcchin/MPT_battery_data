import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import odeint
import itertools
import pprint
from numpy import array2string as a

import time

# data with breakpoints that line up with pulses
T_bp = [0., 20., 30., 45.]
SOC_bp1 = np.linspace(0.08,1,29)
SOC_bp2 = np.linspace(0.083,1,29)
SOC_bp3 = np.linspace(0.017,1,31)
SOC_bp4 = np.linspace(0.012,1,31)


tU_oc1 = [3.123, 3.205, 3.261, 3.338, 3.403, 3.459, 3.505, 3.529, 3.550, 3.601, 3.628, 3.663, 3.693, 3.733, 3.761, 3.798, 3.825, 3.845, 3.864, 3.893, 3.920, 3.967, 4.005, 4.044, 4.070, 4.076, 4.086, 4.106, 4.161]
tU_oc2 = [3.145, 3.205, 3.261, 3.338, 3.403, 3.459, 3.505, 3.529, 3.550, 3.601, 3.648, 3.685, 3.710, 3.748, 3.771, 3.798, 3.825, 3.845, 3.878, 3.913, 3.940, 3.987, 4.025, 4.047, 4.073, 4.076, 4.084, 4.106, 4.161]
tU_oc3 = [2.914, 3.055, 3.152, 3.232, 3.275, 3.335, 3.398, 3.459, 3.510, 3.525, 3.550, 3.603, 3.637, 3.685, 3.717, 3.747, 3.775, 3.802, 3.828, 3.855, 3.887, 3.923, 3.952, 3.986, 4.027, 4.059, 4.069, 4.075, 4.082, 4.109, 4.153]
tU_oc4 = [2.875, 3.028, 3.135, 3.212, 3.255, 3.315, 3.378, 3.439, 3.490, 3.522, 3.550, 3.603, 3.624, 3.664, 3.717, 3.747, 3.775, 3.802, 3.828, 3.855, 3.887, 3.923, 3.952, 3.986, 4.027, 4.058, 4.071, 4.078, 4.082, 4.109, 4.153]       

tC_Th1 = [2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000.]
tC_Th2 = [2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000.]
tC_Th3 = [2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000.]
tC_Th4 = [2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000.]        

tR_Th1 = [0.090, 0.090, 0.060, 0.060, 0.060, 0.060, 0.060, 0.060, 0.060, 0.060, 0.090, 0.070, 0.070, 0.070, 0.070, 0.070, 0.070, 0.050, 0.040, 0.060, 0.070, 0.070, 0.075, 0.075, 0.075, 0.055, 0.055, 0.040, 0.040]
tR_Th2 = [0.060, 0.050, 0.047, 0.045, 0.045, 0.045, 0.040, 0.039, 0.035, 0.035, 0.035, 0.040, 0.060, 0.040, 0.035, 0.030, 0.030, 0.030, 0.030, 0.040, 0.040, 0.040, 0.040, 0.030, 0.030, 0.030, 0.028, 0.025, 0.020]
tR_Th3 = [0.060, 0.045, 0.045, 0.045, 0.045, 0.045, 0.040, 0.040, 0.035, 0.035, 0.030, 0.027, 0.027, 0.025, 0.025, 0.020, 0.020, 0.020, 0.050, 0.040, 0.040, 0.040, 0.040, 0.030, 0.028, 0.028, 0.020, 0.020, 0.020, 0.020, 0.001]
tR_Th4 = [0.060, 0.040, 0.040, 0.040, 0.040, 0.040, 0.040, 0.040, 0.030, 0.030, 0.030, 0.030, 0.030, 0.025, 0.025, 0.020, 0.020, 0.020, 0.050, 0.040, 0.040, 0.040, 0.040, 0.030, 0.028, 0.028, 0.020, 0.020, 0.020, 0.020, 0.001]
        
tR_01  = [0.150, 0.110, 0.090, 0.080, 0.080, 0.075, 0.070, 0.070, 0.030, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.065, 0.065]
tR_02  = [0.050, 0.035, 0.030, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]
tR_03  = [0.060, 0.045, 0.030, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.010, 0.020, 0.020,  0.02, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.03]
tR_04  = [0.060, 0.045, 0.030, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.010, 0.010, 0.010,  0.02, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.03]
        

# plot raw maps
fig, (ax1, ax2, ax3) = plt.subplots(3,1)

# ax1.plot(SOC_bp1, tU_oc1, label='0')
# ax1.plot(SOC_bp2, tU_oc2, label='20')
# ax1.plot(SOC_bp3, tU_oc3, label='30')
# ax1.plot(SOC_bp4, tU_oc4, label='45')
# ax2.plot(SOC_bp1, tR_Th1, label='0')
# ax2.plot(SOC_bp2, tR_Th2, label='20')
# ax2.plot(SOC_bp3, tR_Th3, label='30')
# ax2.plot(SOC_bp4, tR_Th4, label='45')
# ax3.plot(SOC_bp1, tR_01, label='0')
# ax3.plot(SOC_bp2, tR_02, label='20')
# ax3.plot(SOC_bp3, tR_03, label='30')
# ax3.plot(SOC_bp4, tR_04, label='45')
# ax1.invert_xaxis()
# ax2.invert_xaxis()
# ax3.invert_xaxis()
# ax1.legend()
# ax1.set_ylabel("V_oc")
# ax2.set_ylabel("R_Th")
# ax3.set_ylabel("R_0")
# plt.show()

# reinterp from 0 to 1 for all 16 variables, cycle through 4 bps
# save dynamic number variable final<X>
 
bps = np.linspace(0,1,31)
bz = [SOC_bp1, SOC_bp2, SOC_bp3, SOC_bp4] #breakpoints
tz = [tU_oc1, tU_oc2, tU_oc3, tU_oc4, tC_Th1, tC_Th2, tC_Th3, tC_Th4, tR_Th1, tR_Th2, tR_Th3, tR_Th4, tR_01, tR_02, tR_03, tR_04] #table, original
for k, (b, t) in enumerate(list(zip(itertools.cycle(bz),tz))):
    #print (len(b),len(t))
    f = interp1d(b, t, fill_value='extrapolate')
    globals()['final%s' % k] = f(bps)

print("battery.T_bp = np.array([0., 20., 30., 45.])") 
print("battery.SOC_bp = np.array(", a(bps, separator=', '),")")
print("battery.tU_oc = np.array([", a(final0, separator=','), ", ", a(final1, separator=','), ", ", a(final2, separator=','), ", ", a(final3, separator=','), "])")
print("battery.tC_Th = np.array([", a(final4, separator=','), ", ", a(final5, separator=','), ", ", a(final6, separator=','), ", ", a(final7, separator=','), "])")
print("battery.tR_Th = np.array([", a(final8, separator=','), ", ", a(final9, separator=','), ", ", a(final10, separator=','), ", ", a(final11, separator=','), "])")
print("battery.tR_0 = np.array([", a(final12, separator=','), ", ", a(final13, separator=','), ", ", a(final14, separator=','), ", ", a(final15, separator=','), "])")

# plot re-interpolated maps
ax1.plot(bps, final0, label='0')
ax1.plot(bps, final1, label='20')
ax1.plot(bps, final2, label='30')
ax1.plot(bps, final3, label='45')
ax2.plot(bps, final8, label='0') #skip C_Th, not exciting
ax2.plot(bps, final9, label='20')
ax2.plot(bps, final10, label='30')
ax2.plot(bps, final11, label='45')
ax3.plot(bps, final12, label='0')
ax3.plot(bps, final13, label='20')
ax3.plot(bps, final14, label='30')
ax3.plot(bps, final15, label='45')
ax1.invert_xaxis()
ax2.invert_xaxis()
ax3.invert_xaxis()
ax1.legend()
ax1.set_ylabel("V_oc")
ax2.set_ylabel("R_Th")
ax3.set_ylabel("R_0")
plt.show()