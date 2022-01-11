import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

working_directory = pathlib.Path(__file__).parent
velo_dat = pd.read_excel(
    working_directory/'velocity_rad.xlsx', sheet_name='Turbulent')  # Read in the data

########## Formatting the data ########################################################################################
# Reformatting the data to plot the full velocity profile.

vel = np.array(velo_dat.iloc[1:, 0])  # Velocity from exact solution
rad = np.array(velo_dat.iloc[1:, 1])  # Radius values from exact solution
rad_2 = np.flip(rad*-1)  # Flip the array and make the radius values negative.
vel_ne = np.flip(vel)  # Flip the array
# Join the two sets of velocity data --> plot full velocity profile.
vel_fi = np.concatenate((vel, vel_ne))
rad_fi = np.concatenate((rad, rad_2))  # Join the two radius arrays.

########## Analytical solution calculations ###########################################################################
Re = 100000  # Reynolds number
ki_vi = 8e-8  # Kinematic viscosity
D = 8e-3  # Diameter of pipe
v_av = (Re*ki_vi)/D  # Calculate the average velocity
# Velocity from analytical solution power 7
ve_an = v_av*(1-(abs(rad_fi)/(D/2)))**(1/7)
# Velocity from analytical solution power 5
ve_an_2 = v_av*(1-(abs(rad_fi)/(D/2)))**(1/5)

########### Plot of graph ###########################################################################################

plt.figure(figsize=(10, 10))
plt.plot(vel_fi, rad_fi)
plt.plot(ve_an, rad_fi)
plt.plot(ve_an_2, rad_fi)
plt.legend(['solution', 'analytical(n=7)', 'analytical(n=5)'], fontsize=15)
plt.xlabel('Velocity magnitude (m/s)', fontsize=12)
plt.ylabel('Radial distance from axis (m)', fontsize=12)
plt.title('Fully developed flow (solution vs analytical)', fontsize=17)
plt.ylim(-0.0042, 0.0042)
plt.xlim(0, 2.1)
plt.show()

########### Error between solution and anlytical solution ##########################################################################################

# Distribution of the error
err = (abs(ve_an[ve_an != 0] - vel_fi[vel_fi != 0])/ve_an[ve_an != 0])*100
err = err.astype(np.float64)
plt.hist(err, 100, density=True)
plt.xlabel('Residual error (%)')
plt.show

# Kernel density plot of the pointwise error
sns.kdeplot(err, bw=.1)
plt.show()

# Root mean square for continous variables.
error = ve_an - vel_fi
RMSE = np.sqrt(np.mean(error**2))
