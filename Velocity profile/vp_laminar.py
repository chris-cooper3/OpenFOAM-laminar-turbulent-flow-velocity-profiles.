import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import sys
np.set_printoptions(threshold=np.inf)

working_directory = pathlib.Path(__file__).parent
velo_dat = pd.read_excel(
    working_directory/'velocity_rad.xlsx', sheet_name='Laminar')  # Read in the data

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
Re = 1000  # Reynolds number
ki_vi = 8e-6  # Kinematic viscosity
D = 8e-3  # Diameter of pipe
v_av = (Re*ki_vi)/D  # Calculate the average velocity
ve_an = 2*v_av*(1-(rad_fi**2/(D/2)**2))  # Velocity from analytical solution

########### Plot of graph ###########################################################################################

plt.figure(figsize=(10, 10))
plt.plot(vel_fi, rad_fi)
plt.plot(ve_an, rad_fi)
plt.legend(['solution', 'analytical'], fontsize=15)
plt.xlabel('Velocity magnitude (m/s)', fontsize=12)
plt.ylabel('Radial distance from axis (m)', fontsize=12)
plt.title('Fully developed flow (solution vs analytical)', fontsize=17)
plt.ylim(-0.0042, 0.0042)
plt.xlim(0, 2.1)
plt.show()

########### Error between solution and anlytical solution ##########################################################################################

#Distribution of the error 
err = (abs(ve_an[ve_an !=0] - vel_fi[vel_fi!=0])/ve_an[ve_an !=0])*100
err = err.astype(np.float64)
plt.hist(err,100,density=True)
plt.xlabel('Residual error (%)')
plt.show

# Kernel density plot of the pointwise error 
import seaborn as sns 
sns.kdeplot(err,bw=.1)
plt.show()

#Root mean square for continous variables.
error = ve_an - vel_fi
RMSE = np.sqrt(np.mean(error**2))
























len(rad_fi)
len(ve_an)

#rad_fi[ve_an !=0]

plt.figure(figsize=(10, 10))
plt.plot(err, rad_fi[ve_an != 0])
plt.xlabel('error (m)', fontsize=12)
plt.ylabel('Radius (m)', fontsize=12)
plt.title('Error between anayltical and experimental', fontsize=17)
plt.ylim(-0.004, 0.004)
plt.xlim(0)
plt.show()














# 1st try at fixing issue ()
###if np.logical_or(vel_fi == 0, ve_an == 0):
   # err = 0
#else:
   # err = ((abs(vel_fi-ve_an))/vel_fi)*100

# 2nd try
#for i in len(vel_fi), len(ve_an):
   # if np.logical_or(vel_fi[i] == 0, ve_an[i] == 0):
       # err = 0
   # else:
       ### err = ((abs(vel_fi[i]-ve_an[i]))/vel_fi[i])*100
# 3rd try


#def function(vel_fi, ve_an):
    #return print(((abs(vel_fi-ve_an))/vel_fi)*100) if vel_fi else 0


#plt.figure(figsize=(10, 10))
#plt.plot(rad_fi, SD)
#plt.xlabel('Radius (m)', fontsize=12)
#plt.ylabel('Standard deviation', fontsize=12)
#plt.title('Standard deviatiob', fontsize=17)
# plt.ylim(0)
#plt.xlim(0, 2.1)
# plt.show()

########## Initial conditons for tubulent flow ##########################################################################
Re_t = 1e5
U = 1
D_2 = 8e-3
c_mu = 0.09
I = 0.16*(Re_t)**(-1/8)
k = (3/2)*(U*I)**2
l = 0.07*D_2
e = ((c_mu)**(3/4))*(k**(3/2))*(l**-1)

print(k, I, l, e)
