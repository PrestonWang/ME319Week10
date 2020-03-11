'''
ME319 Week 10 Code
3/11/2020
'''

import numpy as np
from WrenchUtils import PTrans 
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
from scipy import spatial as sp_spatial
from sympy import nsolve, symbols


# Functions to help with plotting the limit surface and cones
def plot_ellipsoid(A, ax): 
	# your ellispsoid and center in matrix form
	center = [0,0,0]

	# find the rotation matrix and radii of the axes
	U, S, V = np.linalg.svd(A)
	radii = 1.0/np.sqrt(S)

	# now carry on with EOL's answer
	u = np.linspace(0.0, 2.0 * np.pi, 100)
	v = np.linspace(0.0, np.pi, 100)
	x = radii[0] * np.outer(np.cos(u), np.sin(v))
	y = radii[1] * np.outer(np.sin(u), np.sin(v))
	z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
	for i in range(len(x)):
	    for j in range(len(x)):
	        [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], V) + center

	# plot
	ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
    
def draw_polygon(points,ax):
    hull = sp_spatial.ConvexHull(points)
    indices = hull.simplices
    faces = points[indices]
    for f in faces:
        face = a3.art3d.Poly3DCollection([f])
        face.set_edgecolor('k')
        face.set_alpha(0.5)
        ax.add_collection3d(face)
    plt.show()

# DEFINE CONSTANTS
m = 0.25        # mass of the block
g = 9.8         # gravitational acceleration
mu_s = 0.5      # coefficient of friction between support and block
N = 10          # normal force from the gripper
r = 1           # radius of contact (meters)
k = 0.6*r       # support moment resistance constant
Fn = 5          # normal force from pusher (N)
Ft = Fn*mu_s    # corresponding tangential force


# STEP 1: CALCULATE LIMIT SURFACE
A = np.diag([1/(mu_s*N)**2, 1/(mu_s*N)**2, 1/(k*mu_s*N)**2])


# STEP 2: CALCULATE GENERALIZED FRICTION CONE (set of wrenches the pusher can apply on the object) 
# Substep a: Calculate Jp: Jacobian that maps object velocity (in object frame) to the velocity at the pusher point contact frames
Px = -0.15
Pz1 = -0.0625
Pz2  = 0.0625
Prot = 0
Jp1  = PTrans(Px, Pz1, Prot)[0:2,:]  
Jp2  = PTrans(Px, Pz2, Prot)[0:2,:]
Jp = np.vstack((Jp1, Jp2, Jp2, Jp1))

# Substep b: Calculate pusher wrench 
# See figure in Table 1 for reference
FP1 = np.array([Fn, -Ft])
FP2 = np.array([Fn, Ft])
Fp = np.hstack((FP1, FP1, FP2, FP2))

W1 = Jp[0:2,:].T.dot(Fp[0:2])
W2 = Jp[2:4,:].T.dot(Fp[2:4])
W3 = Jp[4:6,:].T.dot(Fp[4:6])
W4 = Jp[6:,:].T.dot(Fp[6:])
W = np.row_stack((W1,W2,W3,W4))

# STEP 3: CALCULATE OBJECT VELOCITY
# Calculate Js
Sx = 0.0625
Sz   = 0
Srot = 0
Js = PTrans(Sx, Sz, Srot)  # Jacobian that maps v_obj (in object frame) to v_s (in support frame)

# Calculate gravity wrench 
w_gravity = np.array([0,m*g,0])

# Define B, wrench on a unit limit surface 
B = np.diag((1, 1, k**(-2)))
k = symbols('k',real = True)
# function to calculate object velocities
def calculate_object_velocity(W): 
    wLength = W.shape[0]
    V = np.zeros(W.shape)
    for i in range(wLength):
        wpusher = W[i]/np.linalg.norm(W[i])
        ##############################
        # TODO: 
        # Calculate v_object for a given pusher unit wrench. v_obj should be a 3x1 unit vector
        ws_temp = np.linalg.inv(Js.T).dot(-w_gravity - k*wpusher)/(mu_s*N)
        ksol = float(nsolve(ws_temp[0]**2 + ws_temp[1]**2 + (ws_temp[2]**2)/(0.6*r)**2 - 1, k, 1))
        ws_bar = -np.linalg.inv(Js.T).dot(-w_gravity - ksol*wpusher)/(mu_s*N)
        v_obj = (mu_s*N)*np.linalg.inv(Js.T).dot(B).dot(ws_bar)
        ##############################
        V[i] = v_obj
    return V

V = calculate_object_velocity(W)


#%% Plotting Limit Surface, Pusher Wrench and 
# Plot motion cones
origin = [0,0,0]
velocityVec = np.row_stack((origin,V))
wrenchVec = np.row_stack((origin,W1,W2,W3,W4))
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
draw_polygon(wrenchVec,ax)
plot_ellipsoid(A,ax)
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('My')
plt.title('Pusher Wrench and Limit Surface')

fig = plt.figure(2)
plt.clf()
X = velocityVec[:,0]
Y = velocityVec[:,1]
Z = velocityVec[:,2]
ax2 = fig.add_subplot(111, projection='3d')
ax2.plot(X,Y,Z)
draw_polygon(velocityVec,ax2)
ax2.set_xlabel('Tx')
ax2.set_ylabel('Tz')
ax2.set_zlabel('Wy')
plt.title('Motion Cone')



