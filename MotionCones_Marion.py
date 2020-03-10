import numpy as np
from WrenchUtils import PTrans 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as a3
from scipy import spatial as sp_spatial


def plot_ellipsoid(A): 
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
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
	plt.show()



# DEFINE CONSTANTS
m = 1       # mass of the block
g = 10      # gravitational acceleration
mu_s = 0.5  # coefficient of friction between support and block
N = 10      # normal force from the gripper
r = 0.05    # radius of contact (meters)
k = 0.6*r


# STEP 1: CALCULATE LIMIT SURFACE
A = np.diag([1/(mu_s*N)**2, 1/(mu_s*N)**2, 1/(k*mu_s*N)**2])
# plot_ellipsoid(A)


# STEP 2: CALCULATE GENERALIZED FRICTION CONE (set of wrenches the pusher can apply on the object) 
# Substep a: Calculate Jp: Jacobian that maps object velocity (in object frame) to the velocity at the pusher point contact frames
Px = -0.25
Pz1 = -0.25
Pz2  = 0.25
Prot = 0
Jp1  = PTrans(Px, Pz1, Prot)[0:2,:]  
Jp2  = PTrans(Px, Pz2, Prot)[0:2,:]
Jp = np.vstack((Jp1, Jp1, Jp2, Jp2))

# create generalized friction cone
Fn = m*g/mu_s
Ft = Fn*mu_s

# Substep b: Calculate pusher wrench 
# See figure in Table 1 for reference
FP1 = np.array([Fn, -Ft])
FP2 = np.array([Fn, Ft])
Fp = np.hstack((FP1, FP2, FP1, FP2))

W1 = Jp[0:2,:].T.dot(Fp[0:2])
W2 = Jp[2:4,:].T.dot(Fp[2:4])
W3 = Jp[4:6,:].T.dot(Fp[4:6])
W4 = Jp[6:,:].T.dot(Fp[6:])


# STEP 3: CALCULATE OBJECT VELOCITY
# Calculate Js
Sx = 0.125
Sz   = 0
Srot = 0
Js = PTrans(Sx, Sz, Srot)  # Jacobian that maps v_obj (in object frame) to v_s (in support frame)

# Calculate gravity wrench 
w_gravity = np.array([0,m*g,0])

# Define B, wrench on a unit limit surface 
B = np.diag((1, 1, k**(-2)))

def calculate_object_velocity(W): 
	w_s = np.linalg.inv(Js.T).dot(-w_gravity - W)
	ws_bar = w_s / (mu_s * N)
	v_obj = np.linalg.inv(Js).dot(B).dot(ws_bar)
	v_obj = v_obj / np.linalg.norm(v_obj)
	return v_obj

v_obj1 = calculate_object_velocity(W1)
v_obj2 = calculate_object_velocity(W2)
v_obj3 = calculate_object_velocity(W3)
v_obj4 = calculate_object_velocity(W4)


# Plot motion cones
origin = [0,0,0]
points = np.row_stack((origin,v_obj1, v_obj2, v_obj3, v_obj4))
hull = sp_spatial.ConvexHull(points)
indices = hull.simplices
faces2 = points[indices]
fig = plt.figure(2)
plt.clf()
ax = Axes3D(fig)
for f in faces2:
    face = a3.art3d.Poly3DCollection([f])
    face.set_edgecolor('k')
    face.set_alpha(0.5)
    ax.add_collection3d(face)

plt.show()




