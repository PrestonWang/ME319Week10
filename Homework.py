# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:56:13 2020

@author: Preston Wang
"""

import numpy as np
from WrenchUtils import PTrans 
from matplotlib import pyplot as plt
from scipy import spatial as sp_spatial
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as a3

m = 1 # mass of the block
g = -10 # gravitational acceleration
mu_s = 0.5 # coefficient of friction between support and block
N = 1 # normal force from the gripper
r = 0.05 # radius of contact (meters)
k = 0.6*r
# calculating limit surface
A = np.diag([1/(mu_s*N)**2, 1/(mu_s*N)**2, 1/(k*mu_s*N)**2])
s = np.array([0.125,0,0])
Js = np.transpose(PTrans(s[0],s[1],s[2]))


# calculate generalized friction cone
P1 = np.array([-.25, 0.25]) # location of pusher contact
P2 = np.array([-.25,-0.25]) # location of pusher contact 2

def calcJpt(P1,P2):
    # calculate the Jacobian from the pusher contact frames to the object
    Jp1t = np.transpose(PTrans(P1[0], P1[1], 0)) # Jacobian from Pusher point 1 to object
    Jp2t = np.transpose(PTrans(P2[0], P2[1], 0)) # Jacobian from Pusher point 2 to object
    return np.column_stack((Jp1t[:,0:2], Jp2t[:,0:2])) # combined Jacobian
    
Jpt = calcJpt(P1,P2)
# create generalized friction cone
Fn = -m*g/mu_s
Ft = Fn*mu_s

F1 = np.transpose(np.array([Fn,Ft,0,0]))
F2 = np.transpose(np.array([Fn,-Ft,0,0]))
F3 = np.transpose(np.array([0,0,Fn,Ft]))
F4 = np.transpose(np.array([0,0,Fn,-Ft]))

W1 = np.dot(Jpt,F1)
W2 = np.dot(Jpt,F2)
W3 = np.dot(Jpt,F3)
W4 = np.dot(Jpt,F4)
origin = np.array([0,0,0])
points = np.row_stack((origin,W1,W2,W3,W4))
hull = sp_spatial.ConvexHull(points)
indices = hull.simplices
faces = points[indices]

# plotting friction cone
fig = plt.figure(1)
plt.clf()
ax = Axes3D(fig)
for f in faces:
    face = a3.art3d.Poly3DCollection([f])
    face.set_edgecolor('k')
    face.set_alpha(0.5)
    ax.add_collection3d(face)
ax.autoscale_view()
# %% creating motion cone
G = np.transpose(np.array([0,g,0]))
Js_inv = np.linalg.inv(Js)
B = np.diag([1,1,k**-2])
def calcVobj(W):
    Ws = np.dot(Js_inv,(-W-m*G)/(mu_s*N))
    Vobj = np.dot(np.dot(Js_inv,B),Ws)
    return Vobj/np.linalg.norm(Vobj)
Vobj1 = calcVobj(W1)
Vobj2 = calcVobj(W2)
Vobj3 = calcVobj(W3)
Vobj4 = calcVobj(W4)

# plotting motion cones
points = np.row_stack((origin,Vobj1,Vobj2,Vobj3,Vobj4))
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
