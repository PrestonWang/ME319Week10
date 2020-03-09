# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:56:13 2020

@author: Preston Wang
"""

import numpy as np
from WrenchUtils import PTrans 

m = 1 # mass of the block
g = 10 # gravitational acceleration
mu_s = 0.5 # coefficient of friction between support and block
N = 10 # normal force from the gripper
r = 0.05 # radius of contact (meters)
k = 0.6*r
# calculating limit surface
A = np.diag([1/(mu_s*N)**2, 1/(mu_s*N)**2, 1/(k*mu_s*N)**2])

# calculate generalized friction cone
Px = 2
Pz = 0
Prot = 0
Jpt = PTrans(Px, Pz, Prot)
Fp = np.array([[0,0,0],[0,0,0],[]])