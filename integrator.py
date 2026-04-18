#starship\integrator.py
from helper import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Rk4_integrator:
    def __init__(self,get_forceMoment,inertia_matrix=np.array([[1,0,0],[0,1,0],[0,0,1]],dtype="float64"),l=7,a=0.3,mass=85000,g=9.8,step_size=0.001):
        self.inertia_matrix=inertia_matrix
        self.l=l
        self.a=a
        self.mass=mass
        self.g=g
        self.step=step_size
        self.get_forceMoment=get_forceMoment

    def angular_velocity_next(self,forceMoment:forces_moments_body_frame,angular_velocity:angular_velocity):
      try:
        inertia_matrix_inverse=np.linalg.inv(self.inertia_matrix)
      except np.linalg.LinAlgError as e:
        print(f"\nError: Could not calculate inverse. {e}")
      result=inertia_matrix_inverse@(forceMoment[3:]-np.cross(angular_velocity,self.inertia_matrix@angular_velocity))
      return result
    def quaternions_next(self,angular_velocity:angular_velocity,quaternions:quaternions):
      quaternion=(1/2)*construct_omega_matrix_4x4(angular_velocity)@quaternions
      return quaternion
    def velocity_next(self,forceMoment:forces_moments_body_frame,quaternions:quaternions,gravity:gravity_fixed_frame):
      T=forceMoment[:3]
      R=quat_to_rotmat(quaternions)
      return (1/self.mass)*(R@T+self.mass*gravity)
    def position_next(self,velocity):
      return velocity

    def forward(self,state:StateVector): #input format [[tetha1,phi1,t1],[tetha2,phi2,t2],[tetha3,phi3,t3]
      #declaring the ks array and initializing next_state
      K=np.zeros((13,4),dtype="float64")
      next_state=np.zeros(13,dtype="float64")
      gravity=np.array([0,0,self.g],dtype="float64")
      forceMoment=self.get_forceMoment()
      #computing k1
      K[0:3, 0] = self.position_next(state[7:10])
      K[3:7, 0] = self.quaternions_next(state[10:],state[3:7])
      K[7:10, 0] = self.velocity_next(forceMoment,state[3:7],gravity)
      K[10:13, 0] = self.angular_velocity_next(forceMoment,state[10:])
      #computing k2
      state2=state+self.step*(K[:,0]/2)
      K[0:3, 1] = self.position_next(state2[7:10])
      K[3:7, 1] = self.quaternions_next(state2[10:],state2[3:7])
      K[7:10, 1] = self.velocity_next(forceMoment,state2[3:7],gravity)
      K[10:13, 1] = self.angular_velocity_next(forceMoment,state2[10:])
      #computing k3
      state3=state+self.step*(K[:,1]/2)
      K[0:3, 2] = self.position_next(state3[7:10])
      K[3:7, 2] = self.quaternions_next(state3[10:],state3[3:7])
      K[7:10, 2] = self.velocity_next(forceMoment,state3[3:7],gravity)
      K[10:13, 2] = self.angular_velocity_next(forceMoment,state3[10:])
      #computing k4
      state4=state+self.step*K[:,2]
      K[0:3, 3] = self.position_next(state4[7:10])
      K[3:7, 3] = self.quaternions_next(state4[10:],state4[3:7])
      K[7:10, 3] = self.velocity_next(forceMoment,state4[3:7],gravity)
      K[10:13, 3] = self.angular_velocity_next(forceMoment,state4[10:])
      #next state
      next_state=state+self.step*(K@np.array([1/6,1/3,1/3,1/6]))
      return next_state