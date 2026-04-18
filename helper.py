#starship\helper.py
#imports
from typing import Annotated
import numpy as np
import numpy.typing as npt
from math import sqrt,sin,cos
from math import pi
from scipy.spatial.transform import Rotation as R

StateVector = Annotated[npt.NDArray[np.float64], (13,)]
Engines_thrust_Array = Annotated[npt.NDArray[np.float64], (3, 3)] #3 rows each for 1 engine ( total 3 gimabled engines)for each engine: [tetha, phi ,thrust]
thrust_forces_and_moments_body_frame = Annotated[npt.NDArray[np.float64], (6,)]
angular_velocity = Annotated[npt.NDArray[np.float64], (3,)]
moments = Annotated[npt.NDArray[np.float64], (3,)]
quaternions = Annotated[npt.NDArray[np.float64], (4,)]
moments = Annotated[npt.NDArray[np.float64], (3,)]  #[x,y,z]
forces_moments_body_frame=Annotated[npt.NDArray[np.float64], (6,)]  #[Fx,Fy,Fz,Mx,My,Mz]
gravity_fixed_frame=Annotated[npt.NDArray[np.float64], (3,)]  #[gx,gy,gz]

def construct_omega_matrix_4x4(angular_velocity_vector):
    """
    Constructs the 4x4 Omega matrix used in the quaternion kinematic
    differential equation (d𝗊/dt = 1/2 * Ω(𝜔) * 𝗊) from the 3D angular
    velocity vector [wx, wy, wz].
    """
    # Unpack the angular velocity components
    if angular_velocity_vector.shape != (3,):
        raise ValueError("Input must be a 3-element vector [wx, wy, wz].")

    wx, wy, wz = angular_velocity_vector

    # Construct the 4x4 Omega matrix
    # The first row/column relates to the scalar component (qw)
    Omega = np.array([
        [ 0, -wx, -wy, -wz],
        [ wx,   0,  wz, -wy],
        [ wy, -wz,   0,  wx],
        [ wz,  wy, -wx,   0]
    ])
    return Omega

def engine_frame_to_body_frame(spherical): #converts engine orientation and thrust expressed in engine frame in spherical cordinates into forces expressed in rocket frame cartesian cordinates
    """
    Same behavior as your original function,
    but takes a single numpy array [theta, phi, r].
    """
    theta = spherical[..., 0]
    phi   = spherical[..., 1]
    r     = spherical[..., 2]

    x = r * np.cos(phi)
    z = -r * np.cos(theta) * np.sin(phi)
    y = -r * np.sin(phi) * np.sin(theta)

    return np.array([x, y, z])


def quat_to_rotmat(q):
    """
    Convert quaternion [w, x, y, z] to a 3×3 rotation matrix.
    R rotates a vector from BODY frame → INERTIAL frame:
        v_inertial = R @ v_body
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("q must be length-4: [w, x, y, z]")

    # Normalize
    q = q / np.linalg.norm(q)

    w, x, y, z = q  # <-- reordered

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),       2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),   1 - 2*(xx + yy)]
    ])

    return R


def compute_thrust_forces_and_moments(engine_thrust:Engines_thrust_Array,a:float,l:float) -> thrust_forces_and_moments_body_frame:
  """computes sum of thrusts and moments of the 3 engines expressed in the rocket coordinate system. expressed at the center of mass G"
  """
  #resultant computation
  theta = engine_thrust[:, 0]
  phi = engine_thrust[:, 1]
  thrust = engine_thrust[:, 2]
  resultant = np.array([
      +np.sum(thrust * np.cos(phi)),
      -np.sum(thrust * np.sin(phi) * np.sin(theta)),
      -np.sum(thrust * np.sin(phi) * np.cos(theta))
  ], dtype=np.float64)
  #moment in G computation
  GE1=np.array([-l,-a*cos(pi/6),a*cos(2*pi/6)],dtype="float64")
  GE2=np.array([-l,a*cos(pi/6),a*cos(2*pi/6)],dtype="float64")
  GE3=np.array([-l,0,-a],dtype="float64")
  TE1=engine_frame_to_body_frame(engine_thrust[0])
  TE2=engine_frame_to_body_frame(engine_thrust[1])
  TE3=engine_frame_to_body_frame(engine_thrust[2])
  total_moment=np.cross(GE1,TE1)+np.cross(GE2,TE2)+np.cross(GE3,TE3)
  result=np.concatenate((resultant,total_moment))
  return result
#returns [Fx,Fy,Fz,Mx,My,Mz]
compute_thrust_forces_and_moments(np.array([[0,pi/2,5],
                                            [0,pi/2,5],
                                            [0,pi/2,5],
                                            ],dtype="float64"),0.5,7)

def construct_omega_matrix_4x4(angular_velocity_vector):
    """
    Constructs the 4x4 Omega matrix used in the quaternion kinematic
    differential equation (d𝗊/dt = 1/2 * Ω(𝜔) * 𝗊) from the 3D angular
    velocity vector [wx, wy, wz].
    """
    # Unpack the angular velocity components
    if angular_velocity_vector.shape != (3,):
        raise ValueError("Input must be a 3-element vector [wx, wy, wz].")

    wx, wy, wz = angular_velocity_vector

    # Construct the 4x4 Omega matrix
    # The first row/column relates to the scalar component (qw)
    Omega = np.array([
        [ 0, -wx, -wy, -wz],
        [ wx,   0,  wz, -wy],
        [ wy, -wz,   0,  wx],
        [ wz,  wy, -wx,   0]
    ])
    return Omega

def engine_frame_to_body_frame(spherical): #converts engine orientation and thrust expressed in engine frame in spherical cordinates into forces expressed in rocket frame cartesian cordinates
    """
    Same behavior as your original function,
    but takes a single numpy array [theta, phi, r].
    """
    theta = spherical[..., 0]
    phi   = spherical[..., 1]
    r     = spherical[..., 2]

    x = r * np.cos(phi)
    z = -r * np.cos(theta) * np.sin(phi)
    y = -r * np.sin(phi) * np.sin(theta)

    return np.array([x, y, z])


def quat_to_rotmat(q):
    """
    Convert quaternion [w, x, y, z] to a 3×3 rotation matrix.
    R rotates a vector from BODY frame → INERTIAL frame:
        v_inertial = R @ v_body
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("q must be length-4: [w, x, y, z]")

    # Normalize
    q = q / np.linalg.norm(q)

    w, x, y, z = q  # <-- reordered

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),       2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),   1 - 2*(xx + yy)]
    ])

    return R

def compute_thrust_forces_and_moments(engine_thrust:Engines_thrust_Array,a:float,l:float) -> thrust_forces_and_moments_body_frame:
  """computes sum of thrusts and moments of the 3 engines expressed in the rocket coordinate system. expressed at the center of mass G"
  """
  #resultant computation
  theta = engine_thrust[:, 0]
  phi = engine_thrust[:, 1]
  thrust = engine_thrust[:, 2]
  resultant = np.array([
      +np.sum(thrust * np.cos(phi)),
      -np.sum(thrust * np.sin(phi) * np.sin(theta)),
      -np.sum(thrust * np.sin(phi) * np.cos(theta))
  ], dtype=np.float64)
  #moment in G computation
  GE1=np.array([-l,-a*cos(pi/6),a*cos(2*pi/6)],dtype="float64")
  GE2=np.array([-l,a*cos(pi/6),a*cos(2*pi/6)],dtype="float64")
  GE3=np.array([-l,0,-a],dtype="float64")
  TE1=engine_frame_to_body_frame(engine_thrust[0])
  TE2=engine_frame_to_body_frame(engine_thrust[1])
  TE3=engine_frame_to_body_frame(engine_thrust[2])
  total_moment=np.cross(GE1,TE1)+np.cross(GE2,TE2)+np.cross(GE3,TE3)
  result=np.concatenate((resultant,total_moment))
  return result
#returns [Fx,Fy,Fz,Mx,My,Mz]
compute_thrust_forces_and_moments(np.array([[0,pi/2,5],
                                            [0,pi/2,5],
                                            [0,pi/2,5],
                                            ],dtype="float64"),0.5,7)