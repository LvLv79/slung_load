import time
import numpy as np
import json

import mujoco
import mujoco.viewer
import pickle

from simple_pid import PID

class PositionPID:
  """Dummy sensor data. So the control code remains intact."""
  def __init__(self, target, vel_limit = 2) -> None:
    self.target = target
    self.pid_x = PID(2, 0.15, 1.5, setpoint = self.target[0],
                output_limits = (-vel_limit, vel_limit))
    self.pid_y = PID(2, 0.15, 1.5, setpoint = self.target[1],
                output_limits = (-vel_limit, vel_limit))

  def __call__(self,location):
    output = np.array([0,0])
    output[0] = self.pid_x(location[0])
    output[1] = self.pid_y(location[1])
    return output
  
  def update_target(self,target):
    self.pid_x.setpoint = target[0]
    self.pid_y.setpoint = target[1]
    self.target = target

class VelocityPID:
  """Dummy sensor data. So the control code remains intact."""
  def __init__(self, target) -> None:
    self.pid_v_x = PID(0.1, 0.003, 0.02, setpoint = 0,
                output_limits = (-0.1, 0.1))
    self.pid_v_y = PID(0.1, 0.003, 0.02, setpoint = 0,
                  output_limits = (-0.1, 0.1))
    self.target = target

  def __call__(self, v):
    output = np.array([0,0])
    # print("velocity:", v)
    a = self.pid_v_x(v[0])
    b = - self.pid_v_y(v[1])
    output = np.array((a, b))
    return output
  
  def update_target(self, target):
    self.pid_v_x.setpoint = target[0]
    self.pid_v_y.setpoint = target[1]

class AnglesPID:
  """Dummy sensor data. So the control code remains intact."""
  def __init__(self, target) -> None:
    self.pid_roll = PID(2.6785,0.56871, 1.2508, setpoint=0, output_limits = (-1,1) )
    self.pid_pitch = PID(2.6785,0.56871, 1.2508, setpoint=0, output_limits = (-1,1) )
    self.pid_yaw =  PID(0.54, 0, 5.358333, setpoint=1, output_limits = (-3,3) )
    self.target = target

  def __call__(self, angles):
    output = np.array([0,0])
    cmd_roll = - self.pid_roll(angles[1])
    cmd_pitch = self.pid_pitch(angles[2])
    cmd_yaw = - self.pid_yaw(angles[0])
    output = np.array((cmd_roll, cmd_pitch,cmd_yaw))
    return output
  
  def update_target(self, target):
    self.pid_pitch.setpoint= target[0]
    self.pid_roll.setpoint = target[1]

class PositionPlanner:

  def __init__(self, target, target_1, target_2, target_3, vel_limit = 2) -> None:
    # TODO: MPC
    self.target = target 
    self.target_1 = target_1
    self.target_2 = target_2
    self.target_3 = target_3 
    self.vel_limit = vel_limit
    # setpoint target location, controller output: desired velocity.
    self.pid_pos_0 = PositionPID(target)
    self.pid_pos_1 = PositionPID(target_1)
    self.pid_pos_2 = PositionPID(target_2)
    self.pid_pos_3 = PositionPID(target_3)
  
  def __call__(self, loc: np.array, loc_1: np.array, loc_2: np.array, loc_3: np.array):
    """Calls planner at timestep to update cmd_vel"""
    velocities = np.array([0,0,0,0,0,0,0,0])
    velocities_0 = self.pid_pos_0(loc)
    velocities_1 = self.pid_pos_1(loc_1)
    velocities_2 = self.pid_pos_2(loc_2)
    velocities_3 = self.pid_pos_3(loc_3)
    velocities = np.concatenate([velocities_0, velocities_1, velocities_2, velocities_3])
    return velocities
  
  def update_target(self, target, target_1, target_2, target_3):
    """Update targets"""
    self.target = target 
    self.target_1 = target_1
    self.target_2 = target_2
    self.target_3 = target_3  
    # setpoint target location, controller output: desired velocity.
    self.pid_pos_0.update_target(target)
    self.pid_pos_1.update_target(target_1)
    self.pid_pos_2.update_target(target_2)
    self.pid_pos_3.update_target(target_3)

  def get_alt_setpoint(self, loc: np.array, target_x) -> float:

    target = target_x
    distance = target[2] - loc[2]
    
    # maps drone velocities to one.
    if distance > 0.5:
        time_sample = 1/4
        time_to_target =  distance / self.vel_limit
        number_steps = int(time_to_target/time_sample)
        # compute distance for next update
        delta_alt = distance / number_steps

        # 2 times for smoothing
        alt_set = loc[2] + 2 * delta_alt
    
    else:
        alt_set = target[2]

    return alt_set
    
  def get_all_alt_setpoint(self,loc: np.array,loc_1: np.array,loc_2: np.array,loc_3: np.array):
    alt_set = np.array([0,0,0,0])
    alt_set[0] = self.get_alt_setpoint(loc,self.target)
    alt_set[1] = self.get_alt_setpoint(loc_1,self.target_1)
    alt_set[2] = self.get_alt_setpoint(loc_2,self.target_2)
    alt_set[3] = self.get_alt_setpoint(loc_3,self.target_3)
    return alt_set


class Sensor:
  """Dummy sensor data. So the control code remains intact."""
  def __init__(self, d):
    self.position = d.qpos
    self.velocity = d.qvel

  def get_position(self):
    return self.position
  
  def get_velocity(self):
    return self.velocity
    
class drone:
  """Simple drone classe."""
  def __init__(self, target=np.array((0,0,0)), target_1=np.array((0,0,0)), target_2=np.array((0,0,0)), target_3=np.array((0,0,0))):
    self.m = mujoco.MjModel.from_xml_path('mujoco_menagerie-main/skydio_x2/scene.xml')
    self.d = mujoco.MjData(self.m)

    self.planner = PositionPlanner(target=target, target_1=target_1, target_2=target_2, target_3=target_3)
    self.sensor = Sensor(self.d)

    # instantiate controllers

    # inner control to stabalize inflight dynamics
    self.pid_alt = PID(5.50844,0.57871, 1.2,setpoint=0)
    self.pid_angle = AnglesPID(0)
    
    self.pid_alt_1 = PID(5.50844,0.57871, 1.2,setpoint=0)
    self.pid_angle_1 = AnglesPID(0)
    
    self.pid_alt_2 = PID(5.50844,0.57871, 1.2,setpoint=0)
    self.pid_angle_2 = AnglesPID(0)
    
    self.pid_alt_3 = PID(5.50844,0.57871, 1.2,setpoint=0)
    self.pid_angle_3 = AnglesPID(0)

    # outer control loops
    self.pid_v = VelocityPID(0)
    self.pid_v_1 = VelocityPID(0)
    self.pid_v_2 = VelocityPID(0)
    self.pid_v_3 = VelocityPID(0)                                        
                  

  def update_outer_conrol(self):
    """Updates outer control loop for trajectory planning"""
    v = self.sensor.get_velocity()[0:6]
    location = self.sensor.get_position()[0:3]
    
    v_1 = self.sensor.get_velocity()[6:12]
    location_1 = self.sensor.get_position()[7:10]
    
    v_2 = self.sensor.get_velocity()[12:18]
    location_2 = self.sensor.get_position()[14:17]
    
    v_3 = self.sensor.get_velocity()[18:24]
    location_3 = self.sensor.get_position()[21:24]
    
    # Compute target velocities
    velocites_x2 = self.planner(loc=location,loc_1=location_1,loc_2=location_2,loc_3=location_3)
    velocities = velocites_x2[0:2]
    velocities_1 = velocites_x2[2:4]
    velocities_2 = velocites_x2[4:6]
    velocities_3 = velocites_x2[6:8]
    
    all_alt = self.planner.get_all_alt_setpoint(location,location_1,location_2,location_3)
    
    # In this example the altitude is directly controlled by a PID
    self.pid_alt.setpoint = all_alt[0]
    self.pid_v.update_target(velocities)
    
    self.pid_alt_1.setpoint = all_alt[1]
    self.pid_v_1.update_target(velocities_1)
    
    self.pid_alt_2.setpoint = all_alt[2]
    self.pid_v_2.update_target(velocities_2)
    
    self.pid_alt_3.setpoint = all_alt[3]
    self.pid_v_3.update_target(velocities_3)
    

    # Compute angles and set inner controllers accordingly
    angle_target = self.pid_v(v)
    self.pid_angle.update_target(angle_target)
    
    angle_target_1 = self.pid_v_1(v_1)
    self.pid_angle_1.update_target(angle_target_1)

    angle_target_2 = self.pid_v_2(v_2)
    self.pid_angle_2.update_target(angle_target_2)

    angle_target_3 = self.pid_v_3(v_3)
    self.pid_angle_3.update_target(angle_target_3)


  def update_inner_control(self):
    """Upates inner control loop and sets actuators to stabilize flight
    dynamics"""
    alt = self.sensor.get_position()[2]
    alt_1 = self.sensor.get_position()[9]
    alt_2 = self.sensor.get_position()[16]
    alt_3 = self.sensor.get_position()[23]
    angles = self.sensor.get_position()[3:7] # roll, yaw, pitch
    angles_1 = self.sensor.get_position()[10:14] # roll, yaw, pitch
    angles_2 = self.sensor.get_position()[17:21] # roll, yaw, pitch
    angles_3 = self.sensor.get_position()[24:28] # roll, yaw, pitch
    
    # apply PID
    cmd_thrust = self.pid_alt(alt) + 3.2495
    cmd_euler_angle = self.pid_angle(angles)
    cmd_roll = cmd_euler_angle[0]
    cmd_pitch = cmd_euler_angle[1]
    cmd_yaw = cmd_euler_angle[2]
    
    cmd_thrust_1 = self.pid_alt_1(alt_1) + 3.2495
    cmd_euler_angle_1 = self.pid_angle_1(angles_1)
    cmd_roll_1 = cmd_euler_angle_1[0]
    cmd_pitch_1 = cmd_euler_angle_1[1]
    cmd_yaw_1 = cmd_euler_angle_1[2]
    
    cmd_thrust_2 = self.pid_alt_2(alt_2) + 3.2495
    cmd_euler_angle_2 = self.pid_angle_2(angles_2)
    cmd_roll_2 = cmd_euler_angle_2[0]
    cmd_pitch_2 = cmd_euler_angle_2[1]
    cmd_yaw_2 = cmd_euler_angle_2[2]
    
    cmd_thrust_3 = self.pid_alt_3(alt_3) + 3.2495
    cmd_euler_angle_3 = self.pid_angle_3(angles_3)
    cmd_roll_3 = cmd_euler_angle_3[0]
    cmd_pitch_3 = cmd_euler_angle_3[1]
    cmd_yaw_3 = cmd_euler_angle_3[2]

    #transfer to motor control
    out_0 = self.compute_motor_control(cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw)
    out_1 = self.compute_motor_control(cmd_thrust_1, cmd_roll_1, cmd_pitch_1, cmd_yaw_1)
    out_2 = self.compute_motor_control(cmd_thrust_2, cmd_roll_2, cmd_pitch_2, cmd_yaw_2)
    out_3 = self.compute_motor_control(cmd_thrust_3, cmd_roll_3, cmd_pitch_3, cmd_yaw_3)
    
    self.d.ctrl[:16] = [
      out_0[0],
      out_0[1],
      out_0[2],
      out_0[3],
      out_1[0],
      out_1[1],
      out_1[2],
      out_1[3],
      out_2[0],
      out_2[1],
      out_2[2],
      out_2[3],
      out_3[0],
      out_3[1],
      out_3[2],
      out_3[3]
    ]

  #  as the drone is underactuated we set
  def compute_motor_control(self, thrust, roll, pitch, yaw):
    motor_control = [
      thrust + roll + pitch - yaw,
      thrust - roll + pitch + yaw,
      thrust - roll -  pitch - yaw,
      thrust + roll - pitch + yaw
    ]
    return motor_control