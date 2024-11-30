import time
import numpy as np
import json

import mujoco
import mujoco.viewer
import pickle

from simple_pid import PID

from drone import drone


# -------------------------- Initialization ----------------------------------
slung_load = drone(target=np.array((0.6, 0.6, 1)),target_1=np.array((0.6, -0.6, 1)),target_2=np.array((-0.6, 0.6, 1)),target_3=np.array((-0.6, -0.6, 1)))

with mujoco.viewer.launch_passive(slung_load.m, slung_load.d) as viewer:
  time.sleep(2)
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  step = 1

  while viewer.is_running() and time.time() - start < 20:
    step_start = time.time()
    

    # outer control loop
    if step % 20 == 0:
      slung_load.update_outer_conrol()
    # Inner control loop
       
    slung_load.update_inner_control()	
    
    # flight program
    if time.time()- start > 4:
      #slung_load.planner.update_target(np.array((1.6, 0.6, 2.)),np.array((1.6, -0.6, 2.)),np.array((-1.6, 0.6, 2.)),np.array((-1.6, -0.6, 2.)))
      slung_load.planner.update_target(np.array((0.6, 0.6, 2)),np.array((0.6, -0.6, 2)),np.array((-0.6, 0.6, 2)),np.array((-0.6, -0.6, 2)))

    if time.time()- start > 8:
      #slung_load.planner.update_target(np.array((2.6, 0.6, 3)),np.array((2.6, -0.6, 3)),np.array((-2.6, 0.6, 3)),np.array((-2.6, -0.6, 3)))
      slung_load.planner.update_target(np.array((1.6, 0.6, 3)),np.array((1.6, -0.6, 3)),np.array((0.4, 0.6, 3)),np.array((0.4, -0.6, 3)))

    if time.time()- start > 12:
      slung_load.planner.update_target(np.array((1.6, 1.6, 3.)),np.array((1.6, 0.4, 3.)),np.array((0.4, 1.6, 3.)),np.array((0.4, 0.4, 3.)))
    
    if time.time()- start > 16:
      slung_load.planner.update_target(np.array((2.6, 1.6, 3.)),np.array((2.6, 0.4, 3.)),np.array((1.4, 1.6, 3.)),np.array((1.4, 0.4, 3.)))

    mujoco.mj_step(slung_load.m, slung_load.d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(slung_load.d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()
    
    # Increment to time slower outer control loop
    step += 1
    
    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = slung_load.m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
