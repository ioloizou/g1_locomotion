import numpy as np

def PD_controller(data, q_desired, q_init):
  " A simple PD controller"

  # Ignoring floating base
  q_current = data.qpos[7:]
  q_desired = q_init[7:]
  dq_current = data.qvel[6:]
  dq_desired = np.zeros(dq_current.shape)

  # PID gains

  scale = 1.5 

  Kp = np.zeros(23)
  Kd = np.zeros(23)
  Kp[0:6] = [530.0, 570.0, 550.0, 270.0, 130.0, 30.0]
  Kd[0:6] = [60.0, 100.0, 2.0, 20.0, 100.5, 5.]

  Kp[6:12] = Kp[0:6]
  Kd[6:12] = Kd[0:6]

  Kp[12] = 150.0
  Kd[12] = 10.0

  Kp[13:17] = Kp[18:22] = 20.0
  Kp[17] = Kp[22] = 11.0

  Kd[13:17] = Kd[18:22] = 5.0
  Kd[17] = Kd[22] = 0.1

  # Compute feedforward torque
  tau_ff = 0

  tau = tau_ff + 1.5*scale*Kp * (q_desired - q_current) + scale/3*Kd * (dq_desired - dq_current)
  data.ctrl = tau