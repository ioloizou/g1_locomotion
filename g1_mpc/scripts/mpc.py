import osqp
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
import time



# MPC class
class MPC:
  # Constructor
  def __init__(self, mu=0.3, fz_min = 10., fz_max = 666., dt = 0.04, HORIZON_LENGTH = 15):
    # Initialize the MPC class
    self.g = -9.80665 # m/s^2 Gravity
    self.ROBOT_MASS = 34.13385728 # kg
    self.dt = dt # seconds 
    self.NUM_CONTACTS = 4 # Number of contacts
    self.HORIZON_LENGTH = HORIZON_LENGTH # Number of nodes in the MPC horizon
    self.NUM_STATES = 13  
    self.NUM_CONTROLS = 3*self.NUM_CONTACTS
    self.NUM_BOUNDS = 5
    
    # Inertia matrix of the G1 through pinocchio in home configuration
    self.INERTIA_BODY = np.array([
      [3.20564,    4.35027e-05,  0.425526],
      [4.35027e-05, 3.05015,     -0.00065082],
      [0.425526,   -0.00065082,  0.552353]
    ]) # kg*m^2
    
    # x = [roll, pitch, yaw, x, y, z, wx, wy, wz, vx, vy, vz, g]
    self.x = np.zeros((self.NUM_STATES, 1))
    
    # u = [f_left_x, f_left_y, f_left_z, f_right_x, f_right_y, f_right_z]
    self.u = np.zeros((self.NUM_CONTROLS, 1))
    
    # The last weight is for the gravity term
    # [roll, pitch, yaw, x, y, z, wx, wy, wz, vx, vy, vz, g]
    
    # Walking in place weights
    # self.Q_WEIGHTS = np.diag([2e3, 9e3, 5e2, 
    #                           3e3, 2e4, 2e4, 
    #                           5e2, 5e2, 1e1, 
    #                           1e1, 9e2, 1e1, 0])
    # self.R_WEIGHTS = np.diag(np.repeat([0.001, 0.001, 0.001], self.NUM_CONTACTS))
    
    # # Standing double support weights
    self.Q_WEIGHTS = np.diag([7e5, 7e4, 1e4, 
                              5e5, 5e5, 3e6, 
                              3e3, 3e3, 3e3, 
                              5e3, 1e3, 1e4, 0])
    self.R_WEIGHTS = np.diag(np.repeat([0.001, 0.001, 0.001], self.NUM_CONTACTS))
    
    # # # Whole body weights
    # self.Q_WEIGHTS = np.diag([4e4, 5e4, 1e4, 
    #                           1e6, 5e5, 3e6, 
    #                           3e1, 3e2, 3e1, 
    #                           5e3, 1e3, 1e4, 0])
    # self.R_WEIGHTS = np.diag(np.repeat([0.1, 0.1, 0.1], self.NUM_CONTACTS))  

    # # MIT humanoid Weights
    # self.Q_WEIGHTS = np.diag([75e1, 75e0, 125e1, 
    #                           8e2, 2e3, 3e4, 
    #                           8e2, 2e3, 3e4, 
    #                           5e2, 5e3, 5e2, 0])
    # self.R_WEIGHTS = np.diag(np.repeat([0.01, 0.01, 0.1], self.NUM_CONTACTS))
    
    # Testing weights
    # self.Q_WEIGHTS = np.diag([75e1, 75e1, 125e1, 
    #                           2e2, 2e3, 2e4, 
    #                           3e2, 3e3, 3e4, 
    #                           5e2, 5e3, 5e2, 0])
    # self.R_WEIGHTS = np.diag(np.repeat([0.001, 0.001, 0.001], self.NUM_CONTACTS))
    

    self.mu = mu # Coefficient of friction
    self.fz_min = fz_min # Newton, Minimum normal force
    self.fz_max = fz_max # Newton, Maximum normal force


  def init_matrices(self):
    # Initialize the matrices to speed up the online computation
    self.A_continuous = np.zeros((self.NUM_STATES, self.NUM_STATES))
    self.A_discrete = np.zeros((self.NUM_STATES, self.NUM_STATES))
    self.Aqp = np.zeros((self.NUM_STATES * self.HORIZON_LENGTH, self.NUM_STATES))
    
    self.B_continuous = np.zeros((self.NUM_STATES, self.NUM_CONTROLS))
    self.B_continuous_hor= np.zeros((self.NUM_STATES * self.HORIZON_LENGTH, self.NUM_CONTROLS))
    self.B_discrete = np.zeros((self.NUM_STATES, self.NUM_CONTROLS))
    self.B_discrete_hor = np.zeros((self.NUM_STATES * self.HORIZON_LENGTH, self.NUM_CONTROLS))
    self.Bqp = np.zeros((self.NUM_STATES * self.HORIZON_LENGTH, self.NUM_CONTROLS * self.HORIZON_LENGTH))
    
    self.Q = np.zeros((self.NUM_STATES*self.HORIZON_LENGTH, self.NUM_STATES*self.HORIZON_LENGTH))

    self.c_horizon = np.zeros((self.HORIZON_LENGTH, self.NUM_CONTACTS*3))
    self.p_com_horizon = np.zeros((self.HORIZON_LENGTH, self.NUM_STATES))
    
    self.R = np.zeros((self.NUM_CONTROLS*self.HORIZON_LENGTH, self.NUM_CONTROLS*self.HORIZON_LENGTH))
    self.Ac = np.zeros((5 * self.NUM_CONTACTS * self.HORIZON_LENGTH, self.NUM_CONTROLS * self.HORIZON_LENGTH))
    
    self.Hessian = self.Bqp.T @ self.Q @ self.Bqp + self.R
    self.rotation_z_T = np.zeros((3, 3))
    self.x0 = np.zeros((self.NUM_STATES, 1))
    self.x_ref_hor = np.zeros((self.HORIZON_LENGTH, self.NUM_STATES))
    self.u_ref_hor = np.zeros((self.HORIZON_LENGTH, self.NUM_CONTROLS))
    self.u_opt = np.zeros((self.HORIZON_LENGTH, self.NUM_CONTROLS))
    self.x_opt = np.zeros((self.HORIZON_LENGTH, self.NUM_STATES))
    self.u_opt0 = np.zeros((self.NUM_CONTROLS, 1))

  def extract_psi(self):
    # Extract the average yaw angle from the reference trajectory
    yaw_sum = 0
    for i in range(self.HORIZON_LENGTH):
      yaw_sum += self.x_ref_hor[i, 2]
    self.psi = yaw_sum / self.HORIZON_LENGTH
    # print("yaw sum:", yaw_sum)
    return self.psi

  def vector_to_skew_symmetric_matrix(self, v):
    # Convert a vector to a skew symmetric matrix
    return np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])
  
  def rotation_matrix_T(self):
    # Rotation matrix
    self.rotation_z_T = np.array([[np.cos(self.psi), np.sin(self.psi), 0], 
                                [-np.sin(self.psi), np.cos(self.psi), 0], 
                                [0, 0, 1]])

  def set_Q(self):
    for i in range(self.HORIZON_LENGTH):
      self.Q[i*self.NUM_STATES:(i+1)*self.NUM_STATES, i*self.NUM_STATES:(i+1)*self.NUM_STATES] = self. Q_WEIGHTS.copy()
    # print("Q: \n", self.Q)

  def set_R(self):
    for i in range(self.HORIZON_LENGTH):
      self.R[i*self.NUM_CONTROLS:(i+1)*self.NUM_CONTROLS, i*self.NUM_CONTROLS:(i+1)*self.NUM_CONTROLS] = self.R_WEIGHTS.copy()
    # print("R: \n", self.R)
  def calculate_A_continuous(self):
    # Calculate the continuous A matrix
    
    # Populate the A matrix
    self.A_continuous[0:3, 6:9] = self.rotation_z_T.T
    self.A_continuous[3:6, 9:12] = np.eye(3)
    # The augmented gravity part
    self.A_continuous[11, 12] = 1

    # Assert the dimensions of the A matrix
    assert self.A_continuous.shape == (self.NUM_STATES, self.NUM_STATES)
    
    # print("A_continuous: \n", self.A_continuous)
    # print("A_continuous Shape: \n", self.A_continuous.shape)
  
  def calculate_A_discrete(self):
    # Calculate the discrete A matrix
    self.A_discrete = np.eye(13) + self.A_continuous * self.dt
    
    # print("A_discrete: \n", self.A_discrete)
    # print("A_discrete Shape: \n", self.A_discrete.shape)

    assert self.A_discrete.shape == (self.NUM_STATES, self.NUM_STATES)
    
  def calculate_B_continuous(self, c_horizon, p_com_horizon):

    self.c_horizon = np.array(c_horizon)
    self.p_com_horizon = np.array(p_com_horizon)
   
    INERTIA_WORLD = self.rotation_z_T @ self.INERTIA_BODY @ self.rotation_z_T.T

    for i in range(self.HORIZON_LENGTH):
      for j in range(self.NUM_CONTACTS):
        # Vector from the center of mass to the contact point

        # print("c_horizon: \n", c_horizon[i, 3*j:3*j+3]) 
        # print("p_com_horizon: \n", p_com_horizon[i, :])
        self.r = self.c_horizon[i, 3*j:3*j+3] - self.p_com_horizon[i, :]
        # print("r: \n", r)
        r_skew = self.vector_to_skew_symmetric_matrix(self.r)
        self.B_continuous[6:9, 3*j:(3*j+3)] = np.linalg.inv(INERTIA_WORLD) @ r_skew
        self.B_continuous[9:12, 3*j:(3*j+3)] = np.eye(3)/self.ROBOT_MASS
        assert self.B_continuous.shape == (self.NUM_STATES, self.NUM_CONTROLS)

      self.B_continuous_hor[i*self.NUM_STATES:(i+1)*self.NUM_STATES, 0:self.NUM_CONTROLS] = self.B_continuous
    assert self.B_continuous_hor.shape == (self.NUM_STATES * self.HORIZON_LENGTH, self.NUM_CONTROLS)   

    # print("B_continuous_hor: \n", self.B_continuous_hor)

  def calculate_B_discrete(self):
    for i in range(self.HORIZON_LENGTH):
      current_B = self.B_continuous_hor[i*self.NUM_STATES:(i+1)*self.NUM_STATES, 0:self.NUM_CONTROLS]
      self.B_discrete = current_B * self.dt
      self.B_discrete_hor[i*self.NUM_STATES:(i+1)*self.NUM_STATES, 0:self.NUM_CONTROLS] = self.B_discrete.copy()
    
    # self.B_discrete_hor = self.B_continuous_hor * self.dt
    assert self.B_discrete.shape == (self.NUM_STATES, self.NUM_CONTROLS)

  def calculate_Aqp(self):
    for i in range(self.HORIZON_LENGTH):
      if i == 0:
        self.Aqp[0:self.NUM_STATES, 0:self.NUM_STATES] = self.A_discrete.copy()
      else:
        self.Aqp[i*self.NUM_STATES:(i+1)*self.NUM_STATES, 0:self.NUM_STATES] = self.Aqp[(i-1)*self.NUM_STATES:i*self.NUM_STATES, 0:self.NUM_STATES] @ self.A_discrete

    assert self.Aqp.shape == (self.NUM_STATES * self.HORIZON_LENGTH, self.NUM_STATES)
  
  def calculate_Bqp(self):
    for i in range(self.HORIZON_LENGTH):
      for j in range(i + 1):
          if i == j:
            self.Bqp[i*self.NUM_STATES:(i+1)*self.NUM_STATES, j*self.NUM_CONTROLS:(j+1)*self.NUM_CONTROLS] = self.B_discrete_hor[j*self.NUM_STATES:(j+1)*self.NUM_STATES, 0:self.NUM_CONTROLS]
          else:
            self.Bqp[i*self.NUM_STATES:(i+1)*self.NUM_STATES, j*self.NUM_CONTROLS:(j+1)*self.NUM_CONTROLS] = \
            self.Aqp[(i-j-1)*self.NUM_STATES:(i-j)*self.NUM_STATES, 0:self.NUM_STATES] @ self.B_discrete_hor[j*self.NUM_STATES:(j+1)*self.NUM_STATES, 0:self.NUM_CONTROLS]
    
    # np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=2, suppress=True)
    # print("Bqp: \n", self.Bqp[:, 60:])
    # print("Bqp shape: \n", self.Bqp.shape)



    # if np.allclose(self.Bqp, np.tril(self.Bqp)):
    #   print("Bqp is lower triangular.")
    # else:
    #   print("Bqp is not lower triangular.")
    
    assert self.Bqp.shape == (self.NUM_STATES * self.HORIZON_LENGTH, self.NUM_CONTROLS * self.HORIZON_LENGTH)

  def calculate_Ac(self):
      # Define the block for one contact force (5 rows x 3 columns)
      g_block = np.array([
          [1, 0, self.mu],
          [1, 0, -self.mu],
          [0, 1, self.mu],
          [0, 1, -self.mu],
          [0, 0, 1]
      ])

      # Determine the number of blocks over the horizon for all legs
      num_blocks = self.NUM_CONTACTS * self.HORIZON_LENGTH

      # Create a list of blocks to form a block-diagonal matrix
      blocks = [g_block for _ in range(num_blocks)]
      
      # Each block is (NUM_BOUNDS x 3) so the overall Ac should be (NUM_BOUNDS*num_blocks x 3*num_blocks)
      self.Ac = sparse.block_diag(blocks, format='csr')

      expected_rows = self.NUM_BOUNDS * num_blocks
      expected_cols = 3 * num_blocks
      
      # np.set_printoptions(threshold=np.inf)
      # np.set_printoptions(linewidth=np.inf)
      # print(self.Ac.todense())
      # print("Ac shape: \n", self.Ac.shape)
      # exit()
      
      assert self.Ac.shape == (expected_rows, expected_cols)

  def calculate_bounds(self, contact):
    lower_bounds_list = []
    upper_bounds_list = []
    
    for horizon_step in range(self.HORIZON_LENGTH):
        for leg in range(self.NUM_CONTACTS):
            contact_active = contact[horizon_step][leg]
            # Lower bounds: [fx + mu*fz, fx - mu*fz, fy + mu*fz, fy - mu*fz, fz_min]
            lower_bounds_list.extend([
                0,                      # fx + mu*fz lower bound
                -np.inf,                # fx - mu*fz lower bound
                0,                      # fy + mu*fz lower bound
                -np.inf,                # fy - mu*fz lower bound
                self.fz_min * contact_active
            ])
            # Upper bounds: [fx + mu*fz, fx - mu*fz, fy + mu*fz, fy - mu*fz, fz_max]
            upper_bounds_list.extend([
                np.inf,                 # fx + mu*fz upper bound
                0,                      # fx - mu*fz upper bound
                np.inf,                 # fy + mu*fz upper bound
                0,                      # fy - mu*fz upper bound
                self.fz_max * contact_active
            ])
    
    self.lower_bounds_horizon = np.array(lower_bounds_list)
    self.upper_bounds_horizon = np.array(upper_bounds_list)

    # # Print shapes of the bounds
    # print("Lower bounds shape: \n", self.lower_bounds_horizon.shape)
    # print("Upper bounds shape: \n", self.upper_bounds_horizon.shape)

    # print("Lower bounds: \n", self.lower_bounds_horizon)
    # print("Upper bounds: \n", self.upper_bounds_horizon)
    # # Show the whole matrix
    # np.set_printoptions(threshold=np.inf)
    # np.set_printoptions(linewidth=np.inf)
    
    expected_length = self.HORIZON_LENGTH * self.NUM_CONTACTS * self.NUM_BOUNDS

    assert self.lower_bounds_horizon.shape == (expected_length,)
    assert self.upper_bounds_horizon.shape == (expected_length,)

  def calculate_hessian(self):
    self.Hessian = self.Bqp.T @ self.Q @ self.Bqp + self.R
  
  def calculate_gradient(self):
    # Print the shape of all matrices and vectors
    # print("Aqp: \n", self.Aqp.shape)
    # print("Bqp: \n", self.Bqp.shape)
    # print("Q: \n", self.Q.shape)
    # print("R: \n", self.R.shape)
    
    # Create a flattened version for gradient computation while preserving x_ref_hor's original shape
    self.x_ref_hor_vec = self.x_ref_hor.reshape((self.NUM_STATES * self.HORIZON_LENGTH, 1))
    self.x0 = self.x0.reshape((self.NUM_STATES, 1)) # If i dont reshape i have math problem
    # print("Gradient x0: \n", self.x0)
    # print("x_ref_hor: \n", self.x_ref_hor.shape)
    

    self.gradient = self.Bqp.T @ self.Q @ ((self.Aqp @ self.x0) - self.x_ref_hor_vec) 
    # + self.R @ self.*self.HORIZON_LENGTHu_ref

    # print(((self.Aqp@self.x0)).shape)
   

  def solve_qp(self):
    # Convert problem data to sparse matrices
    P = sparse.csc_matrix(self.Hessian)
    A = sparse.csc_matrix(self.Ac)
    q = self.gradient
    l = self.lower_bounds_horizon
    u = self.upper_bounds_horizon

    # print("P: \n", P.shape)
    # print("A: \n", A.shape)
    # print("q: \n", q.shape)
    # print("l: \n", l.shape)
    # print("u: \n", u.shape)

    t0 = time.time()

    # # Initialize the OSQP solver if it hasn't been created yet
    # if not hasattr(self, "qp_solver"):
    self.qp_solver = osqp.OSQP()
    self.qp_solver.setup(P, q, A, l, u, verbose=False, warm_start=True)
    # else:
    #   # Update the OSQP problem data with new values
    #   self.qp_solver.update(q=q, l=l, u=u, Px=P.data, Ax=A.data)

    t1 = time.time()
    result = self.qp_solver.solve()
    t2 = time.time()

    # Three values left leg then three values right leg. Then next horizon step and again the same
    self.u_opt = result.x.reshape((self.HORIZON_LENGTH, self.NUM_CONTROLS))
    # print("Optimal solution: \n", self.u_opt)

    # print("Solver init time: {:.2f}ms".format((t1-t0)*1000))
    # print("Solve time: {:.2f}ms".format((t2-t1)*1000))

    return result

  def update(self, contact, c_horizon, p_com_horizon, x_current=None, one_rollout=False):
    self.extract_psi()
    self.rotation_matrix_T()
    self.set_Q()
    self.set_R()
    self.calculate_A_continuous()
    self.calculate_A_discrete()
    self.calculate_B_continuous(c_horizon, p_com_horizon)
    self.calculate_B_discrete()
    self.calculate_Aqp()
    self.calculate_Bqp()
    self.calculate_Ac()
    self.calculate_bounds(contact)
    self.calculate_hessian()
    self.calculate_gradient()
    self.solve_qp()
    self.compute_rollout(x_current, one_rollout)
    return self.u_opt, self.x_opt

  def compute_rollout(self, x_current=None, only_first_step=False):
    # This overeload is to use with the whole body
    if x_current is not None:      
      self.x_opt[0, :] = x_current.reshape((self.NUM_STATES,))
    
    for i in range(1, self.HORIZON_LENGTH):    
      self.x_opt[i, :] = self.A_discrete @ self.x_opt[i-1, :] + self.B_discrete_hor[(i-1)*self.NUM_STATES:i*self.NUM_STATES, 0:self.NUM_CONTROLS] @ self.u_opt[i-1, :].T
      if only_first_step:
        return

    print("x_opt: \n", self.x_opt)
    print("u_opt: \n", self.u_opt)
    print("Total force on the robot: \n", np.sum(self.u_opt, axis=1))