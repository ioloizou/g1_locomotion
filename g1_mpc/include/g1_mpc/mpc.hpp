#pragma once

#include "g1_mpc/parameters.hpp"

#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <memory>

namespace g1_mpc
{

/// Convex single-rigid-body MPC over ground reaction forces.
///
/// State (13):  [roll pitch yaw | x y z | wx wy wz | vx vy vz | g]
/// Input (12):  [f_Lheel f_Ltoe f_Rheel f_Rtoe], each xyz, world frame.
///
/// The last state is a constant gravity augmentation so the affine gravity
/// term lives inside the linear A matrix and the problem stays a plain QP.
/// Its Q weight is zero.
///
/// Contact-point order is fixed everywhere in this stack:
/// L-heel, L-toe, R-heel, R-toe.
///
/// Linearised continuous dynamics, about the measured yaw, small roll/pitch,
/// dropping the gyroscopic w x Iw term:
///
///     theta_dot = Rz(psi)^T w
///     p_dot     = v
///     w_dot     = Iw^-1 sum_i [r_i]x f_i     with Iw = Rz Ibody Rz^T
///     v_dot     = (sum_i f_i) / m + [0, 0, g]
class MPC
{
public:
  static constexpr int kNumStates = 13;
  static constexpr int kNumContacts = 4;
  static constexpr int kNumControls = 3 * kNumContacts;
  /// Friction pyramid rows per contact: +-fx, +-fy against mu*fz, plus fz.
  static constexpr int kNumBounds = 5;

  explicit MPC(const Parameters &params);

  struct Result
  {
    bool solved = false;
    Eigen::MatrixXd u_opt;  ///< horizon x 12
    Eigen::MatrixXd x_opt;  ///< horizon x 13
    double solve_time = 0.0;
    int iterations = 0;
  };

  /// Solve one instance.
  /// @param x0          current state (13)
  /// @param x_ref       reference trajectory, horizon x 13
  /// @param contact     contact flags, horizon x 4, entries 0 or 1
  /// @param contact_pos world contact-point positions, horizon x 12
  Result solve(const Eigen::VectorXd &x0,
               const Eigen::MatrixXd &x_ref,
               const Eigen::MatrixXd &contact,
               const Eigen::MatrixXd &contact_pos);

  int horizon() const { return horizon_; }
  double dt() const { return dt_; }

private:
  void allocate();
  void buildConstraintMatrix();
  void buildContinuousA(double yaw);
  void buildDiscreteA();
  void buildContinuousB(const Eigen::MatrixXd &contact_pos,
                        const Eigen::MatrixXd &x_ref, double yaw);
  void buildDiscreteB();
  void condense();
  void buildBounds(const Eigen::MatrixXd &contact);
  void buildCost(const Eigen::VectorXd &x0, const Eigen::MatrixXd &x_ref);
  void refreshHessianValues();
  bool runSolver(Eigen::VectorXd &solution, int &iterations);
  /// Static weight split across whatever is in contact, used when the QP fails.
  Eigen::MatrixXd fallbackForces(const Eigen::MatrixXd &contact) const;
  void rollout(const Eigen::VectorXd &x0);

  static Eigen::Matrix3d skew(const Eigen::Vector3d &v);
  static Eigen::Matrix3d rotZ(double yaw);

  Parameters params_;
  double dt_;
  int horizon_;
  int num_vars_;
  int num_constraints_;

  Eigen::Matrix3d inertia_body_;
  double mass_;

  Eigen::MatrixXd A_continuous_, A_discrete_;
  Eigen::MatrixXd B_continuous_hor_, B_discrete_hor_;
  Eigen::MatrixXd Aqp_, Bqp_;
  Eigen::VectorXd q_weights_;
  double r_weight_;

  Eigen::MatrixXd hessian_dense_;
  Eigen::SparseMatrix<double> hessian_;
  Eigen::SparseMatrix<double> constraints_;
  Eigen::VectorXd gradient_, lower_, upper_, x_ref_vec_;

  Eigen::MatrixXd u_opt_, x_opt_;

  std::unique_ptr<OsqpEigen::Solver> solver_;
  bool solver_ready_ = false;
};

} // namespace g1_mpc
