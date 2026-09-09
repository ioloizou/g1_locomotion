#include "g1_mpc/mpc.hpp"

#include <ros/console.h>

#include <chrono>
#include <cmath>
#include <limits>

namespace g1_mpc
{

MPC::MPC(const Parameters &params)
    : params_(params),
      dt_(params.dt),
      horizon_(params.horizon),
      inertia_body_(params.inertia),
      mass_(params.mass),
      q_weights_(params.q_weights),
      r_weight_(params.r_weight)
{
  num_vars_ = kNumControls * horizon_;
  num_constraints_ = kNumBounds * kNumContacts * horizon_;
  allocate();
  buildConstraintMatrix();
}

Eigen::Matrix3d MPC::skew(const Eigen::Vector3d &v)
{
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
      -v.y(), v.x(), 0.0;
  return m;
}

Eigen::Matrix3d MPC::rotZ(double yaw)
{
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  Eigen::Matrix3d R;
  R << c, -s, 0.0,
       s,  c, 0.0,
       0.0, 0.0, 1.0;
  return R;
}

void MPC::allocate()
{
  A_continuous_ = Eigen::MatrixXd::Zero(kNumStates, kNumStates);
  A_discrete_ = Eigen::MatrixXd::Zero(kNumStates, kNumStates);
  B_continuous_hor_ = Eigen::MatrixXd::Zero(kNumStates * horizon_, kNumControls);
  B_discrete_hor_ = Eigen::MatrixXd::Zero(kNumStates * horizon_, kNumControls);
  Aqp_ = Eigen::MatrixXd::Zero(kNumStates * horizon_, kNumStates);
  Bqp_ = Eigen::MatrixXd::Zero(kNumStates * horizon_, num_vars_);

  hessian_dense_ = Eigen::MatrixXd::Zero(num_vars_, num_vars_);
  gradient_ = Eigen::VectorXd::Zero(num_vars_);
  lower_ = Eigen::VectorXd::Zero(num_constraints_);
  upper_ = Eigen::VectorXd::Zero(num_constraints_);
  x_ref_vec_ = Eigen::VectorXd::Zero(kNumStates * horizon_);

  u_opt_ = Eigen::MatrixXd::Zero(horizon_, kNumControls);
  x_opt_ = Eigen::MatrixXd::Zero(horizon_, kNumStates);

  // The condensed Hessian is dense.  Build the sparsity pattern once and only
  // ever overwrite the values, so OSQP can be updated in place instead of
  // rebuilt: changing the pattern would force a full re-initialisation and
  // throw away the warm start.
  hessian_.resize(num_vars_, num_vars_);
  hessian_.reserve(Eigen::VectorXi::Constant(num_vars_, num_vars_));
  for (int col = 0; col < num_vars_; ++col)
  {
    for (int row = 0; row < num_vars_; ++row)
    {
      hessian_.insert(row, col) = 0.0;
    }
  }
  hessian_.makeCompressed();
}

void MPC::buildConstraintMatrix()
{
  // Friction pyramid per contact point, in world frame on flat ground:
  //   -inf <= fx - mu fz <= 0        (with the +mu row, |fx| <= mu fz)
  //      0 <= fx + mu fz <= inf
  //   likewise for fy
  //   fz_min <= fz <= fz_max, scaled by the contact flag
  const double mu = params_.friction_coefficient;
  Eigen::MatrixXd block(kNumBounds, 3);
  block << 1.0, 0.0, mu,
           1.0, 0.0, -mu,
           0.0, 1.0, mu,
           0.0, 1.0, -mu,
           0.0, 0.0, 1.0;

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(static_cast<size_t>(horizon_) * kNumContacts * 9);
  for (int k = 0; k < horizon_ * kNumContacts; ++k)
  {
    const int row0 = k * kNumBounds;
    const int col0 = k * 3;
    for (int i = 0; i < kNumBounds; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        if (block(i, j) != 0.0)
        {
          triplets.emplace_back(row0 + i, col0 + j, block(i, j));
        }
      }
    }
  }
  constraints_.resize(num_constraints_, num_vars_);
  constraints_.setFromTriplets(triplets.begin(), triplets.end());
  constraints_.makeCompressed();
}

void MPC::buildContinuousA(double yaw)
{
  A_continuous_.setZero();
  // Euler-rate map.  theta_dot = Rz(psi)^T w, NOT Rz(psi) w: the transpose is
  // silent at yaw = 0 and only shows up as a diverging attitude once the robot
  // turns.
  A_continuous_.block<3, 3>(0, 6) = rotZ(yaw).transpose();
  A_continuous_.block<3, 3>(3, 9) = Eigen::Matrix3d::Identity();
  // vz_dot picks up the constant gravity state.
  A_continuous_(11, 12) = 1.0;
}

void MPC::buildDiscreteA()
{
  // With the structure above A is nilpotent: A^3 == 0 exactly, so the
  // exponential series terminates and this *is* the exact zero-order-hold
  // discretisation, not a first-order approximation.
  const Eigen::MatrixXd A2 = A_continuous_ * A_continuous_;
  A_discrete_ = Eigen::MatrixXd::Identity(kNumStates, kNumStates) +
                A_continuous_ * dt_ + 0.5 * dt_ * dt_ * A2;
}

void MPC::buildContinuousB(const Eigen::MatrixXd &contact_pos,
                           const Eigen::MatrixXd &x_ref, double yaw)
{
  // World inertia is Rz * Ibody * Rz^T.  The other order is a different
  // matrix as soon as the robot is not facing along x, and it silently
  // mis-scales every angular term.
  const Eigen::Matrix3d Rz = rotZ(yaw);
  const Eigen::Matrix3d inertia_world = Rz * inertia_body_ * Rz.transpose();
  const Eigen::Matrix3d inertia_world_inv = inertia_world.inverse();
  const Eigen::Matrix3d mass_inv = Eigen::Matrix3d::Identity() / mass_;

  B_continuous_hor_.setZero();
  for (int k = 0; k < horizon_; ++k)
  {
    const Eigen::Vector3d com = x_ref.row(k).segment<3>(3);
    for (int c = 0; c < kNumContacts; ++c)
    {
      const Eigen::Vector3d point = contact_pos.row(k).segment<3>(3 * c);
      const Eigen::Vector3d r = point - com;
      B_continuous_hor_.block<3, 3>(k * kNumStates + 6, 3 * c) =
          inertia_world_inv * skew(r);
      B_continuous_hor_.block<3, 3>(k * kNumStates + 9, 3 * c) = mass_inv;
    }
  }
}

void MPC::buildDiscreteB()
{
  // Exact ZOH input matrix for the same nilpotent A:
  //   Bd = (I dt + A dt^2/2 + A^2 dt^3/6) B
  const Eigen::MatrixXd A2 = A_continuous_ * A_continuous_;
  const Eigen::MatrixXd integral =
      Eigen::MatrixXd::Identity(kNumStates, kNumStates) * dt_ +
      A_continuous_ * (0.5 * dt_ * dt_) + A2 * (dt_ * dt_ * dt_ / 6.0);

  for (int k = 0; k < horizon_; ++k)
  {
    B_discrete_hor_.block(k * kNumStates, 0, kNumStates, kNumControls) =
        integral * B_continuous_hor_.block(k * kNumStates, 0, kNumStates, kNumControls);
  }
}

void MPC::condense()
{
  for (int i = 0; i < horizon_; ++i)
  {
    if (i == 0)
    {
      Aqp_.block(0, 0, kNumStates, kNumStates) = A_discrete_;
    }
    else
    {
      Aqp_.block(i * kNumStates, 0, kNumStates, kNumStates) =
          Aqp_.block((i - 1) * kNumStates, 0, kNumStates, kNumStates) * A_discrete_;
    }
  }

  Bqp_.setZero();
  for (int i = 0; i < horizon_; ++i)
  {
    for (int j = 0; j <= i; ++j)
    {
      const auto Bj = B_discrete_hor_.block(j * kNumStates, 0, kNumStates, kNumControls);
      if (i == j)
      {
        Bqp_.block(i * kNumStates, j * kNumControls, kNumStates, kNumControls) = Bj;
      }
      else
      {
        Bqp_.block(i * kNumStates, j * kNumControls, kNumStates, kNumControls) =
            Aqp_.block((i - j - 1) * kNumStates, 0, kNumStates, kNumStates) * Bj;
      }
    }
  }
}

void MPC::buildBounds(const Eigen::MatrixXd &contact)
{
  const double inf = std::numeric_limits<double>::infinity();
  for (int k = 0; k < horizon_; ++k)
  {
    for (int c = 0; c < kNumContacts; ++c)
    {
      const int row = (k * kNumContacts + c) * kNumBounds;
      const double active = contact(k, c) > 0.5 ? 1.0 : 0.0;

      lower_(row + 0) = 0.0;   upper_(row + 0) = inf;   // fx + mu fz >= 0
      lower_(row + 1) = -inf;  upper_(row + 1) = 0.0;   // fx - mu fz <= 0
      lower_(row + 2) = 0.0;   upper_(row + 2) = inf;   // fy + mu fz >= 0
      lower_(row + 3) = -inf;  upper_(row + 3) = 0.0;   // fy - mu fz <= 0
      // A swing foot gets fz in [0, 0]; the friction rows then squeeze the
      // tangential components to zero too, so no separate handling is needed.
      lower_(row + 4) = params_.fz_min * active;
      upper_(row + 4) = params_.fz_max * active;
    }
  }
}

void MPC::buildCost(const Eigen::VectorXd &x0, const Eigen::MatrixXd &x_ref)
{
  for (int k = 0; k < horizon_; ++k)
  {
    x_ref_vec_.segment(k * kNumStates, kNumStates) = x_ref.row(k).transpose();
  }

  // H = 2 (Bqp' Q Bqp + R), g = 2 Bqp' Q (Aqp x0 - xref), with Q and R
  // diagonal so the products are scalings rather than matrix multiplies.
  Eigen::MatrixXd qbqp(kNumStates * horizon_, num_vars_);
  for (int k = 0; k < horizon_; ++k)
  {
    for (int s = 0; s < kNumStates; ++s)
    {
      qbqp.row(k * kNumStates + s) =
          q_weights_(s) * Bqp_.row(k * kNumStates + s);
    }
  }

  hessian_dense_.noalias() = 2.0 * (Bqp_.transpose() * qbqp);
  hessian_dense_.diagonal().array() += 2.0 * r_weight_;

  const Eigen::VectorXd error = Aqp_ * x0 - x_ref_vec_;
  Eigen::VectorXd weighted(kNumStates * horizon_);
  for (int k = 0; k < horizon_; ++k)
  {
    for (int s = 0; s < kNumStates; ++s)
    {
      weighted(k * kNumStates + s) = q_weights_(s) * error(k * kNumStates + s);
    }
  }
  gradient_.noalias() = 2.0 * (Bqp_.transpose() * weighted);

  refreshHessianValues();
}

void MPC::refreshHessianValues()
{
  // The pattern was built dense and column-major, so values map straight onto
  // the value array without touching the structure.
  double *values = hessian_.valuePtr();
  for (int col = 0; col < num_vars_; ++col)
  {
    for (int row = 0; row < num_vars_; ++row)
    {
      values[static_cast<size_t>(col) * num_vars_ + row] = hessian_dense_(row, col);
    }
  }
}

bool MPC::runSolver(Eigen::VectorXd &solution, int &iterations)
{
  if (!solver_ready_)
  {
    solver_ = std::unique_ptr<OsqpEigen::Solver>(new OsqpEigen::Solver());
    solver_->settings()->setVerbosity(false);
    solver_->settings()->setWarmStart(true);
    solver_->settings()->setMaxIteration(params_.osqp_max_iterations);
    solver_->settings()->setAbsoluteTolerance(params_.osqp_absolute_tolerance);
    solver_->settings()->setRelativeTolerance(params_.osqp_relative_tolerance);
    solver_->data()->setNumberOfVariables(num_vars_);
    solver_->data()->setNumberOfConstraints(num_constraints_);
    if (!solver_->data()->setHessianMatrix(hessian_) ||
        !solver_->data()->setGradient(gradient_) ||
        !solver_->data()->setLinearConstraintsMatrix(constraints_) ||
        !solver_->data()->setLowerBound(lower_) ||
        !solver_->data()->setUpperBound(upper_) ||
        !solver_->initSolver())
    {
      ROS_ERROR_THROTTLE(1.0, "g1_mpc: failed to initialise the QP solver");
      return false;
    }
    solver_ready_ = true;
  }
  else
  {
    // Reuse the factorisation and the warm start.  Rebuilding the solver every
    // tick, as an earlier version did, threw both away and made the warm-start
    // setting a no-op.
    if (!solver_->updateHessianMatrix(hessian_) ||
        !solver_->updateGradient(gradient_) ||
        !solver_->updateBounds(lower_, upper_))
    {
      ROS_ERROR_THROTTLE(1.0, "g1_mpc: failed to update the QP");
      solver_ready_ = false;
      return false;
    }
  }

  const OsqpEigen::ErrorExitFlag flag = solver_->solveProblem();
  if (flag != OsqpEigen::ErrorExitFlag::NoError)
  {
    ROS_WARN_THROTTLE(1.0, "g1_mpc: QP solver reported an error");
    return false;
  }

  const OsqpEigen::Status status = solver_->getStatus();
  const bool acceptable = status == OsqpEigen::Status::Solved ||
                          status == OsqpEigen::Status::SolvedInaccurate ||
                          status == OsqpEigen::Status::MaxIterReached;
  if (!acceptable)
  {
    ROS_WARN_THROTTLE(1.0, "g1_mpc: QP not solved (status %d)",
                      static_cast<int>(status));
    return false;
  }

  solution = solver_->getSolution();
  iterations = static_cast<int>(solver_->workspace()->info->iter);
  return solution.allFinite();
}

Eigen::MatrixXd MPC::fallbackForces(const Eigen::MatrixXd &contact) const
{
  // Hold the robot up with an even split across whatever is down.  Never a
  // good plan, but far better than publishing whatever the failed solve left
  // in the buffer.
  Eigen::MatrixXd u = Eigen::MatrixXd::Zero(horizon_, kNumControls);
  const double weight = mass_ * std::abs(params_.gravity);
  for (int k = 0; k < horizon_; ++k)
  {
    int active = 0;
    for (int c = 0; c < kNumContacts; ++c)
    {
      active += contact(k, c) > 0.5 ? 1 : 0;
    }
    if (active == 0)
    {
      continue;
    }
    for (int c = 0; c < kNumContacts; ++c)
    {
      if (contact(k, c) > 0.5)
      {
        u(k, 3 * c + 2) = weight / active;
      }
    }
  }
  return u;
}

void MPC::rollout(const Eigen::VectorXd &x0)
{
  x_opt_.row(0) = x0.transpose();
  for (int k = 1; k < horizon_; ++k)
  {
    x_opt_.row(k) =
        (A_discrete_ * x_opt_.row(k - 1).transpose() +
         B_discrete_hor_.block((k - 1) * kNumStates, 0, kNumStates, kNumControls) *
             u_opt_.row(k - 1).transpose())
            .transpose();
  }
}

MPC::Result MPC::solve(const Eigen::VectorXd &x0,
                       const Eigen::MatrixXd &x_ref,
                       const Eigen::MatrixXd &contact,
                       const Eigen::MatrixXd &contact_pos)
{
  const auto start = std::chrono::high_resolution_clock::now();
  Result result;

  // Linearise about the *measured* yaw.  Using the reference yaw makes the
  // model disagree with the plant exactly when the robot is off-heading, which
  // is when the controller matters most.
  const double yaw = x0(2);

  buildContinuousA(yaw);
  buildDiscreteA();
  buildContinuousB(contact_pos, x_ref, yaw);
  buildDiscreteB();
  condense();
  buildBounds(contact);
  buildCost(x0, x_ref);

  Eigen::VectorXd solution;
  int iterations = 0;
  if (runSolver(solution, iterations))
  {
    for (int k = 0; k < horizon_; ++k)
    {
      u_opt_.row(k) = solution.segment(k * kNumControls, kNumControls).transpose();
    }
    result.solved = true;
  }
  else
  {
    u_opt_ = fallbackForces(contact);
    result.solved = false;
  }

  rollout(x0);

  result.u_opt = u_opt_;
  result.x_opt = x_opt_;
  result.iterations = iterations;
  result.solve_time =
      std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start)
          .count();
  return result;
}

} // namespace g1_mpc
