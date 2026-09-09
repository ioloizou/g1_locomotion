#include "g1_mpc/footstep_planner.hpp"

#include <algorithm>
#include <cmath>

namespace g1_mpc
{

FootstepPlanner::FootstepPlanner(const Parameters &params)
    : params_(params), half_stance_width_(0.5 * params.stance_width)
{
  velocity_gain_ = params.use_capture_point_gain
                       ? std::sqrt(params.com_height / std::abs(params.gravity))
                       : params.velocity_gain;
}

Eigen::Matrix3d FootstepPlanner::rotZ(double yaw)
{
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  Eigen::Matrix3d R;
  R << c, -s, 0.0,
       s,  c, 0.0,
       0.0, 0.0, 1.0;
  return R;
}

double FootstepPlanner::clamp(double v, double lo, double hi)
{
  return std::max(lo, std::min(hi, v));
}

Eigen::Vector3d FootstepPlanner::nominalStance(int leg, const State &state) const
{
  const double side = (leg == LEG_LEFT) ? 1.0 : -1.0;
  Eigen::Vector3d offset(params_.foot_offset_x, side * half_stance_width_, 0.0);
  Eigen::Vector3d p = state.com_position + rotZ(state.yaw) * offset;
  p.z() = state.ground_height;
  return p;
}

Eigen::Vector3d FootstepPlanner::plan(int leg, const State &state,
                                      double time_to_touchdown) const
{
  const double side = (leg == LEG_LEFT) ? 1.0 : -1.0;

  // Heading and hip position predicted at the moment of touchdown.  Turning is
  // handled here: the nominal stance offset is expressed in the frame the
  // robot will be facing when the foot lands.
  const double yaw_td = state.yaw + state.yaw_rate * time_to_touchdown;
  const Eigen::Matrix3d R_td = rotZ(yaw_td);

  Eigen::Vector3d com_td = state.com_position + state.desired_velocity * time_to_touchdown;
  // Nominal foot placement relative to the CoM, in the heading frame. Using
  // the true stance offset rather than "directly under the CoM" means a zero
  // command reproduces the posture the robot was designed to stand in.
  const Eigen::Vector3d hip_offset(params_.foot_offset_x,
                                   side * half_stance_width_, 0.0);
  const Eigen::Vector3d hip_td = com_td + R_td * hip_offset;

  // Raibert, split into feed-forward and feedback.
  //
  // The half-stance travel term uses the *commanded* velocity, not the
  // measured one. A biped's lateral velocity swings back and forth every step
  // by design; feeding that oscillation through a term as large as half a
  // stance duration treats the normal sway as an error, widens the step, and
  // the lateral oscillation grows until the robot steps itself over.
  // Deviation from the command is handled by the feedback term below, which
  // is the capture-point gain and is an order of magnitude smaller.
  Eigen::Vector3d step = 0.5 * params_.stanceDuration() * state.desired_velocity;

  Eigen::Vector3d feedback =
      velocity_gain_ * (state.com_velocity - state.desired_velocity);

  // Centrifugal compensation: turning while translating shifts the capture
  // point sideways.
  const Eigen::Vector3d omega(0.0, 0.0, state.yaw_rate);
  feedback += 0.5 * std::sqrt(params_.com_height / std::abs(params_.gravity)) *
              state.com_velocity.cross(omega);

  feedback.z() = 0.0;
  const double feedback_norm = feedback.head<2>().norm();
  if (feedback_norm > params_.max_feedback)
  {
    feedback *= params_.max_feedback / feedback_norm;
  }

  Eigen::Vector3d target = hip_td + step + feedback;
  target.z() = state.ground_height;

  // Clamp the step relative to the predicted hip, in the heading frame.
  Eigen::Vector3d local = R_td.transpose() * (target - hip_td);
  local.x() = clamp(local.x(), -params_.max_step_x, params_.max_step_x);
  local.y() = clamp(local.y(), -params_.max_step_y, params_.max_step_y);
  target = hip_td + R_td * local;

  // Keep the foot outside a minimum lateral offset from the body centre line
  // so the swing leg cannot cross the stance leg.
  Eigen::Vector3d from_com = R_td.transpose() * (target - com_td);
  const double lateral = side * from_com.y();
  from_com.y() = side * clamp(lateral, params_.min_lateral_offset,
                              params_.max_lateral_offset);
  target = com_td + R_td * from_com;
  target.z() = state.ground_height;

  return target;
}

} // namespace g1_mpc
