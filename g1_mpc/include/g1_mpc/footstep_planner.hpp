#pragma once

#include "g1_mpc/gait_scheduler.hpp"
#include "g1_mpc/parameters.hpp"

#include <Eigen/Dense>

namespace g1_mpc
{

/// Raibert-style footstep placement with hard clamps.
///
/// The target for a swing foot is the point where the hip will be at
/// touchdown, plus half a stance of body travel, plus a velocity-error
/// correction.  Everything is built in the yaw-aligned body frame and rotated
/// into the world at the end, which is what lets the robot turn: the nominal
/// stance offset follows the heading instead of staying pinned to the world y
/// axis.
///
/// Three clamps keep the result reachable and the gait safe:
///   * the feedback correction is capped on its own, so a velocity spike
///     cannot fling the foot away;
///   * the step is capped relative to the predicted hip position;
///   * the foot is held outside a minimum lateral offset from the body's
///     centre line, so the swing leg can never cross the stance leg.
class FootstepPlanner
{
public:
  explicit FootstepPlanner(const Parameters &params);

  struct State
  {
    Eigen::Vector3d com_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d com_velocity = Eigen::Vector3d::Zero();       ///< world frame
    Eigen::Vector3d desired_velocity = Eigen::Vector3d::Zero();   ///< world frame
    double yaw = 0.0;
    double yaw_rate = 0.0;
    double ground_height = 0.0;
  };

  /// World-frame landing position for the foot frame of `leg`, arriving
  /// `time_to_touchdown` seconds from now.
  Eigen::Vector3d plan(int leg, const State &state, double time_to_touchdown) const;

  /// Nominal (command-free) stance position, used while standing so the feet
  /// have a sensible target even when no step is scheduled.
  Eigen::Vector3d nominalStance(int leg, const State &state) const;

  /// Capture-point gain sqrt(h/|g|) or the configured constant.
  double velocityGain() const { return velocity_gain_; }

private:
  static Eigen::Matrix3d rotZ(double yaw);
  static double clamp(double v, double lo, double hi);

  Parameters params_;
  double velocity_gain_;
  double half_stance_width_;
};

} // namespace g1_mpc
