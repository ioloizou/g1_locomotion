#pragma once

#include <Eigen/Dense>
#include <ros/ros.h>

#include <string>
#include <vector>

namespace g1_mpc
{

/// Every tunable number the MPC node needs, read once from the parameter
/// server (populated from config/g1_locomotion.yaml by the launch file).
///
/// Loading is strict on purpose: a missing key is a configuration error, not
/// something to paper over with a built-in default that then silently
/// disagrees with the value the simulation is using.
struct Parameters
{
  // --- robot ---------------------------------------------------------------
  double mass = 0.0;
  Eigen::Matrix3d inertia = Eigen::Matrix3d::Identity();
  double com_height = 0.0;
  double foot_heel_x = 0.0;
  double foot_toe_x = 0.0;
  double stance_width = 0.0;
  double foot_offset_x = 0.0;

  // --- physics -------------------------------------------------------------
  double gravity = -9.80665;
  double friction_coefficient = 0.6;
  double fz_min = 0.0;
  double fz_max = 0.0;

  // --- gait ----------------------------------------------------------------
  int period_ticks = 20;
  int stance_ticks = 13;
  int right_offset_ticks = 10;
  double stand_still_velocity = 0.02;
  double stand_still_yaw_rate = 0.05;
  double swing_height = 0.06;
  /// How far the CoM reference leans towards the stance foot during single
  /// support, as a fraction of the offset to the support centroid.
  double sway_gain = 0.6;
  /// Seconds a walk command must persist before the first step.
  double startup_delay = 1.0;

  // --- footstep ------------------------------------------------------------
  bool use_capture_point_gain = true;
  double velocity_gain = 0.1;
  double max_step_x = 0.25;
  double max_step_y = 0.12;
  double min_lateral_offset = 0.09;
  double max_lateral_offset = 0.30;
  double max_feedback = 0.15;

  // --- mpc -----------------------------------------------------------------
  double dt = 0.04;
  int horizon = 12;
  Eigen::VectorXd q_weights;
  double r_weight = 1e-3;
  double max_position_error = 0.10;
  double max_yaw_error = 0.20;
  int osqp_max_iterations = 200;
  double osqp_absolute_tolerance = 1e-4;
  double osqp_relative_tolerance = 1e-4;

  // --- command -------------------------------------------------------------
  double default_linear_x = 0.0;
  double default_linear_y = 0.0;
  double default_angular_z = 0.0;
  double max_linear_acceleration = 0.5;
  double max_angular_acceleration = 1.0;
  double max_linear_velocity = 0.6;
  double max_angular_velocity = 0.6;

  /// Read every field from `nh`.  Throws std::runtime_error naming the first
  /// key that is missing or malformed.
  static Parameters load(const ros::NodeHandle &nh);

  /// Read the same fields straight from config/g1_locomotion.yaml, for tools
  /// that run without a ROS master.  Keeping this in step with the file the
  /// nodes actually use is the whole point: a second copy of the numbers
  /// compiled into a test drifts out of date and then tests the wrong thing.
  static Parameters loadFromFile(const std::string &path);

  /// Duration of one stance phase, used by the footstep heuristic.
  double stanceDuration() const { return stance_ticks * dt; }
  double swingDuration() const { return (period_ticks - stance_ticks) * dt; }
};

} // namespace g1_mpc
