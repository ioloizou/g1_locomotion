/// Model-predictive controller node.
///
/// Owns the gait clock, the footstep plan and the convex MPC, and publishes a
/// complete plan the whole-body controller can follow: the state trajectory,
/// the ground reaction forces for the first step, and per-foot contact flags,
/// swing phase and landing targets.
///
/// The node is driven by the simulation clock and does exactly one solve per
/// MPC tick.  An earlier version re-solved in a free-running 400 Hz loop, which
/// both wasted the machine and decoupled the solve rate from the gait phase.

#include "g1_mpc/footstep_planner.hpp"
#include "g1_mpc/gait_scheduler.hpp"
#include "g1_mpc/mpc.hpp"
#include "g1_mpc/parameters.hpp"

#include <ros/ros.h>

#include <geometry_msgs/Twist.h>
#include <rosgraph_msgs/Clock.h>
#include <std_msgs/Float64.h>

#include "g1_msgs/ContactPoint.h"
#include "g1_msgs/FootPlan.h"
#include "g1_msgs/SRBD_state.h"
#include "g1_msgs/State.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

double clamp(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }

/// Move `current` towards `target` by at most `max_delta`.
double rateLimit(double current, double target, double max_delta)
{
  const double delta = clamp(target - current, -max_delta, max_delta);
  return current + delta;
}

Eigen::Matrix3d rotZ(double yaw)
{
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  Eigen::Matrix3d R;
  R << c, -s, 0.0,
       s,  c, 0.0,
       0.0, 0.0, 1.0;
  return R;
}

} // namespace

class MpcNode
{
public:
  MpcNode(const ros::NodeHandle &nh, const g1_mpc::Parameters &params)
      : nh_(nh),
        params_(params),
        mpc_(params),
        scheduler_(params),
        footstep_(params)
  {
    contact_names_ = {"left_foot_line_contact_lower", "left_foot_line_contact_upper",
                      "right_foot_line_contact_lower", "right_foot_line_contact_upper"};
    leg_names_ = {"left", "right"};

    // Start from rest and let the rate limiter ramp up to the default. Coming
    // up at full commanded speed makes the scheduler step on the first tick,
    // before the robot has shifted any weight onto a stance foot.
    command_.setZero();
    command_target_ << params_.default_linear_x, params_.default_linear_y,
        params_.default_angular_z;

    foot_position_[g1_mpc::LEG_LEFT].setZero();
    foot_position_[g1_mpc::LEG_RIGHT].setZero();
    landing_target_[g1_mpc::LEG_LEFT].setZero();
    landing_target_[g1_mpc::LEG_RIGHT].setZero();

    x0_ = Eigen::VectorXd::Zero(g1_mpc::MPC::kNumStates);
    measured_contacts_ = Eigen::MatrixXd::Zero(g1_mpc::MPC::kNumContacts, 3);

    pub_solution_ = nh_.advertise<g1_msgs::SRBD_state>("/mpc_solution", 1);
    pub_solve_time_ = nh_.advertise<std_msgs::Float64>("/mpc_solve_time", 1);
    sub_state_ = nh_.subscribe("/srbd_current", 1, &MpcNode::onState, this);
    sub_clock_ = nh_.subscribe("/simulation_time", 1, &MpcNode::onClock, this);
    sub_cmd_ = nh_.subscribe("/cmd_vel", 1, &MpcNode::onCommand, this);
  }

  void onState(const g1_msgs::SRBD_state::ConstPtr &msg)
  {
    if (msg->states_horizon.empty() ||
        msg->contacts.size() != static_cast<size_t>(g1_mpc::MPC::kNumContacts))
    {
      ROS_WARN_THROTTLE(1.0, "g1_mpc: malformed state message, ignoring");
      return;
    }

    const g1_msgs::State &s = msg->states_horizon.front();
    x0_ << s.orientation.x, s.orientation.y, s.orientation.z,
        s.position.x, s.position.y, s.position.z,
        s.angular_velocity.x, s.angular_velocity.y, s.angular_velocity.z,
        s.linear_velocity.x, s.linear_velocity.y, s.linear_velocity.z,
        s.gravity;

    for (int c = 0; c < g1_mpc::MPC::kNumContacts; ++c)
    {
      measured_contacts_.row(c) << msg->contacts[c].position.x,
          msg->contacts[c].position.y, msg->contacts[c].position.z;
    }

    // Recover the foot frame from the two contact points. The heel and toe are
    // not symmetric about it (heel -5 cm, toe +12 cm), so their midpoint sits
    // 3.5 cm ahead of the frame. Using the midpoint makes the MPC believe its
    // feet are further forward than they are, and every force distribution it
    // computes then puts the real centre of pressure behind the intended one,
    // which drives the robot forwards even when it is asked to stand still.
    const Eigen::Matrix3d R = rotZ(x0_(2));
    const Eigen::Vector3d midpoint_offset(
        0.5 * (params_.foot_heel_x + params_.foot_toe_x), 0.0, 0.0);
    for (int leg = 0; leg < g1_mpc::NUM_LEGS; ++leg)
    {
      const Eigen::Vector3d midpoint =
          0.5 * (measured_contacts_.row(2 * leg) + measured_contacts_.row(2 * leg + 1))
                    .transpose();
      foot_position_[leg] = midpoint - R * midpoint_offset;
    }

    if (!have_state_)
    {
      // Anchor the integrating reference to wherever the robot actually is,
      // so the first solve does not start with a large fabricated error.
      reference_position_ = x0_.segment<3>(3);
      reference_yaw_ = x0_(2);
      landing_target_[g1_mpc::LEG_LEFT] = foot_position_[g1_mpc::LEG_LEFT];
      landing_target_[g1_mpc::LEG_RIGHT] = foot_position_[g1_mpc::LEG_RIGHT];
      ground_height_ = 0.5 * (foot_position_[0].z() + foot_position_[1].z());
      have_state_ = true;
    }
  }

  void onClock(const rosgraph_msgs::Clock &msg) { sim_time_ = msg.clock.toSec(); }

  void onCommand(const geometry_msgs::Twist &msg)
  {
    command_target_ << clamp(msg.linear.x, -params_.max_linear_velocity,
                             params_.max_linear_velocity),
        clamp(msg.linear.y, -params_.max_linear_velocity, params_.max_linear_velocity),
        clamp(msg.angular.z, -params_.max_angular_velocity, params_.max_angular_velocity);
  }

  /// Run one MPC tick if the clock has advanced past the next tick boundary.
  void spinOnce()
  {
    if (!have_state_)
    {
      return;
    }
    if (!scheduler_.advanceTo(sim_time_))
    {
      return;
    }

    // Start walking only once a non-trivial command has been standing for a
    // moment. Stepping on the first tick after a command arrives lifts a foot
    // while the robot is still centred between them, and it falls towards the
    // side it just unloaded.
    const bool want_walk =
        command_target_.head<2>().norm() >= params_.stand_still_velocity ||
        std::abs(command_target_(2)) >= params_.stand_still_yaw_rate;
    if (!want_walk)
    {
      walk_requested_since_ = -1.0;
    }
    else if (walk_requested_since_ < 0.0)
    {
      walk_requested_since_ = sim_time_;
    }
    const bool ready =
        want_walk && (sim_time_ - walk_requested_since_) >= params_.startup_delay;
    scheduler_.requestStanding(!ready);

    // Only accumulate commanded motion once the feet are actually stepping.
    // Integrating it while the gait holds double support marches the CoM
    // reference away from the planted feet, and the robot is already toppling
    // by the time the first foot leaves the ground.
    applyCommandRateLimit(!scheduler_.standing());

    Eigen::MatrixXd x_ref = buildReference();
    Eigen::MatrixXd contact(params_.horizon, g1_mpc::MPC::kNumContacts);
    Eigen::MatrixXd contact_pos(params_.horizon, g1_mpc::MPC::kNumControls);
    buildContactHorizon(x_ref, contact, contact_pos);
    applySupportSway(x_ref, contact, contact_pos);

    const g1_mpc::MPC::Result result = mpc_.solve(x0_, x_ref, contact, contact_pos);
    publish(result, x_ref, contact, contact_pos);

    std_msgs::Float64 solve_time;
    solve_time.data = result.solve_time;
    pub_solve_time_.publish(solve_time);

    if (!result.solved)
    {
      ROS_WARN_THROTTLE(1.0, "g1_mpc: falling back to a static force split");
    }
  }

private:
  void applyCommandRateLimit(bool walking)
  {
    const Eigen::Vector3d target = walking ? command_target_ : Eigen::Vector3d::Zero();
    const double dt = params_.dt;
    command_(0) = rateLimit(command_(0), target(0),
                            params_.max_linear_acceleration * dt);
    command_(1) = rateLimit(command_(1), target(1),
                            params_.max_linear_acceleration * dt);
    command_(2) = rateLimit(command_(2), target(2),
                            params_.max_angular_acceleration * dt);
  }

  /// Reference trajectory over the horizon.
  ///
  /// The position and yaw references are *integrated* from the velocity
  /// command rather than pinned to the current measurement, so a steady
  /// tracking error still produces a restoring term.  They are clamped to stay
  /// within a bounded distance of the measurement, which is what stops the
  /// integrator winding up while the robot is held back.
  Eigen::MatrixXd buildReference()
  {
    const int H = params_.horizon;
    const double dt = params_.dt;

    const Eigen::Vector3d com = x0_.segment<3>(3);
    const double yaw = x0_(2);

    reference_position_.x() = clamp(reference_position_.x(),
                                    com.x() - params_.max_position_error,
                                    com.x() + params_.max_position_error);
    reference_position_.y() = clamp(reference_position_.y(),
                                    com.y() - params_.max_position_error,
                                    com.y() + params_.max_position_error);
    reference_yaw_ = clamp(reference_yaw_, yaw - params_.max_yaw_error,
                           yaw + params_.max_yaw_error);

    const double height = ground_height_ + params_.com_height;

    Eigen::MatrixXd x_ref = Eigen::MatrixXd::Zero(H, g1_mpc::MPC::kNumStates);
    Eigen::Vector3d position = reference_position_;
    double heading = reference_yaw_;

    for (int k = 0; k < H; ++k)
    {
      // The command is body frame; rotating it by the *predicted* heading is
      // what keeps "forward" forward through a turn.
      const Eigen::Vector3d velocity =
          rotZ(heading) * Eigen::Vector3d(command_(0), command_(1), 0.0);

      x_ref(k, 0) = 0.0;               // roll
      x_ref(k, 1) = 0.0;               // pitch
      x_ref(k, 2) = heading;
      x_ref(k, 3) = position.x();
      x_ref(k, 4) = position.y();
      x_ref(k, 5) = height;
      x_ref(k, 6) = 0.0;
      x_ref(k, 7) = 0.0;
      x_ref(k, 8) = command_(2);
      x_ref(k, 9) = velocity.x();
      x_ref(k, 10) = velocity.y();
      x_ref(k, 11) = 0.0;
      x_ref(k, 12) = params_.gravity;

      position += velocity * dt;
      heading += command_(2) * dt;
    }

    // Advance the persistent reference by exactly one tick.
    const Eigen::Vector3d step =
        rotZ(reference_yaw_) * Eigen::Vector3d(command_(0), command_(1), 0.0) * dt;
    reference_position_ += step;
    reference_position_.z() = height;
    reference_yaw_ += command_(2) * dt;

    return x_ref;
  }

  /// Contact flags and contact-point positions over the horizon.
  ///
  /// Future stance positions come from the footstep planner evaluated against
  /// the reference trajectory, so the MPC optimises against the footholds it is
  /// actually going to get rather than assuming the feet stay where they are.
  void buildContactHorizon(const Eigen::MatrixXd &x_ref, Eigen::MatrixXd &contact,
                           Eigen::MatrixXd &contact_pos)
  {
    const int H = params_.horizon;
    Eigen::Vector3d foothold[g1_mpc::NUM_LEGS] = {foot_position_[0], foot_position_[1]};

    for (int leg = 0; leg < g1_mpc::NUM_LEGS; ++leg)
    {
      if (!scheduler_.inContact(leg, 0))
      {
        // Mid-swing: this foot is heading for its planned target.
        landing_target_[leg] = planFor(leg, x_ref, scheduler_.ticksToTouchdown(leg));
      }
      else
      {
        landing_target_[leg] = foothold[leg];
      }
    }

    for (int k = 0; k < H; ++k)
    {
      for (int leg = 0; leg < g1_mpc::NUM_LEGS; ++leg)
      {
        const bool now = scheduler_.inContact(leg, k);
        const bool before = (k == 0) ? scheduler_.inContact(leg, 0)
                                     : scheduler_.inContact(leg, k - 1);
        if (now && !before)
        {
          // Touchdown at step k: from here on the leg stands on the foothold
          // planned for the swing that just ended.
          foothold[leg] = planFor(leg, x_ref, k);
        }

        contact(k, 2 * leg) = now ? 1.0 : 0.0;
        contact(k, 2 * leg + 1) = now ? 1.0 : 0.0;

        // Heel and toe sit along the foot's x axis, so they rotate with the
        // heading. This is what makes the moment arms correct while turning.
        const Eigen::Matrix3d R = rotZ(x_ref(k, 2));
        const Eigen::Vector3d heel =
            foothold[leg] + R * Eigen::Vector3d(params_.foot_heel_x, 0.0, 0.0);
        const Eigen::Vector3d toe =
            foothold[leg] + R * Eigen::Vector3d(params_.foot_toe_x, 0.0, 0.0);
        contact_pos.row(k).segment<3>(6 * leg) = heel.transpose();
        contact_pos.row(k).segment<3>(6 * leg + 3) = toe.transpose();
      }
    }
  }

  Eigen::Vector3d planFor(int leg, const Eigen::MatrixXd &x_ref, int ticks_ahead) const
  {
    const int k = std::min(std::max(ticks_ahead, 0), params_.horizon - 1);
    g1_mpc::FootstepPlanner::State state;
    state.com_position = x_ref.row(k).segment<3>(3);
    // Track the *measured* velocity for the feedback term; the reference
    // velocity is what the step is meant to achieve, not what it must correct.
    state.com_velocity = x0_.segment<3>(9);
    state.desired_velocity = x_ref.row(k).segment<3>(9);
    state.yaw = x_ref(k, 2);
    state.yaw_rate = command_(2);
    state.ground_height = ground_height_;

    if (scheduler_.standing())
    {
      return footstep_.nominalStance(leg, state);
    }
    return footstep_.plan(leg, state, ticks_ahead * params_.dt);
  }



  /// Lean the CoM reference towards whatever is actually going to be in
  /// contact.
  ///
  /// A straight-line CoM reference is only right in double support. During
  /// single support the body has to be over the stance foot, and a reference
  /// that insists on staying between the feet fights exactly the weight shift
  /// the step depends on: the robot never loads the stance leg and rolls off
  /// it. The offset is filtered along the horizon because the support polygon
  /// jumps at each contact change and the CoM cannot.
  /// Lean the CoM reference towards whatever is in contact.
  ///
  /// A straight-line CoM reference is only right in double support. During
  /// single support the body has to be over the stance foot, and a reference
  /// that insists on staying between the feet fights exactly the weight shift
  /// the step depends on: the robot never loads the stance leg and topples
  /// towards the foot it just lifted.
  ///
  /// The profile is a trapezoid in gait phase: hold over the stance foot
  /// through single support, and ramp across during double support, which is
  /// precisely the interval the gait sets aside for the transfer. Smoothing the
  /// square wave with a filter instead gets the timing wrong in both
  /// directions - a causal filter arrives after the foot has already lifted,
  /// and one that previews the whole horizon starts unloading the stance foot
  /// while it is still the only thing holding the robot up.
  void applySupportSway(Eigen::MatrixXd &x_ref, const Eigen::MatrixXd &contact,
                        const Eigen::MatrixXd &contact_pos)
  {
    const int H = params_.horizon;

    // Lateral offset from the feet midpoint to the support centroid, per step,
    // in the heading frame. Zero in double support, half a stance width in
    // single support.
    std::vector<double> lateral(H, 0.0);
    std::vector<bool> single(H, false);
    for (int k = 0; k < H; ++k)
    {
      Eigen::Vector2d support = Eigen::Vector2d::Zero();
      Eigen::Vector2d all = Eigen::Vector2d::Zero();
      int active = 0;
      for (int c = 0; c < g1_mpc::MPC::kNumContacts; ++c)
      {
        const Eigen::Vector2d point = contact_pos.row(k).segment<2>(3 * c);
        all += point;
        if (contact(k, c) > 0.5)
        {
          support += point;
          ++active;
        }
      }
      if (active == 0 || active == g1_mpc::MPC::kNumContacts)
      {
        continue;
      }
      all /= g1_mpc::MPC::kNumContacts;
      support /= active;
      const Eigen::Matrix3d R = rotZ(x_ref(k, 2));
      const Eigen::Vector3d delta(support.x() - all.x(), support.y() - all.y(), 0.0);
      lateral[k] = (R.transpose() * delta).y();
      single[k] = true;
    }

    // Fill the double-support gaps by interpolating between the single-support
    // values on either side, so the reference crosses over during the transfer
    // rather than sitting between the feet and then jumping.
    std::vector<double> filled(lateral);
    for (int k = 0; k < H; ++k)
    {
      if (single[k])
      {
        continue;
      }
      int before = -1, after = -1;
      for (int j = k - 1; j >= 0; --j)
      {
        if (single[j]) { before = j; break; }
      }
      for (int j = k + 1; j < H; ++j)
      {
        if (single[j]) { after = j; break; }
      }
      if (before < 0 && after < 0)
      {
        filled[k] = 0.0;
      }
      else if (before < 0)
      {
        // Start of the horizon: head towards the upcoming stance side.
        filled[k] = lateral[after] * static_cast<double>(k + 1) /
                    static_cast<double>(after + 1);
      }
      else if (after < 0)
      {
        filled[k] = lateral[before];
      }
      else
      {
        const double u = static_cast<double>(k - before) /
                         static_cast<double>(after - before);
        filled[k] = (1.0 - u) * lateral[before] + u * lateral[after];
      }
    }

    for (int k = 0; k < H; ++k)
    {
      const Eigen::Vector3d offset =
          rotZ(x_ref(k, 2)) * Eigen::Vector3d(0.0, params_.sway_gain * filled[k], 0.0);
      x_ref(k, 3) += offset.x();
      x_ref(k, 4) += offset.y();
    }
  }

  static void fillState(g1_msgs::State &s, int k, const Eigen::MatrixXd &x)
  {
    s.trajectory_index = k;
    s.orientation.x = x(k, 0);
    s.orientation.y = x(k, 1);
    s.orientation.z = x(k, 2);
    s.position.x = x(k, 3);
    s.position.y = x(k, 4);
    s.position.z = x(k, 5);
    s.angular_velocity.x = x(k, 6);
    s.angular_velocity.y = x(k, 7);
    s.angular_velocity.z = x(k, 8);
    s.linear_velocity.x = x(k, 9);
    s.linear_velocity.y = x(k, 10);
    s.linear_velocity.z = x(k, 11);
    s.gravity = x(k, 12);
  }

  void publish(const g1_mpc::MPC::Result &result, const Eigen::MatrixXd &x_ref,
               const Eigen::MatrixXd &contact, const Eigen::MatrixXd &contact_pos)
  {
    g1_msgs::SRBD_state msg;
    msg.header.stamp = ros::Time(sim_time_);
    msg.header.frame_id = "world";
    msg.solver_ok = result.solved;

    msg.states_horizon.resize(params_.horizon);
    for (int k = 0; k < params_.horizon; ++k)
    {
      fillState(msg.states_horizon[k], k, result.x_opt);
    }

    msg.reference_horizon.resize(params_.horizon);
    for (int k = 0; k < params_.horizon; ++k)
    {
      fillState(msg.reference_horizon[k], k, x_ref);
    }

    msg.contacts.resize(g1_mpc::MPC::kNumContacts);
    for (int c = 0; c < g1_mpc::MPC::kNumContacts; ++c)
    {
      g1_msgs::ContactPoint &p = msg.contacts[c];
      p.name = contact_names_[c];
      p.force.x = result.u_opt(0, 3 * c);
      p.force.y = result.u_opt(0, 3 * c + 1);
      p.force.z = result.u_opt(0, 3 * c + 2);
      p.active = contact(0, c) > 0.5;
      p.position.x = contact_pos(0, 3 * c);
      p.position.y = contact_pos(0, 3 * c + 1);
      p.position.z = contact_pos(0, 3 * c + 2);
    }

    msg.foot_plan.resize(g1_mpc::NUM_LEGS);
    for (int leg = 0; leg < g1_mpc::NUM_LEGS; ++leg)
    {
      g1_msgs::FootPlan &f = msg.foot_plan[leg];
      f.name = leg_names_[leg];
      f.in_contact = scheduler_.inContact(leg, 0);
      f.swing_phase = scheduler_.swingPhase(leg);
      f.landing_position.x = landing_target_[leg].x();
      f.landing_position.y = landing_target_[leg].y();
      f.landing_position.z = landing_target_[leg].z();
    }

    pub_solution_.publish(msg);
  }

  ros::NodeHandle nh_;
  ros::Publisher pub_solution_;
  ros::Publisher pub_solve_time_;
  ros::Subscriber sub_state_;
  ros::Subscriber sub_clock_;
  ros::Subscriber sub_cmd_;

  g1_mpc::Parameters params_;
  g1_mpc::MPC mpc_;
  g1_mpc::GaitScheduler scheduler_;
  g1_mpc::FootstepPlanner footstep_;

  std::vector<std::string> contact_names_;
  std::vector<std::string> leg_names_;

  bool have_state_ = false;
  double sim_time_ = 0.0;
  double ground_height_ = 0.0;

  Eigen::VectorXd x0_;
  Eigen::MatrixXd measured_contacts_;
  Eigen::Vector3d foot_position_[g1_mpc::NUM_LEGS];
  Eigen::Vector3d landing_target_[g1_mpc::NUM_LEGS];

  Eigen::Vector3d reference_position_ = Eigen::Vector3d::Zero();
  double reference_yaw_ = 0.0;

  double walk_requested_since_ = -1.0;
  Eigen::Vector3d command_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d command_target_ = Eigen::Vector3d::Zero();
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "g1_mpc");
  ros::NodeHandle nh;

  g1_mpc::Parameters params;
  try
  {
    params = g1_mpc::Parameters::load(nh);
  }
  catch (const std::exception &e)
  {
    ROS_FATAL("%s", e.what());
    return 1;
  }

  MpcNode node(nh, params);

  // Poll a little faster than the MPC rate so a tick is never missed; the
  // scheduler decides when a solve is actually due.
  ros::Rate rate(4.0 / params.dt);
  while (ros::ok())
  {
    ros::spinOnce();
    node.spinOnce();
    rate.sleep();
  }
  return 0;
}
