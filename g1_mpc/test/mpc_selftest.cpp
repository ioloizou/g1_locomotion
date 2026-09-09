/// Standalone checks on the MPC, the gait scheduler and the footstep planner.
///
/// Runs without a ROS master: the parameters are built in code to match
/// config/g1_locomotion.yaml. Useful for telling "the controller is wrong" from
/// "the simulation is wrong", which is otherwise hard once both are running.

#include "g1_mpc/footstep_planner.hpp"
#include "g1_mpc/gait_scheduler.hpp"
#include "g1_mpc/mpc.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <cstdlib>

using namespace g1_mpc;

namespace
{

/// Parameters as the running stack sees them: straight from the YAML the
/// launch file loads, so this test cannot drift away from the real setup.
Parameters nominal()
{
  const char *env = std::getenv("G1_LOCOMOTION_CONFIG");
  const std::string path =
      env ? env
          : std::string("/home/forest_ws/src/g1_locomotion/g1_mujoco_sim/config/"
                        "g1_locomotion.yaml");
  return Parameters::loadFromFile(path);
}

struct Scene
{
  Eigen::VectorXd x0;
  Eigen::MatrixXd x_ref;
  Eigen::MatrixXd contact;
  Eigen::MatrixXd contact_pos;
};

/// Robot standing still, both feet down, state exactly on the reference.
Scene standing(const Parameters &p, double com_x = 0.0527, double com_y = 0.0)
{
  const int H = p.horizon;
  Scene s;
  s.x0 = Eigen::VectorXd::Zero(MPC::kNumStates);
  s.x0 << 0, 0, 0, com_x, com_y, p.com_height, 0, 0, 0, 0, 0, 0, p.gravity;

  s.x_ref = Eigen::MatrixXd::Zero(H, MPC::kNumStates);
  for (int k = 0; k < H; ++k)
  {
    s.x_ref.row(k) = s.x0.transpose();
  }

  s.contact = Eigen::MatrixXd::Ones(H, MPC::kNumContacts);
  s.contact_pos = Eigen::MatrixXd::Zero(H, MPC::kNumControls);
  const double foot_x = com_x + p.foot_offset_x;
  const double half = 0.5 * p.stance_width;
  for (int k = 0; k < H; ++k)
  {
    // L-heel, L-toe, R-heel, R-toe
    s.contact_pos.row(k) << foot_x + p.foot_heel_x, com_y + half, 0.0,
        foot_x + p.foot_toe_x, com_y + half, 0.0,
        foot_x + p.foot_heel_x, com_y - half, 0.0,
        foot_x + p.foot_toe_x, com_y - half, 0.0;
  }
  return s;
}

void report(const char *label, const Parameters &p, const Scene &s,
            const MPC::Result &r)
{
  Eigen::Vector3d total = Eigen::Vector3d::Zero();
  double moment_x = 0.0, moment_y = 0.0;
  for (int c = 0; c < MPC::kNumContacts; ++c)
  {
    total += r.u_opt.row(0).segment<3>(3 * c).transpose();
    moment_x += s.contact_pos(0, 3 * c + 1) * r.u_opt(0, 3 * c + 2);
    moment_y += s.contact_pos(0, 3 * c) * r.u_opt(0, 3 * c + 2);
  }
  const double weight = p.mass * std::abs(p.gravity);
  printf("%-26s solved=%d  fz=[%6.1f %6.1f %6.1f %6.1f]  sum=(%+7.1f %+7.1f %7.1f)\n",
         label, static_cast<int>(r.solved),
         r.u_opt(0, 2), r.u_opt(0, 5), r.u_opt(0, 8), r.u_opt(0, 11),
         total.x(), total.y(), total.z());
  printf("%-26s cop=(%+.4f %+.4f)  com=(%+.4f %+.4f)  weight=%.1f  a_com=(%+.3f %+.3f %+.3f)\n",
         "", moment_y / total.z(), moment_x / total.z(), s.x0(3), s.x0(4), weight,
         total.x() / p.mass, total.y() / p.mass, total.z() / p.mass + p.gravity);
}

} // namespace

int main(int argc, char **argv)
{
  Parameters p = nominal();
  // Optional: 13 state weights on the command line, for quick sweeps.
  if (argc == 1 + 13)
  {
    for (int i = 0; i < 13; ++i) p.q_weights(i) = std::atof(argv[1 + i]);
  }
  std::cout << "Q = " << p.q_weights.transpose() << "\n";

  {
    MPC mpc(p);
    Scene s = standing(p);
    // A stationary robot on its reference should hold the centre of pressure
    // under the centre of mass and produce exactly its own weight.
    report("standing, zero error", p, s, mpc.solve(s.x0, s.x_ref, s.contact, s.contact_pos));
  }
  {
    MPC mpc(p);
    Scene s = standing(p);
    s.x0(3) += 0.03;  // pushed 3 cm forward of the reference
    report("standing, +3cm x error", p, s, mpc.solve(s.x0, s.x_ref, s.contact, s.contact_pos));
  }
  {
    MPC mpc(p);
    Scene s = standing(p);
    s.x0(9) = 0.2;  // drifting forward
    report("standing, +0.2 m/s vx", p, s, mpc.solve(s.x0, s.x_ref, s.contact, s.contact_pos));
  }
  {
    MPC mpc(p);
    Scene s = standing(p);
    s.x0(1) = 0.05;  // pitched forward
    report("standing, +0.05 pitch", p, s, mpc.solve(s.x0, s.x_ref, s.contact, s.contact_pos));
  }
  {
    MPC mpc(p);
    Scene s = standing(p);
    s.x0(9) = 0.3;   // moving forward
    s.x0(1) = 0.03;  // and pitched forward
    report("drifting: vx .3 pitch .03", p, s, mpc.solve(s.x0, s.x_ref, s.contact, s.contact_pos));
  }
  {
    // Single support on the left foot only.
    MPC mpc(p);
    Scene s = standing(p);
    for (int k = 0; k < p.horizon; ++k) { s.contact(k, 2) = 0; s.contact(k, 3) = 0; }
    s.x0(4) = 0.5 * p.stance_width;
    for (int k = 0; k < p.horizon; ++k) s.x_ref(k, 4) = s.x0(4);
    report("left single support", p, s, mpc.solve(s.x0, s.x_ref, s.contact, s.contact_pos));
  }

  printf("\ngait schedule (tick: L R, swing phase):\n");
  GaitScheduler gait(p);
  gait.requestStanding(false);
  for (int t = 0; t < 24; ++t)
  {
    gait.advanceTo(t * p.dt);
    printf("  %2d: %c %c   swing L=%.2f R=%.2f%s\n", t,
           gait.inContact(LEG_LEFT) ? 'X' : '.',
           gait.inContact(LEG_RIGHT) ? 'X' : '.',
           gait.swingPhase(LEG_LEFT), gait.swingPhase(LEG_RIGHT),
           (gait.isLiftoffTick(LEG_LEFT) || gait.isLiftoffTick(LEG_RIGHT)) ? "  liftoff" : "");
  }

  printf("\nfootstep targets (gain %.3f):\n", FootstepPlanner(p).velocityGain());
  FootstepPlanner planner(p);
  FootstepPlanner::State st;
  st.com_position << 0.0, 0.0, p.com_height;
  for (double vx : {0.0, 0.3, 0.6})
  {
    st.com_velocity << vx, 0.0, 0.0;
    st.desired_velocity = st.com_velocity;
    const Eigen::Vector3d l = planner.plan(LEG_LEFT, st, 0.3);
    const Eigen::Vector3d r = planner.plan(LEG_RIGHT, st, 0.3);
    printf("  vx=%.1f  L=(%+.3f %+.3f)  R=(%+.3f %+.3f)\n", vx, l.x(), l.y(), r.x(), r.y());
  }
  st.com_velocity << 0.3, 0.0, 0.0;
  st.desired_velocity = st.com_velocity;
  st.yaw_rate = 0.5;
  st.yaw = 0.4;
  const Eigen::Vector3d l = planner.plan(LEG_LEFT, st, 0.3);
  const Eigen::Vector3d r = planner.plan(LEG_RIGHT, st, 0.3);
  printf("  turning yaw=0.4 wz=0.5  L=(%+.3f %+.3f)  R=(%+.3f %+.3f)\n",
         l.x(), l.y(), r.x(), r.y());
  return 0;
}
