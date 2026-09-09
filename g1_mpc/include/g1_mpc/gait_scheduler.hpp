#pragma once

#include "g1_mpc/parameters.hpp"

#include <array>

namespace g1_mpc
{

enum Leg
{
  LEG_LEFT = 0,
  LEG_RIGHT = 1,
  NUM_LEGS = 2
};

/// Alternating-support gait clock.
///
/// The schedule is a pure function of an integer tick counter, where one tick
/// is one MPC step.  That is deliberate: with a float-time phase test, contact
/// transitions fall *between* two solves, which leaves a foot that the plan
/// believes is in stance carrying stale forces for part of a tick.  Keeping
/// every transition on a tick boundary makes the plan and the executed
/// schedule agree exactly, and makes the horizon's contact table a simple
/// lookup of future ticks.
///
/// Because the schedule depends only on the tick index, the simulation can
/// reconstruct the same phase from the same clock without any extra
/// synchronisation.
class GaitScheduler
{
public:
  explicit GaitScheduler(const Parameters &params);

  /// Advance to the tick containing `time`.  Returns true when the tick index
  /// changed, i.e. when a new MPC solve is due.
  bool advanceTo(double time);

  int tick() const { return tick_; }
  double tickTime() const { return tick_ * dt_; }

  /// Hold both feet down.  Entering standing takes effect only once the
  /// current swing has finished, so a foot is never abandoned mid-air.
  void requestStanding(bool standing);
  bool standing() const { return standing_; }

  /// Contact flag for `leg` at `ticks_ahead` steps beyond the current tick.
  bool inContact(int leg, int ticks_ahead = 0) const;

  /// Progress through the current swing, in [0, 1).  Zero while in stance.
  double swingPhase(int leg) const;

  /// True on the tick where `leg` lifts off, i.e. the tick a new footstep
  /// target must be committed.
  bool isLiftoffTick(int leg) const;

  /// Number of ticks until `leg` next touches down.  Zero when in stance.
  int ticksToTouchdown(int leg) const;

private:
  /// Phase of `leg` within the gait period, in [0, period_ticks).
  int legPhase(int leg, int ticks_ahead) const;

  double dt_;
  int period_ticks_;
  int stance_ticks_;
  std::array<int, NUM_LEGS> offset_ticks_;

  int tick_ = 0;
  /// Tick the current walking phase started from.
  int phase_origin_ = 0;
  bool initialised_ = false;
  bool standing_ = true;
  bool standing_request_ = true;
};

} // namespace g1_mpc
