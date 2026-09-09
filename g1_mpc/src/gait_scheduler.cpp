#include "g1_mpc/gait_scheduler.hpp"

#include <cmath>

namespace g1_mpc
{

GaitScheduler::GaitScheduler(const Parameters &params)
    : dt_(params.dt),
      period_ticks_(params.period_ticks),
      stance_ticks_(params.stance_ticks)
{
  offset_ticks_[LEG_LEFT] = 0;
  offset_ticks_[LEG_RIGHT] = params.right_offset_ticks % params.period_ticks;
}

bool GaitScheduler::advanceTo(double time)
{
  const int tick = static_cast<int>(std::floor(time / dt_ + 1e-9));
  if (!initialised_)
  {
    initialised_ = true;
    tick_ = tick;
    return true;
  }
  if (tick == tick_)
  {
    return false;
  }
  tick_ = tick;

  // Only switch between walking and standing at a moment when both feet are
  // down, so no foot is left in the air by a mode change.
  const bool both_down = inContact(LEG_LEFT) && inContact(LEG_RIGHT);
  if (standing_ != standing_request_ && (standing_ || both_down))
  {
    if (standing_ && !standing_request_)
    {
      // Restart the cycle from phase zero so the first liftoff is a whole
      // double-support interval away. Picking up mid-cycle can put it one tick
      // later, which leaves no time to shift the weight across.
      phase_origin_ = tick_;
    }
    standing_ = standing_request_;
  }
  return true;
}

void GaitScheduler::requestStanding(bool standing) { standing_request_ = standing; }

int GaitScheduler::legPhase(int leg, int ticks_ahead) const
{
  int phase = (tick_ + ticks_ahead - phase_origin_ - offset_ticks_[leg]) % period_ticks_;
  if (phase < 0)
  {
    phase += period_ticks_;
  }
  return phase;
}

bool GaitScheduler::inContact(int leg, int ticks_ahead) const
{
  if (standing_)
  {
    return true;
  }
  return legPhase(leg, ticks_ahead) < stance_ticks_;
}

double GaitScheduler::swingPhase(int leg) const
{
  if (standing_ || inContact(leg))
  {
    return 0.0;
  }
  const int swing_ticks = period_ticks_ - stance_ticks_;
  const int elapsed = legPhase(leg, 0) - stance_ticks_;
  return static_cast<double>(elapsed) / static_cast<double>(swing_ticks);
}

bool GaitScheduler::isLiftoffTick(int leg) const
{
  return !standing_ && legPhase(leg, 0) == stance_ticks_;
}

int GaitScheduler::ticksToTouchdown(int leg) const
{
  if (standing_ || inContact(leg))
  {
    return 0;
  }
  return period_ticks_ - legPhase(leg, 0);
}

} // namespace g1_mpc
