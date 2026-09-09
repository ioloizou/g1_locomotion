#include "g1_mpc/parameters.hpp"

#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace g1_mpc
{
namespace
{

[[noreturn]] void missing(const std::string &key)
{
  throw std::runtime_error("g1_mpc: required parameter '" + key +
                           "' is missing or has the wrong type. Did the launch "
                           "file load config/g1_locomotion.yaml?");
}

/// Where parameters come from.
///
/// The field list below is written once and read through this interface, so
/// the ROS parameter server and the YAML file cannot end up describing
/// different sets of parameters. Two hand-maintained copies of the same list
/// is how a self-test ends up quietly testing a configuration nobody runs.
struct Source
{
  virtual ~Source() = default;
  virtual double number(const std::string &section, const std::string &key) const = 0;
  virtual int integer(const std::string &section, const std::string &key) const = 0;
  virtual bool boolean(const std::string &section, const std::string &key) const = 0;
  virtual std::vector<double> numbers(const std::string &section,
                                      const std::string &key) const = 0;
  virtual Eigen::Matrix3d matrix3(const std::string &section,
                                  const std::string &key) const = 0;
  virtual double nested(const std::string &section, const std::string &group,
                        const std::string &key) const = 0;
};

class RosSource : public Source
{
public:
  explicit RosSource(const ros::NodeHandle &nh) : nh_(nh) {}

  double number(const std::string &s, const std::string &k) const override
  {
    return get<double>(s + "/" + k);
  }
  int integer(const std::string &s, const std::string &k) const override
  {
    return get<int>(s + "/" + k);
  }
  bool boolean(const std::string &s, const std::string &k) const override
  {
    return get<bool>(s + "/" + k);
  }
  std::vector<double> numbers(const std::string &s, const std::string &k) const override
  {
    return get<std::vector<double>>(s + "/" + k);
  }
  double nested(const std::string &s, const std::string &g,
                const std::string &k) const override
  {
    return get<double>(s + "/" + g + "/" + k);
  }
  Eigen::Matrix3d matrix3(const std::string &s, const std::string &k) const override
  {
    const std::string key = s + "/" + k;
    XmlRpc::XmlRpcValue rows;
    if (!nh_.getParam(key, rows) || rows.getType() != XmlRpc::XmlRpcValue::TypeArray ||
        rows.size() != 3)
    {
      missing(key);
    }
    Eigen::Matrix3d m;
    for (int i = 0; i < 3; ++i)
    {
      if (rows[i].getType() != XmlRpc::XmlRpcValue::TypeArray || rows[i].size() != 3)
      {
        missing(key);
      }
      for (int j = 0; j < 3; ++j)
      {
        XmlRpc::XmlRpcValue &entry = rows[i][j];
        if (entry.getType() == XmlRpc::XmlRpcValue::TypeDouble)
        {
          m(i, j) = static_cast<double>(entry);
        }
        else if (entry.getType() == XmlRpc::XmlRpcValue::TypeInt)
        {
          m(i, j) = static_cast<int>(entry);
        }
        else
        {
          missing(key);
        }
      }
    }
    return m;
  }

private:
  template <typename T>
  T get(const std::string &key) const
  {
    T value;
    if (!nh_.getParam(key, value))
    {
      missing(key);
    }
    return value;
  }

  ros::NodeHandle nh_;
};

class YamlSource : public Source
{
public:
  explicit YamlSource(const std::string &path) : root_(YAML::LoadFile(path)) {}

  double number(const std::string &s, const std::string &k) const override
  {
    return node(s, k).as<double>();
  }
  int integer(const std::string &s, const std::string &k) const override
  {
    return node(s, k).as<int>();
  }
  bool boolean(const std::string &s, const std::string &k) const override
  {
    return node(s, k).as<bool>();
  }
  std::vector<double> numbers(const std::string &s, const std::string &k) const override
  {
    return node(s, k).as<std::vector<double>>();
  }
  double nested(const std::string &s, const std::string &g,
                const std::string &k) const override
  {
    const YAML::Node n = root_[s][g][k];
    if (!n)
    {
      missing(s + "/" + g + "/" + k);
    }
    return n.as<double>();
  }
  Eigen::Matrix3d matrix3(const std::string &s, const std::string &k) const override
  {
    const YAML::Node rows = node(s, k);
    Eigen::Matrix3d m;
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        m(i, j) = rows[i][j].as<double>();
      }
    }
    return m;
  }

private:
  YAML::Node node(const std::string &s, const std::string &k) const
  {
    const YAML::Node n = root_[s][k];
    if (!n)
    {
      missing(s + "/" + k);
    }
    return n;
  }

  YAML::Node root_;
};

/// The one place the parameter set is enumerated.
Parameters read(const Source &src)
{
  Parameters p;

  p.mass = src.number("robot", "mass");
  p.inertia = src.matrix3("robot", "inertia");
  p.com_height = src.number("robot", "com_height");
  p.foot_heel_x = src.number("robot", "foot_heel_x");
  p.foot_toe_x = src.number("robot", "foot_toe_x");
  p.stance_width = src.number("robot", "stance_width");
  p.foot_offset_x = src.number("robot", "foot_offset_x");

  p.gravity = src.number("physics", "gravity");
  p.friction_coefficient = src.number("physics", "friction_coefficient");
  p.fz_min = src.number("physics", "fz_min");
  p.fz_max = src.number("physics", "fz_max");

  p.period_ticks = src.integer("gait", "period_ticks");
  p.stance_ticks = src.integer("gait", "stance_ticks");
  p.right_offset_ticks = src.integer("gait", "right_offset_ticks");
  p.swing_height = src.number("gait", "swing_height");
  p.sway_gain = src.number("gait", "sway_gain");
  p.startup_delay = src.number("gait", "startup_delay");
  p.stand_still_velocity = src.number("gait", "stand_still_velocity");
  p.stand_still_yaw_rate = src.number("gait", "stand_still_yaw_rate");

  p.use_capture_point_gain = src.boolean("footstep", "use_capture_point_gain");
  p.velocity_gain = src.number("footstep", "velocity_gain");
  p.max_step_x = src.number("footstep", "max_step_x");
  p.max_step_y = src.number("footstep", "max_step_y");
  p.min_lateral_offset = src.number("footstep", "min_lateral_offset");
  p.max_lateral_offset = src.number("footstep", "max_lateral_offset");
  p.max_feedback = src.number("footstep", "max_feedback");

  p.dt = src.number("mpc", "dt");
  p.horizon = src.integer("mpc", "horizon");
  const std::vector<double> q = src.numbers("mpc", "q_weights");
  if (q.size() != 13)
  {
    throw std::runtime_error("g1_mpc: mpc/q_weights must have 13 entries, got " +
                             std::to_string(q.size()));
  }
  p.q_weights = Eigen::Map<const Eigen::VectorXd>(q.data(), q.size());
  p.r_weight = src.number("mpc", "r_weight");
  p.max_position_error = src.number("mpc", "max_position_error");
  p.max_yaw_error = src.number("mpc", "max_yaw_error");
  p.osqp_max_iterations = static_cast<int>(src.nested("mpc", "osqp", "max_iterations"));
  p.osqp_absolute_tolerance = src.nested("mpc", "osqp", "absolute_tolerance");
  p.osqp_relative_tolerance = src.nested("mpc", "osqp", "relative_tolerance");

  p.default_linear_x = src.number("command", "default_linear_x");
  p.default_linear_y = src.number("command", "default_linear_y");
  p.default_angular_z = src.number("command", "default_angular_z");
  p.max_linear_acceleration = src.number("command", "max_linear_acceleration");
  p.max_angular_acceleration = src.number("command", "max_angular_acceleration");
  p.max_linear_velocity = src.number("command", "max_linear_velocity");
  p.max_angular_velocity = src.number("command", "max_angular_velocity");

  if (p.stance_ticks <= 0 || p.stance_ticks >= p.period_ticks)
  {
    throw std::runtime_error("g1_mpc: gait/stance_ticks must be in (0, period_ticks)");
  }
  if (2 * p.stance_ticks <= p.period_ticks)
  {
    throw std::runtime_error(
        "g1_mpc: gait/stance_ticks must exceed half of gait/period_ticks, "
        "otherwise the gait has a flight phase and a single rigid body driven "
        "only by contact forces cannot track it");
  }
  if (p.horizon <= 1)
  {
    throw std::runtime_error("g1_mpc: mpc/horizon must be at least 2");
  }

  return p;
}

} // namespace

Parameters Parameters::load(const ros::NodeHandle &nh) { return read(RosSource(nh)); }

Parameters Parameters::loadFromFile(const std::string &path)
{
  return read(YamlSource(path));
}

} // namespace g1_mpc
