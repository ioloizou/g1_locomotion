# Standalone Python SRBD prototype

These scripts are the original standalone Python single-rigid-body MPC and its
matplotlib visualiser. They are kept for reference and are **not** part of the
running stack: the locomotion nodes use the C++ implementation in `../src`.

Nothing here is installed or launched. Run them directly with `python3` if you
want to experiment with the formulation outside ROS.

| File | What it is |
|---|---|
| `mpc.py` | Python version of the convex MPC |
| `gait_planner.py` | Phase-based gait plan with a matplotlib timeline |
| `srbd_plotting.py` | 3D visualiser for the single-rigid-body state |
| `main.py` | Driver that ties the three together |
| `mpc_to_wbid.py` | Superseded Python version of the ROS node |
