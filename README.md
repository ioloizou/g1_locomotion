# Humanoid Bipedal Locomotion Stack through Linear Model Predictive Control

This repository contains the work conducted for my **Master’s Thesis** in *Robotics Engineering* at **Centrale Nantes**, and during my research visit at **HuCeBot, INRIA Nancy**, under the supervision of **Dr. Enrico Mingo Hoffman**. The thesis manuscript can be found (here, link)

The objective of this project is to demonstrate a **linear locomotion control strategy** for the **Unitree G1 humanoid robot**.  
The proposed framework combines **Single Rigid Body Dynamics (SRBD)** with **Whole-Body Inverse Dynamics (WBID)** in a **cascaded control architecture**, permitting a fully **linear control pipeline**.


> ⚠️ *Note:* This implementation has not yet been tested on the physical robot.

---

## Visuals

**Straight-Line Walking experiment with Unitree G1 in Mujoco**

<p align="center">
  <img src="https://github.com/user-attachments/assets/7a4bb8f7-156c-4fc0-b141-280bc0a94c6b" 
       alt="Straight Line Walking Visualization" width="400">
</p>

---

## Table of Contents

  1. [Getting Started](#getting-started)
     - [Dependencies](#dependencies)
     - [Installation](#installation)
        - [Clone the Repository](#clone-the-repository)
        - [Build Using Docker](#build-using-docker)
        - [Run the Docker Container](#run-the-docker-container)
        - [Build the ROS Packages](#build-the-ros-packages)
  3. [Usage](#usage)
     - [Running an Experiment](#running-an-experiment)
     - [Running Plotjuggler](#running-plotjuggler)

  5. [Maintainer](#maintainer)
  6. [License](#license)



---

## Getting Started

### Dependencies

- **Docker**: To containerize the environment with all the important dependencies.
- **NVIDIA Container Toolkit**: Needed if you have an NVIDIA GPU.

---

### Installation

#### Clone the Repository

```bash
git clone https://github.com/hucebot/opensot_docker.git
cd opensot_docker
git checkout g1-locomotion
```

#### Build Using Docker

```bash
cd docker
sh build.sh
```

> 🕒 Building may take up to **30 minutes**.  
> The resulting Docker image will be named **`opensot`**.

#### Run the Docker Container

```bash
sh run.sh
```

#### Build the ROS Packages

Once inside the container:

```bash
cd g1_locomotion
```

Switch both packages to the walking demo branch:

```bash
git checkout walking-demo
cd g1_mpc && git checkout walking-demo
```

Then build everything:

```bash
cd ../ && make all
```

Finally, source the setup file:

```bash
cd ../../ && source setup.bash
```
---

## Usage

### Running an Experiment

Run the straight-line walking simulation:

```bash
roslaunch g1_mujoco_sim mpc_wbid_simulation.launch
```

The simulation will execute a few walking steps and then stop.

### Running Plotjuggler

On a different terminal in Docker:
```bash
rosrun plotjuggler plotjuggler
```
and then can load the MPC_QP_layout found in  **`g1_mujoco_sim/config/`** to have a dashboard of important values

## Maintainer

**Ioannis Loizou**

For questions, feedback, or collaborations, please contact:

- 📧 **Email:** [yiannisloizou@gmail.com](mailto:yiannisloizou@gmail.com)  
- 🔗 **LinkedIn:** [Ioannis Loizou](https://www.linkedin.com/in/ioannis-loizou-80b64615a/)

---

## License

This project is released under the **MIT License**.  
See the [LICENSE](LICENSE) file for more information.

---

## Reference

> *"Humanoid Bipedal through Linear Model Predictive Control"*,  
> Centrale Nantes, 2025.

If you use this work in your research, please cite or acknowledge this repository.

---
