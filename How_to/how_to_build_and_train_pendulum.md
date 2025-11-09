# How to Build and Train Double Pendulum

This guide walks you through the complete process of building a double pendulum robot in OnShape, converting it to URDF format, setting it up in Isaac Sim, and training it using reinforcement learning in Isaac Lab.

## Overview

The double pendulum consists of:
- A **rail** (fixed base)
- A **cart** that slides along the rail (controlled by the RL agent)
- A **first arm** attached to the cart (passive joint)
- A **second arm** attached to the first arm (passive joint)

The RL agent can only control the horizontal force on the cart and must learn to balance both pendulum arms in the upright position.

## Converting OnShape to URDF

The double pendulum robot was designed in OnShape using simple sketches and extrusions. The URDF file and associated assets (STL files) are already included in this repository at `DoublePendulumURDF/robot.urdf`.

If you want to create your own design or modify the existing one, you can use the following tools to convert your OnShape CAD model to URDF format.

### Using onshape-to-robot

Use [onshape-to-robot](https://onshape-to-robot.readthedocs.io/en/latest/index.html) to convert your OnShape design to URDF format.

This tool automatically extracts the robot's geometry, joints, and physical properties from your OnShape CAD model and generates a URDF file that Isaac Sim can import.

> **Alternative:** There is also an [extension to directly load from OnShape to Isaac Sim](https://docs.omniverse.nvidia.com/extensions/latest/ext_onshape.html) (not tested in this project).

### URDF File Location

After conversion, place your URDF file and associated assets (STL files, etc.) in:
```
DoublePendulumIsaacLab/DoublePendulumURDF/
```

The main URDF file should be named `robot.urdf` and placed at:
```
DoublePendulumIsaacLab/DoublePendulumURDF/robot.urdf
```

## Modifying the URDF to Fix Material Issues

Isaac Sim requires materials to be declared at the beginning of the URDF file to properly render them. Manually modify the URDF to declare materials at the top level (before the links) so they are properly taken into account.

### Example Material Declaration

Add material definitions right after the `<robot>` tag:

```xml
<?xml version="1.0" ?>
<!-- Generated using onshape-to-robot -->
<!-- Onshape https://cad.onshape.com/documents/23ff1b6c89f9297b7a0e9b8a/w/75eb753e20548f3cea30a049/e/ffe591d2a8c2ca2f335417a0 -->
<robot name="DoublePendulum">
  <material name="rail_material">
    <color rgba="0.0627451 0.266667 0.490196 1"/>
  </material>
  <material name="cart_material">
    <color rgba="0.0627451 0.266667 0.490196 1"/>
  </material>
  <material name="arm_material">
    <color rgba="0.0627451 0.266667 0.490196 1"/>
  </material>
  <material name="arm_2_material">
    <color rgba="0.0627451 0.266667 0.490196 1"/>
  </material>
  <material name="payload_material">
    <color rgba="0.980392 0.713725 0.00392157 1"/>
  </material>
  <!-- ... rest of URDF ... -->
</robot>
```

## Importing into Isaac Sim

### Importing the URDF

1. Open Isaac Sim
2. Go to **File > Import > robot.urdf**
3. Navigate to your `DoublePendulumURDF/robot.urdf` file
4. Click **Import**

The robot should appear in the scene. You may need to adjust the camera view to see it.

### Configuring Joint Physics

To make the joints behave realistically, you need to configure the physics properties:

1. Go to **Tools > Physics > Gain Tuner**
2. In the "Tune Gains" section, select each joint and adjust the **Stiffness** and **Damping** values

#### Joint Configuration Guidelines

For the double pendulum:
- **Cart joint (prismatic "slide")**: 
  - Set appropriate stiffness and damping for realistic sliding behavior
  - This joint will be controlled by the RL agent, so ensure it responds smoothly to forces
- **First arm joint (revolute "revolute")**: 
  - Set **stiffness = 0** and **damping = 0** to make it a passive joint
  - The arm should hang freely and move only due to physics (gravity, cart motion)
- **Second arm joint (revolute "revolute2")**: 
  - Set **stiffness = 0** and **damping = 0** to make it a passive joint
  - The second arm should also hang freely

> **Important:** Both revolute joints must be passive (stiffness=0, damping=0) for the double pendulum to behave correctly. If these joints have non-zero stiffness, they will try to control the arms, which is not what we want.

#### Testing the Physics

1. Press **Play** on the simulation
2. Click **"Run Test"** in the "Test Gain Settings" section
3. Observe the robot's dynamics
4. Adjust Stiffness and Damping values as needed to achieve realistic behavior

> **Tip:** To manually control and apply forces to the robot during testing, press **Shift + Left Click** and drag the cart. This helps verify that the physics behaves correctly.

## Creating a New Project in Isaac Lab

Isaac Lab provides a template generator to create a new RL training project. This sets up all the necessary structure and boilerplate code.

### Generating the Project Template

> **Important:** If you're using this repository (`DoublePendulumIsaacLab`), the project has already been created and configured. You can skip this section and proceed directly to [Installation](#installation). The following instructions are only needed if you want to create a new project from scratch.

If you're creating a new project from scratch:

1. Navigate to your Isaac Lab repository root
2. Run the template generator:
   ```bash
   ./isaaclab.sh
   ```
3. Follow the interactive prompts and choose the following options:
   - **Project type**: **External** (creates a project outside the Isaac Lab repo)
   - **RL workflow**: **Manager-based** (single-agent, easier for our use case)
   - **RL library**: **skrl** (we'll use PPO from skrl)
   - **RL algorithm**: **PPO** (Proximal Policy Optimization)

After generating the template, you'll need to modify the configuration files (see [Main Configuration Files](#main-configuration-files) below) to set up the double pendulum environment. In this repository, these files have already been configured.

> **Note:** For detailed instructions, see the [official tutorial](https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/template.html).

### Project Structure

The template generator creates the `DoublePendulumTraining` folder with the following structure:
```
DoublePendulumTraining/
├── source/
│   └── DoublePendulumTraining/
│       ├── DoublePendulumTraining/
│       │   └── tasks/
│       │       └── manager_based/
│       │           └── doublependulumtraining/
│       │               ├── doublependulumtraining_env_cfg.py  <- Main config file
│       │               └── agents/
│       │                   └── skrl_ppo_cfg.yaml              <- Training hyperparameters
│       └── setup.py
└── scripts/
    ├── list_envs.py
    ├── zero_agent.py
    └── skrl/
        ├── train.py
        └── play.py
```

### Main Configuration Files

> **Note:** In this repository, these files are already configured for the double pendulum task. The following information is provided for reference or if you're creating a new project from scratch.

The two main configuration files in the project are:

1. **Environment Configuration** (defines the RL problem):
   ```
   DoublePendulumTraining/source/DoublePendulumTraining/DoublePendulumTraining/tasks/manager_based/doublependulumtraining/doublependulumtraining_env_cfg.py
   ```
   This file contains:
   - Robot configuration (URDF path, actuators, physics properties)
   - Scene setup (ground, lighting)
   - Action space (what the agent can control)
   - Observation space (what the agent can see)
   - Reward function (what the agent optimizes)
   - Termination conditions (when episodes end)
   - Event randomizations (for robust training)

2. **Training Hyperparameters** (defines the learning algorithm):
   ```
   DoublePendulumTraining/source/DoublePendulumTraining/DoublePendulumTraining/tasks/manager_based/doublependulumtraining/agents/skrl_ppo_cfg.yaml
   ```
   This file contains:
   - PPO algorithm parameters (learning rate, batch size, etc.)
   - Network architecture (policy and value networks)
   - Training schedule (number of iterations, checkpoints)
   - Logging configuration

## Installation

Before you can use the training environment, you need to install the project as a Python package.

### Install the Package

From the `DoublePendulumIsaacLab` repository root, run:

```bash
uv pip install -e DoublePendulumTraining/source/DoublePendulumTraining
```

The `-e` flag installs the package in "editable" mode, meaning changes to the source code are immediately available without reinstalling.

> **Note:** Make sure you have activated your Isaac Lab conda environment before running this command. If you're using the Isaac Lab setup, the environment should already be configured.

## Usage

All commands should be run from the `DoublePendulumIsaacLab` repository root.

### List Available Tasks

Verify that your environment is properly registered:

```bash
python DoublePendulumTraining/scripts/list_envs.py
```

You should see `Template-Doublependulumtraining-v0` in the list of available tasks.

### Test Loading the Environment

Before training, test that the environment loads correctly:

```bash
python DoublePendulumTraining/scripts/zero_agent.py --task=Template-Doublependulumtraining-v0
```

This script:
- Loads the environment
- Runs a "zero agent" (outputs zero actions)
- Verifies that the simulation runs without errors
- Displays the environment in the viewer

If you see the double pendulum in the viewer and the simulation runs, the environment is configured correctly.


### Launch Training

Start the reinforcement learning training:

```bash
python DoublePendulumTraining/scripts/skrl/train.py --task=Template-Doublependulumtraining-v0
```

This will:
- Create multiple parallel environments (default: 4096)
- Train the PPO agent to balance the double pendulum
- Save checkpoints periodically
- Log training metrics to TensorBoard

Training can take several hours depending on your hardware. The agent learns to balance both pendulum arms by controlling only the cart's horizontal force.

> **Note:** Training parameters (learning rate, batch size, number of iterations, etc.) are defined in `DoublePendulumTraining/source/DoublePendulumTraining/DoublePendulumTraining/tasks/manager_based/doublependulumtraining/agents/skrl_ppo_cfg.yaml`

### Analyze Training Results

Monitor training progress using TensorBoard:

```bash
tensorboard --logdir logs/skrl/doublependulumtraining
```

Then open your browser to `http://localhost:6006` to view:
- Episode rewards (should increase over time)
- Episode length (should increase as the agent learns to balance longer)
- Policy loss and value loss
- Other training metrics

> **Note:** The log directory path is relative to where you run the command. If you run it from the repository root, use `logs/skrl/doublependulumtraining`. If you run it from `DoublePendulumTraining/`, use `../logs/skrl/doublependulumtraining`.

### Checkpoint Location

I have already done training and exported the best agent at:
```
DoublePendulumTraining/best_agent.pt
```

This checkpoint contains the trained policy network weights and can be loaded to run the trained agent.

### Run the Trained Policy

Once training is complete, you can run the trained policy:

```bash
python DoublePendulumTraining/scripts/skrl/play.py \
    --task=Template-Doublependulumtraining-v0 \
    --algorithm=PPO \
    --num_envs=1 \
    --checkpoint=DoublePendulumTraining/best_agent.pt \
    --device cpu
```

Parameters:
- `--task`: The environment name
- `--algorithm`: The RL algorithm used (PPO)
- `--num_envs`: Number of parallel environments (use 1 for visualization)
- `--checkpoint`: Path to the trained model
- `--device cpu`: Use CPU instead of GPU (required for interactive control)

> **Tips:**
> - **For longer episodes**: Increase `episode_length_s` in `doublependulumtraining_env_cfg.py` if you want to interact with the trained robot for longer periods (default is 10 seconds).
> - **For interactive control**: Use `--device cpu` to enable interactive control. You can then press **Shift + Left Click** and drag to "push" the pendulum and see how the trained agent responds.
> - **For headless mode**: Remove `--device cpu` to run on GPU (faster, but no interactive control).