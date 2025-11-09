"""
Double Pendulum Training Environment Configuration

This file defines the complete configuration for training an inverted double pendulum
to stay balanced using reinforcement learning. The environment uses a manager-based
RL system where the MDP (Markov Decision Process) components (actions, observations,
rewards, terminations) are defined declaratively.

The inverted double pendulum consists of:
- A rail (fixed base)
- A cart that slides along the rail (prismatic joint: "slide")
- A first arm attached to the cart (revolute joint: "revolute")
- A second arm attached to the first arm (revolute joint: "revolute2")

The RL agent controls only the cart's horizontal force, and must learn to balance
both pendulum arms in the upright position (pointing straight UP, unstable equilibrium).
This is the classic inverted double pendulum control problem.
"""

import math
import os

# Isaac Lab imports for simulation utilities
import isaaclab.sim as sim_utils

# Actuator configuration for controlling robot joints
from isaaclab.actuators import ImplicitActuatorCfg

# Asset configurations for defining robots and scene elements
from isaaclab.assets import ArticulationCfg, AssetBaseCfg

# Environment base class for manager-based RL
from isaaclab.envs import ManagerBasedRLEnvCfg

# MDP managers for defining events, observations, rewards, and terminations
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

# Scene configuration for setting up the simulation environment
from isaaclab.scene import InteractiveSceneCfg

# Terrain importer for better ground planes
from isaaclab.terrains import TerrainImporterCfg

# Decorator for creating configuration classes
from isaaclab.utils import configclass

# Asset paths for materials
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# Import MDP helper functions (defined in mdp/__init__.py and mdp/rewards.py)
from . import mdp

##
# Pre-defined configs
##

# ============================================================================
# URDF File Path Resolution
# ============================================================================
# Calculate the absolute path to the robot URDF file.
#
# Directory structure:
#   DoublePendulumIsaacLab/
#     ├── DoublePendulumURDF/
#     │   └── robot.urdf          <- Target file
#     └── DoublePendulumTraining/
#         └── source/
#             └── DoublePendulumTraining/
#                 └── DoublePendulumTraining/
#                     └── tasks/
#                         └── manager_based/
#                             └── doublependulumtraining/
#                                 └── doublependulumtraining_env_cfg.py  <- This file
#
# Navigate up 7 directory levels from this file to reach DoublePendulumIsaacLab root.
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
for _ in range(7):
    _CONFIG_DIR = os.path.dirname(_CONFIG_DIR)
# Construct the full path to the URDF file
_URDF_PATH = os.path.join(_CONFIG_DIR, "DoublePendulumURDF", "robot.urdf")

# ============================================================================
# Double Pendulum Robot Configuration
# ============================================================================
# Defines how the double pendulum robot is loaded, initialized, and controlled.
# Specifies the URDF file, physics properties, initial state, and actuator models.
#
DOUBLE_PENDULUM_CFG = ArticulationCfg(
    # ========================================================================
    # URDF Spawn Configuration
    # ========================================================================
    # Defines how the robot URDF file is loaded and converted to USD format
    spawn=sim_utils.UrdfFileCfg(
        # Path to the URDF file (calculated above)
        asset_path=_URDF_PATH,
        # Fix the base link (rail) to the world - prevents it from moving
        # This makes the rail a fixed base, which is correct for this setup
        fix_base=True,
        # Joint drive configuration for URDF conversion
        # When loading URDF, we need to specify default joint drive properties
        # We set these to zero because we'll define actuators separately below
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            # PD (Proportional-Derivative) gains for the joint drive
            # These are set to zero because we use ImplicitActuatorCfg below
            # which will override these values
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,  # No position control from URDF loader
                damping=0.0,  # No velocity damping from URDF loader
            ),
            # Target type "none" means joints won't try to reach target positions
            # This allows our actuators to have full control
            target_type="none",
        ),
        # ====================================================================
        # Rigid Body Physics Properties
        # ====================================================================
        # These properties control the physics simulation of rigid bodies
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Enable rigid body physics (required for dynamics)
            rigid_body_enabled=True,
            # Maximum linear velocity (m/s) - prevents unrealistic speeds
            # Set high to allow rapid cart movements when needed
            max_linear_velocity=1000.0,
            # Maximum angular velocity (rad/s) - prevents unrealistic rotations
            # Set high to allow rapid pendulum swings
            max_angular_velocity=1000.0,
            # Maximum depenetration velocity (m/s) - controls collision response
            # Prevents objects from interpenetrating too quickly
            max_depenetration_velocity=100.0,
            # Enable gyroscopic forces - accounts for conservation of angular momentum
            # Important for accurate pendulum physics
            enable_gyroscopic_forces=True,
        ),
        # ====================================================================
        # Articulation Root Properties
        # ====================================================================
        # These properties control the entire articulation (multi-body system)
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Disable self-collisions - prevents robot parts from colliding with each other
            # Not needed for this simple pendulum system
            enabled_self_collisions=False,
            # Solver position iterations - number of iterations for position constraint solving
            # Higher values = more accurate but slower (4 is a good default)
            solver_position_iteration_count=4,
            # Solver velocity iterations - number of iterations for velocity constraint solving
            # Set to 0 for efficiency (not critical for this system)
            solver_velocity_iteration_count=0,
            # Sleep threshold - bodies slower than this may be put to sleep for efficiency
            # Lower values = more bodies can sleep (0.005 is standard)
            sleep_threshold=0.005,
            # Stabilization threshold - tolerance for position stabilization
            # Lower values = more stable but potentially slower (0.001 is standard)
            stabilization_threshold=0.001,
        ),
    ),
    # ========================================================================
    # Initial State Configuration
    # ========================================================================
    # Defines the starting position and joint states when environments are reset
    init_state=ArticulationCfg.InitialStateCfg(
        # World position (x, y, z) of the robot's root (rail)
        # x=0.0, y=0.0: centered
        # z=1.5: Raised 1.5m above ground to allow full arm rotation without hitting ground
        pos=(0.0, 0.0, 1.5),
        # Initial joint positions (in joint space)
        # All joints start at zero (URDF-defined zero position):
        # - "slide": cart at center of rail (0.0 m)
        # - "revolute": first arm at 0.0 rad (pointing straight UP, verified in Isaac Sim)
        # - "revolute2": second arm at 0.0 rad (pointing straight UP, aligned with first arm)
        # Note: At angle 0, both arms are pointing straight up (unstable equilibrium).
        # Events will randomize these positions during training to learn from various starting states.
        joint_pos={"slide": 0.0, "revolute": 0.0, "revolute2": 0.0},
    ),
    # ========================================================================
    # Actuator Configuration
    # ========================================================================
    # Defines how each joint is controlled. ImplicitActuatorCfg means the physics
    # engine directly applies forces/torques based on the policy's actions.
    actuators={
        # --------------------------------------------------------------------
        # Cart Actuator (Prismatic Joint: "slide")
        # --------------------------------------------------------------------
        # Controls the horizontal movement of the cart along the rail
        # This is the ONLY actuator the RL policy controls
        "cart_actuator": ImplicitActuatorCfg(
            # Joint name pattern - matches the "slide" joint from URDF
            joint_names_expr=["slide"],
            # Maximum force (N) that can be applied to the cart
            # 400N is sufficient for rapid cart movements while preventing instability
            # This is the physics solver limit (applied in simulation)
            effort_limit_sim=400.0,
            # Stiffness = 0.0: Force control (no position control)
            # Why force control? For inverted pendulums, we need direct force application
            # to achieve quick, responsive control. Position control would add lag
            # and make balancing harder. The RL policy outputs forces directly.
            stiffness=0.0,
            # Damping: Velocity-dependent damping (N·s/m)
            # Evolution: Initial 10.0 → Increased to 20.0 (current) to reduce wiggling/oscillations
            # Higher damping adds more viscous resistance to cart movement,
            # which helps smooth out rapid left-right oscillations
            # This is a common technique for reducing control chatter
            damping=20.0,
        ),
        # --------------------------------------------------------------------
        # First Arm Actuator (Revolute Joint: "revolute")
        # --------------------------------------------------------------------
        # This joint is PASSIVE - not directly controlled by the RL policy
        # The arm moves in response to the cart's motion and gravity
        "arm_actuator": ImplicitActuatorCfg(
            # Joint name pattern - matches the "revolute" joint from URDF
            joint_names_expr=["revolute"],
            # Effort limit (if somehow torque is applied, cap it)
            # Set high since this joint should be passive
            effort_limit_sim=400.0,
            # Stiffness = 0.0: No position control
            # The arm is free to rotate based on physics
            stiffness=0.0,
            # Damping = 0.0: No artificial damping
            # We want natural pendulum dynamics - no artificial energy dissipation
            # The arm should swing naturally based on physics
            damping=0.0,
        ),
        # --------------------------------------------------------------------
        # Second Arm Actuator (Revolute Joint: "revolute2")
        # --------------------------------------------------------------------
        # This joint is also PASSIVE - not directly controlled
        # The second arm moves in response to the first arm and cart motion
        "arm2_actuator": ImplicitActuatorCfg(
            # Joint name pattern - matches the "revolute2" joint from URDF
            joint_names_expr=["revolute2"],
            # Effort limit (passive joint, but limit exists)
            effort_limit_sim=400.0,
            # Stiffness = 0.0: No position control
            # Natural pendulum dynamics
            stiffness=0.0,
            # Damping = 0.0: No artificial damping
            # Natural pendulum motion
            damping=0.0,
        ),
    },
)



##
# Scene definition
##


# ============================================================================
# Scene Configuration
# ============================================================================
# Defines all assets in the simulation scene (ground, robot, lights, etc.)
# This is where we assemble the complete environment that the RL agent will interact with.
#
@configclass
class DoublependulumtrainingSceneCfg(InteractiveSceneCfg):
    """Configuration for a double pendulum training scene.

    This scene contains:
    - A terrain ground plane with marble tile material (for visual reference and potential collisions)
    - The double pendulum robot (multiple instances for parallel training)
    - Lighting (for visualization)
    """

    # ------------------------------------------------------------------------
    # Ground Plane (Terrain with Material)
    # ------------------------------------------------------------------------
    # A large flat plane with a nice visual material (marble tiles)
    # Uses TerrainImporterCfg for better visual appearance than the default grid
    # Provides visual reference and can be used for collision detection
    # The marble tile material gives a clean, professional look
    terrain = TerrainImporterCfg(
        # Prim path in USD stage where ground will be created
        prim_path="/World/ground",
        # Terrain type: "plane" creates a simple flat plane
        terrain_type="plane",
        # Collision group: -1 means it collides with everything
        collision_group=-1,
        # Physics material properties for realistic interactions
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        # Visual material: marble tiles for a nice appearance
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),  # Scale the texture appropriately
        ),
        # Disable debug visualization (origin markers)
        debug_vis=False,
    )

    # ------------------------------------------------------------------------
    # Robot (Double Pendulum)
    # ------------------------------------------------------------------------
    # The main robot asset - multiple instances will be created for parallel training
    # The prim_path uses "{ENV_REGEX_NS}" which is a placeholder that gets replaced
    # with environment-specific paths like "/World/envs/env_0/Robot", "/World/envs/env_1/Robot", etc.
    # This allows multiple robots to exist in parallel (one per training environment)
    robot: ArticulationCfg = DOUBLE_PENDULUM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # ------------------------------------------------------------------------
    # Lighting
    # ------------------------------------------------------------------------
    # Dome light provides omnidirectional lighting for visualization
    # Important for rendering and seeing the robot in the viewer
    # Improved lighting for better visibility of the grid floor and robot
    dome_light = AssetBaseCfg(
        # Prim path for the light in USD stage
        prim_path="/World/DomeLight",
        # Spawn configuration: create a dome light
        # color=(0.75, 0.75, 0.75): Neutral white (RGB values) for good contrast
        # intensity=2000.0: Increased brightness for better visibility of grid and robot
        # Higher intensity helps see the grid pattern and robot details clearly
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )


##
# MDP settings
##

# ============================================================================
# Markov Decision Process (MDP) Components
# ============================================================================
# The MDP defines the reinforcement learning problem:
# - Actions: What the agent can do
# - Observations: What the agent can see
# - Rewards: What the agent should optimize for
# - Terminations: When episodes end
# - Events: Randomization and reset behaviors
#


# ============================================================================
# Action Space Configuration
# ============================================================================
# Defines what actions the RL policy can take
#
@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    The action space defines what the RL agent can control. In this environment,
    the agent can only control the cart's horizontal force.
    """

    # ------------------------------------------------------------------------
    # Joint Effort Action
    # ------------------------------------------------------------------------
    # The RL policy outputs a force value (typically in range [-1, 1])
    # This gets scaled and applied as a force to the specified joint
    joint_effort = mdp.JointEffortActionCfg(
        # Which robot asset to apply actions to (must match scene asset name)
        asset_name="robot",
        # Which joint(s) to control - only the "slide" joint (cart)
        # The policy outputs a single scalar value for this joint
        joint_names=["slide"],
        # Scale factor: multiplies the policy output by this value
        # scale=100.0 means policy output of 1.0 → 100N force
        # This scales the normalized [-1, 1] action to actual Newtons
        # The actual force is clamped by effort_limit_sim=400.0 in the actuator (safety limit)
        scale=100.0,
    )
    # Note: The two pendulum joints ("revolute" and "revolute2") are NOT controlled
    # by the policy. They are passive and move based on physics alone.


# ============================================================================
# Observation Space Configuration
# ============================================================================
# Defines what information the RL policy receives about the environment state
# The policy uses these observations to decide what actions to take
#
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.

    The observation space defines what the RL agent can "see" about the system state.
    For the double pendulum, we provide joint positions and velocities.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy network.

        This observation group is fed directly to the RL policy network.
        The observations are concatenated into a single vector.
        """

        # --------------------------------------------------------------------
        # Joint Position Observations (Relative)
        # --------------------------------------------------------------------
        # Returns the relative joint positions for all joints in the robot
        # For our robot, this includes:
        #   - slide: cart position along rail (m)
        #   - revolute: first arm angle (rad)
        #   - revolute2: second arm angle (rad)
        # "Relative" means positions are relative to some reference (often zero)
        # This is a vector of length 3 (one value per joint)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)

        # --------------------------------------------------------------------
        # Joint Velocity Observations (Relative)
        # --------------------------------------------------------------------
        # Returns the relative joint velocities for all joints
        # For our robot, this includes:
        #   - slide: cart velocity along rail (m/s)
        #   - revolute: first arm angular velocity (rad/s)
        #   - revolute2: second arm angular velocity (rad/s)
        # Velocities are crucial for control - they tell the policy how fast
        # things are moving and in what direction
        # This is also a vector of length 3 (one value per joint)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            """Configure observation group settings."""
            # Disable observation corruption (no noise/dropout added)
            self.enable_corruption = False

            # Concatenate all observation terms into a single vector
            # Final observation: [cart_pos, arm1_angle, arm2_angle,
            #                     cart_vel, arm1_vel, arm2_vel]
            # Total dimension: 6 (3 positions + 3 velocities)
            self.concatenate_terms = True

    # ------------------------------------------------------------------------
    # Observation Groups
    # ------------------------------------------------------------------------
    # The policy receives observations from this group
    # Multiple groups could be defined (e.g., for different policies in multi-agent setups)
    policy: PolicyCfg = PolicyCfg()


# ============================================================================
# Event Configuration
# ============================================================================
# Events define randomization and reset behaviors that occur during training
# These help the policy learn to handle diverse initial conditions and improve
# generalization. Events are triggered at the start of each episode.
#
@configclass
class EventCfg:
    """Configuration for environment events (randomization and resets).

    Events are triggered at specific times (e.g., episode reset) to randomize
    the initial state. This helps the RL agent learn robust policies that work
    in various conditions.
    """

    # ------------------------------------------------------------------------
    # Reset Cart Position (Event)
    # ------------------------------------------------------------------------
    # Randomizes the cart's starting position along the rail at each episode reset
    # This forces the policy to learn to balance from different cart positions
    reset_cart_position = EventTerm(
        # Function to call: reset_joints_by_offset adds random offsets to joint states
        func=mdp.reset_joints_by_offset,
        # When to trigger: "reset" means this happens at the start of each episode
        mode="reset",
        # Parameters for the reset function
        params={
            # Which robot and joint to modify
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slide"]),
            # Position range: cart starts at random position between -1.0m and +1.0m
            # This is 1 meter on each side of center - enough to test policy robustness
            "position_range": (-1.0, 1.0),
            # Velocity range: cart starts with random velocity between -0.5 and +0.5 m/s
            # Small initial velocity adds challenge and improves generalization
            "velocity_range": (-0.5, 0.5),
        },
    )

    # ------------------------------------------------------------------------
    # Reset First Arm Position (Event)
    # ------------------------------------------------------------------------
    # Randomizes the first arm's starting angle at each episode reset
    # The policy must learn to stabilize from various initial arm angles
    reset_arm_position = EventTerm(
        # Function to reset joint with random offset
        func=mdp.reset_joints_by_offset,
        # Trigger at episode reset
        mode="reset",
        params={
            # Target the "revolute" joint (first arm)
            "asset_cfg": SceneEntityCfg("robot", joint_names=["revolute"]),
            # Position range: ±45° from vertical (upright)
            # -0.25π to +0.25π radians = -45° to +45°
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            # Velocity range: ±45°/s initial angular velocity
            # Small initial rotation adds realism (pendulum might be moving)
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )

    # ------------------------------------------------------------------------
    # Reset Second Arm Position (Event)
    # ------------------------------------------------------------------------
    # Randomizes the second arm's starting angle at each episode reset
    # Similar to first arm, but independent - adds more diversity to initial states
    reset_arm2_position = EventTerm(
        # Function to reset joint with random offset
        func=mdp.reset_joints_by_offset,
        # Trigger at episode reset
        mode="reset",
        params={
            # Target the "revolute2" joint (second arm)
            "asset_cfg": SceneEntityCfg("robot", joint_names=["revolute2"]),
            # Position range: ±45° from aligned position (same as first arm)
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            # Velocity range: ±45°/s initial angular velocity
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )
    # Note: All three joints are randomized independently, creating 3D space of
    # initial conditions. This helps the policy learn robust balancing from many
    # different starting configurations.


# ============================================================================
# Reward Configuration
# ============================================================================
# Defines the reward function that guides the RL agent's learning
# The reward is a weighted sum of multiple terms, each encouraging specific behaviors
# The agent tries to maximize the total reward over time
#
@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    The reward function is composed of multiple terms:
    - Primary task rewards: Keep both arms upright (main objective)
    - Shaping rewards: Encourage smooth, stable motion
    - Survival rewards: Encourage staying in the episode
    - Penalties: Discourage failure

    All terms are added together: total_reward = Σ(weight_i X term_i)
    """

    # We have not added the survival reward because we have no termination conditions
    # The episode will end after 10 seconds by default
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # ------------------------------------------------------------------------
    # (1) First Arm Position Reward (Primary Task)
    # ------------------------------------------------------------------------
    # Penalizes deviation of the first arm from upright (target angle)
    # This is the MAIN objective: keep the first arm balanced
    # Uses L2 loss (squared error): penalty = (wrapped_angle - target)²
    # The joint angles are wrapped to (-π, π) before comparison
    # The negative weight means deviation is penalized (reward decreases)
    arm_pos = RewTerm(
        # Function: joint_pos_target_l2 computes squared distance from target
        # Note: joint_pos_target_l2 wraps angles to (-π, π) using wrap_to_pi()
        func=mdp.joint_pos_target_l2,
        weight=-1.5,
        # Parameters: which joint and what target value
        params={
            # Target the first arm joint ("revolute")
            "asset_cfg": SceneEntityCfg("robot", joint_names=["revolute"]),
            # Target angle: 0.0 radians = arms pointing straight UP (vertical upward)
            # VERIFIED: When joint angle is 0.0, the first arm is pointing straight up.
            # For an INVERTED double pendulum, this is the unstable equilibrium position
            # we want to balance - both arms vertical and pointing upward.
            # The reward function will penalize deviations from this upright position.
            "target": 0.0,
        },
    )

    # ------------------------------------------------------------------------
    # (2) Second Arm Position Reward (Primary Task)
    # ------------------------------------------------------------------------
    # Penalizes deviation of the second arm from upright (aligned with first arm)
    # Same as first arm - this is the other MAIN objective
    # Balanced signal helps learn to balance both arms simultaneously
    arm2_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.5,
        params={
            # Target the second arm joint ("revolute2")
            "asset_cfg": SceneEntityCfg("robot", joint_names=["revolute2"]),
            # Target angle: 0.0 radians = second arm pointing straight UP (vertical upward)
            # VERIFIED: When joint angle is 0.0, the second arm is pointing straight up.
            # For an INVERTED double pendulum, this aligns with the first arm (both upright).
            # The goal is to balance both arms in the unstable upright position.
            # The reward function will penalize deviations from this upright position.
            "target": 0.0,
        },
    )

    # ------------------------------------------------------------------------
    # (3) Upright Bonus Reward (Positive Shaping)
    # ------------------------------------------------------------------------
    # Rewards the agent for keeping the arms upright using exponential reward
    # This provides positive feedback when close to the target, making learning easier
    # Uses exponential: exp(-angle_error² / (2*sigma²)) - gives high reward when close to upright
    # The smaller the deviation, the higher the reward (peaks at 1.0 when perfectly upright)
    # This complements the L2 penalty by providing positive signal for good behavior
    upright_bonus = RewTerm(
        func=mdp.joint_pos_target_l2_exp,  # exp(-angle_error² / (2*sigma²))
        weight=2.0,  # Positive weight to reward being upright
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["revolute", "revolute2"]),
            "target": 0.0,
            "sigma": 0.3,  # Controls falloff - larger sigma = wider reward region
        },
    )

    # ------------------------------------------------------------------------
    # (4) First Arm Angular Velocity Penalty
    # ------------------------------------------------------------------------
    # Penalizes high angular velocities of the first arm to encourage smooth motion
    # Helps prevent wild oscillations and encourages stable, controlled balancing
    # Similar to cartpole example which uses -0.005 for pole angular velocity
    arm_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["revolute"]),
        },
    )

    # ------------------------------------------------------------------------
    # (5) Second Arm Angular Velocity Penalty
    # ------------------------------------------------------------------------
    # Penalizes high angular velocities of the second arm to encourage smooth motion
    # Helps prevent wild oscillations and encourages stable, controlled balancing
    arm2_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["revolute2"]),
        },
    )

    # ------------------------------------------------------------------------
    # (6) Cart Velocity Shaping Reward
    # ------------------------------------------------------------------------
    # Penalizes high cart velocities to encourage smooth, controlled motion
    # This is a "shaping" reward - helps guide learning toward desired behavior
    # Uses L1 loss (absolute value): penalty = |velocity|
    # Higher penalty encourages smoother, slower movements which reduces oscillations
    cart_vel = RewTerm(
        # Function: joint_vel_l1 computes absolute velocity
        func=mdp.joint_vel_l1,
        # Weight to discourage rapid cart movements
        weight=-0.01,
        params={
            # Target the cart's sliding joint
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slide"]),
        },
    )


# ============================================================================
# Termination Configuration
# ============================================================================
# Defines conditions that end an episode (success or failure)
# When a termination condition is met, the episode stops and a new one begins
#
@configclass
class TerminationsCfg:
    """Termination terms for the MDP.

    Terminations define when episodes end. There are two types:
    - Time out: Episode ends after a fixed duration (success)
    - Failure: Episode ends due to constraint violation (failure)
    """

    # ------------------------------------------------------------------------
    # (1) Time Out Termination
    # ------------------------------------------------------------------------
    # Episode ends after a fixed duration (defined in episode_length_s)
    # This is a "success" termination - the agent survived the full episode
    # time_out=True: This is not a failure, just a normal episode end
    # The episode_length_s is set in __post_init__ to 10 seconds
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Note: We do NOT add position-based terminations because:
    # - The URDF already defines joint limits, which the physics engine enforces


##
# Environment configuration
##


# ============================================================================
# Main Environment Configuration
# ============================================================================
# This is the top-level configuration that brings together all components
# (scene, actions, observations, rewards, terminations, events) into a complete
# RL environment. This class inherits from ManagerBasedRLEnvCfg, which provides
# the framework for manager-based RL environments.
#
@configclass
class DoublependulumtrainingEnvCfg(ManagerBasedRLEnvCfg):
    """Complete configuration for the Double Pendulum Training environment.

    This configuration class defines:
    - The simulation scene (ground, robot, lights)
    - The action space (what the agent can do)
    - The observation space (what the agent can see)
    - The reward function (what the agent should optimize)
    - Termination conditions (when episodes end)
    - Event randomizations (for robust training)
    - Simulation parameters (timestep, episode length, etc.)
    """

    # ========================================================================
    # Scene Configuration
    # ========================================================================
    # Defines the physical simulation scene with all assets
    scene: DoublependulumtrainingSceneCfg = DoublependulumtrainingSceneCfg(
        # num_envs=4096: Number of parallel environments to simulate
        # Running many environments in parallel speeds up training significantly
        # Each environment runs independently, providing diverse experiences
        # 4096 is a common value for Isaac Lab environments (good GPU utilization)
        num_envs=4096,
        # env_spacing=4.0: Distance between parallel environments (meters)
        # Prevents environments from interfering with each other
        # 4.0m spacing is sufficient for this robot size
        env_spacing=4.0,
    )

    # ========================================================================
    # MDP Component Configurations
    # ========================================================================
    # These define the reinforcement learning problem structure

    # Observation space: What the agent observes
    observations: ObservationsCfg = ObservationsCfg()

    # Action space: What the agent can control
    actions: ActionsCfg = ActionsCfg()

    # Events: Randomization and reset behaviors
    events: EventCfg = EventCfg()

    # Reward function: What the agent optimizes
    rewards: RewardsCfg = RewardsCfg()

    # Termination conditions: When episodes end
    terminations: TerminationsCfg = TerminationsCfg()

    # ========================================================================
    # Post-Initialization Configuration
    # ========================================================================
    # These settings are applied after the base configuration is loaded
    # They override or supplement default values from the parent class
    #
    def __post_init__(self) -> None:
        """Post-initialization configuration.

        This method is called after the configuration object is created.
        It sets additional parameters that may depend on other configuration values
        or need to be set programmatically.
        """

        # --------------------------------------------------------------------
        # General Environment Settings
        # --------------------------------------------------------------------

        # Decimation: Skip this many physics steps between policy updates
        # decimation=2: Policy runs at half the simulation frequency
        # If simulation runs at 120Hz, policy runs at 60Hz
        # This is more computationally efficient and often sufficient for control
        self.decimation = 2

        # Episode length: Maximum duration of each training episode (seconds)
        # episode_length_s=10: Episodes last up to 10 seconds
        # Increased from 5s to give more time to practice balancing
        # Longer episodes allow the agent to learn sustained balancing behavior
        # If termination conditions are met earlier, episode ends sooner
        self.episode_length_s = 10

        # --------------------------------------------------------------------
        # Viewer Settings
        # --------------------------------------------------------------------
        # Camera position for visualization (when using the viewer)
        # eye=(8.0, 0.0, 5.0): Camera location in world coordinates (x, y, z)
        # - x=8.0: 8 meters to the right (viewing from the side)
        # - y=0.0: Centered in y-axis
        # - z=5.0: 5 meters high (looking down at the robot)
        # This provides a good side view of the pendulum system
        self.viewer.eye = (8.0, 0.0, 5.0)

        # --------------------------------------------------------------------
        # Simulation Settings
        # --------------------------------------------------------------------

        # Simulation timestep: Duration of each physics step (seconds)
        # dt = 1/120: 120Hz simulation frequency (8.33ms per step)
        # Higher frequency = more accurate physics but slower simulation
        # 120Hz is a good balance for most robotics simulations
        self.sim.dt = 1 / 120

        # Render interval: How often to update the visualization
        # render_interval = decimation: Render at the same rate as policy updates
        # This matches the visual updates to the control frequency
        # Reduces unnecessary rendering overhead
        self.sim.render_interval = self.decimation

        # Note: The effective control frequency is:
        #   control_freq = sim_freq / decimation = 120Hz / 2 = 60Hz
        # This means the policy makes decisions 60 times per second, which is
        # sufficient for balancing a double pendulum.
