# How to Install Isaac Lab and Isaac Sim

- Follow the official guide: [Isaac Lab — Pip Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)

### 1. Create and activate a virtual environment (Python 3.11)
```bash
uv venv --python 3.11 env_isaaclab
source env_isaaclab/bin/activate
```

### 2. Install Isaac Sim (pip)
```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

### 3. Install CUDA-enabled PyTorch
```bash
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

### 4. Verify Isaac Sim CLI is available
```bash
isaacsim
```

### 5. Clone the repository
```bash
git clone https://github.com/isaac-sim/IsaacLab
cd IsaacLab
```

### 6. Install system dependencies
```bash ha
sudo apt install -y cmake build-essential
```

### (Optional) Ensure pip is available in the venv
```bash
python -m ensurepip --default-pip
```

### 7. Install Isaac Lab
```bash
./isaaclab.sh --install
```

### 8. Verify the installation
- This command should open a CartPole scene:
```bash
./isaaclab.sh -p scripts/tutorials/03_envs/create_cartpole_base_env.py
```

# Isaac Sim vs Isaac Lab

- **Isaac Sim**: Omniverse‑based robotics simulation application.
  - Focus: High‑fidelity physics, rendering, sensors, USD workflows, interactive GUI.
  - Usage: Build scenes, import assets, simulate robots, author environments.
  - Install: As a standalone application (selector/launcher), used by other tools.

- **Isaac Lab**: Research and training framework built on top of Isaac Sim.
  - Focus: RL/robotics environments, vectorized simulation, task definitions, training scripts.
  - Usage: Create Gym‑style tasks, batch simulations, integrate with learning libraries.
  - Install: Python project (repo + script) that uses Isaac Sim as the simulator backend.

- **Relationship**: Isaac Lab depends on Isaac Sim to run simulations. You can use Isaac Sim alone for scene creation and simulation. Use Isaac Lab when you need structured tasks, reproducible experiments, and training pipelines.