## Installation

### Prerequisites
- Linux (tested on Ubuntu 20.04+)
- Conda (Miniconda or Anaconda)

### Setup Instructions

**Step 1: Create and activate the conda environment**
```bash
conda create --name DRL -c pytorch -c conda-forge --file conda-requirements.txt
conda activate DRL
```

**Step 2: Install additional Python packages**
```bash
pip install tqdm==4.66.1 \
            stable-baselines3==2.2.1 \
            sb3-contrib==2.2.1 \
            rich==13.7.0 \
            pytz==2023.3.post1 \
            pygments==2.17.2 \
            mdurl==0.1.2 \
            markdown-it-py==3.0.0 \
            fsspec==2023.12.0
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.11 | Runtime |
| PyTorch | 2.1.1 (CPU) | DRL training backend |
| stable-baselines3 | 2.2.1 | PPO, A2C, DQN |
| sb3-contrib | 2.2.1 | TRPO, QR-DQN |
| gymnasium | via conda | RL environment interface |
| numpy | 1.26.0 | Numerical operations |

### Verification

To confirm the installation is successful, run:
```bash
python -c "import gymnasium; import stable_baselines3; print('Setup complete!')"
```
