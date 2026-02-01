# Sets up cares_reinforcement_learning, gymnasium_envrionments, drone_gym, and CrazySim with a unified .venv in a specified base directory.
# Usage: bash setup.sh <base_directory>
base_dir=$1
if [ -z "$base_dir" ]; then
  echo "Usage: bash $0 <base_directory>"
  exit 1
fi

set -e # Exit on error
venv_dir="$base_dir/.venv/drone"
echo "Setting up the drone environment..."

# .venv SETUP
cd $base_dir
python3 -m venv $venv_dir  # Create virtual environment
source $venv_dir/bin/activate # Or change to your virtual environment path

# Dependencies
sudo apt update
sudo apt install git

# # Clone repositories (COMMENT OUT if already cloned)
cd $base_dir
git clone https://github.com/UoA-CARES/cares_reinforcement_learning.git
git clone https://github.com/UoA-CARES/gymnasium_envrionments.git
git clone https://github.com/UoA-CARES/drone_gym.git
git clone https://github.com/gtfactslab/CrazySim.git --recursive
cd CrazySim
git clone https://github.com/llanesc/crazyflie-clients-python.git

# cares_reinforcement_learning SETUP
cd $base_dir/cares_reinforcement_learning
git checkout main
git pull
pip install -r requirements.txt
pip install -e .

# gymnasium_envrionments SETUP
cd $base_dir/gymnasium_envrionments
git checkout main
git pull
pip install -r requirements.txt

# drone_gym SETUP
cd $base_dir/drone_gym
git checkout drone-sim
git pull
pip install -r requirements.txt
pip install -e .

# CrazySim SETUP
cd $base_dir/CrazySim

cd crazyflie-lib-python # CFLib
pip install -e .

cd ../crazyflie-clients-python # cfclient
git checkout sitl-release
pip install -e .

# Force reinstall specific package versions to avoid conflicts
pip install --force-reinstall pandas==2.3.3
pip install --force-reinstall opencv-python==4.7.0.72
pip install --force-reinstall numpy==1.24.4

# Startup
cd $base_dir/drone_gym
docker compose up -d
docker compose exec cares bash
# Now run `python run.py train cli drone --task move_2d SAC`
