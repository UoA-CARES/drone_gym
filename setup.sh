base_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_dir=$base_dir/.venv/drone

echo "Setting up the drone environment..."

# # Install Gazebo and dependencies (COMMENT OUT if already installed)
sudo apt-get update
sudo apt-get install lsb-release curl gnupg
sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install gz-garden
sudo apt install cmake build-essential # This is for the `make all` command
sudo apt install python3.10 python3.10-venv # Python 3.10 and venv module

# # Clone repositories (COMMENT OUT if already cloned)
cd $base_dir
git clone https://github.com/UoA-CARES/cares_reinforcement_learning.git
git clone https://github.com/UoA-CARES/gymnasium_envrionments.git
git clone https://github.com/UoA-CARES/drone_gym.git
git clone https://github.com/gtfactslab/CrazySim.git --recursive
cd CrazySim
git clone https://github.com/llanesc/crazyflie-clients-python.git

# .venv SETUP
cd $base_dir
python3.10 -m venv $venv_dir  # Create virtual environment
source $venv_dir/bin/activate # Or change to your virtual environment path

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
git checkout main
git pull
pip install -r requirements.txt
pip install -e .

# CrazySim SETUP
cd $base_dir/CrazySim

cd crazyflie-lib-python # CFLib
# Patch pyproject.toml to relax dependency versions for compatibility
sed -i 's/scipy~=1.14/scipy>=1.10/' pyproject.toml
sed -i 's/numpy~=2.2/numpy>=1.20/' pyproject.toml
sed -i 's/packaging~=25.0/packaging>=21.0/' pyproject.toml
# Install with explicit version to avoid version detection issues
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.31 pip install -e .

cd ../crazyflie-clients-python # cfclient
git checkout sitl-release
pip install -e .

cd ../crazyflie-firmware # Simulator (Gazebo)
pip install Jinja2
mkdir -p sitl_make/build && cd $_
cmake ..
make all

# Force reinstall specific package versions to avoid conflicts
pip install --force-reinstall pandas==2.3.3
pip install --force-reinstall opencv-python==4.7.0.72
# numpy must be >=1.20 and <1.25 for compatibility with cflib and cfclient
pip install --force-reinstall "numpy>=1.20,<1.25"

# Run training
echo "Setup complete. Running a test training session..."
cd $base_dir

# Clean up any existing simulator processes and temporary files
sudo pkill -9 -f crazyflie 2>/dev/null || true
sudo pkill -9 -f gz 2>/dev/null || true
rm -rf /tmp/crazyflie*

# Run training
echo 'To run simulation:"cd CrazySim/crazyflie-firmware; bash tools/crazyflie-simulation/simulator_files/gazebo/launch/sitl_singleagent.sh -m crazyflie -x 0 -y 0; exec bash"'
echo "Training example: run.py train cli drone --task move_random_2d SAC"