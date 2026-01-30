# This script rebuilds the CrazySim simulator because builds are flaky and often fail silently, needing to be rebuilt.
# Usage: bash rebuild.sh <base_directory>
base_dir="$1"
if [ -z "$base_dir" ]; then
  echo "Usage: bash $0 <base_directory>"
  exit 1
fi

current_dir=$(pwd)

cd "$base_dir/CrazySim/crazyflie-firmware" # Simulator (Gazebo)
echo "//" >> tools/crazyflie-simulation/simulator_files/gazebo/plugins/CrazySim/crazysim_plugin.cpp # Bypass caching
mkdir -p sitl_make/build && cd $_
cmake ..
make all
cd "$current_dir"
echo "Rebuilt CrazySim."
