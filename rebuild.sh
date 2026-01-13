# This script rebuilds the CrazySim simulator because builds are flakey and often fail silently, needing to be rebuilt.
# Usage: bash rebuild.sh <base_directory>
base_dir="$1"
current_dir=$(pwd)

cd $base_dir/CrazySim/crazyflie-firmware # Simulator (Gazebo)
echo "//" >> tools/crazyflie-simulation/simulator_files/gazebo/plugins/CrazySim/crazysim_plugin.cpp # Bypass caching
mkdir -p sitl_make/build && cd $_
cmake ..
make all
cd $current_dir
echo "Rebuilt CrazySim."
