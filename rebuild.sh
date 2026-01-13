# This script rebuilds the CrazySim simulator because builds are flakey and often fail silently, needing to be rebuilt.

echo "//" >> ../CrazySim/crazyflie-firmware/tools/crazyflie-simulation/simulator_files/gazebo/plugins/CrazySim/crazysim_plugin.cpp # Bypass caching
current_dir=$(pwd)
cd ../CrazySim/crazyflie-firmware # Simulator (Gazebo)
mkdir -p sitl_make/build && cd $_
cmake ..
make all
cd $current_dir
echo "Rebuilt CrazySim."
