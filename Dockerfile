# docker build -t oculux314/cares:drone .
# docker run -it --gpus all oculux314/cares:drone
FROM oculux314/cares:base
WORKDIR /app

# -------------------------------------------------------------------
# Installation
# -------------------------------------------------------------------

# Install Gazebo Garden + build tools
RUN apt-get update && \
    apt-get install -y \
        lsb-release curl gnupg cmake build-essential && \
    curl "https://packages.osrfoundation.org/gazebo.gpg" \
        --output "/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg" && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
        https://packages.osrfoundation.org/gazebo/ubuntu-stable \
        $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/gazebo-stable.list && \
    apt-get update && \
    apt-get install -y gz-garden

# -------------------------------------------------------------------
# Clone additional repos
# -------------------------------------------------------------------

# CrazySim (recursive)
RUN git clone https://github.com/gtfactslab/CrazySim.git --recursive
RUN cd CrazySim && git checkout 3246b07269c20413530269955ca054dab4426c15
RUN cd CrazySim/crazyflie-lib-python && git checkout 9f50dce2dbba7d3836a86f6965095dbb43242a07
RUN cd CrazySim/crazyflie-firmware && git checkout 6a4d2d006ba3c394949582c1c2eedd4b84fce17e

# Drone Gym
WORKDIR /app
RUN git clone https://github.com/UoA-CARES/drone_gym.git

# -------------------------------------------------------------------
# Set up dependencies
# -------------------------------------------------------------------

# CrazySim python libs
WORKDIR /app/CrazySim/crazyflie-lib-python
RUN pip install -e .

# Build CrazySim Firmware (Simulator)
WORKDIR /app/CrazySim/crazyflie-firmware
RUN pip install Jinja2 && \
    mkdir -p sitl_make/build && \
    cd sitl_make/build && \
    cmake .. && \
    make all

# -------------------------------------------------------------------
# Setup drone_gym
# -------------------------------------------------------------------

WORKDIR /app/drone_gym
RUN git checkout main && \
    git pull && \
    pip install -r requirements.txt && \
    pip install -e .

# -------------------------------------------------------------------
# Force reinstall specific versions to avoid conflicts
# -------------------------------------------------------------------

RUN pip install --force-reinstall pandas==2.3.3 && \
    pip install --force-reinstall opencv-python==4.7.0.72 && \
    pip install --force-reinstall numpy==1.24.4

# -------------------------------------------------------------------
# Runtime
# -------------------------------------------------------------------

# Disable GUI
ENV GZ_SIM_DISABLE_GUI=1
RUN sed -i \
    -e 's/^gz sim -g$/wait/' \
    /app/CrazySim/crazyflie-firmware/tools/crazyflie-simulation/simulator_files/gazebo/launch/sitl_singleagent.sh
# Simple alias to start simulator
RUN echo 'alias sim="cd /app/CrazySim/crazyflie-firmware && \
    bash tools/crazyflie-simulation/simulator_files/gazebo/launch/sitl_singleagent.sh -m crazyflie -x 0 -y 0"' \
    >> /etc/bash.bashrc

WORKDIR /app/gymnasium_envrionments/scripts
CMD ["bash", "-c", "echo '======================================================================\nRun `sim &` to start the simulator in background mode (close with `kill %<pid>`) then `python run.py train cli drone --task move_2d SAC` to start a training run.\n======================================================================' && bash"]
