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
# CFClient (for manual control)
WORKDIR /app/CrazySim
RUN git clone https://github.com/llanesc/crazyflie-clients-python.git

# Drone Gym
WORKDIR /app
RUN git clone https://github.com/UoA-CARES/drone_gym.git

# -------------------------------------------------------------------
# Install Python packages for each component
# -------------------------------------------------------------------

# drone_gym
WORKDIR /app/drone_gym
RUN git checkout drone-sim && \
    git pull && \
    pip install -r requirements.txt && \
    pip install -e .

# CrazySim python libs
WORKDIR /app/CrazySim/crazyflie-lib-python
RUN pip install -e .

WORKDIR /app/CrazySim/crazyflie-clients-python
RUN git checkout sitl-release && \
    pip install -e .

# -------------------------------------------------------------------
# Build CrazySim Firmware (SIM)
# -------------------------------------------------------------------

WORKDIR /app/CrazySim/crazyflie-firmware
RUN pip install Jinja2 && \
    mkdir -p sitl_make/build && \
    cd sitl_make/build && \
    cmake .. && \
    make all

# -------------------------------------------------------------------
# Force reinstall specific versions to avoid conflicts
# -------------------------------------------------------------------

RUN pip install --force-reinstall pandas==2.3.3 && \
    pip install --force-reinstall opencv-python==4.7.0.72 && \
    pip install --force-reinstall numpy==1.24.4

# -------------------------------------------------------------------
# Runtime
# -------------------------------------------------------------------

WORKDIR /app/gymnasium_envrionments/scripts
CMD ["bash"]
