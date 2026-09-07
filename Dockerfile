FROM ubuntu:22.04

# --- ENVIRONMENT & CACHE SETUP ---
ENV HOME=/tmp
ENV GZ_VERSION=garden
ENV CARES_LOG_BASE_DIR=/workspace/output
ENV GZ_FUEL_CACHE_DIR=/tmp/gz/fuel
ENV GZ_IP=127.0.0.1
ENV GZ_PARTITION=drone_sim
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MPLCONFIGDIR=/tmp/matplotlib_cache
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Headless OpenGL for cv2 and matplotlib (no display needed)
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility

ARG USER_NAME=drone
ARG USER_ID=1000
ARG GROUP_ID=1000

ARG CARES_RL_REF=main
ARG DRONE_GYM_REF=main
ARG CRAZYSIM_REF=main
ARG CFLIB_REF=master

WORKDIR /drone_ws

# Install core system and Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    gnupg \
    ca-certificates \
    lsb-release \
    python3.10 \
    python3-pip \
    build-essential \
    libusb-1.0-0 \
    wget \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgles2 \
    libosmesa6 \
    libglfw3 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libx11-6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxxf86vm1 \
    ffmpeg \
    cmake \
    pkg-config \
    python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Gazebo Garden
RUN wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" > /etc/apt/sources.list.d/gazebo-stable.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    gz-garden \
    libgz-sim7-dev \
    libgz-plugin2-dev \
    libgz-transport12-dev \
    libgz-msgs9-dev && \
    rm -rf /var/lib/apt/lists/*

# Install CrazySim from source.
RUN git clone https://github.com/gtfactslab/CrazySim.git /drone_ws/CrazySim && \
    cd /drone_ws/CrazySim && \
    git checkout "${CRAZYSIM_REF}" && \
    git submodule update --init --recursive && \
    cd /drone_ws/CrazySim/crazyflie-firmware && \
    mkdir -p sitl_make/build && \
    cd sitl_make/build && \
    cmake .. && \
    make all -j"$(nproc)"

# Patch CrazySim's launch script to use python3 and not launch sim GUI.
RUN sed -i \
    -e 's/\bpython\b/python3/g' \
    -e 's/^gz sim -g/wait/' \
    /drone_ws/CrazySim/crazyflie-firmware/tools/crazyflie-simulation/simulator_files/gazebo/launch/sitl_multiagent_square.sh

# Install cflib (Crazyflie Python library) from source.
RUN git clone https://github.com/bitcraze/crazyflie-lib-python.git /drone_ws/cflib_src && \
    cd /drone_ws/cflib_src && \
    git checkout "${CFLIB_REF}" && \
    sed -i 's/scipy~=1.14/scipy>=1.10/' pyproject.toml && \
    sed -i 's/numpy~=2.2/numpy>=1.20/' pyproject.toml && \
    sed -i 's/packaging~=25.0/packaging>=21.0/' pyproject.toml && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.1.31 \
        python3 -m pip install --no-cache-dir -e . && \
    python3 -c "import cflib.crtp; print('cflib OK')"

# Install CARES RL
# Jinja2 is required at runtime by the CrazySim launch script's jinja_gen.py (model.sdf templating).
RUN python3 -m pip install --no-cache-dir pettingzoo Jinja2 && \
    git clone \
        https://github.com/UoA-CARES/cares_reinforcement_learning.git \
        /opt/cares_reinforcement_learning && \
    cd /opt/cares_reinforcement_learning && \
    git checkout "${CARES_RL_REF}" && \
    python3 -m pip install --no-cache-dir -e ".[gym]"

# Install drone_gym from source
RUN git clone \
        https://github.com/UoA-CARES/drone_gym.git \
        /drone_ws/drone_gym && \
    cd /drone_ws/drone_gym && \
    git checkout "${DRONE_GYM_REF}" && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip install --no-cache-dir -e .

# Force reinstall versions known to work with cflib/CrazySim (mirrors setup.sh).
RUN python3 -m pip install --force-reinstall --no-cache-dir \
    pandas==2.3.3 \
    opencv-python==4.7.0.72 \
    "numpy>=1.20,<1.25"


# Create a non-root runtime user, following the CARES RL image pattern.
RUN groupadd --gid "${GROUP_ID}" "${USER_NAME}" && \
    useradd \
        --uid "${USER_ID}" \
        --gid "${GROUP_ID}" \
        --create-home \
        --shell /bin/bash \
        "${USER_NAME}"

# Create directories that need to be writable at runtime.
RUN mkdir -p \
        /workspace/output \
        /drone_ws/cache \
        /drone_ws/logs \
        /tmp/matplotlib_cache \
        /tmp/gz/fuel && \
    chown "${USER_NAME}:${USER_NAME}" /workspace/output && \
    chmod a+rwx \
        /workspace/output \
        /drone_ws/cache \
        /drone_ws/logs \
        /tmp/matplotlib_cache \
        /tmp/gz \
        /tmp/gz/fuel \
        /drone_ws/CrazySim/crazyflie-firmware/sitl_make/build

USER ${USER_NAME}

WORKDIR /drone_ws

CMD ["bash"]
