# docker build -t oculux314/cares:drone . (use --no-cache to rebuild from start)
# docker run -it --gpus all --net host oculux314/cares:drone
FROM oculux314/cares:base
WORKDIR /app

# -------------------------------------------------------------------
# Setup cflib
# -------------------------------------------------------------------

RUN git clone https://github.com/gtfactslab/CrazySim.git --recursive
WORKDIR /app/CrazySim/crazyflie-lib-python
RUN git checkout sitl-release
RUN pip install -e .

# -------------------------------------------------------------------
# Setup drone_gym
# -------------------------------------------------------------------

# Clone repo
WORKDIR /app
RUN git clone https://github.com/UoA-CARES/drone_gym.git


# Install dependencies
WORKDIR /app/drone_gym
RUN pip install -r requirements.txt && \
    pip install -e .

# Force reinstall specific versions to avoid conflicts
RUN pip install --force-reinstall pandas==2.3.3 && \
    pip install --force-reinstall opencv-python==4.7.0.72 && \
    pip install --force-reinstall numpy==1.24.4

# -------------------------------------------------------------------
# Port Forwarding
# -------------------------------------------------------------------

# Port forwarding: our code expects CrazySim to be broadcasting at 0.0.0.0:19850. However, in the docker network this is actually crazysim:19850
RUN apt-get update && apt-get install -y socat && apt-get clean && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# Runtime
# -------------------------------------------------------------------

WORKDIR /app/gymnasium_envrionments/scripts
CMD ["bash", "-c", "\
    socat -v UDP-LISTEN:19850,fork,reuseaddr UDP:crazysim:19850 & \
    echo -e '======================================================================\nRun `python run.py train cli drone --task move_2d SAC` to start a training run.\n======================================================================' && \
    bash"]
