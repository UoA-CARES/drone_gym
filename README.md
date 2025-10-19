# drone_gym
Gym for Reinforcement Learning control of Drones

# Connecting Vicon system
Before you start, make sure you are inducted in the Motion Capture Lab!

1. Turn the cameras on by clicking on the two buttons shown in the picture. Both top and bottom should be on
![Image](https://github.com/user-attachments/assets/721651c8-8104-4b07-b570-9dceae6f8fc1)

2. Open the Vicon Tracker app (the green one). On the Systems tab, click on the "Local Vicon System" tab, and then press "SHOW ADVANCED"
![Image](https://github.com/user-attachments/assets/a9cbd0de-4a1a-4e6b-9bcf-9ccdb84f2231)

3. Scroll down to the "UDP OBJECT STREAM" section and enable the UDP stream. Change the IP address to 10.104.144.214
![Image](https://github.com/user-attachments/assets/edff9550-3619-4c04-8682-79550698886c)

4. Double check if the Crazyflie object is on. Navigate to the OBJECT tab and make sure "Crzayme" is checked
![Image](https://github.com/user-attachments/assets/d97d3e0d-b75d-451b-9609-e2a5ca4da27f)

# Useful shortcuts for Vicon
Zoom in/out - press on the right hand of the mouse and drag
pan - press down on scroll wheel of mouse

# Setting up the drone position
Before turning on the drone, make sure the marker board is on the ground. This stabilises the drone takeoff/flight as the optical flow deck relies on the change of texture (verified by trial and error). Turn the drone on while it is on the ground (there is a small push button at the side of the drone). In terms of coordinates, the direction towards forward (facing towards the window) is the positive y-axis shown as green, and the right is the positive x-axis (red). The drone "Front" should be in the positive x-direction.
![Image](https://github.com/user-attachments/assets/93923ad5-50b7-4fd1-ad6a-6e8a362e8e78)
![Image](https://github.com/user-attachments/assets/d8177450-425e-4ec1-803f-796a6c5fd4ef)

# Fixing drone issues
Check the "Issues" tab for known issues. If the drone keeps flipping in one direction (i.e. tips at one motor), it is likely due to either the motor mount being too loose or the motor overheating. First check if the motor mount is loose by pulling the motor mount off. If there is almost no resistance and the motor mount slides, you should swap it with another one. There is a Crazyflie kit in the drone lab (ask the technician to find it) If the motor is hot at touch, replace the motors (should be extras labelled "New" in the CARES container) For motor replacement, refer to: https://www.bitcraze.io/documentation/tutorials/getting-started-with-crazyflie-2-x/
