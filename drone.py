import threading
import queue
import time
import random
from vicon_connection_class import ViconInterface as vi
from djitellopy import Tello


class Drone:

    def __init__(self):
        self.command_queue = queue.Queue()
        self.velocity = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

        # Drone Properties
        # self.tello = Tello()
        # self.tello.connect()
        
        # Vicon Integration
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.drone_name = "AtlasCrazyflie" 
        self.vicon = vi()
        self.position_thread = threading.Thread(target=self._update_position)
        self.position_thread.start()

        # Drone Safety
        self.boundaries = {"x": 3, "y": 3, "z": 3} 
        self.safety_thread = threading.Thread(target=self._check_boundaries)
        self.safety_thread.start()
        self.in_boundaries = True

        # Objective related
        self.target_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        

    def _check_boundaries(self):
        time.sleep(5)
        while self.running:
            try:
                # Check if position is within boundaries for each axis
                x_in_bounds = self.boundaries["x"] >= abs(self.position["x"])
                y_in_bounds = self.boundaries["y"] >= abs(self.position["y"])
                z_in_bounds = self.boundaries["z"] >= abs(self.position["z"])
                
                # Set in_boundaries status
                current_status = x_in_bounds and y_in_bounds and z_in_bounds
                
                if current_status != self.in_boundaries:
                    self.in_boundaries = current_status
                    if self.in_boundaries:
                        print("Drone currently in bounds")
                    else:
                        print(f"[Drone] WARNING: Out of bounds!")
                        # Instead of calling stop() directly, send a special emergency command
                        # This avoids the self join thread issue
                        self.send_command("emergency_stop")
                        print("Emergency stop command sent")
                time.sleep(0.01)
                
            except Exception as e:
                print(f"[Drone] Error in boundary checking: {str(e)}")
                time.sleep(0.1)  # Longer delay on error
            
    def _update_position(self):
        # It takes some time for the vicon to get values
        time.sleep(3)
        vicon_thread = threading.Thread(target=self.vicon.main_loop)
        vicon_thread.start()
        while self.running:
            try:
                position_array = self.vicon.getPos(self.drone_name)
                if position_array is not None:
                    self.position = {
                        "x": position_array[0],
                        "y": position_array[1],
                        "z": position_array[2]
                    }
                else:
                    print("Drone position is not being updated")

                time.sleep(0.090) # 90 Hz
            except Exception as e:
                print(f"[Drone] Error: Position data could not be parsed correctly - {str(e)}")
        # Signal the vicon thread to join
        self.vicon.run_interface = False
        vicon_thread.join()

    def _run(self):
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.1)
                if command == "exit":
                    self.running = False
                    print("[Drone] Shutting down.")
                elif command == "emergency_stop":
                    self.running = False
                    print("[Drone] EMERGENCY STOP initiated. Shutting down.")
                    # self.tello.land()
                else:
                    self._handle_command(command)
                self.command_queue.task_done()
            except queue.Empty:
                pass  # No command received; keep running
            # Display current position
            print(f"[Drone] Current position: {self.position}")
            # print(f"[Drone] Target position: {self.target_position}")

            time.sleep(0.5)

    def _handle_command(self, command):
        # Handle string commands first
        if not isinstance(command, dict):
            print(f"[Drone] String command received: {command}")
            return
            
        # From here on, we know command is a dictionary
        
        if "velocity" in command:
            self.velocity = command["velocity"]
            print(f"[Drone] Velocity set to: {self.velocity}")
        elif "position" in command:
            # This is a position command
            x = command["position"].get("x", self.position["x"])
            y = command["position"].get("y", self.position["y"])
            z = command["position"].get("z", self.position["z"])
            self.target_position = {"x": x, "y": y, "z": z}
            print(f"[Drone] Target position set: x={x}, y={y}, z={z}")
        elif "take_off" in command:
            try:
                print("[Drone] Executing take-off command")
                # self.tello.takeoff()
                print("[Drone] Take-off successful")
            except Exception as e:
                print(f"[Drone] Take-off failed: {str(e)}")
        elif "land" in command:
            try:
                print("[Drone] Landing drone")
                # self.tello.land()
                print("[Drone] Landing successful")
            except Exception as e:
                print(f"[Drone] Landing failed: {str(e)}")
        else:
            print(f"[Drone] Unknown command: {command}")

    def take_off(self):
        self.send_command({"take_off": True})

    def land(self, command):
        self.send_command({"land": True})
    
    def set_target_position(self, x: float, y: float, z: float) -> None:
        # Check the target position is within boundaries
        if not (0 <= x <= self.boundaries["x"] and 
                0 <= y <= self.boundaries["y"] and 
                0 <= z <= self.boundaries["z"]):
            print(f"[Drone] WARNING: Target position {x}, {y}, {z} is outside safe boundaries. Command rejected.")
            return
            
        # Create and send a position command
        position_command = {
            "position": {
                "x": x,
                "y": y,
                "z": z
            }
        }

        # Send the command to the queue
        self.send_command(position_command)
        print(f"[Drone] Target position command sent: x={x}, y={y}, z={z}")


    def get_position(self):
        return list(self.position)

    def send_command(self, command):
        self.command_queue.put(command)

    def stop(self):
        self.send_command("exit")
        # Join the run thread
        self.thread.join()
        # Join the vicon thread
        self.position_thread.join()
        # Join the safety thread
        self.safety_thread.join()


# def decide_next_action():
#     # Simulate decision-making logic (e.g., from a policy or sensor input)
#     return {"velocity": random.uniform(-2.0, 2.0)}
# Instantiate and control the drone
# drone = Drone()
# try:
#     for _ in range(5):
#         action = decide_next_action()
#         print(f"[Controller] Decided action: {action}")
#         drone.send_command(action)
#         time.sleep(1)  # Simulate time between decisions
# finally:
#     drone.stop()
#     print("[Controller] Drone stopped.")

if __name__ == "__main__":

    # Testing instructions
    # Face the drone away from you towards the windows
    # The window should continously move to the right and stop when it's out of bounds
    # Read the console for it's position
    
    drone = Drone()
    print("drone class initiated")
    # print("Taking off")
    # drone.take_off()
    # time.sleep(3)

    # print("Starting boundary test - going to continuously move to the right")
    # for i in range(10):
    #     drone.safety_landing_check()
    #     time.sleep(1.5)

    #     if not drone.in_boundaries:
    #         print("the drone is now out of bounds - check fo landing behaviour")
    #         break
    
    # time.sleep(10)

    

