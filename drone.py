import threading
import queue
import time
import random
from vicon_connection_class import ViconInterface as vi


class Drone:

    def __init__(self):
        self.command_queue = queue.Queue()
        self.velocity = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

        # Vicon Integration
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.drone_name = "AtlasCrazyflie" 
        self.vicon = vi()
        self.position_thread = threading.Thread(target=self._update_position)
        self.position_thread.start()

        # Drone Safety
        self.boundaries = {"x": 6, "y": 5.5, "z": 4} 
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
                x_in_bounds = 0 <= self.position["x"] <= self.boundaries["x"]
                y_in_bounds = 0 <= self.position["y"] <= self.boundaries["y"]
                z_in_bounds = 0 <= self.position["z"] <= self.boundaries["z"]
                
                # Set in_boundaries status
                current_status = x_in_bounds and y_in_bounds and z_in_bounds
                
                if current_status != self.in_boundaries:
                    self.in_boundaries = current_status
                    if self.in_boundaries:
                        print("Drone currently in bounds")
                    else:
                        print(f"[Drone] WARNING: Out of bounds!")

                        print("placeholder emergency landing")

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
        # Signal the thread to join
        self.vicon.run_interface = False
        vicon_thread.join()

    def _run(self):
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.1)
                if command == "exit":
                    self.running = False
                    print("[Drone] Shutting down.")
                else:
                    self._handle_command(command)
                self.command_queue.task_done()
            except queue.Empty:
                pass  # No command received; keep running
            # Simulate ongoing drone operation
            print(f"[Drone] Current position: {self.position}")
            time.sleep(0.5)

    def _handle_command(self, command):
        if isinstance(command, dict):
            if "velocity" in command:
                self.velocity = command["velocity"]
                # print(f"[Drone] Velocity set to: {self.velocity}")
            elif "command_type" in command and command["command_type"] == "target_position":
                # Extract position data
                pos = command.get("position", {})
                x = pos.get("x", self.position["x"])
                y = pos.get("y", self.position["y"])
                z = pos.get("z", self.position["z"])
                
            else:
                print(f"[Drone] Unknown command: {command}")
        else:
            print(f"[Drone] Unknown command: {command}")

    def take_off(self, command):
        pass

    def land(self, command):
        pass

    def set_target_position(self, x: float, y: float, z: float) -> None:
 
        # Check the target position is within boundaries
        if not (0 <= x <= self.boundaries["x"] and 
                0 <= y <= self.boundaries["y"] and 
                0 <= z <= self.boundaries["z"]):
            print(f"[Drone] WARNING: Target position {x}, {y}, {z} is outside safe boundaries. Command rejected.")
            return
            
        # Update the target position attribute
        self.target_position = {"x": x, "y": y, "z": z}
        
        # Create and send a target position command
        position_command = {
            "command_type": "target_position",
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
        
        return list(self.postion)


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

    

def decide_next_action():
    # Simulate decision-making logic (e.g., from a policy or sensor input)
    return {"velocity": random.uniform(-2.0, 2.0)}
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
    
    drone = Drone()

    
