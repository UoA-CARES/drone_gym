import numpy as np
import time
from drone_gym.move_to_2d_position import MoveToPosition

env = MoveToPosition()

for episode in range(3):
    print(f"\n=== Episode {episode + 1} ===")
    state = env.reset(training=True)
    
    step = 0
    while not env.done and step < 50:
        pos = env.drone.get_position()
        goal = env.goal_position
        
        # Calculate direction to goal
        dx = goal[0] - pos[0]
        dy = goal[1] - pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Create action pointing towards goal
        if(distance > 0.01):
            action = np.array([dx / distance, dy / distance])
        else:
            action = np.array([0.0, 0.0])
        
        # Execute step
        state, reward, done, truncated, info = env.step(action)
        step += 1
        
        print(f"Step {step}: Position=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
              f"Distance={distance:.3f}, Reward={reward:.1f}, Done={done}")
        
        if done:
            print(f"GOAL REACHED in {step} steps!")
            break
        
        time.sleep(0.1)

env.close()
print("\nTest complete!")