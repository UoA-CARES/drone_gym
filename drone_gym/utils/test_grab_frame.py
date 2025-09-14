import numpy as np
import matplotlib.pyplot as plt
import io
import cv2
from mpl_toolkits.mplot3d import Axes3D

def grab_frame(episode_positions, goal_position, xy_limit, z_limit, reset_position, steps, successful_episodes_count, _is_evaluating, height: int = 240, width: int = 300) -> np.ndarray:
    """
    Generate a 3D plot of the drone's trajectory, convert to image array for video recording
    """
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Return black frame if no positions recorded yet
    if not episode_positions:
        plt.close(fig)
        return np.zeros((height, width, 3), dtype=np.uint8)

    # Convert positions to numpy array for easier manipulation
    pos_array = np.array(episode_positions)
    x, y, z = pos_array[:, 0], pos_array[:, 1], pos_array[:, 2]

    # Plot the drone's trajectory
    ax.plot(x, y, z, label='Drone Path', color='cyan', linewidth=2)

    # Mark important points
    ax.scatter(x[0], y[0], z[0], color='green', s=100, label='Start', depthshade=False)
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, label='Current', depthshade=False)
    ax.scatter(goal_position[0], goal_position[1], goal_position[2],
               color='blue', marker='*', s=200, label='Goal', depthshade=False)

    # Set consistent plot limits based on environment boundaries
    ax.set_xlim([-xy_limit, xy_limit])
    ax.set_ylim([-xy_limit, xy_limit])
    ax.set_zlim([0, z_limit + reset_position[2]])

    # Labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    # Add episode info to title
    success_info = f"Successes: {successful_episodes_count}" if _is_evaluating else ""
    ax.set_title(f'Episode Trajectory (Step {steps}) {success_info}')
    ax.legend()

    # Convert matplotlib figure to image array
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)

    # Decode the PNG buffer to numpy array
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close(fig)

    # Use cv2 to decode and resize
    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if frame is not None:
        frame = cv2.resize(frame, (width, height))
        # Convert BGR to RGB for consistency
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        # Fallback to black frame if decode fails
        frame = np.zeros((height, width, 3), dtype=np.uint8)

    return frame

if __name__ == '__main__':
    # Generate some hypothetical data points for a 3D drone trajectory
    num_points = 100
    t = np.linspace(0, 4 * np.pi, num_points)
    x = np.sin(t)
    y = np.cos(t)
    z = np.linspace(0, 5, num_points)
    episode_positions = list(zip(x, y, z))

    # Mock data that would normally come from the environment class
    goal_position = np.array([0, 0, 5])
    xy_limit = 1.5
    z_limit = 5
    reset_position = np.array([0, 0, 0])
    steps = num_points
    successful_episodes_count = 3
    _is_evaluating = True

    # Generate the frame
    frame = grab_frame(
        episode_positions=episode_positions,
        goal_position=goal_position,
        xy_limit=xy_limit,
        z_limit=z_limit,
        reset_position=reset_position,
        steps=steps,
        successful_episodes_count=successful_episodes_count,
        _is_evaluating=_is_evaluating,
        height=480,
        width=600
    )

    # Save the generated frame as an image
    # Convert RGB to BGR for cv2.imwrite
    output_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite('trajectory_plot.png', output_image)
    print("Trajectory plot saved as trajectory_plot.png")
