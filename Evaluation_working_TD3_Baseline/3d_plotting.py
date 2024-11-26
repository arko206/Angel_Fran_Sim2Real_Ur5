import matplotlib.pyplot as plt
import numpy as np




import ast  # For safely evaluating string representations of lists or arrays

import ast


##### Function for seperating lines 
##### with space
# def separate_epochs(file_path):
#     with open(file_path, 'r') as file:
#         content = file.read()
    
#     # Insert a newline before each 'epoch' keyword, except the first occurrence
#     updated_content = content.replace('epoch: ', '\nepoch: ')
    
#     # Write the updated content back to the file
#     with open(file_path, 'w') as file:
#         file.write(updated_content)

# # Replace with your file path
# file_path = "/home/robocupathome/arkaur5_ws/src/contexualaffordance/Ur5_DRL/Test28Nov_End_Effector_trajectory_baseline.txt"
# separate_epochs(file_path)
# print("Epochs have been separated into new lines.")


import numpy as np

def extract_epochs_and_trajectories(file_path):
    epochs_and_trajectories = []  # List to store the results
    epochs_list = []
    trajectory_list = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        if line.startswith("epoch:"):  # Identify lines starting with "epoch:"
            # Split the line by '||' to separate components
            parts = line.split('||')
            
            # Extract epoch number
            epoch_str = parts[0].split(':')[1].strip()
            epoch = int(epoch_str)

            ###Appending the epoch
            epochs_list.append(epoch)
            
            # Extract End_Effector_position
            position_str = parts[1].split(':')[1].strip()
            
            # Parse positions manually to handle 'array([...])'
            positions = []
            for segment in position_str.split('array(')[1:]:  # Split and skip the first non-array part
                array_data = segment.split(')')[0]  # Extract inside of 'array(...)'
                position = np.fromstring(array_data.strip("[]"), sep=',')  # Convert to numpy array
                positions.append(position)

            trajectory_list.append(positions)
            
            # Append the tuple (epoch, positions) to the list
            epochs_and_trajectories.append((epoch, positions))
    
    return epochs_and_trajectories, epochs_list, trajectory_list

# Replace with your file path
file_path = "/home/robocupathome/arkaur5_ws/src/contexualaffordance/Ur5_DRL/Test28Nov_End_Effector_trajectory_baseline.txt"
epochs_and_trajectories, epochs_list, trajectory_list = extract_epochs_and_trajectories(file_path)

# Example: Print the first two entries
for epoch, trajectory in epochs_and_trajectories[:2]:
    print(f"Epoch {epoch}: {trajectory}")

print(epochs_list[0])

print(trajectory_list[0])


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_3d_trajectory(episodes, trajectory, target_position):
    """
    Plots the 3D trajectory of the end-effector for multiple episodes.

    Args:
        episodes_data (list of list of tuples): List containing episode data. Each episode is a list of (x, y, z) coordinates.
        target_position (tuple): Coordinates of the target position (x, y, z).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot target position
    target_x, target_y, target_z = target_position
    ax.scatter(target_x, target_y, target_z, color='red', s=100, label='Target Position', alpha= 0.2)

    # # Plot trajectories for each episode
    for episode_idx, trajectory_item in zip(episodes, trajectory):

        if episode_idx%10 == 0:
            trajectory_item = np.array(trajectory_item)

            # Extract x, y, and z coordinates
            x_coords = [pos[0] for pos in trajectory_item]
            y_coords = [pos[1] for pos in trajectory_item]
            z_coords = [pos[2] for pos in trajectory_item]

            ax.plot(x_coords, y_coords, z_coords, label=f'Episode {episode_idx + 1}')

    # x_coords = [pos[0] for pos in trajectory[499]]
    # print(x_coords)
    # y_coords = [pos[1] for pos in trajectory[499]]
    # z_coords = [pos[2] for pos in trajectory[499]]
    # print(len(trajectory[499]))

    # ax.plot(x_coords, y_coords, z_coords, label=f'Episode {500}')



    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('End-Effector Trajectories')
    ax.legend()
    plt.savefig('Test_2000_End_Effectory_Trajectory.png')

    plt.show()


target_position = [-0.525000,0.00,0.854007]
plot_3d_trajectory(epochs_list, trajectory_list, target_position)
