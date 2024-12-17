from NewMoveit_Sheng_Env import Ur5
#from env2 import Ur5_vision
#from DDPG import DDPG
from TD3_Baseline import TD3
#from TD3_vision import TD3_vision
import numpy as np 
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import time
import argparse
import os
import numpy as np
import rospy
import actionlib
import time
from control_msgs.msg import *
from trajectory_msgs.msg import *
from sensor_msgs.msg import JointState
from tf import TransformListener
from math import pi
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import sys, tf
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
from copy import deepcopy
import math
import utils


# Define the base directory where you want to save log files
base_log_dir = "/home/robocupathome/arkaur5_ws/src/contexualaffordance/Ur5_DRL"

# Make sure the directory exists, if not, create it
if not os.path.exists(base_log_dir):
    os.makedirs(base_log_dir)



# File paths for logging rewards and storing end effector co-ordinates
train_reward_filename = os.path.join(base_log_dir, 'DomainRandomization_Dec17_Collisionchecking_training_rewards_whole.txt')

end_effector_position_filename = os.path.join(base_log_dir, 'DRandomization_Dec17End_Effector_trajectory_baseline.txt')

# max_ts = 200
##time duration in seconds
DELTA = 0.02

##maximum allowed latency
MAX_LATENCY = 1
MAX_NOISE = 0.1

#### Function to determine the amount of time
#### needed to execute a particualr timestep
def get_time(start_time):
    m, s = divmod(int(time.time()-start_time), 60)
    h, m = divmod(m, 60)

    print ('Total time spent: %d:%02d:%02d' % (h, m, s))


#### Training Function

def train(args, env, model):

    ###setting the saving path for the trained model
    if not os.path.exists(args.path_to_model+args.model_name+args.model_date):
        os.makedirs(args.path_to_model+args.model_name+args.model_date)

  
    start_time = time.time()

    ### Loading the Pretrained Weights
    if args.pre_train:
        try:
            base_path = "/home/robocupathome/rosur5_ws/src/contexualaffordance/DRLMODELTD3TD3/26_11_2024/epoch_2000/"
            actor_path = os.path.join(base_path, "model_epoch_2000_actor.pth")
            critic_path = os.path.join(base_path, "model_epoch_2000_critic.pth")
            optim_a_path = os.path.join(base_path, "model_epoch_2000_actor_optimizer.pth")
            optim_c_path = os.path.join(base_path, "model_epoch_2000_critic_optimizer.pth")
            model.load_model(actor_path, critic_path, optim_a_path, optim_c_path)
            print('Model loaded successfully')
        except Exception as e:
            rospy.logwarn(e)
            print ('fail to load model, check the path of models')


    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
        
    #torch.manual_seed(0)
    ### initializing the Replay Buffer
    replay_buffer = utils.ReplayBuffer(env.state_dim, env.action_dim)
    
    print ('start training')
    model.mode(mode='train')

    #training for observation with TD3 Algorithm
    max_action = 0.2

    expl_noise = 0.1

    train_episodes_list = []
    total_reward_list = []
    dis_to_target_list = []
    smoothness_action_cost = []
    num_of_steps = []
    end_eff_traj_final = []
    
    ### variable keeping track of number of timesteps has been counted
    total_num_of_steps = 0
   
    
    for epoch in range(0, 2000):

        ###creating an empty list for storing END-EFFECTOR TRAJECTORIES

        episode_timesteps = 0

        end_eff_history = []

       
        
        ### reset the environment
        if args.model_name == 'TD3':
            state, object_position, initial_end_eff_pos = env.reset()
            
            ### appending the initial position of the end-effector
            end_eff_history.append(initial_end_eff_pos)
            
            ###Creating a History of Storing Actions and distances in an Episode
            action_history = []
            distance_history = []
            state_history = []
            
            ### Parameters for Domain Randomization and each episode runs for 
            ### maximum 100 timesteps
            random_latencies = np.random.uniform(0, MAX_LATENCY, (100, 1))
            random_noises = np.random.uniform(-MAX_NOISE, MAX_NOISE, (100,env.state_dim))
            previous_latency = 0
            
            

            

        total_reward = 0
    
        
        not_done =  True 
        while not_done:
            if args.model_name == 'TD3':

                total_num_of_steps += 1

                state_history.append(state)

                if episode_timesteps > 0 and episode_timesteps < 100:
                    latency = random_latencies[episode_timesteps][0]

                    ###latency cannot be greater that the last latency
                    #### and delta times since then
                    latency = min(episode_timesteps * DELTA, latency)
                    latency = min(previous_latency+DELTA, latency)

                    ##storing the latest latency for the next step
                    previous_latency = latency

                    if latency == 0:
                        lat_state = state
                    else:
                        # the latency is a linear interpolation between two continuous states 
                        ratio = round(latency/DELTA,6)
                        earlier_state = state_history[episode_timesteps-math.ceil(ratio)]
                        earlier_state_portion = (ratio-math.floor(ratio))
                        later_state = state_history[episode_timesteps-math.floor(ratio)]
                        try:
                            lat_state = [x * earlier_state_portion + y * (1 - earlier_state_portion) for x, y in zip(earlier_state, later_state)]
                            state = lat_state
                        except:
                            print("Error happened")

                if episode_timesteps < 100:
                    state = np.array(state) + random_noises[episode_timesteps]
                    state_clipped = np.clip(state, a_min= -2*np.pi, a_max= 2*np.pi)
                
                ### selection of an action from a given state
                action = (model.select_action(np.array(state_clipped)) + np.random.normal(0, max_action * expl_noise, size= env.action_dim)).clip(-max_action, max_action)

                #### adding action to the list
                action_history.append(action)

                state_, reward, end, truncated, terminal, collision_happened, target_pos, end_eff_pos = env.step(action, object_position,  episode_timesteps)
                end_eff_history.append(end_eff_pos)
                
                ### adding to the replay buffer
                replay_buffer.add(state, action, state_, reward, end)
                
                state = state_

                total_reward = total_reward + (0.99) * reward

                
                dis_diff = (np.linalg.norm(end_eff_pos - target_pos))**2
                distance_history.append(dis_diff)
                
                if total_num_of_steps  >= args.random_exploration:
                    print("Training of Networks Started")
                    model.train(replay_buffer, 256)


                ### end conditions looks if the end-effector reaches desired goal or not
                ### collision_happened checks the present state of the arm and truncated 
                #### for if the agent reaches maximum timestep or goal position
                if end or collision_happened or truncated:
                    # state, object_position = env.reset()
                    not_done = False
                
                ### counting the numer of steps when moveit fails
                ### and moveit plans to acheive sub-optimal pos
                # if count_steps % 50 == 0:
                #       state, object_position = env.reset()

                episode_timesteps += 1
        
        if terminal == True:
            save_path = f"{args.path_to_model}{args.model_name}/{args.model_date}/epoch_{epoch + 1}/"
            os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist
            # Call the save function
            model.save(save_path, f"model_epoch_{epoch + 1}")
                

        end_eff_traj_final.append(end_eff_history)       
        total_reward_list.append(total_reward)
        train_episodes_list.append(epoch)
        distance_half_history = len(distance_history)//2
        final_distance_to_target = (2/len(distance_history)) * sum(distance_history[distance_half_history:])

        num_of_steps.append(episode_timesteps)

        dis_to_target_list.append(final_distance_to_target)
        
        ###Evaluation of Continuity Cost of an Episode
        deltas = [None for _ in range(episode_timesteps-1)]
        for i in range(0, episode_timesteps-1):
            deltas[i] = np.mean(np.power(action_history[i] - action_history[i+1], 2) / (2 * max_action))
        deltas = np.array(deltas)
        final_continuity_cost = 100 * np.mean(deltas)
        smoothness_action_cost.append(final_continuity_cost)

        print ('epoch:', epoch,  '||',  'Reward:', total_reward, '||', 'Dis_to_Target:', final_distance_to_target, '||', 'Continuity_Cost:', final_continuity_cost, '||', 'Number_of_steps:', episode_timesteps)
        with open(train_reward_filename, "a") as file:
            file.write(f"epoch: {epoch} || Train_reward: {total_reward} || Final_Dis_Target:{final_distance_to_target} || Continuity_Cost:{final_continuity_cost} || Number_of_steps:{episode_timesteps}\n")

        with open(end_effector_position_filename, "a") as file:
            file.write(f"epoch: {epoch} || End_Effector_position: {end_eff_history} || Distance_History: {distance_history}\n")

        # #begin testing and record the evalation metrics
        if (epoch + 1) % 10 == 0:
            save_path = f"{args.path_to_model}{args.model_name}/{args.model_date}/epoch_{epoch + 1}/"
            os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist
            # Call the save function
            model.save(save_path, f"model_epoch_{epoch + 1}")

            

        
    ####Plotting for Training Rewards
    plt.figure()
    plt.plot(train_episodes_list, total_reward_list, color='g')
    plt.ylabel('Total_reward')
    plt.xlabel('training epoch')
    plt.title('Training_reward_total')
    plt.savefig(args.path_to_model+args.model_name+args.model_date+f'/training_reward_{epoch+1}.png')
    plt.close()
    
    

    ####Plotting for Distance To Target
    plt.figure()
    plt.plot(train_episodes_list, dis_to_target_list, color='violet')
    plt.ylabel('Distance to the Target')
    plt.xlabel('training epoch')
    plt.title('Training_Reaching')
    plt.savefig(args.path_to_model+args.model_name+args.model_date+f'/training_Dis_to_Target{epoch+1}.png')
    plt.close()

    ####Plotting for Continuity of Smoothness
    plt.figure()
    plt.plot(train_episodes_list, smoothness_action_cost, color='brown')
    plt.ylabel('Continuity of Action Smoothness')
    plt.xlabel('training epoch')
    plt.title('Training_Smoothness')
    plt.savefig(args.path_to_model+args.model_name+args.model_date+f'/training_Smoothness{epoch+1}.png')
    plt.close()

    ####Plotting for Number of Steps
    plt.figure()
    plt.plot(train_episodes_list, num_of_steps, color='orange')
    plt.ylabel('Number of Steps in an Episode')
    plt.xlabel('training epoch')
    plt.title('Number_of_Steps')
    plt.savefig(args.path_to_model+args.model_name+args.model_date+f'/Number_of_Steps{epoch+1}.png')
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #select env to be used
    parser.add_argument('--env_name', default='empty')
    #select model to be used
    parser.add_argument('--model_name', default='TD3')
    #Folder name saved as date
    parser.add_argument('--model_date', default='/17_12_2024')
    #Folder stored with trained model weights, which are used for transfer learning
    parser.add_argument('--model_date_', default='/17_12_2024')
    parser.add_argument('--pre_train', default=False)

    #####has to be changed
    parser.add_argument('--path_to_model', default='/home/robocupathome/rosur5_ws/src/contexualaffordance/DRLMODELTD3')
    #The maximum action limit

    ####has to be chnaged
    # parser.add_argument('--action_bound', default=np.pi/72, type=float) #pi/36 for reachings
    parser.add_argument('--train_epoch', default=2000, type=int)
    parser.add_argument('--train_step', default=200, type=int)
    parser.add_argument('--test_epoch', default=10, type=int)
    parser.add_argument('--test_step', default=200, type=int)
    #exploration (randome action generation) steps before updating the model
    parser.add_argument('--random_exploration', default=1000, type=int)
    #store the model weights and plots after epoch number
    parser.add_argument('--epoch_store', default=10, type=int)
    #Wether to use GPU
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--mode', default='train')
    args = parser.parse_args()

    #assert args.env_name == 'empty' or 'vision', 'env name: 1.empty 2.vision'
    assert args.env_name == 'empty', 'env name: 1.empty'
    if args.env_name == 'empty': env = Ur5()
    #if args.env_name == 'vision': env = Ur5_vision()

    #assert args.model_name == 'TD3_vision' or 'TD3' or 'DDPG', 'model name: 1.TD3_vision 2.TD3 3.DDPG'
    #assert args.model_name == 'DDPG', 'model name: 1.DDPG '
    #if args.model_name == 'TD3_vision': model = TD3_vision(a_dim=env.action_dim,s_dim=env.state_dim,cuda=args.cuda)
    if args.model_name == 'TD3': model = TD3(state_dim=env.state_dim, action_dim=env.action_dim)
    #if args.model_name == 'DDPG': model = DDPG(a_dim=env.action_dim,s_dim=env.state_dim,cuda=args.cuda)

    assert args.mode == 'train' or 'test', 'mode: 1.train 2.test'
    if args.mode == 'train': 
        train(args, env, model)

    # if args.mode == 'test': 
    #     env.duration = 0.1
    #     test(args, env, model)
