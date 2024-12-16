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



# File paths for logging rewards
test_reward_filename = os.path.join(base_log_dir, 'Decemebr15_rewards_whole_TestOnly.txt')
end_effector_position_filename_test = os.path.join(base_log_dir, 'December15_End_Effector_trajectory_baseline.txt')

# max_ts = 50
##time duration in seconds
DELTA = 0.02

##maximum allowed latency
MAX_LATENCY = 1
MAX_NOISE = 0.1



def get_time(start_time):
    m, s = divmod(int(time.time()-start_time), 60)
    h, m = divmod(m, 60)

    print ('Total time spent: %d:%02d:%02d' % (h, m, s))



def train(args, env, model):

    ###setting the saving path for the trained model
    if not os.path.exists(args.path_to_model+args.model_name+args.model_date):
        os.makedirs(args.path_to_model+args.model_name+args.model_date)

    start_time = time.time()

    ### Loading the Pretrained Weights
    if args.pre_train:
        try:
            base_path = "/home/robocupathome/rosur5_ws/src/contexualaffordance/DRLMODELTD3TD3/15_12_2024/epoch_2000/"
            actor_path = os.path.join(base_path, "model_epoch_2000_actor.pth")
            critic_path = os.path.join(base_path, "model_epoch_2000_critic.pth")
            optim_a_path = os.path.join(base_path, "model_epoch_2000_actor_optimizer.pth")
            optim_c_path = os.path.join(base_path, "model_epoch_2000_critic_optimizer.pth")
            model.load_model(actor_path, critic_path, optim_a_path, optim_c_path)
            print('Model loaded successfully')
        except Exception as e:
            rospy.logwarn(e)
            print ('fail to load model, check the path of models')
        

        


    print ('start testing')
    

    test_reward, epochs_step = test(args, env, model)
    print ('finish testing')


def test(args,env, model):
    model.mode(mode='test')
    print ('start to test the model')
    try:
            base_path = "/home/robocupathome/rosur5_ws/src/contexualaffordance/DRLMODELTD3TD3/15_12_2024/epoch_2000/"
            actor_path = os.path.join(base_path, "model_epoch_2000_actor.pth")
            critic_path = os.path.join(base_path, "model_epoch_2000_critic.pth")
            optim_a_path = os.path.join(base_path, "model_epoch_2000_actor_optimizer.pth")
            optim_c_path = os.path.join(base_path, "model_epoch_2000_critic_optimizer.pth")
            model.load_model(actor_path, critic_path, optim_a_path, optim_c_path)
            print('Model loaded successfully')
    except Exception as e:
            rospy.logwarn(e)
            print ('fail to load model, check the path of models')
        

    test_reward_list = []
    epochs_list = []
    dis_to_target_list = []
    smoothness_test_cost = []
    num_of_steps_test = []
    end_eff_traj_test = []

    max_action = 0.2
    expl_noise = 0.1

    for epoch in range(0, 50):

        ###creating an empty list for storing training episodes
        ### and cumulative reward in each episode

        episode_timesteps = 0

        end_eff_history = []
        
        ### reset the environment
        if args.model_name == 'TD3':
            state, object_position, initial_end_eff_pos = env.reset()
            end_eff_history.append(initial_end_eff_pos)
            print("Initial end-effector position is: ", initial_end_eff_pos)
            
            ###Creating a History of Storing Actions and distances in an Episode
            action_history = []
            distance_history = []

            

        total_reward = 0
    
        
        not_done =  True 
        while not_done:
            if args.model_name == 'TD3':

                
                ### selection of an action from a given state
                #action = (model.select_action(state)).clip(-max_action, max_action)
                with torch.no_grad():
                     action = model.select_action(state)

                #### adding action to the list
                action_history.append(action)

                state_, reward, end, truncated, terminal, collision_happened, target_pos, end_eff_pos = env.step(action, object_position,  episode_timesteps)
                end_eff_history.append(end_eff_pos)
                print("End Effector position is: ", end_eff_pos)
                print("Traget position is: ", target_pos)
                
                
                state = state_

                total_reward = total_reward + (0.99) * reward

                
                dis_diff = (np.linalg.norm(end_eff_pos - target_pos))**2
                distance_history.append(dis_diff)
                
                


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

        end_eff_traj_test.append(end_eff_history)
        test_reward_list.append(total_reward)
        num_of_steps_test.append(episode_timesteps)

        distance_half_history = len(distance_history)//2
        final_distance_to_target = (2/len(distance_history)) * sum(distance_history[distance_half_history:])
        dis_to_target_list.append(final_distance_to_target)

        

        ###Evaluation of Continuity Cost of an Episode
        deltas = [None for _ in range(episode_timesteps-1)]
        for i in range(0, episode_timesteps-1):
            deltas[i] = np.mean(np.power(action_history[i] - action_history[i+1], 2) / (2 * max_action))
        deltas = np.array(deltas)
        final_continuity_cost = 100 * np.mean(deltas)
        smoothness_test_cost.append(final_continuity_cost)

        print ('testing_epoch:', epoch,  '||',  'Reward:', total_reward, '||', 'Distance_to_target:', final_distance_to_target, '||', 'Continuity_Cost:', final_continuity_cost, 'Episode_Timesteps:', episode_timesteps)
        ###Writing in file
        with open(test_reward_filename, "a") as file:
            file.write(f"epoch: {epoch} || Test_Reward: {total_reward} || Final_Distance_to_Target: {final_distance_to_target} || Continuity_Cost: {final_continuity_cost} || Episode_Timesteps: {episode_timesteps}\n")
        
        with open(end_effector_position_filename_test, "a") as file:
            file.write(f"epoch: {epoch} || End_Effector_position: {end_eff_history} || Distance_History: {distance_history}")

        ####adding each iterations
        epochs_list.append(epoch)


    #average_reward = np.mean(np.array(total_reward_list))
    #average_step = 0 if steps_list == [] else np.mean(np.array(steps_list))

    return test_reward_list, epochs_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #select env to be used
    parser.add_argument('--env_name', default='empty')
    #select model to be used
    parser.add_argument('--model_name', default='TD3')
    #Folder name saved as date
    parser.add_argument('--model_date', default='/15_12_2024')
    #Folder stored with trained model weights, which are used for transfer learning
    parser.add_argument('--model_date_', default='/15_12_2024')
    parser.add_argument('--pre_train', default=True)

    #####has to be changed
    parser.add_argument('--path_to_model', default='/home/robocupathome/rosur5_ws/src/contexualaffordance/DRLMODELTD3')
    #The maximum action limit

    ####has to be chnaged
    parser.add_argument('--action_bound', default=np.pi/72, type=float) #pi/36 for reachings
    #parser.add_argument('--train_epoch', default=600, type=int)
    #parser.add_argument('--train_step', default=200, type=int)
    parser.add_argument('--test_epoch', default=50, type=int)
    parser.add_argument('--test_step', default=50, type=int)
    
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
    assert args.model_name == 'TD3', 'model name: 1.TD3'
    #if args.model_name == 'TD3_vision': model = TD3_vision(a_dim=env.action_dim,s_dim=env.state_dim,cuda=args.cuda)
    #if args.model_name == 'TD3': model = TD3(a_dim=env.action_dim,s_dim=env.state_dim,cuda=args.cuda)
    if args.model_name == 'TD3': model = TD3(state_dim=env.state_dim, action_dim=env.action_dim)

    assert args.mode == 'test' or 'test', 'mode: 1.train 2.test'
    if args.mode == 'train': 
        train(args, env, model)

    if args.mode == 'test': 
        env.duration = 0.1
        test(args, env, model)
