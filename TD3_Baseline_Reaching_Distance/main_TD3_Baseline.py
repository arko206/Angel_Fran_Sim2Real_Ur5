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
train_reward_filename = os.path.join(base_log_dir, 'TD3_18thNov_training_rewards_whole.txt')
# test_reward_filename = os.path.join(base_log_dir, 'TD311THAugust_testing_rewards_whole.txt')

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
            model.load(args.path_to_model+args.model_name, args.model_date_+'/')
            print ('load model successfully')
        except:
            print ('fail to load model, check the path of models')

        
    
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
    
    ### variable keeping track of number of timesteps has been counted
    total_num_of_steps = 0
   
    
    for epoch in range(0, 1000):

        ###creating an empty list for storing training episodes
        ### and cumulative reward in each episode

        episode_timesteps = 0
        
        ### reset the environment
        if args.model_name == 'TD3':
            state, object_position = env.reset()
            
            ###Creating a History of Storing Actions and distances in an Episode
            action_history = []
            distance_history = []

        total_reward = 0
    
        
        not_done =  True 
        while not_done:
            if args.model_name == 'TD3':

                total_num_of_steps += 1
                
                ### selection of an action from a given state
                action = (model.select_action(state) + np.random.normal(0, max_action * expl_noise, size= env.action_dim)).clip(-max_action, max_action)

                #### adding action to the list
                action_history.append(action)

                state_, reward, end, truncated, collision_happened, target_pos, end_eff_pos = env.step(action, object_position,  episode_timesteps)
                
                ### adding to the replay buffer
                replay_buffer.add(state, action, state_, reward, truncated)
                
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
                
                
        total_reward_list.append(total_reward)
        train_episodes_list.append(epoch)
        distance_half_history = len(distance_history)//2
        final_distance_to_target = (2/len(distance_history)) * sum(distance_history[distance_half_history:])

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

        # #begin testing and record the evalation metrics
        if (epoch+1) % 10 == 0:
            model.save(args.path_to_model+args.model_name, args.model_date+'/')

        
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




            
    

def test(args,env, model):
    model.mode(mode='test')
    print ('start to test the model')
    try:
        model.load_model(args.path_to_model+args.model_name, args.model_date_+'/')
        #model.load_model('/home/robocupathome/arkaur5_ws/src/contexualaffordance/RL_trialTD3/06_06_2024')
        print(args.path_to_model+args.model_name, args.model_date_+'/')
        print ('load model successfully')
    except Exception as e:
        rospy.logwarn(e)
        print ('fail to load model, check the path of models')

    total_reward_list = []
    epochs_list = []
    dis_to_target_list = []
    smoothness_test_cost = []
    #testing for vision observation
    for epoch in range(args.test_epoch):
        if args.model_name == 'TD3':
            state, object_pos = env.reset()
            state_history = [None for _ in range(max_ts)]
            random_noises_test = np.random.uniform(-MAX_NOISE, MAX_NOISE, (max_ts, 8))
            random_latencies_test = np.random.uniform(0, MAX_LATENCY, (max_ts, 1))
            previous_latency = 0
            action_history = [None for _ in range(max_ts + 1)]
        total_reward = 0
        distance_to_target = 0
        for step in range(args.test_step):
            state_history[step] = state
            if args.model_name == 'TD3':
                ####latency is added in current state
                if step > 0 and step < 200:
                    latency = random_latencies_test[step][0] #self.np_random.uniform(0, MAX_LATENCY)
                    latency = min(step * DELTA, latency)
                    # latency can't be greater than the last latency and the delta time since then
                    latency = min(previous_latency + DELTA, latency)
                    # store the latest latency for the next step
                    previous_latency = latency

                    if latency == 0:
                       lat_state = state
                    else:
                        # the latency is a linear interpolation between the two discrete states that correspond
                        ratio = round(latency/DELTA, 6)
                        earlier_state = state_history[step-math.ceil(ratio)]
                        earlier_state_portion = (ratio-math.floor(ratio))
                        later_state = state_history[step-math.floor(ratio)]
                        try:
                           lat_state = [x *  earlier_state_portion + y * (1 - earlier_state_portion) for x, y in zip(earlier_state, later_state)]
                           state = lat_state
                        except:
                           print("Error happened")
                ####Noise Randomization is added to the State Space
                if step < 200:
                    state = np.array(state) + random_noises_test[step]
                action = model.choose_action(state,noise=None)
                action_history[step] = action
                state_, reward, terminal, target_pos, end_eff_pos = env.step(action*args.action_bound, object_pos)
                state = state_
                total_reward = total_reward + (0.99)**step * reward

                if step > 100 and step < 200:
                    dis_diff = (np.linalg.norm(end_eff_pos - target_pos))**2
                    distance_to_target += dis_diff

                if terminal:
                    state, object_pos = env.reset()
        total_reward_list.append(total_reward)
        final_distance_to_target = 0.01 * distance_to_target
        dis_to_target_list.append(final_distance_to_target)

        ###Evaluation of Continuity Cost of an Episode
        deltas = [None for _ in range(max_ts-1)]
        for i in range(0, max_ts-1):
            deltas[i] = np.mean(np.power(action_history[i] - action_history[i+1], 2) / 6.28)
        deltas = np.array(deltas)

        final_continuity_cost = 100 * np.mean(deltas)
        smoothness_test_cost.append(final_continuity_cost)

        print ('testing_epoch:', epoch,  '||',  'Reward:', total_reward, '||', 'Distance_to_target:', final_distance_to_target, '||', 'Continuity_Cost:', final_continuity_cost)
        ###Writing in file
        # with open(test_reward_filename, "a") as file:
        #     file.write(f"epoch: {epoch} || Test_Reward: {total_reward} || Final_Distance_to_Target: {final_distance_to_target} || Continuity_Cost: {final_continuity_cost} \n")

        ####adding each iterations
        epochs_list.append(epoch)


    #average_reward = np.mean(np.array(total_reward_list))
    #average_step = 0 if steps_list == [] else np.mean(np.array(steps_list))

    return total_reward_list, epochs_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #select env to be used
    parser.add_argument('--env_name', default='empty')
    #select model to be used
    parser.add_argument('--model_name', default='TD3')
    #Folder name saved as date
    parser.add_argument('--model_date', default='/18_11_2024')
    #Folder stored with trained model weights, which are used for transfer learning
    parser.add_argument('--model_date_', default='/18_11_2024')
    parser.add_argument('--pre_train', default=False)

    #####has to be changed
    parser.add_argument('--path_to_model', default='/home/robocupathome/rosur5_ws/src/contexualaffordance/DRLMODELTD3')
    #The maximum action limit

    ####has to be chnaged
    # parser.add_argument('--action_bound', default=np.pi/72, type=float) #pi/36 for reachings
    parser.add_argument('--train_epoch', default=1000, type=int)
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

    if args.mode == 'test': 
        env.duration = 0.1
        test(args, env, model)
