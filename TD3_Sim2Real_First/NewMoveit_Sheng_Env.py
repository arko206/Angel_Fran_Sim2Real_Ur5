import numpy as np 
import rospy
import actionlib
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
from math import sin, cos
from sensor_msgs.msg import JointState
from moveit_commander import PlanningSceneInterface
from std_srvs.srv import Empty
import time
#from ur_msgs.srv import GetRobotStateRv

import numpy as np
import os



import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
from tf.transformations import euler_from_quaternion
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from tf.transformations import euler_from_quaternion
from moveit_msgs.srv import GetStateValidityRequest, GetStateValidity
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPlanningScene, GetPlanningSceneRequest


JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
DURATION = 3

###Goal contains first 3 co-ordinates of end effector's x, y, z
###Goal contains second 3 c-ordinates of end-effectr's r, p, y
# GOAL = [0.525000, 0.00, 0.854007, 0.00, 0.00, 0.00]

### The Goal Position has been chnaged
GOAL = [0.7, 0.00, 0.854007, 0.00, 0.00, 0.00]

INIT = [0.0, -1.571, 1.571, -1.571, 0.0, 0.0]

##taking this value from Josip's Paper
#max_dist = 2.004



####Class consists of methods to find whether collision
#### exists in the environment or not
class StateValidity():
    def __init__(self):
        # subscribe to topic 'joint_states' of message type 'JointState'
        ### Msgs are passed to callback function 'jointStatesCB'
        rospy.Subscriber("joint_states", JointState, self.jointStatesCB, queue_size=1)

        # prepare service for collision check
        ### ros service 'check_state_validity'
        ## has been called to check validity 
        ## of the state
        self.sv_srv = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
        # wait for service to become available
        self.sv_srv.wait_for_service()
        rospy.loginfo('service is avaiable')
        # prepare msg to interface with moveit
        self.rs = RobotState()
        self.rs.joint_state.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

        self.rs.joint_state.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_states_received = False


    def checkCollision(self):
        '''
        check if robotis in collision
        '''

        collision_hapened = False
        if self.getStateValidity().valid:
            rospy.loginfo('robot not in collision, all ok!')
        else:
            rospy.logwarn('robot in collision')
            collision_hapened = True

        return collision_hapened


    def jointStatesCB(self, msg):
        '''
        update robot state
        '''
        self.rs.joint_state.position = [msg.position[0], msg.position[1], msg.position[2],
        msg.position[3], msg.position[4], msg.position[5]]

        ####checking the joint angle values
        #print(self.rs.joint_state.position)

        ##creating the flag on receiving the joint states
        self.joint_states_received = True


    def getStateValidity(self, group_name='manipulator', constraints=None):
        '''
        Given a RobotState and a group name and an optional Constraints
        return the validity of the State
        '''
        gsvr = GetStateValidityRequest()
        gsvr.robot_state = self.rs
        gsvr.group_name = group_name
        if constraints != None:
            gsvr.constraints = constraints
        result = self.sv_srv.call(gsvr)
        return result


    def start_collision_checker(self):
        while not self.joint_states_received:
            rospy.sleep(0.1)
        rospy.loginfo('joint states received! continue')
        result_collision = self.checkCollision()
        # rospy.spin()

        return result_collision



class Ur5():
    
    # metadata = {"render_modes": ["human"]}

    def __init__(self, init_joints=INIT, goal_pose=GOAL, duration=DURATION, render_mode=None):
        
    
        # Initialize ROS node
        rospy.init_node('ur5_env', anonymous=True)


        ### initialization for moveit connection
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group = MoveGroupCommander("manipulator")

        ### setting the planner duration
        self.duration = DURATION


        self.goal_pose = np.array(goal_pose, dtype=np.float32)

        # self.base_pos = self.get_pos(link_name='base_link')
        
        self.state_dim = 7
        self.action_dim = 6

        self.tf = TransformListener()   

        ### the reference farme is considered as 
        ### the world
        reference_frame = "world"

        # Set the ur5_arm reference frame accordingly
        self.group.set_pose_reference_frame(reference_frame)

        self.counter = 0

        ##getting the name of the end-effector link
        self.end_effector_link = self.group.get_end_effector_link()

        ###setting the planning time for the arm
        self.group.set_planning_time(self.duration)

        # self.group.set_planner_id("LBKPIECE")

        
        # Add table to the MoveIt planning scene
        self.add_table_to_scene()

       
        
        ### object is generated
        self.target_generate()

        ###object for Checking Validity State Class
        self.collision_checker_node = StateValidity()

        # Service to get the current planning scene (which includes the collision matrix)
        self.get_planning_scene = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)

        # Wait for the service to be available
        rospy.wait_for_service('/get_planning_scene')

        # Request the full planning scene, including the allowed collision matrix
        self.request = GetPlanningSceneRequest()
        




    # Method to add the table to the MoveIt planning scene
    def add_table_to_scene(self):
        table_pose = PoseStamped()
        table_pose.header.frame_id = "world"  # Use the correct reference frame
        table_pose.pose.position.x = 0.0  # Set to your table's position (centered at origin)
        table_pose.pose.position.y = 0.0
        table_pose.pose.position.z = 0.005  # Half of the table's height to place it on the ground
        table_pose.pose.orientation.w = 1.0

        # Add the table to the scene with correct size
        self.scene.add_box("table", table_pose, size=(1.27, 0.577, 0.03))  # Set size to match your table dimensions
        rospy.loginfo("Table added to MoveIt planning scene with correct dimensions")


    ###Checking for the Truncation Condition
    def checking_end_effector_pos(self, goal_pose, end_eff_pose, timestamp):

        print("Checking Timestamp is: ", timestamp)
        end = False
        dis = np.linalg.norm(goal_pose[:3] - end_eff_pose)

        print('The Euclidean Distance is:', dis)
        
        ###Checking within 1000 timesteps, the end-effector is able to 
        ###reach the desired goal position
        if dis > 2:
            end = True

        return end

    
    ###Checking the difference of each of the joint angles
    ###Function checks if the robot is stuck in same position for certain time
    def checking_change_of_joints(self, previous_joints, present_joints):
        path_tolertance = 1e-5
        diff_list = []

        for v1, v2 in zip(previous_joints, present_joints):

            sub = v1 - v2

            diff_list.append(sub < path_tolertance)

        print(diff_list)

        return any(diff_list)



    
    def step(self, action, object_position, timestep):

        #### printing the Action value
        print('Given Action is', action)

        action_clipped = np.clip(action, a_min= -0.2, a_max=0.2)

        #### printing the Clipped action values
        print('After Clipping Action is', action_clipped)
        
        ###Obtaining the Present Joint Angles
        present_joint_angles = self.group.get_current_joint_values()

        print('Before Addition:', present_joint_angles)

        action_sent = [0]*6
        
        ### Adding the angle value of the previous state
        ### with clipped action output from the actor network
        for i in range(len(action_clipped)):
            action_sent[i] = action_clipped[i] + present_joint_angles[i]

        print('After addition:', action_sent)

        

        ### Checking whether the limits of the joint
        ### angles are between -pi to +pi
        joint_limits = [
            (-2*np.pi, 2*np.pi),  # shoulder_pan_joint
            (-2*np.pi, 2*np.pi),  # shoulder_lift_joint
            (-np.pi, np.pi),  # elbow_joint
            (-2*np.pi, 2*np.pi),  # wrist_1_joint
            (-2*np.pi, 2*np.pi),  # wrist_2_joint
            (-2*np.pi, 2*np.pi)   # wrist_3_joint
        ]
        

        ####Checking the Joint Limits
        for i, (low, high) in enumerate(joint_limits):
            if action_sent[i] < low:
                action_sent[i] = low

            if action_sent[i] > high:
                action_sent[i] = high

        print("After Limitation: ", action_sent)
           

        ###Moving the Joint angles
        self.group.set_joint_value_target(action_sent)
        self.group.go(wait=True)
        
        self.group.stop()

        ####Checking for Collision
        collision_happened = self.collision_checker_node.start_collision_checker()

        ###Gettting the angles of the arm after the joint movement
        after_moving_angles= self.group.get_current_joint_values()

        print('After movement', after_moving_angles)

        ### checking the change of joint angles in consecutive states
        joint_changes = self.checking_change_of_joints(present_joint_angles, after_moving_angles)

        self.counter += 1

        ### Obtaining the position and rpy of an end-effector
        position, rpy = self.get_pos()
        
        ###Get the New State Acheived
        state = self.get_state(after_moving_angles, position, object_position)

        
        ##Checking the joint movements at the New State
        ###Here 'end' contains the termination condition
        ### whteher the arm must reach to the goal position
        ### or not
        # end = self.checking_end_effector_pos(object_position, position, timestep)
        end = False
        truncated = False

        #####wanting the arm to reach within 100 timesteps to goal position
        if self.counter % 100 == 0:
            truncated = True


        reward, terminal, target_pos, end_eff_pos = self.get_reward(object_position,position, timestep, collision_happened, joint_changes)

        truncated = truncated or terminal
        

        return state, reward, end, truncated , collision_happened, target_pos, end_eff_pos


    ###Method to reset the arm joint position
    def reset(self):
        
        #####current position before INIT gets triggered
        start = self.group.get_current_joint_values()
        stop = [0.0, -1.571, 1.571, -1.571, 0.0, 0.0]

        ###creating an uniform spaced path
        path = np.linspace(start, stop, num=10)

        self.counter = 0

    

        ## making the arm move slowly towards the
        ### initial position
        for item in path:
            ####passing a path
            print(item)
            self.group.set_joint_value_target(item.tolist())
            self.group.go(wait=True)
            self.group.stop()
                    

        after_moving_angles = self.group.get_current_joint_values()
        
        ### printing the angle values after the movement of joints
        print("The angles in the initial state are: ", after_moving_angles)

        object_position = self.target_generate()
        
        ### obtaining the position and rpy
        ### of the end-effector
        position, rpy = self.get_pos()
        
        ###getiing the Present State
        state_obtained = self.get_state(after_moving_angles,position, object_position)

        return state_obtained, object_position
    
    
    ####Function to get the Present State of the Arm
    def get_state(self,angular_values, position, object_position):
        #x, y, z of angular distance
        goal_pose = object_position[:3]

        position_diff = (np.linalg.norm(goal_pose - position))

        state = np.concatenate((angular_values, position_diff),axis=None)
        
        rospy.loginfo(f'State Representation is:  {state}')
        
        return state
    
   ##### Function for Penalizing and Rewarding the Agent
    def get_reward(self,object_position,pos, timestep, collision_happened, joint_changes):
        t = False

        
        rospy.loginfo(f'Entering timestep:  {timestep}')
        
        dis = np.linalg.norm(object_position[:3] - pos)
        rospy.loginfo(f'The distance is:  {dis}')
    

        reward = -dis

        
        if  dis <= 0.1:
            reward += 30
            print ('reach distance')
            t = True
            print('successfully reached distance at timestep', timestep)
            print(pos)
  
        print("Reward for the present state is: ", reward)
        
        return reward, t, object_position[:3], pos

   


    ###Function to get the present position
    ### of the end-effector
    def get_pos(self):
        
      
        current_pose = self.group.get_current_pose(self.end_effector_link).pose

       
        position = np.array([current_pose.position.x, current_pose.position.y, current_pose.position.z])
        orientation = [current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]
        rpy = euler_from_quaternion(orientation)

        return position, np.array(rpy)

    ####Visualizaing the function for displaying  of the cube    
    def target_vis(self,goal):
        rospy.wait_for_service("gazebo/delete_model")
        rospy.wait_for_service("gazebo/spawn_sdf_model")
        delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        
        s = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
        
        orient = Quaternion(*tf.transformations.quaternion_from_euler(1.571, 0, 0))
        origin_pose = Pose(Point(goal[0],goal[1],goal[2]), orient)
        

        with open('/home/robocupathome/arkaur5_ws/src/contexualaffordance/models/box_red/model.sdf',"r") as f:
            reel_xml = f.read()
        
        for row in [1]:
            for col in range(1):
                reel_name = "reel_%d_%d" % (row,col)
                delete_model(reel_name)
                pose = deepcopy(origin_pose)
                pose.position.x = origin_pose.position.x 

                #if pose.position.x > 3:
                    #pose.position.x = 0.506

                pose.position.y = origin_pose.position.y 
                    
                pose.position.z = origin_pose.position.z 
                s(reel_name, reel_xml, "", pose, "world")
                #print("spawnobj")

    def target_generate(self):
        rand_x, rand_y, rand_z= np.random.uniform(-0.04,0.04), np.random.uniform(-0.3,0.1), np.random.uniform(-0.6,0)

        # get_target_pose
        self.goal_pose = np.array(GOAL)
        self.goal_pose[0] += 0
        self.goal_pose[1] += 0
        self.goal_pose[2] += 0
        self.target_vis(self.goal_pose)
        return self.goal_pose
    


if __name__ == '__main__':
    arm = Ur5()

    