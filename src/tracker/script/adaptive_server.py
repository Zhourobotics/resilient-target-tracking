#! /usr/bin/env python

import os
import rospy
import rospkg
import numpy as np
import functools
from model.tracker_manger import TrackerManager
from utils.visualizer import Visualizer
from utils.visualizer_ros import VisualizerROS
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import yaml
from model.problem import Problem

class AdaptiveServer:

    def __init__(self, config_loader, tracker_manager):
        """
        exp_name: .yaml config file to read
        """
    
        ############# General Settings #############
        self.frame_id  = config_loader.frame_id

        self.target_ids = config_loader.targetID
        self.robot_ids = config_loader.robotID
        self.dim = config_loader.dim
        self.weights = config_loader.weights.copy()
        self.weights_adaptive = tracker_manager.trac_prob.weights_adaptive
        self.trac_manager = tracker_manager
        self.I_attack_countdown = np.zeros(len(self.robot_ids))
        self.II_attack_countdown = np.zeros(len(self.robot_ids))

        ######### data #########
        self.his_sub_robot_odom_all = []
        self.his_sub_target_odom_all = []

        for i in range(len(self.robot_ids)):
            self.his_sub_robot_odom_all.append([])
        for i in range(len(self.target_ids)):
            self.his_sub_target_odom_all.append([])

        ######### solver results #########
        self.his_drone_pos = []
        self.his_target_pos = []


    def drone_odom_callback(self, data, index):
        """
        data: Crazyswarm GenericLogData
        id: int, id of the drone
        """
        # extract the position
        x, y, z = data.values[0], data.values[1], data.values[2]
        # update to the tracker manager
        self.his_sub_robot_pos_all[index].append(np.array([x, y]))

        return

    def target_odom_callback(self, data, index):

        x, y, z = data.values[0], data.values[1], data.values[2]
        
        self.his_sub_target_pos_all[index].append(np.array([x, y]))
        return
        

    def sim_adapter(self, results, activate=True):
        """
        update the drone and target positions
        """
        if not activate:
            return
        
        self.his_drone_pos.append(results["robot_pos"])
        self.his_target_pos.append(results["target_pos"])

        ########### write your update strategy here ##############
        # new_weight = TODO
        # self.weights = new_weight


        ##### adapter strategy #####
        ### strategy 1: if attack, optimize escaping ###
        attacked_typeI_flags = results["attacked_typeI_flags"]
        attacked_typeII_flags = results["attacked_typeII_flags"]
        
        # adjust weights for inidividual robots
        # weight is in the form of [control, tracking, slack 1, slack 2]
        # obj is min problem --> small weight in tracking means less performance
        for robot_id in range(len(attacked_typeI_flags)):
            # decrease tracking weight for better tracking quality
            # decrease control weight for better escaping

            ## Peihan: controls duration of adaptation
            if attacked_typeI_flags[robot_id]:
                self.I_attack_countdown[robot_id] = 7
            if attacked_typeII_flags[robot_id]:
                self.II_attack_countdown[robot_id] = 7

            ## Peihan: controls the weight adaptation
            if self.I_attack_countdown[robot_id] != 0:
                # self.weights_adaptive[robot_id][1] *= 10
                self.weights_adaptive[robot_id][2] = 10    # slack 1
                self.I_attack_countdown[robot_id] -= 1
                print("\033[91m [INFO] Robot ", robot_id, " is attacked, updated weights: ", self.weights_adaptive[robot_id], "\033[0m")
            
            if self.II_attack_countdown[robot_id] != 0:
                self.weights_adaptive[robot_id][3] = 10    # slack 2
                self.II_attack_countdown[robot_id] -= 1
                print("\033[91m [INFO] Robot ", robot_id, " is attacked, updated weights: ", self.weights_adaptive[robot_id], "\033[0m")
            # use default weights
            if self.I_attack_countdown[robot_id] == 0 and self.II_attack_countdown[robot_id] == 0:
                # self.weights_adaptive[robot_id] = self.weights.copy()

                # default weights
                self.weights_adaptive[robot_id] = [0.05, 10, 1, 1]

                print("\033[96m [INFO] Robot ", robot_id, " is not attacked, weights: ", self.weights_adaptive[robot_id], "\033[0m")

        ### strategy 2: when stuck, optimize control ###
        # check for last n steps position, if the robot is stuck, increase control weight
        # the threshold for activation should be about the same as the target motion



        # self.weights *= (1 +  0.1 * np.random.randn(len(self.weights)))
        print("\033[95m [INFO] Updated weights: ", self.weights_adaptive, "\033[0m")
        # return (self.weights).copy()
        return


        

    def ros_adapter(self):
        """
        setup ROS parameters
        """

        rospy.init_node('tracker_server', anonymous=True)
        rate = rospy.Rate(1/self.config_loader.dt)

        #subscribe the robot information, the dangerous zone information
        for i in range(len(self.robot_ids)):
            self.drone_odom_subs.append(rospy.Subscriber("/drone" + str(self.robot_ids[i]) + "/odom",
                                        Odometry, functools.partial(self.drone_odom_callback, index=i)))

        for i in range(len(self.target_ids)):
            self.target_odom_subs.append(rospy.Subscriber("/target" + str(self.target_ids[i]) + "/odom",
                                        Odometry, functools.partial(self.target_odom_callback, index=i)))

        
        return







if __name__ == '__main__':

    exp_name = "exp4"

    try :
        path = rospkg.RosPack().get_path('tracker')
    except:
        #use relative path
        path = os.path.dirname(os.path.abspath(__file__)) + "/.."
    
    print("path is: ", path)
    #path = "/home/grasp/ma_ws/src/target_tracking/src/tracker"
    config_path = path + "/config/" + exp_name + ".yaml"
    if not os.path.exists(config_path):
        print("Config file does not exist")
    

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    tracker_server = AdaptiveServer(config)
    #rospy.spin()
