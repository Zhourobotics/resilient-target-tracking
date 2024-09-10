#! /usr/bin/env python

import os
import rospy
import rospkg
import numpy as np
import functools

#### ros packages ####
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8MultiArray
from adaptive_server import AdaptiveServer

#### local packages ####
from model.tracker_manger import TrackerManager
from model.config_loader import ConfigLoader
from utils.visualizer import Visualizer
from utils.visualizer_ros import VisualizerROS

import yaml

class TrackerServer:

    def __init__(self, exp_name, cent=True, adapt=True):
        """
        exp_name: .yaml config file to read
        """
        ######### general parameters #########
        try :
            path = rospkg.RosPack().get_path('tracker')
        except:
            #use relative path
            path = os.path.dirname(os.path.abspath(__file__)) + "/.."
        
        config_path = path + "/config/" + exp_name + ".yaml"
        print("config path is: ", config_path)
        if not os.path.exists(config_path):
            print("Config file does not exist")
            return

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        print("========= loading config file from: ", config_path)

        self.config_loader = ConfigLoader(config)
        self.cent = cent
        self.use_adapt = adapt
        self.config_loader.use_cent_solver = cent
        self.tracker_manager = TrackerManager(self.config_loader)
        self.exp = self.config_loader.exp
        self.frame_id = self.config_loader.frame_id
        self.target_ids = self.config_loader.targetID
        self.robot_ids = self.config_loader.robotID
        self.dim = self.config_loader.dim


        ######### save path #########
        self.save_path = path + "/results/" + str(self.tracker_manager.testID) + "/"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        ######### ROS parameters #########
        self.drone_cmd_pubs = []
        self.target_odom_subs = []
        self.drone_odom_subs = []
        self.drone_odom_latest = np.zeros((len(self.robot_ids), 3))

        self.flags_pub = rospy.Publisher("/flags", Int8MultiArray, queue_size=10)

        ####### only for sim
        self.drone_odom_pubs = []
        self.target_odom_pubs = []

        ######### Experiment parameters #########
        self.drone_ids  = self.config_loader.drone_ids
        self.car_ids = self.config_loader.car_ids

        ############ visualization ############
        self.vis = Visualizer(self.save_path)
        self.vis_ros = None
        self.his_target_pos = []
        self.his_target_vel = []
        self.his_drone_pos = []
        self.his_drone_vel = []
        self.comm_dzones = []
        self.sens_dzones = []
        self.his_drone_cmd = []
        self.his_exitflag = []
        self.his_known_typeI_flags = []
        self.his_known_typeII_flags = []
        self.his_attacked_typeI_flags = []
        self.his_attacked_typeII_flags = []

        # run simulation
        self.adapter = AdaptiveServer(self.config_loader, self.tracker_manager)
 
        if self.exp == "simulation":
            self.simulation()
        elif self.exp == "ros simulation":
            self.ros_simulation()
        elif self.exp == "ros real":
            self.ros_real()
        elif self.exp == "simulation benchmark":
            self.simulation_benchmark()
        else:
            print("Invalid experiment type")
            return
        

    def drone_odom_callback(self, data, index):
        """
        data: Crazyswarm GenericLogData
        id: int, id of the drone
        """
        # extract the position
        x, y, z = data.values[0], data.values[1], data.values[2]
        # update to the tracker manager
        self.tracker_manager.trac_prob.robot_odom[index, :self.dim] = np.array([x, y])
        self.drone_odom_latest[index] = np.array([x, y, z])
        # print(f"drone {index} position: {x}, {y}")
        return
    

    def target_odom_callback(self, data, index):
        """
        data: Crazyswarm GenericLogData
        id: int, id of the target
        """
        # extract the position
        x, y, z = data.values[0], data.values[1], data.values[2]
        # update to the tracker manager
        self.tracker_manager.trac_prob.target_odom[index, :self.dim] = np.array([x, y])
        # print(f"!!!!!!target {index} position: {x}, {y}")
        return
    
    
    def update_results(self, results):

        self.his_drone_pos.append(results["robot_pos"])
        self.his_drone_vel.append(results["robot_vel"])
        self.his_target_pos.append(results["target_pos"])
        self.his_drone_cmd.append(results["robot_cmd"])
        self.his_exitflag.append(results["exitflag"])
        self.his_known_typeI_flags.append(results["known_typeI_flags"])
        self.his_known_typeII_flags.append(results["known_typeII_flags"])
        self.his_attacked_typeI_flags.append(results["attacked_typeI_flags"])
        self.his_attacked_typeII_flags.append(results["attacked_typeII_flags"])

        

    def simulation(self):
        """
        steps: int, number of steps to run the simulation
        """
        # solve the problem
        print(" ========= solving problem =========")

        for i in range(self.tracker_manager.steps):
            results = self.tracker_manager.solve_one_step()
            self.update_results(results)
            
            ###### adapter ######
            # self.tracker_manager.weights = self.adapter.sim_adapter(results)
            self.adapter.sim_adapter(results, self.use_adapt)


        # find the step with changes in known flags
        current_known_flag_count = 0
        typeI_flag_change_step = []
        current_step = 0
        for i in range(self.tracker_manager.steps):
            # print(sum(self.his_known_typeI_flags[i])[0])
            # if self.his_attacked_typeI_flags[:i].sum() > current_known_flag_count:
            #     current_known_flag_count = self.his_known_typeI_flags[i].sum() + self.his_known_typeII_flags[i].sum()
                current_step = i
            #     typeI_flag_change_step.append(i)
        print("changes in known flags at step ", current_step)
        print("typeI_flag_change_step are ", typeI_flag_change_step)


        current_step = self.tracker_manager.steps - 1
        #current_step = 299      # Plot the first n steps

        # load map
        print(" ========= loading map =========")
        self.vis.visualize_map(self.config_loader.x_bounds)
        self.vis.visualize_zones(self.config_loader.typeI_zones,
                                 self.config_loader.typeII_zones,
                                 self.his_known_typeI_flags[current_step], #[-1],
                                 self.his_known_typeII_flags[current_step]) #[-1])

        # visualize results
        print(" ========= visualising results =========")
        self.vis.visualize_target(self.his_target_pos[:current_step], self.config_loader.targetID, self.tracker_manager.steps)
        self.vis.visualize_robot(self.his_drone_pos[:current_step], self.config_loader.robotID, self.tracker_manager.steps)
        # self.vis.plot_dyn(self.his_drone_cmd, 
        #                   self.his_drone_vel)
        # self.vis.plot_cmd(self.his_drone_cmd)

        print("typeI attacked_rate is ", \
              len(self.tracker_manager.typeI_attacked_pos) / self.tracker_manager.steps)
        print("typeII attacked_rate is ", \
              len(self.tracker_manager.typeII_attacked_pos) / self.tracker_manager.steps)
        
        attack_count_I = int(np.sum(self.his_attacked_typeI_flags[:current_step]))
        attack_count_II = int(np.sum(self.his_attacked_typeII_flags[:current_step]))
        print("typeII attack count is ", attack_count_II)
        self.vis.plot_pts(self.tracker_manager.typeI_attacked_pos[:attack_count_I], "maroon")
        self.vis.plot_pts(self.tracker_manager.typeII_attacked_pos[:attack_count_II], "cyan")
        self.vis.plot_trace(self.tracker_manager.trace_list[:current_step])
        self.vis.plot_trace_single(self.tracker_manager.trace_list_single[:current_step])
 
        # self.vis.plot_known_flags(self.his_known_typeI_flags[:current_step], self.his_known_typeII_flags[:current_step])
        # self.vis.plot_exitflag(self.his_exitflag[:current_step])
        # self.vis.plot_attacked_flags(self.his_attacked_typeI_flags[:current_step], self.his_attacked_typeII_flags[:current_step])

        self.vis.show(current_step)
    

    def ros_simulation(self):
        """
        setup ROS parameters
        """
        rospy.init_node('tracker_server', anonymous=True)
        rate = rospy.Rate(1/self.tracker_manager.dt)

        #publishers
        for i in range(len(self.robot_ids)):
            self.drone_cmd_pubs.append(rospy.Publisher("/drone" + str(self.robot_ids[i])+ "/cmd_vel", 
                                                       Twist, queue_size=10))
            self.drone_odom_pubs.append(rospy.Publisher("/drone" +  str(self.robot_ids[i]) + "/odom",
                                                       Odometry, queue_size=10))
            
        for i in range(len(self.target_ids)):
            self.target_odom_pubs.append(rospy.Publisher("/target" + str(self.target_ids[i]) + "/odom",
                                                            Odometry, queue_size=10))
       
        self.vis_ros = VisualizerROS(self.frame_id,
                                     self.drone_cmd_pubs, 
                                     self.drone_odom_pubs,
                                     self.target_odom_pubs)
                                     
        typeI_zones = self.config_loader.typeI_zones
        typeII_zones = self.config_loader.typeII_zones
        while not rospy.is_shutdown() and self.tracker_manager.cur_step < self.tracker_manager.steps:
            results = self.tracker_manager.solve_one_step()

            self.vis_ros.publish_all(results["target_pos"], 
                                     results["robot_pos"], 
                                     results["robot_cmd"],
                                     typeI_zones,
                                     typeII_zones,
                                     results["known_typeI_flags"],
                                     results["known_typeII_flags"])
            # self.tracker_manager.weights = self.adapter.sim_adapter(results)
            self.adapter.sim_adapter(results, self.use_adapt)
            rate.sleep()
        
        return

        
    def ros_real(self):
        """
        setup ROS parameters
        """
        from pycrazyswarm import Crazyswarm
        from crazyswarm.msg import GenericLogData
        # initialize Crazyflie
        print(" ========= initializing Crazyflie =========")
        swarm = Crazyswarm()
        # print(" ========= Crazyflie initialized =========")
        timeHelper = swarm.timeHelper
        cfs = swarm.allcfs

        num_agent = len(self.robot_ids)

        rate = rospy.Rate(10)

        # set subscribers to get the positions, and update to the tracker manager
        print(" ========= setting up ROS subscribers =========")
        for i, id in enumerate(self.drone_ids):
            print(f"cf{id} has index: {i}")
            self.drone_odom_subs.append(rospy.Subscriber("/cf" + str(id) + "/log1", 
                                                         GenericLogData, 
                                                         functools.partial(self.drone_odom_callback, index=i)))
        
        for i, id in enumerate(self.car_ids):
            self.target_odom_subs.append(rospy.Subscriber("/cf" + str(id) + "/log1", 
                                                          GenericLogData, 
                                                          functools.partial(self.target_odom_callback, index=i)))
            self.target_odom_pubs.append(rospy.Publisher("/robot" + str(i+1) + "/cmd_vel", Twist, queue_size=10))
        

        print(" ========= Crazyflie takeoff =========")
        for id, cf in enumerate(self.robot_ids):
            cf = cfs.crazyflies[id]
            cf.takeoff(targetHeight=0.5, duration=2.5)
        timeHelper.sleep(3)

        # run simulation
        print(" ========= solving problem =========")

        while not rospy.is_shutdown():

            results = self.tracker_manager.solve_one_step()
            cmd_vel = results["robot_cmd"]
            print("robot_states are ", results["robot_pos"])
            print("target_states are ", results["target_pos"])
            self.tracker_manager.weights = self.adapter.sim_adapter(results)
            # toc = timeHelper.time()
            # print(f"time for solving one step: {toc-tic} sec, cmd = {cmd_vel}")
            for i in range(num_agent):
                # print(f"cf{i} cmd_vel: {cmd_vel[i, 0]}, {cmd_vel[i, 1]}")
                # cmdVelocityWorld cannot hold altitude when give v_z = 0 somehow...
                print(f"id = {i}")
                if self.drone_odom_latest[i][2] - 0.5 > 0:
                    v_z = -0.05
                else:
                    v_z = 0.05
                cfs.crazyflies[i].cmdVelocityWorld(np.array([cmd_vel[i, 0], cmd_vel[i, 1], v_z]), yawRate=0)
                # print(cfs.crazyflies[i].position())

            for i in range(len(self.target_ids)):
                pub_msg = Twist()
                pub_msg.linear.x = self.tracker_manager.trac_prob.targetStartVel[i][0]
                self.target_odom_pubs[i].publish(pub_msg)

            # limited experiment space
            if results["robot_pos"][0][0] > 1.5  or results["robot_pos"][1][0] > 1.5:
                pub_stop = Twist()
                for i in range(len(self.target_ids)):
                    self.target_odom_pubs[i].publish(pub_stop)
                break
            rate.sleep()



    def simulation_benchmark(self):
        """
        steps: int, number of steps to run the simulation
        """
        # solve the problem
        print(" ========= solving problem =========")
        # trace_single = []
        # self.tracker_manager.use_cent_solver = True
        for i in range(self.tracker_manager.steps):
            results = self.tracker_manager.solve_one_step()
            self.update_results(results)
            ###### adapter ######
            # self.tracker_manager.weights = self.adapter.sim_adapter(results)
            self.adapter.sim_adapter(results, self.use_adapt)




if __name__ == '__main__':

    import sys

    #try
    try:
        exp_name = sys.argv[1]
        print("exp is ", exp_name)
    except:

        exp_name = "exp1"

    tracker_server = TrackerServer(exp_name, cent=True, adapt=True)
