# Resilient and Adaptive Multi-robot Target Tracking

A resilient and adaptive multi-robot target tracking framework with sensing and communication danger zones


## About 

__Authors__: [Peihan Li](https://scholar.google.com/citations?user=Qg7-Gr0AAAAJ&hl=en), 
[Yuwei Wu](https://github.com/yuwei-wu), [Jiazhen Liu](https://scholar.google.com/citations?user=x4OzGCwAAAAJ&hl=en), Gaurav S. Sukhatmem, and Vijay Kumar, and [Lifeng Zhou](https://zhourobotics.github.io/)

__Video Links__:  [Youtube](https://www.youtube.com/watch?v=ARMUzIKwsvc)

__Related Paper__: Peihan Li, Yuwei Wu, Jiazhen Liu, Gaurav S. Sukhatme, Vijay Kumar and Lifeng Zhou. “Resilient and Adaptive Replanning for Multi-Robot Target Tracking with Sensing and Communication Danger Zones.” (2024).

```
@misc{li2024resilientadaptivereplanningmultirobot,
      title={Resilient and Adaptive Replanning for Multi-Robot Target Tracking with Sensing and Communication Danger Zones}, 
      author={Peihan Li and Yuwei Wu and Jiazhen Liu and Gaurav S. Sukhatme and Vijay Kumar and Lifeng Zhou},
      year={2024},
      eprint={2409.11230},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.11230}, 
}
```

## Prerequisites

- [ROS](https://wiki.ros.org/ROS/Installation): Our framework has been tested in ROS Noetic. We also support a non-ros version.

- [Forces Pro](https://www.embotech.com/products/forcespro/overview/): You can request an academic license from [here](https://www.embotech.com/products/forcespro/licensing/).

## Run

### a. Simple Run

```
python tracker_server.py
```




### b. Run with ROS

You can change the parameters in config/exp_.yaml to switch different modes. 
```
exp: "simulation"  # 0: "simulation", 1: "ros simulation", 2: "ros real"
```
To run ROS simulation, you will need to build the package:

```
git clone git@github.com:Zhourobotics/resilient-target-tracking.git
catkin build
source devel/setup.bash

```

then you can run "tracker_server.py" and open a .rviz to visualize the odom and danger zones. 


## File structure of folder 

```

└── tracker
    ├── CMakeLists.txt
    ├── config
    │   ├── exp10.yaml
    │   ├── exp1.yaml
    │   ├── exp2.yaml
    │   ├── exp3.yaml
    │   ├── exp4.yaml
    │   ├── exp5.yaml
    │   ├── exp6.yaml
    │   ├── exp7.yaml
    │   └── sim.rviz
    ├── launch
    │   └── vis_sim.launch
    ├── package.xml
    ├── results   # your result plot here
    └── script
        ├── adaptive_server.py # adaptive replanner
        ├── model
        │   ├── config_loader.py
        │   ├── danger_zones.py # class of danger zones
        │   ├── dynamics.py # all different dynamics
        │   ├── forcepro_centralized.py # solver
        │   ├── forcepro_single.py # solver
        │   ├── problem.py  # problem formulation
        │   └── tracker_manger.py # main manager, take odom, call solver, update
        ├── tracker_server.py # main service, connect ros or call simulation
        └── utils
            ├── visualizer.py
            └── visualizer_ros.py
```


## Maintaince

For any technical issues, please contact Yuwei Wu (yuweiwu@seas.upenn.edu) and Peihan Li (pl525@drexel.edu)
