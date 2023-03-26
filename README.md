# CMPUT 412 Exercise 5 (inference branch)
This repository contains all the code for Exercise 5 (on remote machine).

## Usage
Make sure to have a local installation of ROS.

1. Clone the repository.

2. Set ROS master to the bot: 

`export ROS_MASTER_URI=http://<BOT>.local:11311`

3. In the working directory: 
`source devel/setup.zsh` or `source devel/setup.<any supported shell>`

4. Run the node:
```roslaunch detect inference_node.launch veh:=<BOT>```

## Node
* Inference mode

## License
[Duckietown](https://www.duckietown.org/about/sw-license)
