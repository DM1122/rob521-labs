# Lab 2 Report

## Part A: Simulation

### 1. See the following figure demonstrating successful output from RRT (Daye)

| ![rrt_planning](./results/rrt_planning.png)|
| :--: |
| A rrt planning figure
the triangle on top-left spot is the starting node and the blue cicle on the bottom-right spot is the our goal point. All green circles are the nodes that are collision-free and the blue circles connected by blue lines consists of the final trajectory of the robot. |

### 2. See the following figure illustrating successful output from RRT* (Davin)


### 3. See `trajectory-rollout-sim.mp4` for footage demonstrating a successful run of the trajectory rollout using RRT. (David)

### 4. Algorithm descriptions (Daye + Davin)

#### RRT planning 

       
        <RRT alogrithm>

        1) sample one random point
           - adjusted sampling stratgy fit to the our `willow  garage map`
             - split the map into 10 * 10 small boxes, iterate over the small boxes from left to right and from top to bottom 
             - Since there are bottle-neck region with low visibility in the "willow" map, 
               - Set the extra small bottle neck region 
               - sampling points 10 times in that bottle neck region 
  
        2) find the closest point to the sampled point in the node list
           1) using Euclidean distance

        3) find trajectory to the closest point with collision check together 
            - 
            - if path to NEW_STATE is collision free
                - Add end point
                - Add path from nearest node to end point
        4) retrun success/failure and current tree



## Part B: Real Environment Deployment

1. See `trajectory-rollout.mp4` for footage of the robot conducting trajectory rollout using RRT in the new Myhal map.
2. Open-loop demonstration conducted during lab session


Code is attached