# ROB521 Lab 4 Report


## Part 1: turtlebot3_world localization

AMCL RViz window with a converged filter with default values
![AMCL Default Converged](captures/amcl-default-converged.png)

As we lower the min-max AMCL particle counts, we see poorer localization performance. The image below shows the result of localization with `min_particles=9` and
`max_particles=94`. These values represent the minimum number of particles for successful localization, assuming the robot is driven under 0.1m/s. Any faster and the localization diverges. This was tested over three independent runs with different initial poses.
![Poor Localization with lower particle count](captures/low-particle-count.png)

The maximum odom_alpha parameters were increased until localization diverged. This occurred at values in the neighborhood of 6.0.
![odom-alpha](captures/odom-alpha.png)
