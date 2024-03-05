# Lab 3 Report


## Task 1: Vehicle Calibration (David)
*Design the experiments. How would you drive a robot using teleoperation to determine the parameters?*

There are two calibration experiments: 1) estimation of the wheel radius, and 2) estimation of the wheel separation. 

Experiment 1 procedure (wheel radius):
1. Command robot to travel a known distance of 0.75m
2. Sum the encoder ticks for each wheel at each timestep
3. Solve for radius from the differential drive model
$$
r = \frac{2x_{\text{true}}}{\sum_k (\Delta \phi_r(kh) + \Delta \phi_\epsilon(kh))}
$$

Read 14169 ticks on the left encoder and 14168 ticks on the right encoder. The wheel radius provided in the datasheet is 0.033m. The estimated wheel radius from the experiment yields 0.0345m. This is a percentage error of 4.54%.

Experiment 2 procedure (wheel separation):



## Task 4 (David)

In your report, describe your experiments. Your summary must address these
points:
• What path the robot was driven in each experiment? (1 pt total)
– (0.5 pts) How did you drive the robot to determine the wheel radius? How much
did it rotate? How far did you drive forward?
– (0.5 pts) How did you drive the robot to determine the wheel separation? How
much did you rotate? How far did you drive forward?
• How does your code work or should work? How are the parameters determined? (0.5
pt total)
• What values did you get for the wheel radius and baseline? (0.5 pt total)
• Does these values match those given in the datasheet? Identify one possible source
of uncertainty or bias that made your answer differ from the factory calibration. (1
pt total)
– Identify the source of uncertainty or bias.
– How does the source of uncertainty or bias affect your measurement?
– How could you mitigate this source of uncertainty or bias?