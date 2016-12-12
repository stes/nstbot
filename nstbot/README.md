# Omniarmbot Nengo Interface

## Overview
The Omniarmbot Nengo interface is based on Terry's nstbot python interface.
The omniarmbot is a subclass of nstbot and encapsulates all the low level hardware interface for
data acquisition and control.
Except the basic functions in NstBot the OmniArmBot class contains wrapper functions to access
the specific sensors and send specific control commands to the Omniarmbot platform:

- ```base()``` sets speed for the 3 individual wheels of the base (range ```[-100, 100]```)
- ```base_pos()``` sets the complex driving command 3 values, ```[x, y, heading angle]``` with respect to the current position in 2D space (range ```[-100, 100]```)
- ```arm()``` sets the joints to the 3 joints (counted from the base): shoulder, elbow, hand (without rotation) and gripper (open/close). The input values of three joints are set in the interval ```[0,2xPI]``` while the input for the gripper is 0 resp. 1 for open resp. close.
- ```set_arm_speed()``` sets the speed for the arm motors within the range ```[0,1024]```. Note: 0 does not mean zero velocity, but is instead actually the maximal value (so be careful when setting the speed of the arm motors)
- ```tracker()``` starts the embedded uDVS tracker algorithm (output ```(x,y,radius, certainty)``` of the tracked stimulus). Input to this function is currently a list of frequencies and the name of the retina.
- ```get_tracker_info()``` returns the tracker output for the retina given by ```name```


##Test and usage
To test the basic functionality of the OmniArmBot class you can run in the examples directory python omniarmbot.py. In order to encapsulate all functionality and provide a nengo interface, we created the
OmniArmBotNetwork which extends nengo.Network class and contains nengo.Nodes for all subsystems of the omniarmbot platform. To run a nengo demo (either in GUI or simulator) go into
the ```nstbot/nengo directory``` and run ```nengo omnibot_network_test.py``` or ```python omnibot_network_test.py``` for GUI and simulator respectively.

- you can enable which sensors to be active (an inactive sensor also inactivates the data
streaming, to save bandwidth) when instantiating the OmniArnNetwork class
- you have to provide a dictionary with ```{name:[ip, port]}``` for each subsystem providing data, i.e.each retina and motors.
- you can enable neural representation (a population projection of each sensory stream) by setting the parameter ```n_neurons_p_dim``` (number of neurons per dimension for each neural ensemble) to a scalar value. The default value is ```None```, which disables neural representations entirely.
- you can enable probing for data visualization by setting ```b_probe=True```. Note: probing is currently only possible if neural representations are active (TODO: probing/plotting for embedded tracker is not implemented yet)
- in order to set the frequency of the control signal (sending) parametrize the ```send_msg_period = 0.2``` (5 Hz), which will be propagated to the subsystems to send control signals (i.e. base, arm). Note: 5 Hz is currently the maximum value for the control signal as larger values lead to delayed processing of control commands (possibly a problem in the firmware?)
- in order to set the frequency for the data acquisition (receiving) parametrize the ```receive_msg_period = 0.1``` (10Hz), which will be propagated to all the subsystems to receive data (i.e. retina, sensor streaming, etc.).