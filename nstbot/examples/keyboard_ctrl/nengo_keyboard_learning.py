import nengo
import nstbot
import numpy as np
import udp

bot = nstbot.PushBot()
bot.connect(nstbot.Socket('10.162.177.88'))
#bot.connect(nstbot.Socket('10.162.177.94'))
bot.retina(True)
bot.laser(100)
bot.track_frequencies([100, 100])
bot.show_image()

# use scale factor 1.0 when running this script with the nengo backend
scale=1.0
# use scale factor 1.0 when running this script with the nengo_spinnaker backend 
#(slows down synapses to avoid delay between sensor and motor training input)
#scale=3.0

if not hasattr(udp, 'hack'):
    m = nengo.Network()
    with m:
        udp.hack = udp.UDPReceiver(9999, 2)


def trans_func(x):
    return x[0],x[0]

def turn_func(x):
    if x[0] >=0:
        return x[0],0
    else:
        return 0,-x[0]

model = nengo.Network()
with model:

    model.nodes.append(udp.hack)

    def bot_control(t, x):
        bot.motor(x[0], x[1], msg_period=0.1)

    bot_c = nengo.Node(bot_control, size_in=2)

    # setting up some ensembles 
    
    trans_neurons = nengo.Ensemble(100, 1)
    turn_neurons = nengo.Ensemble(100,1)

    motors_neurons = nengo.Ensemble(200,2)
    desired_trans_neurons = nengo.Ensemble(200,1)
    desired_turn_neurons = nengo.Ensemble(200,1)

    nengo.Connection(udp.hack[0], desired_trans_neurons, synapse=None)
    nengo.Connection(udp.hack[1], desired_turn_neurons, synapse=None)

    nengo.Connection(motors_neurons, bot_c, synapse=0.1*scale)
    
    nengo.Connection(desired_trans_neurons, trans_neurons, synapse=0.005*scale)
    nengo.Connection(trans_neurons, motors_neurons, function=trans_func, synapse=0.1*scale)
    
    nengo.Connection(desired_turn_neurons, turn_neurons, synapse=0.005*scale)
    #nengo.Connection(turn_neurons, motors_neurons, function=turn_func, synapse=0.1*scale)

    #here comes the learning part

    # create a neuron ensemble to represent the error between desired and calculated speed values 
    trans_error = nengo.Ensemble(n_neurons=200, dimensions=1)

    # create a neuron ensemble to represent the error between desired and calculated speed values 
    turn_error = nengo.Ensemble(n_neurons=200, dimensions=1)

    # create a node to control (i.e. start or stop) learning
    stop_learn = nengo.Node(1)
    # connect the node to the error neurons with a large inhibitory connection
    nengo.Connection(stop_learn, trans_error.neurons, transform=-10*np.ones((200,1)))

    # connect the node to the error neurons with a large inhibitory connection
    nengo.Connection(stop_learn, turn_error.neurons, transform=-10*np.ones((200,1)))

    n_sensor_neurons = 4

    # neuron ensemble to represent the coordinates of both laser points
    points_ensemble = nengo.Ensemble(200*n_sensor_neurons, dimensions=n_sensor_neurons, radius=1.4)
    
    # putting both points in one node, since two seperate nodes feeding one population seems to cause a bug in nengo_spinnaker
    both_points = nengo.Node(lambda t: [bot.p_x[0], bot.p_y[0], bot.p_x[1], bot.p_y[1]])

    # feeding the y-coordinates of both points to be represented in the y_coord neuron ensemble
    nengo.Connection(both_points[[1,3]], points_ensemble[:2], transform = 1.0/128, synapse=0.005*scale)

    # do a recurrent connection to have the sensor input from some time ago to make the input for learning richer
    nengo.Connection(points_ensemble[:2], points_ensemble[2:4], synapse=0.05*scale, transform=0.95)

    # # feeding the x-coordinates of both points to be represented in the y_coord neuron ensemble
    # nengo.Connection(both_points[[0,2]], points_ensemble[4:6], transform = 1.0/128, synapse=0.005*scale)

    #  # do a recurrent connection to have the sensor input from some time ago to make the input for learning richer
    # nengo.Connection(points_ensemble[4:6], points_ensemble[6:8], synapse=0.05*scale, transform=0.95)

    #establish the learning connection with zero function as inital function 
    def init_func(x):
        return 0
    trans_learn_conn = nengo.Connection(points_ensemble, trans_neurons, function=init_func,
                                  learning_rule_type=nengo.PES(learning_rate=1e-4/scale, pre_tau=0.005*scale),synapse=0.05*scale)

    # connection from error population to the learning rule of the learning connection
    nengo.Connection(trans_error, trans_learn_conn.learning_rule, synapse=0.005*scale)

    nengo.Connection(desired_trans_neurons, trans_error, transform=-1, synapse=0.05*scale)
    nengo.Connection(trans_neurons, trans_error, transform=1, synapse=0.005*scale)
    
    # turn_learn_conn = nengo.Connection(points_ensemble, turn_neurons, function=init_func,
    #                               learning_rule_type=nengo.PES(learning_rate=1e-4/scale, pre_tau=0.005*scale),synapse=0.05*scale)

    # # connection from error population to the learning rule of the learning connection
    # nengo.Connection(turn_error, turn_learn_conn.learning_rule, synapse=0.005*scale)

    # nengo.Connection(desired_turn_neurons, turn_error, transform=-1, synapse=0.05*scale)
    # nengo.Connection(turn_neurons, turn_error, transform=1, synapse=0.005*scale)