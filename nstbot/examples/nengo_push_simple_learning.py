import nengo
import nstbot
import numpy as np

# use scale factor 1.0 when running this script with the nengo backend
scale=1.0
# use scale factor 1.0 when running this script with the nengo_spinnaker backend 
#(slows down synapses to avoid delay between sensor and motor training input)
#scale=3.0

bot = nstbot.PushBot()
bot.connect(nstbot.Socket('10.162.177.88'))
#bot.connect(nstbot.Socket('10.162.177.94'))
bot.retina(True)
bot.laser(100)
bot.track_frequencies([100, 100])
bot.show_image()

def trans_func(x):
    return x,x

def ang_func(x):
    return x,-x
    
def turn_func(x):
    if x >=0:
        return x,0
    else:
        return 0,-x

model = nengo.Network()
with model:
    
    trans_input_speed = nengo.Node([0])
    def bot_control(t, x):
        bot.motor(x[0], x[1], msg_period=0.1)
    bot_c = nengo.Node(bot_control, size_in=2)

    # setting up some ensembles 

    desired_speed_neurons = nengo.Ensemble(n_neurons=100, dimensions=1)

    # create a neuron ensemble to represent the error between desired and calculated speed values 
    error = nengo.Ensemble(n_neurons=100, dimensions=1, radius =2.0)

    # create a node to control (i.e. start or stop) learning
    stop_learn = nengo.Node(1)
    # connect the node to the error neurons with a large inhibitory connection
    nengo.Connection(stop_learn, error.neurons, transform=-10*np.ones((100,1)))

    trans_neurons = nengo.Ensemble(100, 1)
    motors_neurons = nengo.Ensemble(200,2)

    nengo.Connection(motors_neurons, bot_c, synapse=0.005*scale)
    
    nengo.Connection(trans_input_speed, desired_speed_neurons, synapse=0.005*scale)
    
    nengo.Connection(trans_neurons, motors_neurons, function=trans_func, synapse=0.1*scale)
    
    # neuron ensemble to represent the y-coordinates of both laser points
    y_coord = nengo.Ensemble(200, dimensions=4, radius=1.4)

    # putting both points in one node, since two seperate nodes feeding one population seems to cause a bug in nengo_spinnaker
    both_points = nengo.Node(lambda t: [bot.p_x[0], bot.p_y[0], bot.p_x[1], bot.p_y[1]])

    # feeding the y-coordinates of both points to be represented in the y_coord neuron ensemble
    nengo.Connection(both_points[[1,3]], y_coord[:2], transform = 1.0/128, synapse=0.005*scale)

    # do a recurrent connection to have the sensor input from some time ago to make the input for learning richer
    nengo.Connection(y_coord[:2], y_coord[2:], synapse=0.005*scale)

    #establish the learning connection with zero function as inital function 
    def init_func(x):
        return 0
    learn_conn = nengo.Connection(y_coord, trans_neurons, function=init_func,
                                  learning_rule_type=nengo.PES(learning_rate=1e-4/scale, pre_tau=0.005*scale),synapse=0.05*scale)

    # connection from error population to the learning rule of the learning connection
    nengo.Connection(error, learn_conn.learning_rule, synapse=0.005*scale)

    nengo.Connection(desired_speed_neurons, error, transform=-1, synapse=0.05*scale)
    nengo.Connection(trans_neurons, error, transform=1, synapse=0.005*scale)
    

