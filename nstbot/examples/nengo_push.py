import nengo
import nstbot

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
    #motors = nengo.Node([0]*2)
    
    trans_speed = nengo.Node([0])
    ang_speed = nengo.Node([0])
    turn = nengo.Node([0])

    def bot_control(t, x):
        bot.motor(x[0], x[1], msg_period=0.1)
    bot_c = nengo.Node(bot_control, size_in=2)

    #nengo.Connection(motors, bot_c)
    
    #a = nengo.Ensemble(50, 2)
    #nengo.Connection(motors, a)
    #nengo.Connection(a, bot_c, synapse = 0.1)

    # setting up some ensembles 
    
    trans_neurons = nengo.Ensemble(100, 1)
    ang_neurons = nengo.Ensemble(100, 1)
    turn_neurons = nengo.Ensemble(100,1)

    motors_neurons = nengo.Ensemble(200,2)
    nengo.Connection(motors_neurons, bot_c)
    
    nengo.Connection(trans_speed, trans_neurons)
    
    nengo.Connection(trans_neurons, motors_neurons, function=trans_func, synapse=0.1)
    
    nengo.Connection(ang_speed, ang_neurons)
    
    nengo.Connection(ang_neurons, motors_neurons, function=ang_func, synapse=0.1)

    nengo.Connection(turn, turn_neurons)
    
    nengo.Connection(turn_neurons, motors_neurons, function=turn_func, synapse=0.1)

    # neuron ensemble to represent the y-coordinates of both laser points
    y_coord = nengo.Ensemble(100, dimensions=2, radius=1.4)

    # neuron ensemble to represent the x-coordinates of both laser points
    x_coord = nengo.Ensemble(100, dimensions=2, radius=1.4)
    
    # old buggy version with two nodes feeding into one ensemble (works with nengo backend, but does not with nengo_spinnaker)
    #left_point = nengo.Node(lambda t: [bot.p_x[0], bot.p_y[0]])
    #right_point = nengo.Node(lambda t: [bot.p_x[1], bot.p_y[1]])
    # nengo.Connection(left_point[1], y_coord[0], transform=1.0/128)
    # nengo.Connection(right_point[1], y_coord[1], transform=1.0/128)

    # putting both points in one node, since two seperate nodes feeding one population seems to cause a bug in nengo_spinnaker
    both_points = nengo.Node(lambda t: [bot.p_x[0], bot.p_y[0], bot.p_x[1], bot.p_y[1]])

    # feeding the y-coordinates of both points to be represented in the y_coord neuron ensemble
    nengo.Connection(both_points[[1,3]], y_coord, transform = 1.0/128)

    # feeding the x-coordinates of both points to be represented in the x_coord neuron ensemble
    #nengo.Connection(both_points[[0,2]], x_coord, transform = 1.0/128)
    
    
    def obstacle_backoff(x):
        av = (x[0] + x[1])/2.0
        if av > 0.7:
            #return -(av-0.2)
            return -0.4
        elif av<= 0.7 and av > 0.3:
            #return (1-av)+0.2
            return 0.1
        else:
            return 0.5

    nengo.Connection(y_coord, trans_neurons, function=obstacle_backoff)

    def obstacle_turn(x):
        av = (x[0] + x[1])/2.0
        if av<= 0.7 and av > 0.3:
            if x[0] > x[1]:
                return 0.5
            else:
                return -0.5
        else:
            return 0.0

    nengo.Connection(y_coord, turn_neurons, function=obstacle_turn)

    #sim = nengo.Simulator(model)
    #sim.run(2.0)
    