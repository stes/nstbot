import nengo
import nstbot

bot = nstbot.PushBot()
bot.connect(nstbot.Socket('10.162.177.88'))
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
        return 0, -x

model = nengo.Network()
with model:
    #motors = nengo.Node([0]*2)
    
    trans_speed = nengo.Node([0])
    ang_speed = nengo.Node([0])
    turn = nengo.Node([0])

    def bot_control(t, x):
        bot.motor(x[0], x[1])
    bot_c = nengo.Node(bot_control, size_in=2)

    #nengo.Connection(motors, bot_c)
    
    #a = nengo.Ensemble(50, 2)
    #nengo.Connection(motors, a)
    #nengo.Connection(a, bot_c, synapse = 0.1)
    
    trans_neurons = nengo.Ensemble(100, 2)
    
    nengo.Connection(trans_speed, trans_neurons, function = trans_func)
    
    nengo.Connection(trans_neurons, bot_c, synapse = 0.1)
    
    ang_neurons = nengo.Ensemble(100, 2)
    
    nengo.Connection(ang_speed, ang_neurons, function = ang_func)
    
    nengo.Connection(ang_neurons, bot_c, synapse = 0.1)
    
    turn_neurons = nengo.Ensemble(100,2)
    
    nengo.Connection(turn, turn_neurons, function = turn_func)
    
    nengo.Connection(turn_neurons, bot_c, synapse = 0.1)
    
    left_point = nengo.Node([bot.p_x[0], bot.p_y[0]])
    right_point = nengo.Node([bot.p_x[1], bot.p_y[1]])
    
    y_coord = nengo.Ensemble(100, dimensions=2)
    
    nengo.Connection(left_point[1], y_coord[0], transform=1.0/128)
    
    nengo.Connection(right_point[1], y_coord[1], transform=1.0/128)
    
    def collide(x):
        av = (x[0] + x[1])/2.0
        if av > 0.8:
            return -0.2, -0.2
        else:
            return 0.2, 0.2
    
    nengo.Connection(y_coord, trans_neurons, function=collide)