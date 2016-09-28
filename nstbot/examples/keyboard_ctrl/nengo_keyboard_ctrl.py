import nengo
import nstbot
import udp

bot = nstbot.PushBot()
bot.connect(nstbot.Socket('10.162.177.88'))
#bot.connect(nstbot.Socket('10.162.177.94'))
bot.retina(True)
bot.laser(100)
bot.track_frequencies([100, 100])
bot.show_image()

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

    nengo.Connection(udp.hack[0], trans_neurons, synapse=None)
    nengo.Connection(udp.hack[1], turn_neurons, synapse=None)

    motors_neurons = nengo.Ensemble(200,2)
    nengo.Connection(motors_neurons, bot_c)
    
    nengo.Connection(trans_neurons, motors_neurons, function=trans_func, synapse=0.1)
    
    nengo.Connection(turn_neurons, motors_neurons, function=turn_func, synapse=0.1)