import nengo
import numpy as np
import nstbot

bot = nstbot.PushBot()
bot.connect(nstbot.Socket('10.162.177.108'))
bot.retina(True)
bot.track_frequencies([600])
#bot.show_image()

class Bot(nengo.Network):
    def __init__(self, bot):
        super(Bot, self).__init__()
        with self:
            def bot_control(t, x):
                bot.motor(x[0], x[1], msg_period=0.1)

            self.base_ctrl = nengo.Node(bot_control, size_in=2)

            def get_points(t):
                point_array = np.array([[bot.p_x[ind], bot.p_y[ind]] for ind in range(len(bot.p_x))])
                return point_array.flatten()

            self.tracked_points = nengo.Node(get_points)

class TaskDriveFrontBack(nengo.Network):
    def __init__(self, botnet, strength=0.4):
        super(TaskDriveFrontBack, self).__init__()
        with self:
            self.activation = nengo.Ensemble(n_neurons=50,  dimensions=1, neuron_type=nengo.Direct())

        nengo.Connection(self.activation, botnet.base_ctrl, function=lambda x: [x,x] ,transform=strength)

class TaskTurn(nengo.Network):
    def __init__(self, botnet, strength=1.0):
        super(TaskTurn, self).__init__()
        with self:
            self.activation = nengo.Ensemble(n_neurons=50, dimensions=1, neuron_type=nengo.Direct())

        nengo.Connection(self.activation, botnet.base_ctrl, function=lambda x: [x,-x] ,transform=strength)

class SensorTest(nengo.Network):
    def __init__(self, botnet):
        super(SensorTest, self).__init__()
        with self:
            self.stim = nengo.Ensemble(n_neurons=200, dimensions=2, radius=1.4)

        nengo.Connection(botnet.tracked_points, self.stim, transform=1.0/128.0)


class BehaviourControl(nengo.Network):
    def __init__(self, behaviours):
        super(BehaviourControl, self).__init__()
        with self:
            self.behave = nengo.Node([0]*len(behaviours))
        for i, b in enumerate(behaviours):
            nengo.Connection(self.behave[i], b.activation, synapse=None)


model = nengo.Network(seed=2)
with model:

    motors = nengo.Node([0]*2)
    botnet = Bot(bot)
    nengo.Connection(motors, botnet.base_ctrl)

    drive_fb = TaskDriveFrontBack(botnet)
    turn_lr = TaskTurn(botnet)
    sens = SensorTest(botnet)
    bc = BehaviourControl([drive_fb, turn_lr])
    
    #diff = nengo.Ensemble(n_neurons=700, dimensions=2)
    
    #nengo.Connection(sens.stim, diff)
    #nengo.Connection(diff, diff, synapse = 0.3, function = lambda x: -x)
    
    nengo.Connection(sens.stim[1], drive_fb.activation, synapse=None,
                     function=lambda x: max(2 * (x - 0.5), 0))
    nengo.Connection(sens.stim[0], turn_lr.activation, synapse=None,
                    function=lambda x: 0.5 * (x - 0.5))
