import nengo
import numpy as np
import time
import nstbot

class BaseNode(nengo.Node):
    def __init__(self, bot, msg_period):
        super(BaseNode, self).__init__(self.move_base, size_in=3, size_out=0)
        self.bot = bot
        self.msg_period = msg_period

    def move_base(self, t, x):
        self.bot.base([x[0], x[1], x[2]], msg_period=self.msg_period)

    def get_input_dim(self):
        return self.size_in

class ArmNode(nengo.Node):
    def __init__(self, bot, msg_period):
        super(ArmNode, self).__init__(self.move_arm, size_in=4, size_out=0)
        self.bot = bot
        self.msg_period = msg_period

    def move_arm(self, t, x):
        self.bot.arm([x[0], x[1], x[2], x[3]], msg_period=self.msg_period)

    def get_input_dim(self):
        return self.size_in

class RetinaNode(nengo.Node):
    def __init__(self, bot, name, msg_period):
        super(RetinaNode, self).__init__(self.retina, size_in=0, size_out=128*128)
        self.bot = bot
        self.name = name
        self.msg_period = msg_period
        #self.bot.show_image(self.name)

    def retina(self, t):
        #return self.bot.get_image(self.name).flatten()
        pass

class FrequencyNode(nengo.Node):
    def __init__(self, bot, name, msg_period, freqs):
        super(FrequencyNode, self).__init__(self.freqs, label=name,
                                            size_in=0, size_out=len(freqs)*3)
        self.bot = bot
        self.name = name
        self.bot.track_frequencies(name, freqs=freqs)
        self.msg_period = msg_period
        self.result = np.zeros(3*len(freqs), dtype='float')
        self.n_freqs = len(freqs)

    def freqs(self, t):
        for i in range(self.n_freqs):
            self.result[i * 3 : (i + 1) * 3] = self.bot.get_frequency_info(self.name, i)
        return self.result

    def get_output_dim(self):
        return len(self.result)


class SensorNode(nengo.Node):
    def __init__(self, bot, key):
        self.bot = bot
        self.length = len(bot.get_sensor(key))
        self.key = key
        super(SensorNode, self).__init__(self.sensor,
                                         size_in=0, size_out=self.length)

    def sensor(self, t):
        if self.bot.get_sensor(self.key) is not None or self.bot.get_sensor(self.key) != []:
            if self.length == len(self.bot.get_sensor(self.key)):
                return self.bot.get_sensor(self.key)
            else:
                return np.zeros(self.length)
        else:
            return np.zeros(self.length)

    def get_output_dim(self):
        return self.length



class OmniArmBotNetwork(nengo.Network):
    def __init__(self, connection, msg_period=0.01, label='OmniArmBot',
                 n_neurons_p_dim=100, b_probe = False,
                 base=False, arm=False, retina=False, freqs=[],
                 **sensors):
        super(OmniArmBotNetwork, self).__init__(label=label)
        self.bot = nstbot.OmniArmBot()
        self.bot.connect(connection)
        self.b_probe = b_probe
        self.b_base = base
        self.b_arm = arm
        self.b_retina = retina
        self.b_freqs = False
        self.b_sensors = {}

        for name in self.bot.adress_list:
            if 'retina' in name:
                self.bot.retina(name, True)
                self.bot.track_frequencies(name, freqs=freqs)

        with self:
            if base:
                self.base = BaseNode(self.bot, msg_period=msg_period)
                dim = self.base.get_input_dim()
                self.base_neurons = nengo.Ensemble(n_neurons=dim*n_neurons_p_dim, dimensions=dim)
                nengo.Connection(self.base_neurons, self.base)
                if self.b_probe:
                    self.p_base_neurons_out = nengo.Probe(self.base_neurons, synapse=0.01)
                    self.p_base_neurons_spikes = nengo.Probe(self.base_neurons.neurons, "spikes")
                    self.p_base_neurons_vol = nengo.Probe(self.base_neurons.neurons, "voltage")
                    
            if arm:
                self.arm = ArmNode(self.bot, msg_period=msg_period)
                dim = self.arm.get_input_dim()
                self.arm_neurons = nengo.Ensemble(n_neurons=dim*n_neurons_p_dim, dimensions=dim)
                nengo.Connection(self.arm_neurons, self.arm)
                if self.b_probe:
                    self.p_arm_neurons_out = nengo.Probe(self.arm_neurons, synapse=0.01)
                    self.p_arm_neurons_spikes = nengo.Probe(self.arm_neurons.neurons, "spikes")
                    self.p_arm_neurons_vol = nengo.Probe(self.arm_neurons.neurons, "voltage")
            if retina or freqs:
                names = connection.get_socket_keys()
                for name in names:
                    if "retina" in name:
                        self.bot.retina(name, True)
                        # if retina:
                        #     self.retina = RetinaNode(self.bot, name, msg_period=msg_period)
                        if freqs:
                            self.b_freqs = True
                            self.freqs = FrequencyNode(self.bot, name, msg_period=msg_period,
                                                       freqs=freqs)
                            dim = self.freqs.get_output_dim()
                            self.freqs_neurons = nengo.Ensemble(n_neurons=dim*n_neurons_p_dim, dimensions=dim)
                            nengo.Connection(self.freqs, self.freqs_neurons)

                            if self.b_probe:
                                self.p_freqs_out = nengo.Probe(self.freqs, synapse=0.01)
                                self.p_freqs_neurons_out = nengo.Probe(self.freqs_neurons, synapse=0.01)
                                self.p_freqs_neurons_spikes = nengo.Probe(self.freqs_neurons.neurons, "spikes")
                                self.p_freqs_neurons_vol = nengo.Probe(self.freqs_neurons.neurons, "voltage")

            if len(sensors) > 0:
                self.bot.activate_sensors(period=msg_period, **sensors)
                time.sleep(5.0)
                for k, v in sensors.items():
                    self.b_sensors[k] = v
                    if v:
                        setattr(self, k, SensorNode(self.bot, k))
                        dim = getattr(self, k).get_output_dim()
                        setattr(self, k+"_neurons", nengo.Ensemble(n_neurons=dim*n_neurons_p_dim, dimensions=dim))
                        nengo.Connection(getattr(self, k), getattr(self, k+"_neurons"))
                        
                        if self.b_probe:
                            setattr(self, "p_"+k+"_out", nengo.Probe(getattr(self,k), synapse=0.01))
                            setattr(self, "p_"+k+"_neurons_out", nengo.Probe(getattr(self,k+"_neurons"), synapse=0.01))
                            setattr(self, "p_"+k+"_neurons_spikes", nengo.Probe(getattr(self,k+"_neurons").neurons, "spikes"))
                            setattr(self, "p_"+k+"_neurons_vol", nengo.Probe(getattr(self,k+"_neurons").neurons, "voltage"))


