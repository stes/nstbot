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


class TrackerNode(nengo.Node):
    def __init__(self, name, bot, tr_freq, st_freq):
        super(TrackerNode, self).__init__(self.tracked_freqs, label=name,
                                            size_in=0, size_out=4*len(tr_freq))
        self.bot = bot
        self.name = name
        self.msg_period = st_freq
        self.result = np.zeros(4*len(, dtype='float')

    def tracked_freqs(self, t):
        return self.bot.get_tracker_info(self.name)

    def get_output_dim(self):
        return len(self.result)


class OmniArmBotNetwork(nengo.Network):
    def __init__(self, connection, send_msg_period=0.01, receive_msg_period=0.01,
                 label='OmniArmBot',
                 n_neurons_p_dim=None, b_probe=False,
                 base=False, arm=False, retina=False, freqs=[],
                 tracker=False,
                 **sensors):
        super(OmniArmBotNetwork, self).__init__(label=label)
        self.bot = nstbot.OmniArmBot()
        self.bot.connect(connection)

        self.freqs = {}
        self.freqs_neurons = {}
        self.b_probe = b_probe
        self.b_base = base
        self.b_arm = arm
        self.b_retina = retina
        self.b_freqs = False
        self.b_sensors = {}
        self.b_tracker = tracker
        self.trackers = {}
        self.tracker_neurons = {}

        self.p_freqs_out = {}
        self.p_freqs_neurons_out = {}
        self.p_freqs_neurons_spikes = {}
        self.p_freqs_neurons_vol = {}

        self.p_trackers_out = {}
        self.p_trackers_neurons_out = {}
        self.p_trackers_neurons_spikes = {}
        self.p_trackers_neurons_vol = {}

        # TODO INTEGRATE THE TRACKER IN THE NETWORK
        for name in self.bot.adress_list:
            if 'retina' in name:
                self.bot.retina(name, True)
                if tracker:
                    # TODO: come the information from the embedded tracking on the same port as the DVS, on the motor port or on a completely different port?
                    if "left" in name:
                        # TODO: can only one frequency be tracked on board or many? If only one is possible, we can get rid of this for-loop
                        for freq in freqs:
                            self.bot.tracker(channel=0, active=True, tracking_freq=freq, streaming_period=receive_msg_period)
                    elif "right" in name:
                        # TODO: can only one frequency be tracked on board or many? If only one is possible, we can get rid of this for-loop
                        for freq in freqs:
                            self.bot.tracker(channel=1, active=True, tracking_freq=freq, streaming_period=receive_msg_period)
                    else:
                        # TODO: can only one frequency be tracked on board or many? If only one is possible, we can get rid of this for-loop
                        for freq in freqs:
                            self.bot.tracker(channel=2, active=True, tracking_freq=freq, streaming_period=receive_msg_period)

                else:
                    self.bot.track_frequencies(name, freqs=freqs)


        with self:
            if base:
                self.base = BaseNode(self.bot, msg_period=send_msg_period)
                if n_neurons_p_dim is not None:
                    dim = self.base.get_input_dim()
                    self.base_neurons = nengo.Ensemble(n_neurons=dim*n_neurons_p_dim, dimensions=dim)
                    nengo.Connection(self.base_neurons, self.base)
                    if self.b_probe:
                        self.p_base_neurons_out = nengo.Probe(self.base_neurons, synapse=0.01)
                        self.p_base_neurons_spikes = nengo.Probe(self.base_neurons.neurons, "spikes")
                        self.p_base_neurons_vol = nengo.Probe(self.base_neurons.neurons, "voltage")
                    
            if arm:
                self.arm = ArmNode(self.bot, msg_period=send_msg_period)
                if n_neurons_p_dim is not None:
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
                        #     self.retina = RetinaNode(self.bot, name, msg_period=receive_msg_period)
                        if freqs and not tracker:
                            self.b_freqs = True
                            self.freqs[name] = FrequencyNode(self.bot, name, msg_period=receive_msg_period,
                                                       freqs=freqs)
                            if n_neurons_p_dim is not None:
                                dim = self.tracker_neurons[name].get_output_dim()
                                self.tracker_neurons[name] = nengo.Ensemble(n_neurons=dim*n_neurons_p_dim, dimensions=dim)
                                nengo.Connection(self.freqs[name], self.freqs_neurons[name])

                                if self.b_probe:
                                    self.p_freqs_out[name] = nengo.Probe(self.freqs[name], synapse=0.01)
                                    self.p_freqs_neurons_out[name] = nengo.Probe(self.freqs_neurons[name], synapse=0.01)
                                    self.p_freqs_neurons_spikes[name] = nengo.Probe(self.freqs_neurons[name].neurons, "spikes")
                                    self.p_freqs_neurons_vol[name] = nengo.Probe(self.freqs_neurons[name].neurons, "voltage")
                        if tracker:
                            self.trackers[name] = TrackerNode(self.bot, name, tracking_freq=freqs, streaming_period=receive_msg_period)

                            if n_neurons_p_dim is not None:
                                dim = self.trackers[name].get_output_dim()
                                self.tracker_neurons[name] = nengo.Ensemble(n_neurons=dim*n_neurons_p_dim, dimensions=dim)
                                nengo.Connection(self.trackers[name], self.tracker_neurons[name])

                                if self.b_probe:    
                                    self.p_tracker_out[name] = nengo.Probe(self.tracker[name], synapse=0.01)
                                    self.p_tracker_neurons_out[name] = nengo.Probe(self.tracker_neurons[name], synapse=0.01)
                                    self.p_tracker_neurons_spikes[name] = nengo.Probe(self.tracker_neurons[name].neurons, "spikes")
                                    self.p_tracker_neurons_vol[name] = nengo.Probe(self.tracker_neurons[name].neurons, "voltage")


            if len(sensors) > 0:
                self.bot.activate_sensors(period=receive_msg_period, **sensors)
                time.sleep(5.0)
                for k, v in sensors.items():
                    self.b_sensors[k] = v
                    if v:
                        setattr(self, k, SensorNode(self.bot, k))
                        if n_neurons_p_dim is not None:
                            dim = getattr(self, k).get_output_dim()
                            setattr(self, k+"_neurons", nengo.Ensemble(n_neurons=dim*n_neurons_p_dim, dimensions=dim))
                            nengo.Connection(getattr(self, k), getattr(self, k+"_neurons"))
                            
                            if self.b_probe:
                                setattr(self, "p_"+k+"_out", nengo.Probe(getattr(self,k), synapse=0.01))
                                setattr(self, "p_"+k+"_neurons_out", nengo.Probe(getattr(self,k+"_neurons"), synapse=0.01))
                                setattr(self, "p_"+k+"_neurons_spikes", nengo.Probe(getattr(self,k+"_neurons").neurons, "spikes"))
                                setattr(self, "p_"+k+"_neurons_vol", nengo.Probe(getattr(self,k+"_neurons").neurons, "voltage"))


