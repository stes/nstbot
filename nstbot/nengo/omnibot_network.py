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
        self.bot.base(x[0], x[1], x[2], msg_period=self.msg_period)

class ArmNode(nengo.Node):
    def __init__(self, bot, msg_period):
        super(ArmNode, self).__init__(self.move_arm, size_in=5, size_out=0)
        self.bot = bot
        self.msg_period = msg_period

    def move_arm(self, t, x):
        self.bot.arm(x[0], x[1], x[2], x[3], x[4], msg_period=self.msg_period)

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



class OmniArmBotNetwork(nengo.Network):
    def __init__(self, connection, msg_period=0.01, label='OmniArmBot',
                 base=False, arm=False, retina=False, freqs=[],
                 **sensors):
        super(OmniArmBotNetwork, self).__init__(label=label)
        self.bot = nstbot.OmniArmBot()
        self.bot.connect(connection)

        for name in self.bot.adress_list:
            if 'retina' in name:
                self.bot.retina(name, True)
                self.bot.track_frequencies(name, freqs=freqs)

        with self:
            if base:
                self.base = BaseNode(self.bot, msg_period=msg_period)
            if arm:
                self.arm = ArmNode(self.bot, msg_period=msg_period)
            if retina or freqs:
                names = connection.get_socket_keys()
                for name in names:
                    if "retina" in name:
                        self.bot.retina(name, True)
                        # if retina:
                        #     self.retina = RetinaNode(self.bot, name, msg_period=msg_period)
                        if freqs:
                            self.freqs = FrequencyNode(self.bot, name, msg_period=msg_period,
                                                       freqs=freqs)
            if len(sensors) > 0:
                self.bot.activate_sensors(period=msg_period, **sensors)
                time.sleep(5.0)
                for k, v in sensors.items():
                    if v:
                        setattr(self, k, SensorNode(self.bot, k))


