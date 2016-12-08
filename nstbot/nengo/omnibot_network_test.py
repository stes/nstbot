import nstbot
import numpy as np
import nengo


model = nengo.Network()
with model:
    address_list = {'retina_left': ['10.162.177.29', 54320],
                    'retina_right': ['10.162.177.29', 54321],
                    'retina_arm': ['10.162.177.29', 54323],
                    'motors': ['10.162.177.29', 54322]}

    bot = nstbot.OmniArmBotNetwork(
            nstbot.SocketList(address_list),
            base=True, retina=True, arm=True, freqs=[100, 300, 400],
            accel=True, bump=True, wheel=True, euler=True, servo=True, load=True,
            compass=True, gyro=True, msg_period=0.1)


    ctrl_base = nengo.Node([0,0,0])
    nengo.Connection(ctrl_base, bot.base)

    ctrl_arm = nengo.Node([np.pi, np.pi, np.pi, 0])
    nengo.Connection(ctrl_arm, bot.arm)

