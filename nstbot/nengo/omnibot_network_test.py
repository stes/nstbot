import nstbot
import numpy as np
import nengo


model = nengo.Network()
with model:
    bot = nstbot.OmniArmBotNetwork(
            nstbot.Socket('10.162.177.102', port=56000),
            base=True, retina=False, #freqs=[100, 200, 300],
            accel=True, bump=True, wheel=True, euler=True, servo=True, load=True,
            compass=True, gyro=True, msg_period=0.1)

    ctrl_base = nengo.Node([0, 0, 0])
    nengo.Connection(ctrl_base, bot.base)


    # ctrl_arm = nengo.Node([0.184, 0.172, 0.394, 0.052, 0.134])
    # nengo.Connection(ctrl_arm, bot.arm)

