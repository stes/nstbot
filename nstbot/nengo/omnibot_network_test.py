import nstbot
import numpy as np
import nengo
import signal
import sys
import plot_helpers as ph

model = nengo.Network()
with model:
    address_list = {'retina_left': ['10.162.177.29', 54320],
                    'retina_right': ['10.162.177.29', 54321],
                    'retina_arm': ['10.162.177.29', 54323],
                    'motors': ['10.162.177.29', 54322]}

    bot = nstbot.OmniArmBotNetwork(
            nstbot.SocketList(address_list), b_probe=True,
            base=True, retina=False, arm=True, freqs=[100],
            accel=True, bump=True, wheel=True, euler=True,
            servo=True, load=True,
            compass=True, gyro=True, msg_period=0.1)


    ctrl_base = nengo.Node([0.5, 0.5, 0.5])
    nengo.Connection(ctrl_base, bot.base)

    def ctrl_arm_func(t):
        t_mod = t % 30
        if t_mod > 15:
            return [np.pi, np.pi, np.pi, 0]
        else:
            return [np.pi, np.pi, np.pi/2, 1]

    ctrl_arm = nengo.Node(ctrl_arm_func)

    #ctrl_arm = nengo.Node([np.pi, np.pi, np.pi, 0])

    nengo.Connection(ctrl_arm, bot.arm)

def signal_handler(signal, frame):
    bot.bot.disconnect()
    sys.exit(0)

if __name__ == "__main__":
    backend = "nengo" # select between different backends (nengo, nengo_ocl, nengo_spinnaker)
    sim_time = 1 # set this to None to make the simulation run forever
    sim = None # init sim variable

    # signal handler enable
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


    if backend == "nengo":
      print "Reference"
      sim = nengo.Simulator(model)
    elif backend == "nengo_ocl":
      "print GPU"
      import nengo_ocl
      import os
      os.environ["PYOPENCL_CTX"]="0:0,1,2,3"
      sim = nengo_ocl.Simulator(model)
    elif backend == "nengo_spinnaker":
      print "SpiNNaker"
      import nengo_spinnaker
      sim = nengo_spinnaker.Simulator(model)

    if sim is not None:
        if sim_time is not None:
            sim.run(sim_time)
        else:
            while True:
                sim.run(10)
        print "simulation finished"

    if bot.b_probe:
        ph.plot_function(bot, sim)
    sim.close()
    print "script finished"