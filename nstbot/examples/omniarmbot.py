import time
import nstbot
import numpy as np

bot = nstbot.OmniArmBot()
address_list = {'retina_left': ['10.162.177.29', 54320],
               'retina_right': ['10.162.177.29', 54321],
               'retina_arm': ['10.162.177.29', 54323],
               'motors': [ '10.162.177.29', 54322]}

bot.connect(nstbot.SocketList(address_list))
bot.activate_sensors(period=0.1, euler=True,
                     bump=True, wheel=True,
                     gyro=True, accel=True,
                     compass=True, servo=True,
                     load=True)
for name, val in address_list.iteritems():
    if 'retina' in name:
        bot.retina(name, True)
        bot.track_frequencies(name, freqs=[1000])
        #bot.show_image(name)

while True:
    print 'wheel', bot.sensor['wheel']
    print 'bump', bot.sensor["bump"]
    print 'gyro', bot.sensor['gyro']
    print 'accel', bot.sensor['accel']
    print 'euler', bot.sensor['euler']
    print 'compass', bot.sensor['compass']
    print 'servo', bot.sensor['servo']
    print 'load', bot.sensor['load']
    for key, value in address_list.iteritems():
        if "retina" in key:
            print "tracked point in " + key + ": ", [bot.p_x[key],  bot.p_y[key]]
    print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    bot.base([0.5, 0.5, 0.5])
    bot.arm([np.pi, np.pi, np.pi/2, 0])
    time.sleep(4)
    bot.arm([np.pi, np.pi, np.pi, 1])
    time.sleep(4)