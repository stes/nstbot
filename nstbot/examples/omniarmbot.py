import time
import nstbot
import numpy as np

bot = nstbot.OmniArmBot()
address_list = {'retina_left': ['10.162.177.29', 54320],
               'retina_right': ['10.162.177.29', 54321],
               'retina_arm': ['10.162.177.29', 54323],
               'motors': [ '10.162.177.29', 54322]}

bot.connect(nstbot.SocketList(address_list))
bot.activate_sensors(period=0.1, euler=False,
                     bump=False, wheel=False,
                     gyro=False, accel=False,
                     compass=False, servo=False,
                     load=False)
for name, val in address_list.iteritems():
    if 'retina' in name:
        bot.retina(name, False)
        #bot.track_frequencies(name, freqs=[1000])
        #bot.show_image(name)
        print 'init with regular mode'
        bot.tracker('retina_arm', True, tracking_freqs=[200], streaming_period=10000)

print 'switch to blob mode'
bot.tracker('retina_arm', True, tracking_freqs=[200], streaming_period=10000, mode='blob')

cnt = 0

while cnt < 10:
    # print 'wheel', bot.sensor['wheel']
    # print 'bump', bot.sensor["bump"]
    # print 'gyro', bot.sensor['gyro']
    # print 'accel', bot.sensor['accel']
    # print 'euler', bot.sensor['euler']
    # print 'compass', bot.sensor['compass']
    # print 'servo', bot.sensor['servo']
    # print 'load', bot.sensor['load']
    for key, value in address_list.iteritems():
        if "retina" in key:
            print "tracked point in " + key + ": ", [bot.trk_px[key],  bot.trk_py[key]]
    print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    bot.base([0.5, 0.5, 0.5])
    # bot.arm([np.pi+1, np.pi-1.5, np.pi-1, 1])
    time.sleep(1)
    cnt += 1
    # bot.arm([2.28, 1.36, 4.24, 1])

print 'switch to regular mode'
bot.tracker('retina_arm', True, tracking_freqs=[200], streaming_period=10000)

cnt = 0

while cnt < 10:
    # print 'wheel', bot.sensor['wheel']
    # print 'bump', bot.sensor["bump"]
    # print 'gyro', bot.sensor['gyro']
    # print 'accel', bot.sensor['accel']
    # print 'euler', bot.sensor['euler']
    # print 'compass', bot.sensor['compass']
    # print 'servo', bot.sensor['servo']
    # print 'load', bot.sensor['load']
    for key, value in address_list.iteritems():
        if "retina" in key:
            print "tracked point in " + key + ": ", [bot.trk_px[key],  bot.trk_py[key]]
    print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    bot.base([0.5, 0.5, 0.5])
    # bot.arm([np.pi+1, np.pi-1.5, np.pi-1, 1])
    time.sleep(1)
    cnt += 1
    # bot.arm([2.28, 1.36, 4.24, 1])
