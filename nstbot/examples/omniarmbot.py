import time
import nstbot

bot = nstbot.OmniArmBot()
adress_list = {'retina_left': ['10.162.177.29', 54320], 
               'retina_right': ['10.162.177.29', 54321], 
               'retina_arm': ['10.162.177.29', 54323], 
               'motors': ['10.162.177.29', 54322]}
bot.connect(nstbot.SocketList(adress_list))
bot.activate_sensors(period=1.0, euler=True,
                     bump=True, wheel=True,
                     gyro=True, accel=True,
                     compass=True, servo=True,
                     load=True)

while True:
    print 'wheel', bot.sensor['wheel']
    print 'bump', bot.sensor["bump"]
    print 'gyro', bot.sensor['gyro']
    print 'accel', bot.sensor['accel']
    print 'euler', bot.sensor['euler']
    print 'compass', bot.sensor['compass']
    print 'servo', bot.sensor['servo']
    print 'load', bot.sensor['load']
    for key, value in adress_list.iteritems():
        if "retina" in key:
            print "tracked point in " + key + ": ", [bot.p_x[key],  bot.p_y[key]]
    print '-------------------------'
    bot.base_pos(0.0, 0.0, 0.5)
    time.sleep(0.1)