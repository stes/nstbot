import time
import nstbot

bot = nstbot.OmniArmBot()
bot.connect(nstbot.Socket('10.162.177.29'))
bot.activate_sensors(period=1.0, euler = True, bump=True, wheel=True, gyro=True, accel=True, compass=True, servo=True, load=True)

while True:
    print 'bump', bot.sensor["bump"]
    print 'wheel', bot.sensor['wheel']
    print 'gyro', bot.sensor['gyro']
    print 'accel', bot.sensor['accel']
    print 'euler', bot.sensor['euler']
    print 'compass', bot.sensor['compass']
    print 'servo', bot.sensor['servo']
    print 'load', bot.sensor['load']
    print '-------------------------'
    bot.motor(0.0, 0.0, 0.5)
    time.sleep(2)

