import time
import nstbot

bot = nstbot.OmniArmBot()
bot.connect(nstbot.Socket('10.162.177.29'))
bot.activate_sensors(period=0.1,  wheel=True, gyro=True, accel=True, compass=True, servo=True, load=True)

while True:
    time.sleep(1)
    bot.motor(0,0,0.5)
    #print 'bump', bot.sensor["bump"]
    print 'wheel', bot.sensor['wheel']
    # print 'gyro', bot.sensor['gyro']
    # print 'accel', bot.sensor['accel']
    # print 'euler', bot.sensor['euler']
    # print 'compass', bot.sensor['compass']
    # print 'servo', bot.sensor['servo']
    # print 'load', bot.sensor['load']
