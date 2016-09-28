

import readchar

import nengo
import udp
import timeit
import threading
import time

model = nengo.Network()
with model:

    sender = udp.UDPSender('localhost', 9999, size_in=2, period=0.01)

    class Keyboard(nengo.Node):
        def __init__(self):
            self.value = [0,0]
            super(Keyboard, self).__init__(self.step)

            self.last_key_time = timeit.default_timer()

            t1 = threading.Thread(target=self.reset_thread)
            t2 = threading.Thread(target=self.read_thread)
            t1.daemon = True
            t2.daemon = True
            t1.start()
            t2.start()


        def step(self, t):
            time.sleep(0.001)
            #print self.value
            if self.value is None:
                raise Execption("Quitting :-)")
            return self.value

        def read_thread(self):
            while True:
                time.sleep(0.001)
                c = readchar.readchar()
                if c.lower() == '8':
                    # drive forward
                    self.value[0] = 1
                if c.lower() == '2':
                    # drive backward
                    self.value[0] = -1
                if c.lower() == '6':
                    # turn right
                    self.value[1] = 1
                if c.lower() == '4':
                    # turn left
                    self.value[1] = -1
                if c.lower() == '7':
                    # drive left curve forward
                    self.value[0] = 0.5
                    self.value[1] = -0.5
                if c.lower() == '9':
                    # drive right curve forward
                    self.value[0] = 0.5
                    self.value[1] = 0.5
                if c.lower() == '1':
                    # drive left curve backward
                    self.value[0] = -1
                    self.value[1] = 0.5
                if c.lower() == '3':
                    # drive right curve backward
                    self.value[0] = -1
                    self.value[1] = -0.5
                if c.lower() == 'q':
                    self.value = None
                self.last_key_time = timeit.default_timer()


        def reset_thread(self):
            while True:
                time.sleep(0.001)
                now = timeit.default_timer()

                if now - self.last_key_time > 0.1:
                    self.value[0] = 0
                    self.value[1] = 0
                    self.last_key_time = now


    keybd = Keyboard()

    nengo.Connection(keybd, sender, synapse=None)


sim = nengo.Simulator(model)
while True:
    sim.run(10, progress_bar=False)

