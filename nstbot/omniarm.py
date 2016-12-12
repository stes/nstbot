from . import nstbot
import numpy as np
import threading
import time


class OmniArmBot(nstbot.NSTBot):
    def initialize(self):
        super(OmniArmBot, self).initialize()
        self.retina_packet_size = {}
        self.image = {}
        self.count_spike_regions = {}
        self.track_periods = {}
        self.last_timestamp = {}
        self.p_x = {}
        self.p_y = {}
        self.track_certainty = {}
        self.count_regions = {}
        self.count_regions_scale = {}
        self.track_certainty_scale = {}

        self.track_sigma_t = {}
        self.track_sigma_p = {}
        self.track_eta = {}

        self.last_off = {}
        self.track_certainty = {}
        self.good_events = {}

        self.conn_thread = {}
        self.retina_thread = {}

        self.trk_px = {}
        self.trk_py = {}
        self.trk_radius = {}
        self.trk_certainty = {}

        for name in self.adress_list:
            if "retina" in name:
                self.retina_packet_size[name] = None
                self.image[name] = None
                self.count_spike_regions[name] = None
                self.track_periods[name] = None
                self.p_x[name] = None
                self.p_y[name] = None
                self.track_certainty[name] = None
                self.last_timestamp[name] = None
 
                # initialize variables for embedded tracker           
                self.trk_px[name] = np.array([None]*8)
                self.trk_py[name] = np.array([None]*8)
                self.trk_radius[name] = np.array([None]*8)
                self.trk_certainty[name] = np.array([None]*8)
        self.sensor = {}
        self.sensor_scale = {}
        self.sensor_map = {}
        self.add_sensor('bump', bit=0, range=1, length=1)
        # we have 8 values but we take only the first 3 vals (base motors)
        self.add_sensor('wheel', bit=1, range=100, length=3)  
        # the hex encoded values need conversion
        # we have 4 values but we take only the first 3 vals
        # gyro is measured in deg/s and Q16 encoded
        self.add_sensor('gyro', bit=2, range=2**32-1, length=3)
        # acc is measured in g and Q16 encoded
        self.add_sensor('accel', bit=3, range=2**32-1, length=3)
        # we have 4 values but we take only the first 3 vals
        # euler is measured in deg and Q16 encoded
        # roll [-90,90], pitch [-180,180], yaw [-180,180]
        self.add_sensor('euler', bit=4, range=2**32-1, length=3)
        # compass is measured in uT (micro tesla) and Q16 encoded
        self.add_sensor('compass', bit=5, range=2**32-1, length=3)
        # we have 8 values but we only take the last 5 (arm motors)
        self.add_sensor('servo', bit=7, range=4096, length=5)
        # we have 8 values but we only take the last 5 (arm motors)
        self.add_sensor('load', bit=9, range=4096, length=5) 
        self.sensor_bitmap = {"bump": [19, slice(0, 1)],
                              "wheel": [17,slice(0, 3)],
                              "gyro": [4,slice(0, 3)],
                              "accel": [5, slice(0,3)],
                              "euler": [9, slice(0,3)],
                              "compass": [6, slice(0, 3)],
                              "servo": [16, slice(3, 8)],
                              "load": [14, slice(3, 8)]}
        self.base([0.0, 0.0, 0.0])
        self.arm([np.pi, np.pi, np.pi, 0])

    def base(self, spd, msg_period=None):
        val_range = 100

        x = int(spd[0] * val_range)
        y = int(spd[1] * val_range)
        z = int(spd[2] * val_range)

        if x > val_range: x = val_range
        if x < -val_range: x = -val_range
        if y > val_range: y = val_range
        if y < -val_range: y = -val_range
        if z > val_range: z = val_range
        if z < -val_range: z = -val_range
        cmd = '!P0%d\n!P1%d\n!P2%d\n' % (x, y, z)
        self.send('motors', 'base', cmd, msg_period=msg_period)

    def base_pos(self, pos2d, msg_period=None):
        val_range = 100

        x = int(pos2d[0] * val_range)
        y = int(pos2d[1] * val_range)
        rot = int(pos2d[2] * val_range)

        if x > val_range: x = val_range
        if x < -val_range: x = -val_range
        if y > val_range: y = val_range
        if y < -val_range: y = -val_range
        if rot > val_range: rot = val_range
        if rot < -val_range: rot = -val_range
        cmd = '!D%d,%d,%d\n' % (x, y, rot)
        self.send('motors', 'base_pos', cmd, msg_period=msg_period)

    def arm(self, joints, msg_period=None):
        # motion should be limited by the dynamics of the robot

        # convert to numpy for special ops
        jnt = np.array(joints)

        # apply rad to position map
        pos = (jnt[:3]*10000).astype(dtype=np.int)

        # min pos and max pos in the range
        min_val = 0
        max_val = 62830

        # apply range
        pos[pos < min_val] = min_val
        pos[pos > max_val] = max_val

        # separate each link
        pos = np.append(pos, jnt[3])
        [shoulder, elbow, hand, gripper] = pos

        # indices for motor IDs
        cmd = '!r%d,%d,%d\n' % (shoulder, elbow, hand)
        if abs(float(gripper - 1)) < 0.0001:
            grip = '!y\n'
        else:
            grip = '!x\n'
        cmd += grip
        self.send('motors', 'arm', cmd, msg_period=msg_period)

    def set_arm_speed(self, x, msg_period=None):
        if x > 0:
            self.send('motors', 'arm', '!P3%d\n!P4%d\n!P5%d\n!P750\n' % (x, x, x), msg_period=msg_period)
        else:
            self.send('motors', 'arm', '!P35\n!P45\n!P55\n', msg_period=msg_period)

    def add_sensor(self, name, bit, range, length):
        value = np.zeros(length)
        self.sensor[bit] = value
        self.sensor[name] = value
        self.sensor_map[bit] = name
        self.sensor_map[name] = bit
        self.sensor_scale[bit] = 1.0 / range

    def activate_sensors(self, period=0.1, **names):
        bits = 0
        for name, b_activate in names.items():
            bit = self.sensor_map[name]
            if b_activate:
                bits += 1 << bit
        cmd = '!I1,%d,%d\n' % (int(1.0/period), bits)
        self.connection.send('motors', cmd)

    def send(self, name, key, message, msg_period=None):
        now = time.time()
        if msg_period is None or now > self.last_time.get(key, 0) + msg_period:
            self.connection.send(name, message)
            self.last_time[key] = now

    def get_sensor(self, name):
        return self.sensor[name]

    def connect(self, connection):
        self.adress_list = connection.get_socket_keys()
        super(OmniArmBot, self).connect(connection)
        for name in self.adress_list:
            self.connection.send(name,'R\n')  # reset platform
            time.sleep(1)
            if "retina" not in name:
                self.connection.send(name,'!E0\n')  # disable command echo (save some bandwidth)
                time.sleep(1)
            else:
                self.connection.send(name, 'E+\n')
                time.sleep(1)
            self.set_arm_speed(20)
            time.sleep(0.5)
            self.conn_thread[name] = threading.Thread(target=self.sensor_loop, args=(name,))
            #self.conn_thread[name] = multiprocessing.Process(target=self.sensor_loop, args=(name,))
            self.conn_thread[name].daemon = True
            self.conn_thread[name].start()

    def disconnect(self):
        for name in self.adress_list:
            if 'retina' not in name:
                self.connection.send(name, '!I0\n')
                self.connection.send(name, 'R\n')
            else:
                self.retina(name, False)
            # for process in multiprocessing.active_children():
            #     if process.is_alive():
            #         process.join()
            #     process.terminate()
        self.base([0, 0, 0])
        self.arm([np.pi, np.pi, np.pi, 0])
        super(OmniArmBot, self).disconnect()

    def retina(self, name, active, bytes_in_timestamp=4):
        if active:
            assert bytes_in_timestamp in [0, 2, 3, 4]
            cmd = '!E%d\nE+\n' % bytes_in_timestamp
            self.retina_packet_size[name] = 2 + bytes_in_timestamp
        else:
            cmd = 'E-\n'
            self.retina_packet_size[name] = None
        self.connection.send(name, cmd)

    def tracker(self, name, active, tracking_freqs, streaming_period):
        # calculate tracking period from frequency
        tracking_periods = np.array([int(np.ceil(freq*1000*1000)) for freq in tracking_freqs]) 

        for channel, tracking_period in enumerate(tracking_periods):
            if active:
                cmd = '!TD%d=%d\n!TR=%d\n' % (channel, tracking_period, streaming_period)
            else:
                cmd = '!TD%d=0\n!TR=0\n' % (channel)
            
        self.connection.send(name, cmd)


    def show_image(self, name, decay=0.5, display_mode='quick'):
        if self.image[name] is None:
            self.image[name] = np.zeros((128, 128), dtype=float)
            self.retina_thread[name] = threading.Thread(target=self.image_loop,
                                      args=(name, decay, display_mode))
            # self.retina_thread[name] = multiprocessing.Process(target=self.image_loop,
            #                           args=(name, decay, display_mode))
            self.retina_thread[name].daemon = True
            self.retina_thread[name].start()

    def get_image(self, name):
        return self.image[name]

    def keep_image(self, name):
        if self.image[name] is None:
            self.image[name] = np.zeros((128, 128), dtype=float)

    def image_loop(self, name, decay, display_mode):
        import pylab

        import matplotlib.pyplot as plt
        # using axis for updating only parts of the image that change
        fig, ax = plt.subplots()
        # so quick mode can run on ubuntu
        plt.show(block=False)

        pylab.ion()
        img = pylab.imshow(self.image[name], vmax=1, vmin=-1,
                                       interpolation='none', cmap='binary')
        pylab.xlim(0, 127)
        pylab.ylim(127, 0)

        regions = {}
        if self.count_spike_regions[name] is not None:
            for k, v in self.count_spike_regions[name].items():
                minx, miny, maxx, maxy = v
                rect = pylab.Rectangle((minx - 0.5, miny - 0.5),
                                       maxx - minx,
                                       maxy - miny,
                                       facecolor='yellow', alpha=0.2)
                pylab.gca().add_patch(rect)
                regions[k] = rect

        if self.track_periods[name] is not None:
            colors = ([(0,0,1), (0,1,0), (1,0,0), (1,1,0), (1,0,1)] * 10)[:len(self.p_y)]
            scatter = pylab.scatter(self.p_x[name], self.p_y[name], s=50, c=colors)
        else:
            scatter = None

        while True:

            img.set_data(self.image[name])

            for k, rect in regions.items():
                alpha = self.get_spike_rate(k) * 0.5
                alpha = min(alpha, 0.5)
                rect.set_alpha(0.05 + alpha)
            if scatter is not None:
                scatter.set_offsets(np.array([self.p_x[name], self.p_y[name]]).T)
                c = [(r,g,b,min(self.track_certainty[name][i],1)) for i,(r,g,b) in enumerate(colors)]
                scatter.set_color(c)

            if display_mode == 'quick':
                # this is faster, but doesn't work on all systems
                fig.canvas.draw()
                fig.canvas.flush_events()

            elif display_mode == 'ubuntu_quick':
                # this is even faster, but doesn't work on all systems
                ax.draw_artist(ax.patch)
                ax.draw_artist(img)
                ax.draw_artist(scatter)
                fig.canvas.update()

                fig.canvas.flush_events()
            else:
                # this works on all systems, but is kinda slow
                pylab.pause(1e-8)

            self.image[name] *= decay

    def sensor_loop(self, name):
        """Handle all data coming from the robot."""
        old_data = None
        buffered_ascii = ''
        while True:
            if "retina" in name:
                packet_size = self.retina_packet_size[name]
            else:
                packet_size = None
            # grab the new data
            data = self.connection.receive(name)

            # combine it with any leftover data from last time through the loop
            if old_data is not None:
                data = old_data + data
                old_data = None

            if packet_size is None:
                # no retina events, so everything should be ascii
                buffered_ascii += data
            else:
                # find the ascii events
                data_all = np.fromstring(data, np.uint8)
                ascii_index = np.where(data_all[::packet_size] < 0x80)[0]

                offset = 0
                while len(ascii_index) > 0:
                    # if there's an ascii event, remove it from the data
                    index = ascii_index[0] * packet_size
                    stop_index = np.where(data_all[index:] >= 0x80)[0]
                    if len(stop_index) > 0:
                        stop_index = index + stop_index[0]
                    else:
                        stop_index = len(data)

                    # and add it to the buffered_ascii list
                    buffered_ascii += data[offset + index:offset + stop_index]
                    data_all = np.hstack((data_all[:index],
                                          data_all[stop_index:]))
                    offset += stop_index - index
                    ascii_index = np.where(data_all[::packet_size] < 0x80)[0]

                # handle any partial retina packets
                extra = len(data_all) % packet_size
                if extra != 0:
                    old_data = data[-extra:]
                    data_all = data_all[:-extra]
                if len(data_all) > 0:
                    # now process those retina events
                    if "retina" in name:
                        self.process_retina(name, data_all)
                
            if "retina" not in name:
                # and process the ascii events too from the base and arm sensors
                while '\n\n' in buffered_ascii:
                    cmd, buffered_ascii = buffered_ascii.split('\n\n', 1)
                    if '-I' in cmd:
                        dbg, proc_cmd = cmd.split('-I', 1)
                        self.process_ascii(name, '-I' + proc_cmd)
            else:
                # process ascii events from embedded tracker
                while '\n' in buffered_ascii:
                    cmd, buffered_ascii = buffered_ascii.split('\n', 1)
                    if '-T' in cmd:
                        dbg, proc_cmd = cmd.split('-T', 1)
                        self.process_ascii(name, '-T' + proc_cmd)


    def process_ascii(self, name, message):
        try:
            # handle sensory data
            if message[:2] == '-I':
                data = message[2:].split()
                sp_data = data[1:]
                hdr_idx = [ind for ind, s in enumerate(sp_data) if '-S' in s]
                for ind, el in enumerate(hdr_idx):
                    if ind < len(hdr_idx) - 1:
                        vals = sp_data[hdr_idx[ind] + 1:hdr_idx[ind + 1]]
                    else:
                        vals = sp_data[hdr_idx[ind] + 1:]
                    src = int(sp_data[hdr_idx[ind]][2:])
                    for name, value in self.sensor_bitmap.iteritems():
                        if src == value[0]:
                            sliced = value[1]
                            index = self.sensor_map[name]
                            scale = self.sensor_scale[index]
                            # FIXME Check the correct ranges and conversions
                            if scale < 1./6000 or name is not 'bump':
                                sensors = [float.fromhex(x)*scale for x in vals[sliced]]
                            else:
                                sensors = [float(x)*scale for x in vals[sliced]]
                            self.sensor[index] = sensors
                            self.sensor[self.sensor_map[index]] = sensors
            # handle uDVS tracker data
            if message[:2] == '-T':
                trk_data = message[2:]
                # make sure, that the message is not the DVS confirmation msg:
                # TODO: do we need to track more than one frequency per retina? 
                # if yes, how do we get the information about the current frequency from the incoming message?
                # in this case we need to make adjustments accodringly here, in the get_tracker function and in the firmware
                # Update: up to 8 tracked frequencies are possible for each retina (arcodring changes need test)
                if len(trk_data) > 5:
                    trk_id = trk_data[0]        # uDVS tracker id
                    trk_xpos = trk_data[1:5]    # xpos 4byte HEX, scale = 2^16
                    trk_ypos = trk_data[5:9]    # ypos 4byte HEX, scale = 2^16
                    trk_rad = trk_data[9:11]    # tracking radius 2byte HEX, scale = 2^8
                    trk_cert = trk_data[11:13]  # tracking certainty 2byte HEX, scale = 2^8
                    self.trk_px[name][trk_id] = float.fromhex(trk_xpos)*(2**16-1)
                    self.trk_py[name][trk_id] = float.fromhex(trk_ypos)*(2**16-1)
                    self.trk_radius[name][trk_id] = float.fromhex(trk_rad)*(2**8-1)
                    self.trk_certainty[name][trk_id] = float.fromhex(trk_cert)*(2**8-1)
        except:
            pass
            # print('Error processing "%s"' % message)
            # import traceback
            # traceback.print_exc()

    last_timestamp = None

    def process_retina(self, name, data):
        packet_size = self.retina_packet_size[name]
        y = data[::packet_size] & 0x7f
        x = data[1::packet_size] & 0x7f
        if self.image[name] is not None:
            value = np.where(data[1::packet_size]>=0x80, 1, -1)
            self.image[name][y, x] += value

        if self.count_spike_regions[name] is not None:
            tau = 0.05 * 1000000
            for k, region in self.count_spike_regions[name].items():
                minx, miny, maxx, maxy = region
                index = (minx <= x) & (x<maxx) & (miny <= y) & (y<maxy)
                count = np.sum(index)
                t = (int(data[-2]) << 8) + data[-1]
                if packet_size >= 5:
                    t += int(data[-3]) << 16
                if packet_size >= 6:
                    t += int(data[-4]) << 24

                old_count, old_time = self.count_regions[name][k]

                dt = float(t - old_time)
                if dt < 0:
                    dt += 1 << ((packet_size - 2) * 8)
                count *= self.count_regions_scale[name][k]
                count /= dt / 1000.0

                decay = np.exp(-dt/tau)
                new_count = old_count * (decay) + count * (1-decay)

                self.count_regions[name][k] = new_count, t

        if self.track_periods[name] is not None:
            t = data[2::packet_size].astype(np.uint32)
            t = (t << 8) + data[3::packet_size]
            if packet_size >= 5:
                t = (t << 8) + data[4::packet_size]
            if packet_size >=6:
                t = (t << 8) + data[5::packet_size]

            if self.last_timestamp[name] is not None:
                dt = float(t[-1]) - self.last_timestamp[name]
                if dt < 0:
                    dt += 1 << (8 * (packet_size-2))
            else:
                dt = 1
            self.last_timestamp[name] = t[-1]

            index_off = (data[1::packet_size] & 0x80) == 0

            delta = np.where(index_off, t - self.last_off[name][x, y], 0)

            self.last_off[name][x[index_off],
                         y[index_off]] = t[index_off]

            tau = 0.05 * 1000000
            decay = np.exp(-dt/tau)
            self.track_certainty[name] *= decay

            for i, period in enumerate(self.track_periods[name]):
                eta = self.track_eta[name]
                t_exp = period * 2
                sigma_t = self.track_sigma_t[name]    # in microseconds
                sigma_p = self.track_sigma_p[name]    # in pixels
                t_diff = delta.astype(np.float) - t_exp

                w_t = np.exp(-(t_diff**2)/(2*sigma_t**2))
                px = self.p_x[name][i]
                py = self.p_y[name][i]

                dist2 = (x - px)**2 + (y - py)**2
                w_p = np.exp(-dist2/(2*sigma_p**2))

                ww = w_t * w_p
                c = sum(ww) * self.track_certainty_scale[name] / dt

                self.track_certainty[name][i] += (1-decay) * c

                w = eta * ww
                for j in np.where(w > eta * 0.1)[0]:
                        px += w[j] * (x[j] - px)
                        py += w[j] * (y[j] - py)
                self.p_x[name][i] = px
                self.p_y[name][i] = py

    def track_spike_rate(self,name, **regions):
        self.count_spike_regions[name] = regions
        self.count_regions[name] = {}
        self.count_regions_scale[name] = {}
        for k,v in regions.items():
            self.count_regions[name][k] = [0, 0]
            area = (v[2] - v[0]) * (v[3] - v[1])
            self.count_regions_scale[name][k] = 200.0 / area

    def get_spike_rate(self, name, region):
        return self.count_regions[name][region][0]

    def track_frequencies(self, name, freqs, sigma_t=100, sigma_p=30, eta=0.3,
                                 certainty_scale=10000):
        freqs = np.array(freqs, dtype=float)
        track_periods = 500000 / freqs
        self.track_certainty_scale[name] = certainty_scale

        self.track_sigma_t[name] = sigma_t
        self.track_sigma_p[name] = sigma_p
        self.track_eta[name] = eta

        self.last_off[name] = np.zeros((128, 128), dtype=np.uint32)
        self.p_x[name] = np.zeros_like(track_periods) + 64.0
        self.p_y[name] = np.zeros_like(track_periods) + 64.0
        self.track_certainty[name] = np.zeros_like(track_periods)
        self.good_events[name] = np.zeros_like(track_periods, dtype=int)
        self.track_periods[name] = track_periods

    def get_frequency_info(self, name, index):
        x = self.p_x[name][index] / 64.0 - 1
        y = - self.p_y[name][index] / 64.0 + 1
        return x, y, self.track_certainty[name][index]

    def get_tracker_info(self, name):
        x = self.trk_px[name]/ 64.0 - 1
        y = - self.trk_py[name]/ 64.0 + 1
        return x, y, self.trk_radius[name], self.trk_certainty[name]
