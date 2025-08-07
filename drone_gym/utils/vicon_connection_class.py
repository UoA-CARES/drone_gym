import socket
import time
import builtins
from struct import unpack
from enum import Enum
from datetime import datetime
import math
from random import randint
import queue
from threading import Thread

us_start = int(time.time() * 1000 * 1000)

def micros():
    us_now = int(time.time() * 1000 * 1000)
    return int(us_now - us_start)


def micros():
    us_now = int(time.time() * 1000 * 1000)

    return int(us_now - us_start)

class ViconInterface():

    def __init__(self, udp_ip="0.0.0.0", udp_port=51001):
        # Port and IP to bind UDP listener
        self.udp_port = udp_port
        self.udp_ip = udp_ip

        # Bind the listener
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((udp_ip, udp_port))

        # Used to time packet frequency to ensure FPS
        self.time_last_packet = 0
        self.FPS = 0

        # Array of dictionaries to store the tracked objects being recieved
        self.tracked_object = {}

        # Flag to end the main loop
        self.run_interface = True

        # Flag that the interface has heard from Vicon
        self.have_recv_packet = False
    # Ends main loop on closure
    def end(self):
        self.run_interface = False
        self.sock.close()

    def main_loop(self):
        try:
            while self.run_interface:
                # Block until there is a packet
                b, addr = self.sock.recvfrom(256)

                # First 5 bytes are the frame # and # of items in frame
                FrameNumber = int.from_bytes(b[0:4], byteorder='little')
                ItemsInBlock = b[4]

                byte_offset = 5

                # Loop for each item
                for i in range(0, ItemsInBlock):
                    # Read item id
                    ItemID = b[byte_offset]
                    byte_offset += 1

                    # Read item data size
                    ItemDataSize = int.from_bytes(b[byte_offset:byte_offset+2], byteorder='little')
                    byte_offset += 2

                    # Get this data
                    ItemData = b[byte_offset:byte_offset+ItemDataSize]
                    byte_offset += ItemDataSize

                    # If this is an object item
                    if(ItemID == 0):
                        # Process it
                        name = str(ItemData[0:24], 'utf-8').strip('\x00')
                        data = unpack( 'd d d d d d', b[32:80] )

                        #data_string = str(data)

                        # Rotate to NED
                        x = (data[0] / 1000) # DEPRECATED: ned x = vicon y
                        y = (data[1] / 1000) # DEPRECATED: ned y = vicon x
                        z = (data[2] / 1000) # DEPRECATED: ned z = - vicon z

                        yaw = (data[5]) # DEPRECATED: yaw inverted
                        roll = data[3]
                        pitch = data[4] # DEPRECATED: pitch inverted

                        if name in self.tracked_object:
                            x_vel = (x - self.tracked_object[name][0])/((datetime.now()-self.tracked_object[name][12]).total_seconds())
                            y_vel = (y - self.tracked_object[name][1])/((datetime.now()-self.tracked_object[name][12]).total_seconds())
                            z_vel = (z - self.tracked_object[name][2])/((datetime.now()-self.tracked_object[name][12]).total_seconds())

                            roll_rate = (((roll - self.tracked_object[name][3]+math.pi)%(2*math.pi))-math.pi)/((datetime.now()-self.tracked_object[name][12]).total_seconds())
                            pitch_rate = (((pitch - self.tracked_object[name][4]+math.pi)%(2*math.pi))-math.pi)/((datetime.now()-self.tracked_object[name][12]).total_seconds())
                            yaw_rate = (((yaw - self.tracked_object[name][5]+math.pi)%(2*math.pi))-math.pi)/((datetime.now()-self.tracked_object[name][12]).total_seconds())
                        else:
                            x_vel = 0
                            y_vel = 0
                            z_vel = 0
                            roll_rate = 0
                            pitch_rate = 0
                            yaw_rate = 0

                        self.have_recv_packet = True

                        # Store in public variable
                        self.tracked_object[name] = [x, y, z, roll, pitch, yaw, x_vel, y_vel, z_vel, roll_rate, pitch_rate, yaw_rate, datetime.now()]

                        #print("p{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(ned_x, ned_y, ned_z, ned_roll, ned_pitch, ned_yaw))
        except Exception as e:
            pass
        finally:
            self.sock.close()

    def getPos(self, name):
        try:
            return self.tracked_object[name][0:6]
        except Exception as e:
            return None

    def getVel(self, name):
        try:
            return self.tracked_object[name][6:12]
        except Exception as e:
            return None


if __name__ == "__main__":

    print(1)
