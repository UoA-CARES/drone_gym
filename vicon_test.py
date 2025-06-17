import time
from threading import Thread
from vicon_connection_class import ViconInterface as vi


OBJECT_NAME = "AtlasCrazyflie" 


if __name__ == "__main__":
    print("Starting test")
    vicon = vi()

    vicon_thread = Thread(target=vicon.main_loop)
    vicon_thread.start()
    print(1)

    counter = 0
    while counter < 5:
        position = vicon.getPos(OBJECT_NAME)
        time.sleep(1)
        print(position)
        counter += 1 

    vicon.run_interface = False
    vicon_thread.join()
    print("Exiting")