#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
PLOT_PATH = os.path.join(dir_path, '../../rl_controller/log/episode%d.png')

class Plot():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def callback(self, data):
        self.x.append(data.pose.pose.position.x)
        self.y.append(data.pose.pose.position.y)

    def write(self):
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.plot(self.x, self.y, lw=2.0)

        plt.savefig(PLOT_PATH%eps_no)
        plt.close()

if __name__ == '__main__':
    rospy.init_node('create_plot')
    eps_no = rospy.get_param('~eps_no')

    print("Started plotting")

    start = rospy.Time.now()

    plot = Plot([], [])
    rospy.Subscriber("/rexrov/pose_gt", Odometry, plot.callback)
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print('plotting')

    plot.write()    