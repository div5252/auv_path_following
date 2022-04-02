#!/usr/bin/env python
import rospy
import sys
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt

class Plot():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def callback(self, data):
        self.x.append(data.pose.pose.position.x)
        self.y.append(data.pose.pose.position.y)
        # print("x: {}, y: {}".format(data.pose.pose.position.x, data.pose.pose.position.y))

    def show(self):
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.plot(self.x, self.y, lw=2.0)
        plt.show()

if __name__ == '__main__':
    rospy.init_node('create_plot')
    print("Started plotting")

    start = rospy.Time.now()

    plot = Plot([], [])
    rospy.Subscriber("/rexrov/pose_gt", Odometry, plot.callback)
    
    if rospy.Time.now() - start < rospy.Duration(60):
        rospy.spin()

    plot.show()
    