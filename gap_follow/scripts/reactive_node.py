#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import array
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

class ReactiveFollowGap(Node):
    """ 
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('reactive_node')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        self.lidar_sub = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        
        self.last_processed_ranges = np.zeros((1080,),dtype = np.float32)

    def preprocess_lidar(self, ranges: array) -> array:
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        max_value = 1.4
        proc_ranges = np.array(ranges, dtype=np.float32)
        ranges = (proc_ranges+self.last_processed_ranges)/2
        self.last_processed_ranges = proc_ranges

        np.clip(ranges, 0, max_value, out=ranges)
        return ranges
    
    def find_closest_point(self, ranges: array) -> int:
        """ Return the index of the closest point
        """
        return min(range(len(ranges)), key=lambda x: ranges[x])
    
    def process_bubble(self, ranges: array, point_idx: int) -> array:
        """ Set all the points arround the point_idx to 0
        """
        left, right = max(0, point_idx - 100), min(len(ranges)-1, point_idx + 99)
        ranges[left: right+1] = [0] * (right - left + 1)
        return ranges

    def find_max_gap(self, ranges:array) -> (int, int, array):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        split_idx = np.where(ranges == 0.0)[0]
        sranges = np.split(ranges,split_idx)
        len_sranges = np.array([len(x) for x in sranges])
        max_idx = np.argmax(len_sranges)
        if max_idx == 0:
            start_i = 0
            end_i = len_sranges[0]-1
        else:
            start_i = np.sum(len_sranges[:max_idx])
            end_i = start_i+len_sranges[max_idx]-1
        max_length_ranges = sranges[max_idx]
        return start_i, end_i, max_length_ranges
    
    def find_best_point(self, start_i: int, end_i: int, ranges: array) -> int:
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        idx_list = np.where(ranges == np.max(ranges))[0]
        best_idx = start_i + idx_list[round(len(idx_list)/2)]
        return best_idx
    
    def get_speed(self, steering_angle: float) -> float:
        """ Estimate speed based on steering angle
        """
        if abs(steering_angle) >= np.radians(0) and abs(steering_angle) < np.radians(10):
            return 3.0
        elif abs(steering_angle) >= np.radians(10) and abs(steering_angle) < np.radians(20):
            return 1.0
        else:
            return 0.5

    def lidar_callback(self, data: LaserScan):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        ranges = data.ranges
        proc_ranges = self.preprocess_lidar(ranges)
        
        closest_point = self.find_closest_point(proc_ranges)

        bubble_processed_ranges = self.process_bubble(proc_ranges, closest_point)

        max_gap_start, max_gap_end, max_gap_ranges = self.find_max_gap(bubble_processed_ranges)

        max_point = self.find_best_point(max_gap_start, max_gap_end, max_gap_ranges)

        steering_angle = (max_point * data.angle_increment) + data.angle_min

        drive_message = AckermannDriveStamped()
        drive_message.drive.steering_angle = steering_angle
        drive_message.drive.speed = self.get_speed(steering_angle)
        self.drive_pub.publish(drive_message)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()