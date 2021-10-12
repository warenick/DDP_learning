from geometry_msgs.msg import Point


def arr2point(arr):
    return Point(x=arr[0],y=arr[1],z=0)