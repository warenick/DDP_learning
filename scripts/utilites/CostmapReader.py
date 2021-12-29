import rospy 
import math
from nav_msgs.msg import OccupancyGrid

class CostmapReader:
    def __init__(self, topic="/move_base/global_costmap/costmap") -> None:
        rospy.init_node("ddp_map_reader")
        self.sub = rospy.Subscriber(topic, OccupancyGrid, self.callback)
        self.is_init = False
        self.costmap = OccupancyGrid()

    def callback(self, msg):
    # [nav_msgs/OccupancyGrid]
    # look description in console:
    # rosmsg info OccupancyGrid
        self.costmap = msg
        self.is_init = True

    def check_index_in_map(self, index_xy):
        if index_xy[0]<0 or index_xy[1]<0:
            return False 
        size = self.get_size()
        if index_xy[0]>size[0] or index_xy[1]>size[1]:
            return False 
        return True

    # #########################################
    # some API naming taken from https://github.com/stonier/cost_map/blob/devel/cost_map_core/doc/grid_map_conventions.pdf 
    def get_index(self, xy):
        # cell index at xy coords
        offset = self.get_position()
        resolution = self.get_resolution()
        index_x = int((xy[0]-offset[0])//resolution)
        index_y = int((xy[1]-offset[1])//resolution)
        return [index_x, index_y]


    def at(self, index_xy):
        # value of cell at index_xy[x,y]
        if not self.check_index_in_map(index_xy):
            rospy.WARN("requested index is out of map")
            return False
        size = self.get_size()
        index = index_xy[0]*size[1]+index_xy[1]
        return self.costmap.data[index]
        

    def at_position(self, xy):
        index_xy = self.get_index(xy)
        return self.at(index_xy)

    def get_position(self, xy=None):
        # position of costmap in m.,
        if xy is None:
            x = self.costmap.info.origin.position.x
            y = self.costmap.info.origin.position.y
            yaw = self.__q2yaw(self.costmap.info.origin.orientation)
            return [x, y, yaw]
        # or position of cell in m.
        # TODO: realize that
        x = 0
        y = 0
        return [x,y]

    def get_size(self):
        # length of map in cells for (x,y)
        size_x = self.costmap.info.width
        size_y = self.costmap.info.height
        return [size_x, size_y]

    def get_length(self):
        # length of map in m. for (x,y)
        resolution = self.get_resolution()
        length_x = self.costmap.info.width*resolution
        length_y = self.costmap.info.height*resolution
        return [length_x, length_y]

    def get_resolution(self):
        # resolution/grid size in m./cell
        return self.costmap.info.resolution
    # #########################################
    
    def __q2yaw(self, q):
        # conver quaternion msg to yaw angle
        t3 = +2.0 * (q.w * q.z + q.x * q.y)
        t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        #TODO:check convertion
        return math.degrees(math.atan2(t3, t4)) 

if __name__=="__main__":
    cr = CostmapReader()
    while not cr.is_init:
        rospy.sleep(.1)
    p1 = [0,0] # center
    p2 = [1.0,10.0]
    p3 = [-1.0,-1.0]
    index1 = [0,0] # border
    index2 = [cr.get_size()[0]-1, cr.get_size()[1]-1] # border
    index3 = [cr.get_size()[0]/2, cr.get_size()[1]/2] # center
    # from pprint import pprint
    print("cr.get_position() ", cr.get_position())    
    print("cr.get_length() ", cr.get_length())    
    print("cr.get_resolution() ", cr.get_resolution())    
    print("cr.get_size() ", cr.get_size())    
    print("cr.at(index1) ", index1, cr.at(index1))
    print("cr.at(index2) ", index2, cr.at(index2))
    print("cr.at_position(p1) ", p1, cr.at_position(p1))
    print("cr.at_position(p2) ", p2, cr.at_position(p2))
    print("cr.at_position(p3) ", p3, cr.at_position(p3))

    