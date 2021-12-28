import rospy
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped

class DynamicTfPub:
    def __init__(self, frame_from="map", frame_to="base_link") -> None:
        # self.origin_frame = origin_frame
        # self.map_frame = map_frame
        # self.tf_broadcaster = tf2_ros.transform_broadcaster.TransformBroadcaster()
        self.tf_msg = TransformStamped()
        self.tf_msg.header.frame_id = frame_from
        self.tf_msg.child_frame_id = frame_to
        self.tf_broadcaster = tf2_ros.transform_broadcaster.TransformBroadcaster()

    def pub_tf(self, coords_from, coords_to):
        # [x,y,z]
        x = coords_from[0] - coords_to[0]
        y = coords_from[1] - coords_to[1]
        yaw = coords_from[2] - coords_to[2]

        self.tf_msg.header.stamp = rospy.Time.now()
        self.tf_msg.transform.translation.x = x
        self.tf_msg.transform.translation.y = y
        self.tf_msg.transform.translation.z = 0.

        quat = tf.transformations.quaternion_from_euler(0, 0, yaw)
        self.tf_msg.transform.rotation.x = quat[0]
        self.tf_msg.transform.rotation.y = quat[1]
        self.tf_msg.transform.rotation.z = quat[2]
        self.tf_msg.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(self.tf_msg)

if __name__=="__main__":
    rospy.init_node("my_tf", anonymous=True)
    map_coords = [0,0,0]
    base_link_coords = [0,0,0]
    mytf = DynamicTfPub(frame_from="map",frame_to="base_link")
    while not rospy.is_shutdown():
        # base_link_coords[0]+=1
        mytf.pub_tf(coords_from=map_coords, coords_to=base_link_coords)
        rospy.sleep(.05)
    exit()
