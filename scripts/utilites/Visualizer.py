import torch
import rospy
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, Pose, Vector3, Quaternion
from std_msgs.msg import ColorRGBA
import torch


MARKER_CFG = {
    "robot": {
        "scale": Vector3(0.3, 0.6, 1.8),
        "color": ColorRGBA(0.9, 0.9, 0.9, 1),  # white
        "prediction_color": ColorRGBA(0.9, 0.9, 0.9, 0.2)},
    "agent": {
        "scale": Vector3(0.3, 0.6, 1.8),
        "color": ColorRGBA(0, 0.9, 0, 1),  # green
        "prediction_color": ColorRGBA(0, 0.9, 0, 0.2)},
    "robot_path": {
        "color": ColorRGBA(0, 1, 0, 1),  # green
        "scale": Vector3(0.15, 0.0, 0.0)},  # x - with of line
    "agent_path": {
        "color": ColorRGBA(0, 0.5, 0.5, 1),  # green/blue
        "scale": Vector3(0.1, 0.0, 0.0)},  # x - with of line
    "robot_goal": {
        "scale": Vector3(0.2, 0.35, 0.2),
        "color": ColorRGBA(0, 1, 0, 1)},  # green
    "agent_goal": {
        "scale": Vector3(0.2, 0.35, 0.2),
        "color": ColorRGBA(1, 1, 1, 1)},  # white
    "arrows": {
        "scale": Vector3(0.02, 0.1, 1),
        "colors": [ColorRGBA(0, 1, 0, 1),  # - green
                   ColorRGBA(0, 0, 1, 1),  # - blue]
                   ColorRGBA(1, 0, 0, 1), ]  # - red]
    }}



class Visualizer:
    
    def __init__(self, frame="map", topic="ddp/vis") -> None:
        node_name = rospy.get_name()
        if 'unnamed' in node_name:
            rospy.init_node("ddp")
        self.frame = frame
        self.pub = rospy.Publisher(
            topic, MarkerArray, queue_size=1)

    def pub_agent_state(self, agents_arr):
        msg = MarkerArray()
        for num, agent in enumerate(agents_arr):
            # INITIAL STATE
            # agent.type need to be in MARKER_CFG
            # agent_marker = Marker(
            #     id=num,
            #     type=Marker.SPHERE,
            #     action=Marker.ADD,
            #     scale=MARKER_CFG[agent.type]["scale"],
            #     color=MARKER_CFG[agent.type]["prediction_color"],
            #     pose=self.__arr2pose(agent.state_initial),
            #     ns = "initial_state"
            # )
            # agent_marker.header.frame_id = self.frame
            # agent_marker.pose.position.z = MARKER_CFG[agent.type]["scale"].z/2.
            # msg.markers.append(agent_marker)
            # STATE
            # agent.type need to be in MARKER_CFG
            agent_marker = Marker(
                id=num,
                type=Marker.SPHERE,
                action=Marker.ADD,
                scale=MARKER_CFG[agent.type]["scale"],
                color=MARKER_CFG[agent.type]["color"],
                pose=self.__arr2pose(agent.state),
                ns = "state "+ agent.name
            )
            agent_marker.header.frame_id = self.frame
            agent_marker.pose.position.z = MARKER_CFG[agent.type]["scale"].z/2.
            msg.markers.append(agent_marker)
            # GOAL
            agent_goal_marker = Marker(
                id=num,
                type=Marker.CUBE,
                action=Marker.ADD,
                scale=MARKER_CFG[agent.type+"_goal"]["scale"],
                color=MARKER_CFG[agent.type+"_goal"]["color"],
                pose=self.__arr2pose(agent.goal),
                ns = "goal "+ agent.name
            )
            agent_goal_marker.header.frame_id = self.frame
            agent_goal_marker.pose.position.z = agent_goal_marker.scale.z/2.
            msg.markers.append(agent_goal_marker)
    
            # PREDICTION
            local_id = 0
            for state in agent.prediction["state"]:
                prediction_marker = Marker(
                    id=local_id,
                    type=Marker.SPHERE,
                    action=Marker.ADD,
                    scale=MARKER_CFG[agent.type]["scale"],
                    color=MARKER_CFG[agent.type]["prediction_color"],
                    pose=self.__arr2pose(state),
                    ns = "predicted states " + agent.name
                )
                prediction_marker.header.frame_id = self.frame
                prediction_marker.pose.position.z = MARKER_CFG[agent.type]["scale"].z/2.
                msg.markers.append(prediction_marker)
                local_id += 1

            # PREDICTION AS PATH
            path_marker = Marker(
                id=local_id,
                type=Marker.LINE_STRIP,
                action=Marker.ADD,
                scale=MARKER_CFG[agent.type+"_path"]["scale"],
                color=MARKER_CFG[agent.type+"_path"]["color"],
                pose=self.__arr2pose([0,0,0]),
                points=self.__arr2posearr(agent.prediction["state"]),
                ns = "predicted path " + agent.name
            )
            path_marker.header.frame_id = self.frame
            path_marker.pose.position.z = MARKER_CFG[agent.type]["scale"].z/2.
            msg.markers.append(path_marker)

        self.pub.publish(msg)
        # rospy.sleep(0.001) # just for publish

    def __arr2posearr(self,arr):
        posearr = []
        for p in arr:
            point = Point(x=p[0], y=p[1], z=0) 
            posearr.append(point)
        return posearr

    def __arr2pose(self, arr):
        p = Pose()
        p.position.x = arr[0]
        p.position.y = arr[1]
        p.orientation = self.__yaw2q(0)
        if len(arr)>2:
            p.orientation = self.__yaw2q(arr[2])
        return p

    def __yaw2q(self,yaw):
        # conver yaw angle to quaternion msg 
        try:
            return Quaternion(x=0, y=0, z=np.sin(yaw/2), w=np.cos(yaw/2))
        except:
            if "Tensor" in yaw.type():
                return Quaternion(x=0, y=0, z=torch.sin(yaw/2), w=torch.cos(yaw/2))
            return Quaternion(x=0, y=0, z=0,w=1)


    def __q2yaw(self,q):
        # conver quaternion msg to yaw angle
        t3 = +2.0 * (q.w * q.z + q.x * q.y)
        t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        # return math.degrees(math.atan2(t3, t4))
        return np.atan2(t3, t4)

    
