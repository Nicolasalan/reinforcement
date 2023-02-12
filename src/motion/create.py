#!/usr/bin/env python3

import cv2
import numpy as np
import trimesh
import rospkg
import aspose.threed as a3d
import os

import rospy
from nav_msgs.msg import OccupancyGrid
from utils import Extension

class CreateEnvironment(object):

     def __init__(self, map_topic, threshold=1, height=1.0):
          rospy.Subscriber(map_topic, OccupancyGrid, self.map_callback)
          rospack = rospkg.RosPack()
                    
          self.path_to_package = rospack.get_path('motion')
          self.threshold = threshold
          self.height = height

     def map_callback(self, msg):
          map_dims = (msg.info.height, msg.info.width)
          map_array = np.array(msg.data).reshape(map_dims)

          map_array[map_array < 0] = 0
          contours = self.get_occupied_regions(map_array)
          meshes = [self.contour_to_mesh(c, msg.info) for c in contours]

          corners = list(np.vstack(contours))
          corners = [c[0] for c in corners]
          mesh = trimesh.util.concatenate(meshes)

          # Export DAE
          export_dir = self.path_to_package + "/world"

          with open(export_dir + "/map.dae", 'w') as f:
               f.write(trimesh.exchange.dae.export_collada(mesh).decode())

          # convert DAE to STL
          scene = a3d.Scene.from_file(export_dir + "/map.dae")
          scene.save(export_dir + "/map.stl")

          # remove DAE
          os.remove(export_dir + "/map.dae")      

          # create World file
          self.create_world_file(export_dir)

          # Shut down the program
          rospy.signal_shutdown("Exported map successfully")

     def get_occupied_regions(self, map_array):
          map_array = map_array.astype(np.uint8)
          _, thresh_map = cv2.threshold(
               map_array, self.threshold, 100, cv2.THRESH_BINARY)
          contours, hierarchy = cv2.findContours(
               thresh_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

          hierarchy = hierarchy[0]
          corner_idxs = [i for i in range(
               len(contours)) if hierarchy[i][3] == -1]
          return [contours[i] for i in corner_idxs]

     def contour_to_mesh(self, contour, metadata):
          height = np.array([0, 0, self.height])
          meshes = []
          for point in contour:
               x, y = point[0]
               vertices = []
               new_vertices = [
                    self.coords_to_loc((x, y), metadata),
                    self.coords_to_loc((x, y+1), metadata),
                    self.coords_to_loc((x+1, y), metadata),
                    self.coords_to_loc((x+1, y+1), metadata)]
               vertices.extend(new_vertices)
               vertices.extend([v + height for v in new_vertices])
               faces = [[0, 2, 4],
                         [4, 2, 6],
                         [1, 2, 0],
                         [3, 2, 1],
                         [5, 0, 4],
                         [1, 0, 5],
                         [3, 7, 2],
                         [7, 6, 2],
                         [7, 4, 6],
                         [5, 4, 7],
                         [1, 5, 3],
                         [7, 3, 5]]
               mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
               if not mesh.is_volume:
                    mesh.fix_normals()
               meshes.append(mesh)
          mesh = trimesh.util.concatenate(meshes)
          mesh.remove_duplicate_faces()

          return mesh

     def coords_to_loc(self, coords, metadata):
          x, y = coords
          loc_x = x * metadata.resolution + metadata.origin.position.x
          loc_y = y * metadata.resolution + metadata.origin.position.y
          return np.array([loc_x, loc_y, 0.0])

     def create_world_file(self, path):
          world_file = f"""<?xml version="1.0"?>
          <sdf version="1.7">
               <world name="my_stl_world">
                    <include>
                         <uri>model://sun</uri>
                    </include>
                    <include>
                         <uri>model://ground_plane</uri>
                    </include>
                    <model name="my_stl_model">
                         <pose>0 0 0 0 0 0</pose>
                         <link name="base_link">
                              <collision name="collision">
                                   <geometry>
                                        <mesh>
                                             <uri>file://{path}</uri>
                                             <scale>1 1 1</scale>
                                        </mesh>
                                   </geometry>
                              </collision>
                              <visual name="visual">
                                   <geometry>
                                        <mesh>
                                             <uri>file://{path}</uri>
                                             <scale>1 1 1</scale>
                                        </mesh>
                                   </geometry>
                              </visual>
                         </link>
                    </model>
               </world>
          </sdf>
          """
          save_path = self.path_to_package + '/world' + '/environment.world'
          with open(save_path, "w") as f:
               f.write(world_file)      

if __name__ == "__main__":
     rospy.init_node("create_env")

     CONFIG_PATH = rospy.get_param('config_path')  

     useful = Extension(CONFIG_PATH)
     param = useful.load_config("config.yaml")
     map_topic = param["topic_map"]
     threshold = param["threshold"] 
     height = param["height"] 

     create = CreateEnvironment(map_topic, threshold=threshold, height=height)
     rospy.spin()
