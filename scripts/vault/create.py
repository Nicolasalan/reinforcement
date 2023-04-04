#!/usr/bin/env python3

import cv2
import numpy as np
import trimesh
import rospkg

import rospy
from nav_msgs.msg import OccupancyGrid

class CreateEnvironment(object):

     def __init__(self, map_topic, thresholds=1, height=1.0):
          rospy.Subscriber(map_topic, OccupancyGrid, self.map_callback)
          rospack = rospkg.RosPack()
                    
          self.path_to_package = rospack.get_path('vault')
          self.threshold = thresholds
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
          export_dir = self.path_to_package + "/models/map"

          with open(export_dir + "/map.dae", 'w') as f:
               f.write(trimesh.exchange.dae.export_collada(mesh).decode()) 

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

if __name__ == "__main__":
     rospy.init_node("create_env")

     rospy.loginfo("Creating environment...")

     create = CreateEnvironment("map", thresholds=1, height=1.0)
     rospy.spin()
