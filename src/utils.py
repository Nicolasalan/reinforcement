import math

# funcao para pegar a distancia do alvo 
def getGoalDistace(self):
     # calcular a distancia do alvo em relacao ao robo
     goal_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
     # distancia do alvo no passado se torna a distancia do alvo atual
     self.past_distance = goal_distance
     # retornar a distancia do alvo
     return goal_distance

# funcao para pegar a posicao do robo por meio do topico '/odom' 
def getOdometry(self, odom):
     # pegar a posicao do robo
     self.position = odom.pose.pose.position
     # pegar a orientacao do robo
     orientation = odom.pose.pose.orientation
     # converter a orientacao do robo para euler
     q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
     # converter a orientacao do robo para euler
     yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

     # se o yaw for maior que 0, entao o yaw sera o yaw atual
     if yaw >= 0:
          yaw = yaw
     # se nao for maior que 0, entao o yaw sera o yaw atual mais 360
     else:
          yaw = yaw + 360
     # relacao de distancia entre o robo e o alvo em relacao ao eixo x
     rel_dis_x = round(self.goal_position.position.x - self.position.x, 1)
     # relacao de distancia entre o robo e o alvo em relacao ao eixo y
     rel_dis_y = round(self.goal_position.position.y - self.position.y, 1)

     # Calcule o ângulo entre o robô e o alvo
     if rel_dis_x > 0 and rel_dis_y > 0:
          theta = math.atan(rel_dis_y / rel_dis_x)
     elif rel_dis_x > 0 and rel_dis_y < 0:
          theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
     elif rel_dis_x < 0 and rel_dis_y < 0:
          theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
     elif rel_dis_x < 0 and rel_dis_y > 0:
          theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
     elif rel_dis_x == 0 and rel_dis_y > 0:
          theta = 1 / 2 * math.pi
     elif rel_dis_x == 0 and rel_dis_y < 0:
          theta = 3 / 2 * math.pi
     elif rel_dis_y == 0 and rel_dis_x > 0:
          theta = 0
     else:
          theta = math.pi
     rel_theta = round(math.degrees(theta), 2)

     diff_angle = abs(rel_theta - yaw)

     if diff_angle <= 180:
          diff_angle = round(diff_angle, 2)
     else:
          diff_angle = round(360 - diff_angle, 2)

     self.rel_theta = rel_theta # relacao de angulo entre o robo e o alvo
     self.yaw = yaw # orientacao do robo
     self.diff_angle = diff_angle # diferenca de angulo entre o robo e o alvo