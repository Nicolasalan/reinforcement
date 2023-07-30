# Angle of the robot in relation to the objective

O agente tem como uma das entradas para rede neuralo angulo do robo em relacao objetivo em coordenadas polares.
Para isso utiliza o calculo entre dois vetores, o vetor do robo e o vetor do objetivo.

```python
# Calculate the relative angle between the robot's heading and heading toward the goal
angle_diff = np.arctan2(self.goal_y - self.odom_y, self.goal_x - self.odom_x) - angle

# Normalize the angle to be in the range [-pi, pi]
if angle_diff > np.pi:
     angle_diff -= 2 * np.pi
elif angle_diff < -np.pi:
     angle_diff += 2 * np.pi
```

Um vetor é um medida de direcao e magnitude. Com dois vetores conseguimos caclular com geometria analitica o angulo entre o robo e o objetivo