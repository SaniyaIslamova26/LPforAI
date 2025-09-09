import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import torch
from rl_transformer import TransformerNetwork

class SimpleGazeboEnv(gym.Env):
    def __init__(self):
        super(SimpleGazeboEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # Вперёд, назад, влево, вправо
        self.observation_space = spaces.Box(
        low=np.array([-10, -10], dtype=np.float32),
        high=np.array([10, 10], dtype=np.float32),
        dtype=np.float32
        )

        # Нейросеть для обработки объектов
        self.transformer = TransformerNetwork(input_dim=2, embed_dim=16, num_heads=2, num_layers=1)

        # ROS 2 Инициализация
        rclpy.init()
        self.node = rclpy.create_node("simple_gazebo_env")

        # ROS 2 Топики
        self.cmd_pub = self.node.create_publisher(Twist, "cmd_vel", 10)
        self.marker_pub = self.node.create_publisher(Marker, "visualization_marker", 10)
        self.lidar_sub = self.node.create_subscription(LaserScan, "/scan", self.process_lidar_data, 10)

        # Состояние среды
        self.robot_position = np.array([0.0, 0.0])
        self.objects_positions = np.array([])
        self.episode_reward = 0
        self.episode_steps = 0

    def process_lidar_data(self, msg):
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)
        x_coords = ranges * np.cos(angles)
        y_coords = ranges * np.sin(angles)
        self.objects_positions = np.vstack((x_coords, y_coords)).T
        
    def step(self, action):
    # Управление роботом
        twist = Twist()
        if action == 0:
           twist.linear.x = 0.5  # Вперёд
        elif action == 1:
           twist.linear.x = -0.5  # Назад
        elif action == 2:
           twist.angular.z = 0.5  # Влево
        elif action == 3:
           twist.angular.z = -0.5  # Вправо
        self.cmd_pub.publish(twist)

    # Если данные лидара отсутствуют
        if self.objects_positions.size == 0:
            return self.robot_position, 0, False, False, {}

    # Вычисляем эмбеддинги объектов
        input_data = torch.tensor(self.objects_positions,dtype=torch.float32).unsqueeze(0)
        object_embeddings = self.transformer(input_data).detach().numpy()
        observation = self.robot_position + object_embeddings.mean(axis=1)

    # Награда
        reward = -np.linalg.norm(self.robot_position)
        self.episode_reward += reward
        self.episode_steps += 1

    # Завершение эпизода
        terminated = np.linalg.norm(self.robot_position) < 0.1  # Логическое завершение задачи
        truncated = self.episode_steps >= 1000  # Ограничение на количество шагов
        info = {"episode_steps": self.episode_steps, "episode_reward": self.episode_reward}

        if terminated or truncated:
           self.episode_reward = 0
           self.episode_steps = 0

    # Визуализация в Gazebo
        self.visualize_in_gazebo(self.robot_position)

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
    # Установка начального seed для воспроизводимости
        super().reset(seed=seed)  # Если используется версия Gymnasium >= 1.0.0
        self.robot_position = np.array([0.0, 0.0])  # Сброс позиции робота
        info = {"robot_position": self.robot_position}  # Информация о состоянии
        return self.robot_position, info

    def visualize_in_gazebo(self, position):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "robot_traj"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = 0.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        self.marker_pub.publish(marker)
