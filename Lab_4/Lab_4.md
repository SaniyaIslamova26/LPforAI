# Лабораторная работа №4

## 1. Запустить Gymnasium  

Для выполнения этой лабораторной работы нам потребуется установка 2 библиотек:
```
pip install gymnasium==0.28.1
pip install stable-baselines3[extra]
```
- Документация для библиотеки [gymnasium](https://gymnasium.farama.org/)
- Документация для библиотеки [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/)

Библиотека gymnasium (ранее известная как gym) — это библиотека Python для разработки и сравнения алгоритмов обучения с подкреплением (Reinforcement Learning — RL).

Библиотека stable-baselines3 (сокращённо SB3) — это набор надёжных реализаций алгоритмов обучения с подкреплением (Reinforcement Learning — RL) на основе PyTorch. Она предназначена для того, чтобы сделать RL более доступным, удобным и надёжным для исследователей и разработчиков.

Что такое [extra] зависимости?

Часть [extra] в pip install stable-baselines3[extra] относится к функции «дополнительные возможности» pip. Она указывает pip на установку дополнительных зависимостей, которые не требуются для основных функций stable-baselines3, но обеспечивают дополнительные возможности, удобство и совместимость. Группа extra в stable-baselines3 обычно включает зависимости для:

- Визуализация: часто используется для рендеринга сред и визуализации поведения агентов во время обучения. Сюда могут входить такие библиотеки, как opencv-python для обработки изображений и pygame для рендеринга.
- Интеграция с TensorBoard: инструменты для использования TensorBoard (инструмента визуализации для машинного обучения) для отслеживания показателей обучения.
- Дополнительные зависимости среды: некоторым средам RL могут потребоваться дополнительные зависимости, которые установит эта команда.

В чем разница между stable-baselines3 и gymnasium:

- gymnasium — это библиотека сред, которая предоставляет стандартизированный интерфейс для взаимодействия агента с окружающей средой.
- stable-baselines3 — это библиотека алгоритмов, которая предоставляет готовые реализации алгоритмов RL, которые можно использовать для обучения агентов в этих средах.
- stable-baselines3 часто используется совместно с gymnasium при обучении RL-агента в той или иной среде.

# 2. На ТБ3-4 Запустить планирование маршрута любым из алгоритмов RL (Лучше DDPG\PPO)

#### Алгоритм PPO:

Proximal Policy Optimization (PPO) — это алгоритм обучения с подкреплением, который использует методы оптимизации политики, чтобы обучать агентов принимать действия в среде. 

В PPO используется актёр-критиковый подход, где актёр (политика) обновляется для улучшения выбора действий, а критик (оценщик) оценивает, насколько хорошо действия агента соответствуют ожидаемым наградам. 

#### Цель PPO — максимизировать ожидаемую сумму наград через оптимизацию параметров политики. Это делается с использованием функции цели, называемой «функцией потерь актёра» (Actor Loss). 

#### Преимущества PPO:

- Стабильность обучения. PPO известен своей способностью достичь стабильного обучения даже в сложных средах. 
- Относительная простота. В сравнении с некоторыми другими алгоритмами, PPO имеет относительно простую реализацию и настройку гиперпараметров. 
-Хорошая производительность. PPO часто демонстрирует хорошую производительность на различных задачах обучения с подкреплением. 

#### Недостатки PPO:

- Высокое количество данных. PPO может потребовать большое количество обучающих данных, что может быть проблемой в реальных задачах. 

Для выполнения этого пункта мы уже скачали необходимые библиотеки.

Теперь нужно настроить рабочее пространство. Так как у меня уже есть ros2_ws, то:
```
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python gazebo_rl_project
```

Так будет выглядеть созданный пакет:
![image](https://github.com/user-attachments/assets/5c3521c2-94ab-4fea-bae9-8a08c822ccd3)

Далее переходим в созданный пакет и создаём файлы, в которые помещаем Python-скрипты:

Создаём файл rl_env.py:
```Python
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
    def init(self):
        super(SimpleGazeboEnv, self).init()
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
```

Создаём файл rl_transformer.py:
```Python
import torch
import torch.nn as nn

class TransformerNetwork(nn.Module):
    def init(self, input_dim, embed_dim, num_heads, num_layers):
        super(TransformerNetwork, self).init()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc_out(x.mean(dim=1))
        return x
```

Создаём файл train_rl.py:
```Python
from stable_baselines3 import PPO
from rl_env import SimpleGazeboEnv

def main():
    # Создаём среду
    env = SimpleGazeboEnv()
    # Создаём модель
    model = PPO("MlpPolicy", env, verbose=1)
    # Обучаем модель
    model.learn(total_timesteps=10000, log_interval=10)
    # Сохраняем модель
    model.save("simple_ppo_model.zip")
    print("Обучение завершено!")

if name == "main":
    main()
```

![image](https://github.com/user-attachments/assets/f05c75fa-80e8-4993-bcae-ab04ca48c8ee)


Добавляём строчку в файл setup.py для корректной работы:
```
entry_points={
'console_scripts': [
'train_rl = gazebo_rl_project.train_rl:main',],}
```

![image](https://github.com/user-attachments/assets/924149a9-3514-4585-bdf6-44d642f03e41)

Также добавляем файл requirements.txt с библиотеками и устанавливаем зависимость:
```
pip install -r ~/ros2_ws/src/gazebo_rl_project/requirements.txt
```

Теперь через терминал переходим в рабочую область, собираем наш проект:
```
cd ~/ros2_ws
colcon build --packages-select gazebo_rl_project
source ~/ros2_ws/src/install/setup.bash
```

Открываем второй терминал и запускаем мир Gazebo с роботом:
```
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

Открываем третий терминал параллельно и запускаем обучение:
```
python3 ~/ros2_ws/src/gazebo_rl_project/gazebo_rl_project/train_rl.py
```


У меня выходят некоторые предупреждения, но они не являются ошибками и обучение происходит.

![image](https://github.com/user-attachments/assets/2d71de8d-2cbf-41de-8328-2323f7d6a8bd)


## Запуск планирования маршрута алгоритмом RL - PPO - [ВИДЕО](https://disk.yandex.ru/i/9HVfIk4VCew9Vg)
