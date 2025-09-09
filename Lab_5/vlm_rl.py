import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from stable_baselines3 import PPO
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from cv_bridge import CvBridge
import gym
from gym import spaces


class GazeboEnv(gym.Env):
    def __init__(self):
        super(GazeboEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 4 действия: вперёд, назад, влево, вправо
        self.observation_space = spaces.Box(low=0, high=255, shape=(512,), dtype=np.float32)  # Пространство признаков CLIP
        self.state = np.zeros(512, dtype=np.float32)  # Инициализация состояния
        self.step_counter = 0  # Счётчик шагов для текущего эпизода
        self.max_steps = 100  # Максимальное количество шагов в эпизоде

    def step(self, action):
        self.step_counter += 1

        # Награда за действия: нейтральная награда (0) за вперед/вправо, отрицательная (-1) за назад/влево
        reward = self._calculate_reward(action)

        # Проверка условия завершения эпизода
        done = self._check_done_condition()

        info = {}
        print(f"[Environment] Step: {self.step_counter}, Action: {action}, Reward: {reward}, Done: {done}")
        return self.state, reward, done, info

    def reset(self):
        self.state = np.zeros(512, dtype=np.float32)
        self.step_counter = 0  # Сбрасываем счётчик шагов
        return self.state

    def _calculate_reward(self, action):
        # Награда 0 за вперёд/вправо, -1 за назад/влево
        return 0 if action in [0, 2] else -1

    def _check_done_condition(self):
        # Завершаем эпизод, если достигнут лимит шагов
        return self.step_counter >= self.max_steps


class VLMRLNode(Node):
    def __init__(self):
        super().__init__("vlm_rl_node")
        self.bridge = CvBridge()  # Преобразование ROS-изображений в OpenCV
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.goal_description = "Find a table"
        self.state = None
        self.action = None
        self.step_counter = 0  # Счётчик шагов для текущего эпизода
        self.total_steps = 0  # Суммарное количество шагов
        self.max_total_steps = 200  # Лимит шагов для завершения обучения

        # Подписка и публикация ROS
        self.subscription = self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        # RL и среда
        self.env = GazeboEnv()
        self.rl_model = PPO("MlpPolicy", self.env, verbose=1)

    def image_callback(self, msg):
        # Получаем изображение из ROS-сообщения и преобразуем его в формат OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        image_tensor = self.processor(images=cv_image, return_tensors="pt")["pixel_values"].to(self.device)

        with torch.no_grad():
            # Получаем признаки изображения с помощью модели CLIP
            features = self.model.get_image_features(image_tensor)
            self.state = features.cpu().numpy()

        # Действие робота
        self.act_and_publish()

    def act_and_publish(self):
        if self.state is not None:
            # Прогнозируем следующее действие
            self.action, _ = self.rl_model.predict(self.state)
            self.step_counter += 1
            self.total_steps += 1

            # Публикуем действие в виде команды роботу
            twist = Twist()
            if self.action == 0:  # Вперёд
                twist.linear.x = 0.5
            elif self.action == 1:  # Назад
                twist.linear.x = -0.5
            elif self.action == 2:  # Влево
                twist.angular.z = 0.5
            elif self.action == 3:  # Вправо
                twist.angular.z = -0.5
            self.publisher.publish(twist)
            # Обновляем состояние среды и получаем награду
            _, reward, done, _ = self.env.step(self.action)

            # Логируем шаги
            self.get_logger().info(f"Step: {self.step_counter}, Total Steps: {self.total_steps}, Action: {self.action}, Reward: {reward}")

            # Завершение эпизода, если необходимо
            if done:
                self.get_logger().info("Episode finished. Resetting environment...")
                self.env.reset()
                self.step_counter = 0  # Сбрасываем счётчик эпизода

            # Завершаем обучение, если достигнут лимит шагов
            if self.total_steps >= self.max_total_steps:
                self.get_logger().info("Max steps reached. Shutting down...")
                rclpy.shutdown()

    def train(self, total_timesteps=100000):
        # Обучение RL модели
        self.rl_model.learn(total_timesteps=total_timesteps)


def main():
    rclpy.init()
    node = VLMRLNode()
    try:
        # Запускаем обучение
        node.train(total_timesteps=10000)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
