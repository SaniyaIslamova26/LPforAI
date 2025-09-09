import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import Twist
import time


class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')
        self.get_logger().info("Planner node works")

        # Создаём публишер для отправки команд скорости
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)

        # Определяем последовательность точек
        self.path = ['A','D',' B', 'C']  # Путь от A к D
        self.current_step = 0  # Текущий шаг

        # Таймер для выполнения плана
        self.timer = self.create_timer(1.0, self.execute_plan)

    def execute_plan(self):
        """Выполняет движение по заданному пути."""
        if self.current_step < len(self.path) - 1:
            current_location = self.path[self.current_step]
            next_location = self.path[self.current_step + 1]

            self.get_logger().info(f"Moving from {current_location} to {next_location}")

            # Публикуем команду движения вперёд
            self.move_forward()

            # Пауза для завершения движения
            time.sleep(5)

            # Останавливаем робота
            self.stop_robot()

            self.current_step += 1  # Переходим к следующему шагу
        else:
            self.get_logger().info("Reached destination: D")
            self.timer.cancel()  # Останавливаем таймер

    def move_forward(self):
        """Отправляет команду для движения вперёд."""
        msg = Twist()
        msg.linear.x = 0.5  # Скорость движения вперёд
        msg.angular.z = 0.0  # Движение по прямой
        self.publisher_.publish(msg)
        self.get_logger().info("Moving forward...")

    def stop_robot(self):
        """Отправляет команду для остановки робота."""
        msg = Twist()
        msg.linear.x = 0.0  # Остановка
        msg.angular.z = 0.0
        self.publisher_.publish(msg)
        self.get_logger().info("Robot stopped.")

    def shutdown(self):
        """Завершает работу узла."""
        self.get_logger().info("Shutting down")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()


if __name__ == '__main__':
    main()
