# Лабораторная работа №3

## 1. Использовать 1 из алгоритмов планирования для передвижения робототехнического агента из точки А в точку В

В работе будет использован Метод планирования на основе дискретных точек пути (Waypoints Planning).

Метод планирования на основе дискретных точек пути (Waypoints Planning) — это подход, при котором робот или транспортное средство движется между заранее заданными дискретными точками (точками пути, или waypoint), следуя по прямым отрезкам.

### Основные характеристики метода

1. Дискретизация пути:
   - Путь представляется в виде последовательности точек, каждая из которых задаёт промежуточное состояние робота.
   - Робот движется от одной точки пути к следующей по прямой линии.

2. Движение по заранее определённой траектории:
   - Точки пути задаются заранее и являются фиксированными.
   - Движение осуществляется строго по этим точкам, без оптимизации траектории между ними.

3. Отсутствие динамического реагирования:
   - Метод не включает возможности обхода препятствий в реальном времени (если не используется дополнительное планирование на этапе выполнения).

4. Поддержка простых сред:
   - Подходит для предсказуемых, статических карт, где расположение препятствий известно заранее.


### Преимущества метода

1. Простота реализации:
   - Метод легко реализуется с использованием базовых алгоритмов и команд.
   - Не требует сложных математических вычислений.

2. Низкие вычислительные затраты:
   - Не требует больших вычислительных ресурсов, так как движение осуществляется по заранее заданным прямым отрезкам.

3. Предсказуемость:
   - Путь строго задан, что позволяет легко проверить корректность выполнения.

4. Совместимость с разными платформами:
   - Используется для наземных, воздушных и водных роботов, где движение по прямым линиям достаточно для выполнения задач.
  

Поскольку у меня уже есть директория ros2_ws, то её я создавать не буду. Перейдём в эту директорию и создадим новый пакет:
```
cd ~/ros2_ws
ros2 pkg create --build-type ament_python planner_node
```
Пакет должен содержать такие папки и файлы:

![Снимок экрана от 2024-12-12 23-57-42](https://github.com/user-attachments/assets/efd89952-cf8d-4313-809b-361da7ec968e)


Создаём файл planner_node.py в папке planner_node (я создала вручную).
```Python
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import Twist
import time


class PlannerNode(Node):
    def init(self):
        super().init('planner_node')
        self.get_logger().info("Planner node works")

        # Создаём публишер для отправки команд скорости
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)

        # Определяем последовательность точек
        self.path = ['A', 'B', 'C', 'D']  # Путь от A к D
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


if __name__ == 'main':
    main()
```

![Снимок экрана от 2024-12-12 23-57-54](https://github.com/user-attachments/assets/10dd1563-0a8e-4c60-9aed-82d8f9cb52ed)

Далее для корректной работы добавляем строчку в файл setup.py:
```
entry_points={
'console_scripts': [
'planner_node = planner_node.planner_node:main',],}
```

![Снимок экрана от 2024-12-12 23-58-13](https://github.com/user-attachments/assets/2797ff64-3b71-41ec-9f91-ba2b2f7fa116)

После всего открываем терминал и открываем мир в Gazebo:
```
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

Также параллельно отрываем второй терминал и запускаем узел для передвижения робота:
```
cd ~/ros2_ws
colcon build --packages-select planner_node
source install/setup.bash
ros2 run planner_node planner_node
```

![Снимок экрана от 2024-12-12 23-54-28](https://github.com/user-attachments/assets/53762559-ebd7-4102-8a64-c5ab8b65d99c)

При успешном выполнении узла, мы увидим перемещение робота из точки A в точку D в Gazebo:

![Снимок экрана от 2024-12-12 23-51-20](https://github.com/user-attachments/assets/b71f2509-6c30-4b56-a121-15e9793f2a8c)

![Снимок экрана от 2024-12-12 23-54-19](https://github.com/user-attachments/assets/03d98a8b-0328-4e4c-91fd-07fd3e41ecce)


## Перемещение робота из точки A в точку D в Gazebo - [ВИДЕО](https://disk.yandex.ru/i/wiw3wmVx50ojCg)


# 2. Изучить отличие сплайнового метода построения пути от классических алгоритмов

Сравнение методов: используемого в работе и сплайнового

| Критерий                 | Планирование на основе точек пути     | Сплайновое планирование            |
|--------------------------|---------------------------------------|------------------------------------|
| Тип траектории           | Прямые линии между точками            | Гладкие кривые                     |
| Плавность движения       | Низкая, возможны резкие остановки     | Высокая, движение плавное          |
| Выбор траектории         | Жёстко заданная                       | Оптимизируется для плавности       |
| Вычислительная сложность | Низкая                                | Высокая                            |
| Подходит для             | Простых задач в статических средах    | Динамических или сложных сред      |
| Энергозатраты            | Высокие из-за остановок и ускорений   | Минимальные за счёт плавности      |

