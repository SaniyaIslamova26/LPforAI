# 1. Установка ROS2 Humble

## Установка VirtualBox, Ubuntu

Запуск (установка) [VirtualBox](https://www.virtualbox.org/wiki/Downloads). Oracle VB лучше устанавливать (обновить) до версии 7. Иначе будет некорректная работа Kernel (требоваться кол-во процессоров >1), проблема с colcon build.

Установка [Linux Ubuntu 22.04](https://releases.ubuntu.com/22.04/) на виртуальную машину 
(ROS2 Humble работает только с этой версией, ни в коем случае не обновлять до новейшей версии Ubuntu).

![image](https://github.com/user-attachments/assets/bd2ffd2d-0ffd-4f07-bc3b-c3de2e7061e2)

Проверка версии операционной системы

![image](https://github.com/user-attachments/assets/352d7d64-3f4b-44b6-bb1e-3c80dad3d864)

## Установка ROS2

Документация [ROS2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)

Проверка и настройка локали системы на UTF-8:
```
locale  # проверка UTF-8
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
locale  # проверка настроек
```

Добавляем пакеты для управления репозиториями программного обеспечения в системе Linux:
```
sudo apt install software-properties-common 
sudo add-apt-repository universe  
```

Добавление GPG-ключа (Ключ GPG (GNU Privacy Guard) — это пара криптографических ключей, используемых для цифровых подписей и шифрования в системе GnuPG. Это своего рода цифровой отпечаток, который подтверждает подлинность и целостность данных):
```
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
```

Добавляем репозиторий в свой список источников:
```
$("deb [arch=echodpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

Обновляем кеш репозитория и систему:
```
sudo apt update 
sudo apt upgrade
```

Установка полного рабочего стола:
```
sudo apt install ros-humble-desktop-full
```

Установка базовых компонентов:
```
sudo apt install ros-humble-ros-base
sudo apt install ros-dev-tools
```

Настройка окружения:
```
source /opt/ros/humble/setup.bash
```
Проверка установки и версии ROS2:
![image](https://github.com/user-attachments/assets/01133a30-874c-4da0-90e0-dd5fdc22b421)


Попробуем несколько узлов для проверки работоспособности ROS:
```
ros2 run demo_nodes_cpp talker
ros2 run demo_nodes_py listener
```
![image](https://github.com/user-attachments/assets/effaccf3-2df3-486b-8f7d-f3c9a8ddf0bc)

# 2.Установка Pyperplan

Документация [Pyperplan](https://pypi.org/project/pyperplan/)

Устанавливаем Pyperplan с помощью pip:
```
pip install pyperplan
```
Проверка работоспособности Pyperplan с помощью принимаемых аргументов: файл домена PDDL и файл задачи PDDL:
```
pyperplan benchmarks/tpp/domain.pddl benchmarks/tpp/task01.pddl
```
![image](https://github.com/user-attachments/assets/39bd6b76-d98f-41dc-886b-5616f64c0de1)

# 3.Установка Turtlebot3

Документация по установке [Turtlebot3](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/)

Устанавливаеv пакет для интеграции ROS с Gazebo:
 ```
sudo apt install ros-humble-gazebo-*
```
Устанавливаем пакет Cartographer и пакет, интегрирующий библиотеку Cartographer с платформой ROS:
```
sudo apt install ros-humble-cartographer
sudo apt install ros-humble-cartographer-ros
```
Устанавливаем основной Navigation2 метапакет. И пакет, который предоставляет файлы запуска и конфигурации для простого запуска и работы базовой навигационной системы:
```
sudo apt install ros-humble-navigation2
sudo apt install ros-humble-nav2-bringup
```

Устанавливаем пакеты Turtlebot3:
```
mkdir -p ~/turtlebot3_ws/src
cd ~/turtlebot3_ws/src/
git clone -b humble https://github.com/ROBOTIS-GIT/DynamixelSDK.git  # Пакет предоставляет драйверы и инструменты для управления двигателями Dynamixel в среде ROS
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git # Пакет содержит типы сообщений, которые ROS использует для взаимодействия с роботом TurtleBot3
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3.git # Пакет содержит драйверы, файлы конфигурации и запуска, которые упрощают взаимодействие с роботом TurtleBot3
sudo apt install python3-colcon-common-extensions #
cd ~/turtlebot3_ws
colcon build --symlink-install
echo 'source ~/turtlebot3_ws/install/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

Установка среды ROS для ПК:
```
echo 'export ROS_DOMAIN_ID=30 #TURTLEBOT3' >> ~/.bashrc
echo 'source /usr/share/gazebo/setup.sh' >> ~/.bashrc
source ~/.bashrc
```

Запуск пустого мира в Gazebo:
```
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

![image](https://github.com/user-attachments/assets/e5d73246-b7b4-4a43-bce2-dadd19c0c590)

Устанавливаем недостающие пакеты:
```
sudo apt install ros-humble-ros2-control 
sudo apt install ros-humble-ros2-controllers 
sudo apt install ros-humble-gripper-controllers 
sudo apt install ros-humble-moveit
```
Запуск Turtlebot3 с манипуляторной рукой в Gazebo:
```
ros2 launch turtlebot3_manipulation_bringup gazebo.launch.py
```

![image](https://github.com/user-attachments/assets/9b1aec9a-04b3-4ace-99e1-9e1b16117956)

### Запуск пустого мира в Gazebo и запуск Turtlebot3 с манипуляторной рукой  - [ВИДЕО]( https://disk.yandex.ru/i/M_TdEujrj4rKtQ)

## Лайфхак для работы
Позволяет немедленно получить доступ к обновлённому файлу .bashrc и сделать переменные среды рабочей области доступными в текущем терминале, 

если добавить в файл ./bashrc все сорсированные данные:

![image](https://github.com/user-attachments/assets/c0c58adb-0e1c-4616-8766-105a92442c49)

# 4. Создание ROS-узла

В рабочей области ros2_ws создаём новый пакет pyperplan_node:
```
ros2 pkg create --build-type ament_python pyperplan_node
```
Структура пакета должна быть следующая:
```
pyperplan_node/
├── pyperplan_node/
│   ├── __init__.py
│   └── pyperplan_node.py
├── setup.py
├── setup.cfg
└── package.xml
```
![image](https://github.com/user-attachments/assets/e22c89d0-88b1-4c65-a480-06ecd533d27a)

Создаём Python-скрип для ROS2-узла:
```Python

import rclpy
from rclpy.node import Node


class PyperplanNode(Node):
    def __init__(self):
        super().__init__('pyperplan_node')
        self.get_logger().info('Pyperplan node started')

        # Симуляция выполнения планировщика
        self.run_dummy_planner()

    def run_dummy_planner(self):
        # Логика выполнения
        self.get_logger().info('Running the dummy planner...')
        self.get_logger().info('Planner execution complete. No PDDL task was provided.')


def main(args=None):
    rclpy.init(args=args)
    node = PyperplanNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```
![image](https://github.com/user-attachments/assets/1e1b01ca-5afc-4285-9c7d-85c733316875)

Редактируем setup.py для установки зависимостей и конфигурации пакета:

![photo_5384209343894907694_x](https://github.com/user-attachments/assets/297bc4d9-17f4-4d42-bfe8-73c49e64b08c)

Компилируем пакет:
```
cd ~/ros2_ws
colcon build --packages-select pyperplan_node
source install/setup.bash
```
![image](https://github.com/user-attachments/assets/47344e46-6054-4370-ac0b-3d0845841d70)

Запускаем узел:
```
ros2 run pyperplan_node pyperplan_node
```

![image](https://github.com/user-attachments/assets/4274bac9-08ee-4b37-a0c6-4e062253d295)

Наш ROS2-узел успешно запущен и работает. Лог показывает, что узел был инициализирован, симуляция робота планировщика выполнена, и узел завершил выполнение без ошибок.









