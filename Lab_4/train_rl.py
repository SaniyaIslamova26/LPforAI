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

if __name__ == "__main__":
    main()
