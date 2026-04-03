import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_torcs import TorcsEnv

def main():
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./tensorboard_logs", exist_ok=True)

    env = TorcsEnv(vision=False, throttle=True, gear_change=False)

    # Monitor śledzi statystyki, a DummyVecEnv pozwala algorytmom SB3 działać na środowisku
    env = Monitor(env, "./logs/")
    env = DummyVecEnv([lambda: env])

    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./tensorboard_logs/",
        learning_rate=0.0003,    # bezpieczny krok uczenia
        buffer_size=100000,      # Rozmiar pamięci doświadczeń
        batch_size=256,          # Ile próbek bierze do jednej aktualizacji wag sieci
        ent_coef=0.1,         #regulacja współczynnika entropii
        learning_starts=2000,
    )

    # Kontynuacja treningu z istniejącego modelu
    #model = SAC.load("./models/torcs_sac_290000_steps.zip", env=env, tensorboard_log="./tensorboard_logs/")

    # Wstrzykujemy nowe, bardzo małe parametry do gotowego mózgu (Fine-Tuning)
    custom_params = {
        "learning_rate": 0.0001,
        "ent_coef": 0.01
    }

    model = SAC.load(
        "./models/torcs_sac_290000_steps.zip", 
        env=env, 
        tensorboard_log="./tensorboard_logs/", 
        custom_objects=custom_params
    )

    # Zapisuje stan sieci co 10 000 kroków
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="torcs_sac"
    )

    print("Start treningu")
    # total_timesteps to ilość klatek decyzyjnych
    model.learn(total_timesteps=500000, callback=checkpoint_callback, reset_num_timesteps=False)

    model.save("./models/torcs_sac_final")
    print("Trening zakończony pomyślnie")

if __name__ == "__main__":
    main()