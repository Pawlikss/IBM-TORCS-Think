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
    env = Monitor(env, "./logs/")
    env = DummyVecEnv([lambda: env])

    MODEL_PATH = "./models/torcs_sac_latest.zip"
    REPLAY_BUFFER_PATH = "./models/torcs_sac_latest_replay.pkl"

    if os.path.exists(MODEL_PATH):
        print("Wznawianie treningu z pliku: ", MODEL_PATH)
        model = SAC.load(MODEL_PATH, env=env, tensorboard_log="./tensorboard_logs/")
        
        if os.path.exists(REPLAY_BUFFER_PATH):
            print("Wczytywanie Replay Buffer")
            model.load_replay_buffer(REPLAY_BUFFER_PATH)
        else:
            print("Brak pliku Replay Buffer")
    else:
        model = SAC(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log="./tensorboard_logs/",
            learning_rate=0.0003,    
            buffer_size=100000,      
            batch_size=256,          
            ent_coef="auto",
            learning_starts=2000,
        )

    # Zapis stanu sieci i BUFORA co 10 000 kroków
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="torcs_sac",
        save_replay_buffer=True
    )
    
    # reset_num_timesteps=False łączy ze sobą wykresy na TensorBoardzie po wczytaniu
    model.learn(total_timesteps=1000000, callback=checkpoint_callback, reset_num_timesteps=False)

    # Bezpieczny zapis na koniec
    model.save(MODEL_PATH)
    model.save_replay_buffer(REPLAY_BUFFER_PATH)
    print("Trening zakończony pomyślnie")

if __name__ == "__main__":
    main()