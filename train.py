import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_torcs import TorcsEnv
import torch
import numpy as np

class LiveInfoCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        info = infos[0]
        for key in ["speedX", "trackPos", "reward_total"]:
            if key in info:
                self.logger.record(f"telemetria/{key}", float(info[key]))

        return True

class TerminationStatsCallback(BaseCallback):
    def __init__(self, print_freq=10, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.counts = {
            "off_track": 0,
            "backward": 0,
            "low_progress": 0,
            "server_shutdown": 0,
            "other": 0,
        }
        self.episodes = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is None or infos is None:
            return True

        for done, info in zip(dones, infos):
            if done:
                reason = info.get("terminal_reason", "other")
                if reason not in self.counts:
                    reason = "other"

                self.counts[reason] += 1
                self.episodes += 1
                total = max(self.episodes, 1)

                for key in self.counts:
                    self.logger.record(f"powod_konca/{key}_rate", self.counts[key] / total)

                if self.episodes % self.print_freq == 0:
                    statystyki = " | ".join(f"{k}: {v}" for k, v in self.counts.items() if v > 0)
                    print(f"\n[STATYSTYKI] Epizod {self.episodes} | {statystyki}")
        return True   

def main():
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./tensorboard_logs", exist_ok=True)

    env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    env = Monitor(env, "./logs/")
    env = DummyVecEnv([lambda: env])

    MODEL_PATH = "./models/torcs_sac_300000_steps.zip"
    REPLAY_BUFFER_PATH = "./models/torcs_sac_replay_buffer_xx_steps.pkl"

    if os.path.exists(MODEL_PATH):
        print("Wznawianie treningu z pliku: ", MODEL_PATH)
        #Linie zakomentowane służą do fine tuningu
        # nowe_ustawienia = {
        #     "learning_rate": 0.0001,
        #     "buffer_size": 1000000
        # }
        
        model = SAC.load(
            MODEL_PATH, 
            env=env, 
            tensorboard_log="./tensorboard_logs/",
            # custom_objects=nowe_ustawienia 
        )

        # with torch.no_grad():
        #     model.log_ent_coef.fill_(np.log(0.05))
        
        if os.path.exists(REPLAY_BUFFER_PATH):
            model.load_replay_buffer(REPLAY_BUFFER_PATH)
        else:
            print("Brak pliku Replay Buffer, start z pustą pamięcią!")
    else:
        model = SAC(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log="./tensorboard_logs/",
            learning_rate=0.0003,    # do fine tuning można obniżyć do 0.0001
            buffer_size=1000000,
            batch_size=256,          
            ent_coef="auto",
            learning_starts=10000,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="torcs_sac",
        save_replay_buffer=True
    )
    
    term_stats_callback = TerminationStatsCallback(print_freq=10)
    live_info_callback = LiveInfoCallback()

    callback_list = CallbackList([
        checkpoint_callback, 
        term_stats_callback, 
        live_info_callback
    ])
    
    try:
        model.learn(total_timesteps=3000000, callback=callback_list, reset_num_timesteps=True)
    except KeyboardInterrupt:
        print("\nPrzerwano trening ręcznie. Zapisywanie postępów...")
    finally:
        model.save(MODEL_PATH)
        if hasattr(model, "save_replay_buffer"):
            model.save_replay_buffer(REPLAY_BUFFER_PATH)
        print("Najnowszy stan mózgu i bufor zostały zapisane pomyślnie")

if __name__ == "__main__":
    main()