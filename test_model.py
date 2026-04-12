import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_torcs import TorcsEnv

def main():
    raw_env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    env = DummyVecEnv([lambda: raw_env])

    model_filename = "torcs_sac_850000_steps"  
    model_path = os.path.join(".", "models", model_filename)
    normalizer_path = os.path.join(".", "models", "vec_normalize_850000_steps.pkl")
    
    if os.path.exists(normalizer_path):
        print(f"Wczytuję statystyki normalizatora z: {normalizer_path}")
        env = VecNormalize.load(normalizer_path, env)

        env.training = False      

        env.norm_reward = False   
    else:
        print("Brak pliku normalizatora")

    print(f"Próbuję wczytać model z: {model_path}")
    model = SAC.load(model_path)
    
    obs = env.reset()
    print("\nRozpoczynam jazdę testową! (Kierownica | Pedały | Krokowa Nagroda | Całkowity Wynik)")
    
    total_reward = 0.0

    while True:
        action, _states = model.predict(obs, deterministic=True)
        
        steer = action[0][0]
        pedal = action[0][1]
        
        if pedal >= -0.2:
            gaz_procent = (pedal + 0.2) / 1.2 * 100
            pedal_str = f"GAZ: {gaz_procent:3.0f}%    "
        else:
            hamulec_procent = (-pedal - 0.2) / 0.8 * 100
            pedal_str = f"HAM: {hamulec_procent:3.0f}%"
            
        obs, reward, done, info = env.step(action)
        
        current_reward = reward[0]
        total_reward += current_reward
        
        print(f"Kierownica: {steer:6.2f}  |  {pedal_str}  |  Punkt: {current_reward:6.2f}  |  Suma: {total_reward:8.2f}")
        
        if done[0]:
            print(f"\n--- PRZERWANA SESJA ---")
            print(f"Całkowity wynik epizodu: {total_reward:.2f}")
            print("Restart toru...\n")
            
            total_reward = 0.0
            obs = env.reset()

if __name__ == "__main__":
    main()