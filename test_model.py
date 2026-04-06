import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_torcs import TorcsEnv

def main():
    raw_env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    env = DummyVecEnv([lambda: raw_env])

    model_filename = "torcs_sac_2100000_steps.zip"  
    model_path = os.path.join(".", "models", model_filename)
    
    print(f"Próbuję wczytać model z: {model_path}")
    model = SAC.load(model_path)
    
    obs = env.reset()
    print("Rozpoczynam jazdę testową z telemetrią")
    
    while True:
        action, _states = model.predict(obs, deterministic=True)
        
        steer = action[0][0]
        pedal = action[0][1]
        
        if pedal > 0:
            pedal_str = f"GAZ: {pedal * 100:3.0f}%    "
        else:
            pedal_str = f"HAMULEC: {abs(pedal) * 100:3.0f}%"
            
        print(f"Kierownica: {steer:6.2f}  |  {pedal_str}")
        
        obs, reward, done, info = env.step(action)
        
        if done[0]:
            print("Przerwana sesja")
            obs = env.reset()

if __name__ == "__main__":
    main()