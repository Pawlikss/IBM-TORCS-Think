import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_torcs import TorcsEnv

def main():
    # Czyste środowisko (bez VecNormalize, zgodnie z naszym nowym planem)
    raw_env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    env = DummyVecEnv([lambda: raw_env])

    # TUTAJ WPISZ NAZWĘ PLIKU, KTÓRY CHCESZ PRZETESTOWAĆ
    # np. z nowego folderu fine-tuningu
    model_filename = "torcs_sac_454997.zip"  
    model_path = os.path.join(".", "models", model_filename)

    print(f"Próbuję wczytać model z: {model_path}")
    if os.path.exists(model_path):
        model = SAC.load(model_path)
    else:
        raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")
    
    obs = env.reset()
    print("\nRozpoczynam jazdę testową! (Kierownica | Pedały | Krokowa Nagroda | Całkowity Wynik)")
    
    total_reward = 0.0

    while True:
        action, _states = model.predict(obs, deterministic=True)
        
        steer = action[0][0]
        pedal = action[0][1]
        
        # IDEALNIE ZSYNCHRONIZOWANA MATEMATYKA PEDAŁÓW (Próg na -0.2)
        if pedal >= -0.2:
            # Zakres od -0.2 do 1.0 = rozpiętość 1.2
            gaz_procent = ((pedal + 0.2) / 1.2) * 100.0
            pedal_str = f"GAZ: {gaz_procent:3.0f}%    "
        else:
            # Zakres od -1.0 do -0.2 = rozpiętość 0.8
            hamulec_procent = ((-pedal - 0.2) / 0.8) * 100.0
            pedal_str = f"HAM: {hamulec_procent:3.0f}%"
            
        obs, reward, done, info = env.step(action)
        
        current_reward = reward[0]
        total_reward += current_reward
        
        # Wyświetlanie telemetrii w czasie rzeczywistym
        print(f"Kierownica: {steer:6.2f}  |  {pedal_str}  |  Punkt: {current_reward:6.2f}  |  Suma: {total_reward:8.2f}")
        
        if done[0]:
            print(f"\n--- PRZERWANA SESJA ---")
            
            # Pobranie informacji, DLACZEGO zginął (ściana, niska prędkość, obrócenie)
            powod = info[0].get('terminal_reason', 'nieznany')
            print(f"Powód końca: {powod}")
            print(f"Całkowity wynik epizodu: {total_reward:.2f}")
            print("Restart toru...\n")
            
            total_reward = 0.0
            obs = env.reset()

if __name__ == "__main__":
    main()