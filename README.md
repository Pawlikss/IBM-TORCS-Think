# TORCS Autonomous Driving Agent (SAC)

Projekt wykorzystujący algorytm SAC do nauki autonomicznej jazdy w symulatorze wyścigowym TORCS.

## Użyte technologie

- **Python 3**
- **Stable-Baselines3** (algorytm Soft Actor-Critic - SAC)
- **Gym-TORCS** (środowisko zintegrowane ze standardem OpenAI Gym)
- **TensorBoard** (monitorowanie i wizualizacja procesu uczenia)

### 1. Trening modelu

Aby rozpocząć uczenie agenta od zera, upewnij się, że masz uruchomiony serwer gry TORCS, a następnie odpal skrypt:
python train.py

Proces uczenia korzysta z TensorBoard. Aby śledzić postępy, użyj:
tensorboard --logdir tensorboard_logs

### 2. Testowanie wytrenowanego bota

Aby zobaczyć bota w akcji bez trybu eksploracji (deterministycznie), uruchom:
python test_model.py

## Architektura nagród (Reward System)

Funkcja nagrody (Reward Function) została rygorystycznie zaprojektowana, aby wymusić płynną jazdę:

- **Premia** za prędkość wzdłuż osi toru.
- **Kary** za gwałtowne ruchy kierownicą (zapobieganie saturacji Tanh).
- **Kary** za wyjazd poza tor i jazdę pod prąd.
