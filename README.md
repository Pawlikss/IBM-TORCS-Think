# TORCS Autonomous Driving Agent (SAC)

Projekt wykorzystujący algorytm uczenia ze wzmocnieniem (Reinforcement Learning) do nauki autonomicznej jazdy w symulatorze wyścigowym TORCS. Agent uczy się sterowania kierownicą oraz operowania gazem i hamulcem w celu pokonywania okrążeń bez kolizji.

## Użyte technologie

- **Python 3**
- **Stable-Baselines3** (algorytm Soft Actor-Critic - SAC)
- **Gym-TORCS** (środowisko zintegrowane ze standardem OpenAI Gym)
- **TensorBoard** (monitorowanie i wizualizacja procesu uczenia)

### 1. Trening modelu

Aby rozpocząć uczenie agenta od zera, upewnij się, że masz uruchomiony serwer gry TORCS, a następnie odpal skrypt:
\`\`\`bash
python train.py
\`\`\`
Proces uczenia korzysta z TensorBoard. Aby śledzić postępy, użyj:
\`\`\`bash
tensorboard --logdir tensorboard_logs
\`\`\`

### 2. Testowanie wytrenowanego bota

Aby zobaczyć bota w akcji bez trybu eksploracji (deterministycznie), uruchom:
\`\`\`bash
python test_model.py
\`\`\`

## Architektura nagród (Reward System)

Funkcja nagrody (Reward Function) została rygorystycznie zaprojektowana, aby wymusić płynną jazdę:

- **Premia** za prędkość wzdłuż osi toru.
- **Kary** za gwałtowne ruchy kierownicą (zapobieganie saturacji Tanh).
- **Kary** za wyjazd poza tor i jazdę pod prąd.
