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

### Instalacja i konfiguracja środowiska

1. Zainstaluj wymagane biblioteki Python:
   ```bash
   pip install -r requirements.txt
   ```
2. Upewnij się, że katalog `torcs` znajduje się w katalogu głównym repozytorium.
   To właśnie stamtąd ładowany jest plik `wtorcs.exe`.
3. Na Windows uruchomienie kodu wykonuje polecenia `taskkill` i startuje grę przez `wtorcs.exe`.
   Jeśli gra nie wystartuje poprawnie, zamknij wszystkie istniejące procesy TORCS i spróbuj ponownie.
4. Jeżeli chcesz uruchomić testowy model, użyj:
   ```bash
   python test_model.py
   ```

### Struktura projektu

- `train.py` — trening agenta SAC
- `test_model.py` — testowanie wytrenowanego modelu
- `gym_torcs.py` — wrapper środowiska TORCS dla Gymnasium
- `requirements.txt` — wymagane biblioteki Python
- `torcs/` — lokalna kopia gry TORCS używana przez środowisko

### Użyte pliki TORCS i stare skrypty

- Obecna ścieżka gry jest skonfigurowana do lokalnego katalogu `torcs/`.
- Starsze, nieużywane skrypty zostały przeniesione do folderu `unused/`.

### 2. Testowanie wytrenowanego bota

Aby zobaczyć bota w akcji bez trybu eksploracji (deterministycznie), uruchom:
python test_model.py

## Architektura nagród (Reward System)

Funkcja nagrody (Reward Function) została rygorystycznie zaprojektowana, aby wymusić płynną jazdę:

- **Premia** za prędkość wzdłuż osi toru.
- **Kary** za gwałtowne ruchy kierownicą (zapobieganie saturacji Tanh).
- **Kary** za wyjazd poza tor i jazdę pod prąd.
