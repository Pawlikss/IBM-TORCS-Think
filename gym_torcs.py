import gymnasium as gym
from gymnasium import spaces
import numpy as np
import snakeoil3_gym as snakeoil3
import copy
import collections as col
import os
import time
import pyautogui
import pathlib

class TorcsEnv(gym.Env):
    terminal_judge_start = 500  
    termination_limit_progress = 5  
    default_speed = 50

    initial_reset = True

    def __init__(self, vision=False, throttle=False, gear_change=False):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.initial_run = True

        cwd = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        torcs_dir = os.path.join(script_dir, "torcs")
        
        if not os.path.exists(torcs_dir):
            raise FileNotFoundError(f"Nie znaleziono folderu TORCS w: {torcs_dir}")
            
        os.chdir(torcs_dir)

        os.system('taskkill /f /im wtorcs.exe >nul 2>&1')
        time.sleep(1.0)
        
        if self.vision is True:
            os.system('start "" wtorcs.exe -nofuel -nodamage -nolaptime -vision')
        else:
            os.system('start "" wtorcs.exe -nofuel -nodamage -nolaptime')

        time.sleep(3.0) 
        for key in ['enter', 'enter', 'up', 'up', 'enter', 'enter']:
            pyautogui.press(key)
            time.sleep(0.2)
        time.sleep(5.0)
        
        os.chdir(cwd)

        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        if vision is False:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(70 + 64 * 64 * 3,), dtype=np.float32
            )

    def step(self, u):
        client = getattr(self, "client", None)
        if client is None or getattr(client, "so", None) is None:
            self._force_relaunch_next_reset = True
            return self.get_obs(), 0.0, True, False, {"terminal_reason": "server_shutdown"}

        client = self.client
        this_action = self.agent_to_torcs(u)
        action_torcs = client.R.d

        # Steering
        action_torcs["steer"] = this_action["steer"]  # in [-1, 1]

        if self.throttle is True:
            action_torcs["accel"] = this_action["accel"]
            action_torcs["brake"] = this_action["brake"]

        # Simple Automatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            
            if client.S.d["speedX"] < target_speed:
                client.R.d["accel"] += 0.05
            else:
                client.R.d["accel"] -= 0.05

            if client.S.d["speedX"] < 10:
                client.R.d["accel"] += 0.1 

            if client.R.d["accel"] > 0.4:
                client.R.d["accel"] = 0.4
            if client.R.d["accel"] < 0.0:
                client.R.d["accel"] = 0.0
                
            action_torcs["brake"] = 0.0

        # Automatic Gear Change
        if self.gear_change is True:
            action_torcs["gear"] = this_action["gear"]
        else:
            action_torcs["gear"] = 1
            if client.S.d["speedX"] > 50: action_torcs["gear"] = 2
            if client.S.d["speedX"] > 80: action_torcs["gear"] = 3
            if client.S.d["speedX"] > 110: action_torcs["gear"] = 4
            if client.S.d["speedX"] > 140: action_torcs["gear"] = 5
            if client.S.d["speedX"] > 170: action_torcs["gear"] = 6

        obs_pre = copy.deepcopy(client.S.d)

        client.respond_to_server()
        client.get_servers_input()

        if getattr(client, "so", None) is None:
            self._force_relaunch_next_reset = True
            return self.get_obs(), 0.0, True, False, {"terminal_reason": "server_shutdown"}

        obs = client.S.d
        self.observation = self.make_observaton(obs)

        if not hasattr(self, "last_steer"):
            self.last_steer = 0.0

        angle = float(obs["angle"])
        speed_x = float(obs["speedX"])
        track_pos = float(obs["trackPos"])
        track = np.asarray(obs["track"], dtype=np.float32)
        current_steer = float(this_action["steer"])
        cos_a = np.cos(angle)

        #nagroda za prędkość do przodu
        forward = speed_x * cos_a
        raw_reward = forward * 0.1

        steer_change = abs(current_steer - self.last_steer)
        #kara za szarpanie
        raw_reward -= (steer_change ** 2) * 5.0

        self.last_steer = current_steer

        #Kara za zbliżanie się do zakrętu bez redukcji prędkości
        if track.size > 10:
            front_distance = max(track[8], track[9], track[10]) / 200.0
        else:
            front_distance = 1.0
            
        front_distance = np.clip(front_distance, 0.0, 1.0)
        current_brake = float(this_action.get("brake", 0.0))

        threshold = 0.3

        if front_distance < threshold:
            curve_risk = (threshold - front_distance) / threshold
            
            speed_penalty = 25.0 * curve_risk * ((max(speed_x, 0.0) / 100.0) ** 2)

            if current_brake > 0:
                shield = speed_penalty * (current_brake * 0.8)
                speed_penalty -= shield

            raw_reward -= speed_penalty

        #kara za zjazd ze srodka toru
        deadband = 0.3 
        if abs(track_pos) > deadband:
            pos_penalty = (abs(track_pos) - deadband) * 0.5 
            raw_reward -= pos_penalty

        #normalizacja nagrody
        reward = raw_reward / 100.0

        # Termination judgement
        episode_terminate = False
        terminal_reason = None

        if abs(track_pos) > 1.0 or track.min() < 0:
            reward -= 5.0
            episode_terminate = True
            terminal_reason = "off_track"
            client.R.d["meta"] = True

        if not episode_terminate and cos_a < 0:
            reward -= 5.0
            episode_terminate = True
            terminal_reason = "backward"
            client.R.d["meta"] = True

        if not episode_terminate and self.terminal_judge_start < self.time_step:
            if forward < self.termination_limit_progress:
                episode_terminate = True
                terminal_reason = "low_progress"
                client.R.d["meta"] = True

        if client.R.d["meta"] is True: 
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        info = {
            "terminal_reason": terminal_reason,
            "reward_total": float(reward),
            "episode_terminate": bool(episode_terminate),
            "speedX": float(speed_x),
            "trackPos": float(track_pos)
        }
        
        terminated = bool(client.R.d["meta"])
        truncated = False
        return self.get_obs(), float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_step = 0
        
        self.last_steer = 0.0

        if getattr(self, "_force_relaunch_next_reset", False):
            self.reset_torcs()
            self._force_relaunch_next_reset = False
            self.initial_reset = True

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

        self.client = snakeoil3.Client(p=3001, vision=self.vision)
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  

        obs = client.S.d  
        self.observation = self.make_observaton(obs)

        self.last_u = None
        self.initial_reset = False
        
        return self.get_obs(), {}

    def reset_torcs(self):
        cwd = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        torcs_dir = os.path.join(script_dir, "torcs")
        
        if not os.path.exists(torcs_dir):
            raise FileNotFoundError(f"Nie znaleziono folderu TORCS w: {torcs_dir}")
            
        os.chdir(torcs_dir)

        os.system('taskkill /f /im wtorcs.exe >nul 2>&1')
        time.sleep(1.0)
        
        if self.vision is True:
            os.system('start "" wtorcs.exe -nofuel -nodamage -nolaptime -vision')
        else:
            os.system('start "" wtorcs.exe -nofuel -nodamage -nolaptime')
        
        time.sleep(3.0) 
        for key in ['enter', 'enter', 'up', 'up', 'enter', 'enter']:
            pyautogui.press(key)
            time.sleep(0.2)
        time.sleep(5.0)
        
        os.chdir(cwd)

    def end(self):
        os.system('taskkill /f /im wtorcs.exe >nul 2>&1')

    def get_obs(self):
        obs = self.observation
        # Bierzemy tylko najważniejsze dane (łącznie 24 wartości)
        parts = [
            np.atleast_1d(np.asarray(obs.track, dtype=np.float32)).ravel(),
            np.atleast_1d(np.asarray(obs.speedX, dtype=np.float32)).ravel(),
            np.atleast_1d(np.asarray(obs.speedY, dtype=np.float32)).ravel(),
            np.atleast_1d(np.asarray(obs.angle, dtype=np.float32)).ravel(),
            np.atleast_1d(np.asarray(obs.trackPos, dtype=np.float32)).ravel(),
            np.array([self.last_steer], dtype=np.float32)
        ]
        return np.concatenate(parts).astype(np.float32)

    def agent_to_torcs(self, u):
        a = np.asarray(u, dtype=np.float32).ravel()
        idx = 0
        torcs_action = {'steer': float(a[idx])}
        idx += 1

        if self.throttle is True:
            raw_accel = float(a[idx])
            if raw_accel >= -0.2:
                torcs_action.update({'accel': (raw_accel + 0.2) / 1.2, 'brake': 0.0})
            else:
                torcs_action.update({'accel': 0.0, 'brake': (-raw_accel - 0.2) / 0.8})
            idx += 1
        if self.gear_change is True:
            gear_raw = float(a[idx])  
            gear = int(np.clip(np.round(((gear_raw + 1.0) / 2.0) * 5.0) + 1, 1, 6))
            torcs_action.update({'gear': gear})

        return torcs_action

    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec = obs_image_vec
        rgb = []
        temp = []
        for i in range(0,12286,3):
            temp.append(image_vec[i])
            temp.append(image_vec[i+1])
            temp.append(image_vec[i+2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = ['focus', 'speedX', 'speedY', 'speedZ', 'opponents', 'rpm', 'track', 'wheelSpinVel', 'angle', 'trackPos']
            Observation = col.namedtuple('Observation', names)
            return Observation(
                focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                speedX=np.array([raw_obs['speedX']], dtype=np.float32)/self.default_speed,
                speedY=np.array([raw_obs['speedY']], dtype=np.float32)/self.default_speed,
                speedZ=np.array([raw_obs['speedZ']], dtype=np.float32)/self.default_speed,
                opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                rpm=np.array([raw_obs['rpm']], dtype=np.float32) / 10000.0,
                track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32) / 100.0,
                angle=np.array([raw_obs['angle']], dtype=np.float32) / np.pi,
                trackPos=np.array([raw_obs['trackPos']], dtype=np.float32)
            )
        else:
            names = ['focus', 'speedX', 'speedY', 'speedZ', 'opponents', 'rpm', 'track', 'wheelSpinVel', 'angle', 'trackPos', 'img']
            Observation = col.namedtuple('Observation', names)
            image_rgb = self.obs_vision_to_image_rgb(raw_obs['img'])

            return Observation(
                focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                speedX=np.array([raw_obs['speedX']], dtype=np.float32)/self.default_speed,
                speedY=np.array([raw_obs['speedY']], dtype=np.float32)/self.default_speed,
                speedZ=np.array([raw_obs['speedZ']], dtype=np.float32)/self.default_speed,
                opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                rpm=np.array([raw_obs['rpm']], dtype=np.float32) / 10000.0,
                track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32) / 100.0,
                angle=np.array([raw_obs['angle']], dtype=np.float32) / np.pi,
                trackPos=np.array([raw_obs['trackPos']], dtype=np.float32),
                img=image_rgb
            )