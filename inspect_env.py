import gymnasium as gym
import numpy as np
import re
import mani_skill.envs

def main():
    try:
        env = gym.make('PushCube-v1', obs_mode='rgb', control_mode='pd_joint_pos', render_mode='rgb_array')
        obs, info = env.reset(seed=0)
        unwrapped = env.unwrapped

        print('--- unwrapped attrs ---')
        for a in dir(unwrapped):
            if re.search(r'(goal|target|cube)', a, re.IGNORECASE):
                try:
                    v = getattr(unwrapped, a)
                    if hasattr(v, 'pose'):
                        print(f'{a} (pose.p): {v.pose.p}')
                    elif hasattr(v, 'p'):
                        print(f'{a} (p): {v.p}')
                except: pass

        print('--- obs extra ---')
        if isinstance(obs, dict) and 'extra' in obs:
            for k, v in obs['extra'].items():
                print(f'extra.{k}: {v}')
        
        env.close()
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    main()
