import gymnasium as gym
import numpy as np
import mani_skill.envs
import torch

def main():
    try:
        env = gym.make('PushCube-v1', obs_mode='rgb', control_mode='pd_joint_pos', render_mode='rgb_array')
        obs, info = env.reset(seed=0)
        unwrapped = env.unwrapped

        print('--- unwrapped attributes ---')
        for a in dir(unwrapped):
            v = getattr(unwrapped, a)
            # Look for actors
            if hasattr(v, 'pose'):
                print(f'{a}: {v.pose.p}')
        
        env.close()
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    main()
