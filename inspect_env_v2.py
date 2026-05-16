import gymnasium as gym
import numpy as np
import mani_skill.envs
import torch

def main():
    try:
        env = gym.make('PushCube-v1', obs_mode='rgb', control_mode='pd_joint_pos', render_mode='rgb_array')
        obs, info = env.reset(seed=0)
        unwrapped = env.unwrapped

        print('--- cube and goal ---')
        if hasattr(unwrapped, 'cube'):
            print(f"cube pose: {unwrapped.cube.pose.p}")
        if hasattr(unwrapped, 'goal_region'):
            print(f"goal_region pose: {unwrapped.goal_region.pose.p}")
        
        # In ManiSkill, the cube is often the object being pushed.
        
        env.close()
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    main()
