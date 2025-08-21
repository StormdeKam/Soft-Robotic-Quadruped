import sys
import os
import torch

def print_weights_from_state_dict(state_dict, print_full=False, max_elements=10):
    print("Model weights:")
    for name, param in state_dict.items():
        print(f"\n{name}: shape {param.shape}")
        data = param.cpu().numpy()
        if not print_full and data.size > max_elements:
            print(f"  values (first {max_elements}): {data.flatten()[:max_elements]}")
        else:
            print(f"  values:\n{data}")


def load_pt_state_dict(file_path):
    state = torch.load(file_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        return state['state_dict']
    elif isinstance(state, dict):
        return state
    else:
        raise ValueError("Unsupported .pt format")

def load_multihead_sac_zip(file_path):
    from rl.scripts.tasq_env import TASQEnv
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    from tasq_rl.rl.scripts.tasq_sac import MultiHeadSAC

    def make_env():
        return TASQEnv(urdf_path="rl/urdfs/TASQ_n7.urdf", no_links=7, gui=False)

    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)

    model = MultiHeadSAC.load(file_path, env=env)
    return model.policy.actor.actor_net.state_dict()

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to .pt/.zip model file")
    parser.add_argument("--full", action="store_true", help="Print full weight tensors")
    args = parser.parse_args()

    file_path = args.file
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        sys.exit(1)

    ext = os.path.splitext(file_path)[-1]

    if ext in ['.pt', '.pth']:
        state_dict = load_pt_state_dict(file_path)
    elif ext == '.zip':
        state_dict = load_multihead_sac_zip(file_path)
    else:
        print("Unsupported file type. Only .pt/.pth and .zip are supported.")
        sys.exit(1)

    print_weights_from_state_dict(state_dict, print_full=args.full)


if __name__ == "__main__":
    main()
