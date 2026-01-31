import random
import numpy as np
import argparse

# IMPORTANT: Change this import to point to your new GA logic
from ga.run_fixed import run_ga as run_ga_fixed 

if __name__ == "__main__":
    '''
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    '''
    
    parser = argparse.ArgumentParser(description='Arguments for FIXED ga script')
    parser.add_argument('--exp-name', type=str, default='fixed_ga_test', help='Name of the experiment')
    parser.add_argument('--env-name', type=str, default='Walker-v0', help='Environment name')
    parser.add_argument('--pop-size', type=int, default=3, help='Population size')
    parser.add_argument('--structure_shape', type=tuple, default=(5,5), help='Shape of the structure')
    parser.add_argument('--max-evaluations', type=int, default=6, help='Max robots evaluated')
    parser.add_argument('--num-cores', type=int, default=3, help='Number of parallel robots')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # NOTICE: We removed add_ppo_args(parser) because we don't need 
    # learning rates, clip ranges, or 1 million timesteps anymore.
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Call the new fixed version of the GA
    run_ga_fixed(args)