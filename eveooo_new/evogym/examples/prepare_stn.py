import os
import numpy as np
import argparse

def create_stn_input(folder_name, run_label, base_path='saved_data'):
    run_dir = os.path.join(base_path, folder_name)
    if not os.path.exists(run_dir):
        print(f"Error: Folder '{run_dir}' not found.")
        return

    # 1. Gather all generation folders in numerical order
    gen_dirs = [d for d in os.listdir(run_dir) if d.startswith('generation_')]
    gen_dirs.sort(key=lambda x: int(x.split('_')[1]))

    trajectory_data = []

    # 2. Extract best robot from each generation
    for gen in gen_dirs:
        gen_path = os.path.join(run_dir, gen)
        output_file = os.path.join(gen_path, "output.txt")
        
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                first_line = f.readline().strip()
                if not first_line: continue
                
                parts = first_line.split()
                robot_idx, fitness = parts[0], parts[1]

                # Load the 25-voxel body structure
                npz_path = os.path.join(gen_path, "structure", f"{robot_idx}.npz")
                if os.path.exists(npz_path):
                    data = np.load(npz_path)
                    body = data[data.files[0]].reshape(-1) # Flatten to 25 values
                    trajectory_data.append({
                        "fitness": fitness,
                        "body": " ".join(map(str, body))
                    })

    # 3. Write to the specific STN format
    # Header format: Run Value_Current_Best Current_Best [n] Value_Solution_At_Iteration Solution_At_Iteration
    output_filename = f"{folder_name}_stn_input.txt"
    with open(output_filename, 'w') as out:
        # Note: '25' indicates the dimension of your voxel vector
        header = "Run Value_Current_Best Current_Best 25 Value_Solution_At_Iteration Solution_At_Iteration\n"
        out.write(header)

        # Create edges: Gen(i) -> Gen(i+1)
        for i in range(len(trajectory_data) - 1):
            curr = trajectory_data[i]
            nxt = trajectory_data[i+1]
            
            # Row: [RunLabel] [Fit1] [Body1] [Fit2] [Body2]
            line = f"{run_label} {curr['fitness']} {curr['body']} {nxt['fitness']} {nxt['body']}\n"
            out.write(line)

    print(f"Successfully created STN input file: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_name', type=str, help='Folder containing generations')
    parser.add_argument('run_label', type=str, help='Numerical ID for this trajectory (e.g., 1)')
    args = parser.parse_args()
    create_stn_input(args.folder_name, args.run_label)