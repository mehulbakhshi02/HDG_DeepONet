import numpy as np
import subprocess
import os
    
directories = os.listdir()

Req_dir = []

epsilon_per_core = 1

for directory in directories:
    if directory.find("Dataset_") != -1:
        Req_dir.append(directory)
    
cwd = os.getcwd()

Destination = "Combined_Dataset"

print("Removing common old data directory (if any)")
subprocess.run(f"rm -r {Destination}", shell=True)
print("Removed common old data directory (if any)")

subprocess.run(f"mkdir {Destination}", shell=True)

num_cores = os.cpu_count()

# Just leaving 2 cores idle for other jobs
num_process = 10

command = f"cd {Destination}; mkdir TrainingDataFiles"
subprocess.run(command, shell=True)
    
for i in range(num_process):
    Data_Files = cwd+f"/{Req_dir[i]}/ML/TrainingDataFiles/*"
    # Data_Files = f"{Req_dir[i]}/ML/TrainingDataFiles/*"
    # Data_Files = cwd+f"/{Req_dir[i]}/ML/*.npy"
    
    # command = f"cd {Destination}/Data_sep; mkdir {i}"
    # subprocess.run(command, shell=True)
    
    # command = f"cp {Data_Files} {Destination}/Data_sep/{i}"
    command = f"cp {Data_Files} {Destination}/TrainingDataFiles/"
    subprocess.run(command, shell=True)
    
print(f"All the data files copied to {Destination}")
