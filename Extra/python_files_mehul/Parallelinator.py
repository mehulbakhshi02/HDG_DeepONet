import concurrent.futures
import subprocess
import os
import time

# Define a function to execute a command
def execute_command(command):
    subprocess.run(command, shell=True)

cwd = os.getcwd()
    
commands = []

Base = "HDG_ML_NS"

num_cores = os.cpu_count()

# Just leaving 2 cores idle for other jobs
num_process = 10

for i in range(num_process):
    Script = cwd+f"/Dataset_{i+1}"
    # command = f"cp -r {Base} {Script}"
    # Run the Run_py_File_Changes.py separately
    command = f"cd {Script}; taskset --cpu-list {i} python3 run.py"
    commands.append(command)
    
start = time.time()
# Using ThreadPoolExecutor to run scripts in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
# with concurrent.futures.ProcessPoolExecutor() as executor:
    # Submit each script to the executor
    futures = [executor.submit(execute_command,command) for command in commands]

    # Wait for all scripts to complete
    concurrent.futures.wait(futures)
    
print(time.time() - start)
