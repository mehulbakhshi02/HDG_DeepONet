import numpy as np
import subprocess
import os

def Value_Change_in_pde_file(file_name,str_to_find,value):
    
    pde_file = file_name

    file = open(pde_file)
    file_content = file.read()
    file.close()
    
    start = file_content.find(str_to_find)
    end = file_content.find(str_to_find) + len(str_to_find)
    
    file_content_before = file_content[:end]

    file_content_after = file_content[end:]
    
    New_file_content = file_content_before+str(value)+file_content_after
    
    subprocess.call(f"rm {pde_file}",shell =True)
    
    file = open(pde_file,"x")
    file.write(New_file_content)
    file.close()
    
def Multiple_Value_Change_in_pde_file(file_name,str_to_find,value):
    # Here all the inputs are the list
    
    # Just checking the sizes of the list
    assert len(Files) == len(Strings) == len(Values)
    
    for i in range(len(file_name)):
        Value_Change_in_pde_file(file_name[i],str_to_find[i],value[i])
    print(f"\nValues changed in the files\n")
    
directories = os.listdir()

Req_dir = []


Files = []
Strings = []
Values = []
mach_per_core = 1

for directory in directories:
    if directory.find("Dataset_") != -1:
        Req_dir.append(directory)


Req_dir = sorted(Req_dir)

for i in range(len(Req_dir)):
    Files.append(f"./{Req_dir[i]}/run.py")
    Strings.append("mach_range = np.linspace(0.35, 0.7, N)")
    Values.append(f"[{mach_per_core *i}:{mach_per_core *(i+1)}]")
    # print(f"[{1*i}:{1*(i+1)}]")
print(f"Directories: \n{Req_dir}")
print("Change the last one manually\n")
# Change the last one manually
    
Multiple_Value_Change_in_pde_file(Files,Strings,Values)
