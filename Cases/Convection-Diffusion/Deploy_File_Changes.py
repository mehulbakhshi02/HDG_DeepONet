import numpy as np
import subprocess

def Value_Change_in_pde_file(file_name,str_to_find,value):
    
    pde_file = file_name

    file = open(pde_file)
    file_content = file.read()
    file.close()
    
    start = file_content.find(str_to_find)
    end = file_content.find(str_to_find) + len(str_to_find)
    
    num_digits = file_content[end:].find("\n")
    
    file_content_before = file_content[:end]

    file_content_after = file_content[end+num_digits:]
    
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
    
    
Files = ["test.pde","test.pde","test.pde","test.pde"]

Strings = ["define constant epsilon = ","define constant alpha = ","define constant beta = ","define constant train = "]

Values = [0.01,2,2,0]

Multiple_Value_Change_in_pde_file(Files,Strings,Values)
