import numpy as np
import matplotlib.pyplot as plt
import os

filename_list = os.listdir('./TrainingDataFiles')

input_arrays_list = []
output_arrays_list = []

print("Allocating memory\n")
# To know how much memory to allocate 
data_points = 0
num_inputs = 0
num_outputs = 0
for filename in filename_list:
    # print(f"File Name: {filename}")
    file = open('./TrainingDataFiles/'+filename,'r')
    lines = file.readlines()[1:]
    file.close()
    element_num = 0
    for line_num in range(len(lines)):
        split_list = lines[line_num].split(',')
        if len(split_list) == 3:
            data_points = data_points + 1
            num_inputs = int(split_list[1])
            num_outputs = int(split_list[0])
            
# Memory Allocation
input_arrays = np.zeros((data_points,num_inputs))
output_arrays = np.zeros((data_points,num_outputs))
ctr = 0

print("Populating the memory\n")

for filename in filename_list:
    # print(f"File Name: {filename}")

    file = open('./TrainingDataFiles/'+filename,'r')
    lines = file.readlines()[1:]
    file.close()

    element_num = 0
    for line_num in range(len(lines)):
        split_list = lines[line_num].split(',')

        if len(split_list) == 3:
            num_input_lines = int(split_list[1])
            num_output_lines = int(split_list[0])

            inputs = []
            outputs = []

            for input_line in range(line_num+1,line_num+num_input_lines+1):
                inputs.append(float(lines[input_line]))

            for output_line in range(line_num+num_input_lines+1,line_num+num_input_lines+num_output_lines+1):
                outputs.append(float(lines[output_line]))
            
            file_inputs = np.asarray(inputs).reshape(1,-1)
            file_outputs = np.asarray(outputs).reshape(1,-1)
            # print(file_inputs.shape)
            input_arrays[ctr,:] = file_inputs
            output_arrays[ctr,:] = file_outputs
            ctr = ctr + 1
            
np.save('All_input_data.npy',input_arrays)
np.save('All_output_data.npy',output_arrays)

# KUNAL
# Checking
# print(all_inputs.shape)
# print(all_outputs.shape)
