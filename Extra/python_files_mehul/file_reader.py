import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

filename_list = os.listdir('./TrainingDataFiles')

#Memory Allocation

data_points = 0

for filename in filename_list:
	file = open('./TrainingDataFiles/'+filename,'r')
	lines = file.readlines()[1:]
	file.close()
	
	for line_num in range(len(lines)):
		split_list = lines[line_num].split(',')
		if len(split_list) ==3:
			data_points = data_points + 1
			num_branch_inputs = int(split_list[0])
			num_trunk_inputs = 2
			num_outputs = int(split_list[2])

branch_input_array = np.zeros((data_points, num_outputs, num_branch_inputs))
trunk_input_array = np.zeros((data_points,num_outputs, num_trunk_inputs))
output_array = np.zeros((data_points,num_outputs))

i=0

for filename in filename_list:

	file = open('./TrainingDataFiles/'+filename,'r')
	first_line = file.readlines()[0:1]
	spliting = first_line[0].split(',')
	num_ele = int(spliting[0])
	file.close()
	
	file = open('./TrainingDataFiles/'+filename,'r')
	lines = file.readlines()[1:]
	file.close()
	
	rr = 0

	for element in range(num_ele):

		rr = rr+1
		for k in range(num_branch_inputs):
			branch_input_array[i, 0, k] = lines[rr]
			rr = rr+1
		branch_input_array[i, :, :] = branch_input_array[i, 0, :]
		
		for j in range(num_outputs):
			coordinates = lines[rr].split(',')
			for k in range (num_trunk_inputs):
				trunk_input_array[i, j, k] = float(coordinates[k])
			rr = rr+1
		
		for j in range(num_outputs):
			output_array[i, j] = lines[rr]
			rr = rr+1
		
		i = i+1

print(branch_input_array.shape)
print(trunk_input_array.shape)
print(output_array.shape)

np.save("branch_input.npy", branch_input_array)
np.save("trunk_input.npy", trunk_input_array)
np.save("output.npy", output_array)


# np.savetxt("output.csv",output_array)

# # Save each 2D slice as a separate CSV
# for i in range(branch_input_array.shape[0]):
#     df = pd.DataFrame(branch_input_array[i])  # Convert slice to DataFrame
#     df.to_csv(f"slice_{i}.csv", index=False, header=False)

# print("Saved each slice as a separate CSV file!")


