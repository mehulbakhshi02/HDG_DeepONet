#!/usr/bin/env python
# coding: utf-8

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

mesh_list = [2]
print_list = [2]

for i in mesh_list:
    try:
        df = pd.read_csv("cell_wise_ml_err_"+str(i)+".csv",sep=",",header=None,index_col=False)
        df = df.sort_values(1)
        data = np.array(df)
        plt.semilogy(data)
        plt.title("Number of the elements: "+str(i))
        plt.legend(["Proj_err: Adjoint","tr_err: True Error","ml_err: ML Prediction"])
        plt.grid()
        plt.xlabel("Element Number")
        plt.ylabel(r"$|u-u_h|_{L^2(\Omega)}$")
        if i in print_list:
            plt.savefig(f"Element_Wise_Error_Plot_Mesh_Elemnents_{i}.png")
        # plt.show()
        plt.close()
    except:
        print(f"No Error to print in case of {i}")





