import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, OPTICS
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

# Total
input_data = np.load("All_input_data.npy")
output_data = np.load("All_output_data.npy")

# preproc_input = Pipeline([('stdscaler', StandardScaler()),('minmaxscaler', MinMaxScaler())])
# input_data = preproc_input.fit_transform(input_data)
# scaler_filename = f"ip_scaler.save"
# joblib.dump(preproc_input, scaler_filename)

# preproc_output = Pipeline([('stdscaler', StandardScaler()),('minmaxscaler', MinMaxScaler())])
# output_data = preproc_output.fit_transform(output_data)
# scaler_filename = f"op_scaler.save"
# joblib.dump(preproc_output, scaler_filename)

pca = PCA(n_components=1)
pca_input = pca.fit_transform(input_data)
pca = PCA(n_components=1)
pca_output = pca.fit_transform(output_data)

print(input_data.shape)
pearson_coeff = np.corrcoef(pca_input[:,0],pca_output[:,0])[0,1]
print(f"total relation coeff: {pearson_coeff}")

# Rho
input_data = np.load("All_input_data.npy")
input_data = np.hstack([input_data[:, 0:6], input_data[:, 6:12], input_data[:, 30:42], input_data[:, 78:90], input_data[:, 126:150]])
output_data = np.load("All_output_data.npy")

# preproc_input = Pipeline([('stdscaler', StandardScaler()),('minmaxscaler', MinMaxScaler())])
# input_data = preproc_input.fit_transform(input_data)
# scaler_filename = f"ip_scaler.save"
# joblib.dump(preproc_input, scaler_filename)

# preproc_output = Pipeline([('stdscaler', StandardScaler()),('minmaxscaler', MinMaxScaler())])
# output_data = preproc_output.fit_transform(output_data)
# scaler_filename = f"op_scaler.save"
# joblib.dump(preproc_output, scaler_filename)

pca = PCA(n_components=1)
pca_input = pca.fit_transform(input_data)
pca = PCA(n_components=1)
pca_output = pca.fit_transform(output_data)

print(input_data.shape)
pearson_coeff = np.corrcoef(pca_input[:,0],pca_output[:,0])[0,1]
print(f"rho relation coeff: {pearson_coeff}")

# rho*u
input_data = np.load("All_input_data.npy")
input_data = np.hstack([input_data[:, 0:6], input_data[:, 12:18], input_data[:, 42:54], input_data[:, 90:102], input_data[:, 150:174]])
output_data = np.load("All_output_data.npy")

# preproc_input = Pipeline([('stdscaler', StandardScaler()),('minmaxscaler', MinMaxScaler())])
# input_data = preproc_input.fit_transform(input_data)
# scaler_filename = f"ip_scaler.save"
# joblib.dump(preproc_input, scaler_filename)

# preproc_output = Pipeline([('stdscaler', StandardScaler()),('minmaxscaler', MinMaxScaler())])
# output_data = preproc_output.fit_transform(output_data)
# scaler_filename = f"op_scaler.save"
# joblib.dump(preproc_output, scaler_filename)

pca = PCA(n_components=1)
pca_input = pca.fit_transform(input_data)
pca = PCA(n_components=1)
pca_output = pca.fit_transform(output_data)

print(input_data.shape)
pearson_coeff = np.corrcoef(pca_input[:,0],pca_output[:,0])[0,1]
print(f"rho*u relation coeff: {pearson_coeff}")

# rho*v
input_data = np.load("All_input_data.npy")
input_data = np.hstack([input_data[:, 0:6], input_data[:, 18:24], input_data[:, 54:66], input_data[:, 102:114], input_data[:, 174:198]])
output_data = np.load("All_output_data.npy")

# preproc_input = Pipeline([('stdscaler', StandardScaler()),('minmaxscaler', MinMaxScaler())])
# input_data = preproc_input.fit_transform(input_data)
# scaler_filename = f"ip_scaler.save"
# joblib.dump(preproc_input, scaler_filename)

# preproc_output = Pipeline([('stdscaler', StandardScaler()),('minmaxscaler', MinMaxScaler())])
# output_data = preproc_output.fit_transform(output_data)
# scaler_filename = f"op_scaler.save"
# joblib.dump(preproc_output, scaler_filename)

pca = PCA(n_components=1)
pca_input = pca.fit_transform(input_data)
pca = PCA(n_components=1)
pca_output = pca.fit_transform(output_data)

print(input_data.shape)
pearson_coeff = np.corrcoef(pca_input[:,0],pca_output[:,0])[0,1]
print(f"rho*v relation coeff: {pearson_coeff}")

# rho*E
input_data = np.load("All_input_data.npy")
input_data = np.hstack([input_data[:, 0:6], input_data[:, 24:30], input_data[:, 66:78], input_data[:, 114:126], input_data[:, 198:222]])
output_data = np.load("All_output_data.npy")

# preproc_input = Pipeline([('stdscaler', StandardScaler()),('minmaxscaler', MinMaxScaler())])
# input_data = preproc_input.fit_transform(input_data)
# scaler_filename = f"ip_scaler.save"
# joblib.dump(preproc_input, scaler_filename)

# preproc_output = Pipeline([('stdscaler', StandardScaler()),('minmaxscaler', MinMaxScaler())])
# output_data = preproc_output.fit_transform(output_data)
# scaler_filename = f"op_scaler.save"
# joblib.dump(preproc_output, scaler_filename)

pca = PCA(n_components=1)
pca_input = pca.fit_transform(input_data)
pca = PCA(n_components=1)
pca_output = pca.fit_transform(output_data)

print(input_data.shape)
pearson_coeff = np.corrcoef(pca_input[:,0],pca_output[:,0])[0,1]
print(f"rho*E relation coeff: {pearson_coeff}")
