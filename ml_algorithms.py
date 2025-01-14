import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
	'''Principal Components Analysis

 	Input Data: (m x n) -> Samples (m), Features (n)
  	Output Data: (m x k) -> Samples (m), Components (k)

 	Args:
  		data (np.ndarray): Raw data
		k (int): Number of principle components

  	Returns:
   		pca_data (np.ndarray): PCA transformed data
	'''
	# Standardize the data
	f_mean = data.mean(axis=0)
	f_std = data.std(axis=0)
	
	z_data = (data - f_mean) / f_std
    
    	# Compute the covariance matrix for features
    	z_cov = np.cov(z_data.T)
    
    	# Compute eigenvalues and eigenvectors
    	eigen_values, eigen_vectors = np.linalg.eig(z_cov)
    
    	# Get top k components
    	top_indices = eigen_values.argsort()[::-1]
    	top_k_components = eigen_vectors[:, top_indices[:k]]
    
    	# Project the data
    	pca_data = z_data.dot(top_k_components)

	return pca_data


def lda(data: np.ndarray, y: np.array, k: int) -> np.ndarray
	'''Linear Discriminant Analysis

 	Input Data: (m x n) -> Samples (m), Features (n)
  	Output Data: (m x k) -> samples (m), Components (k)

   	Args:
    		data (np.ndarray): Raw data
      		y (np.array): Labels
		k (int): Number of components

  	Returns:
   		lda_data (np.ndarray): LDA transformed data
     	'''
	# Standardize the data
	f_mean = data.mean(axis=0)
	f_std = data.std(axis=0)
	
	z_data = (data - f_mean) / f_std
	z_mean = z_data.mean(axis=0)

	# Create matrices for Sum of Squared Variances 
	ssb = np.zeros(shape=(data.shape[1], data.shape[1]))
	ssw = np.zeros(shape=(data.shape[1], data.shape[1]))

	for label in np.unique(y):
		y_data = z_data[y == label,:]
		y_data_mean = y_data.mean(axis=0)

		# Sum of Squared Variances Within
		y_data_diff = y_data - y_data_mean
		ssw += y_data_diff.T.dot(y_data_diff)

		# Sum of Squared Variances Between
		data_diff = (y_data_mean - z_mean).reshape(-1, 1)
		ssb += data_diff.dot(data_diff.T) * len(y_data)

	# Compute SSB / SSW
	cov = np.linalg.inv(ssw).dot(ssb)

	# Compute eigenvalues and eigenvectors
	eigen_values, eigen_vectors = np.linalg.eig(cov)

	# Get top k components
	top_indices = eigen_values.argsort()[::-1]
	top_k_components = eigen_vectors[:,top_indices[:k]]

	lda_data = z_data.dot(top_k_components)

	return lda_data
