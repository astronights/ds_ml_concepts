import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
	'''Principal Components Analysis

 	Input Data shape: (m x n) -> Samples (m), Features (n)

 	Args:
  		data (np.ndarray): Raw data
		k (int): Number of principle components

  	Returns:
   		pca_data (np.ndarray): PCA transformed data (m x k)
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
