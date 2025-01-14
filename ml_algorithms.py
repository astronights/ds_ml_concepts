import numpy as np
from collections import Counter

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

	# Project the data
	lda_data = z_data.dot(top_k_components)

	return lda_data


def tf_idf(corpus: list[list[str]], query: list[str]) -> list[list[float]]:
	'''Compute TF-IDF scores

 	Args:
  		corpus (list): List of documents, each a list of words
    		query (list): List of words to query

	Returns:
 		tf_idfs (lis): List of documents, each containing TF-IDF scores for each word
   	'''
	# Create empty Term Frequency Matrix
	tf = np.zeros((len(corpus), len(query)))

	# Iterate over documents
	for doc_ix, doc in enumerate(corpus):
		# Calculate word frequencies
		term_count = Counter(doc)

		# Iterate over words
		for word_ix, word in enumerate(query):
			# Get word count of token
			word_count = term_count.get(word, 0)
			total_terms = sum(term_count.values())

			# Compute Term Frequency
			tf[doc_ix, word_ix] = word_count / total_terms

	# Compute document frequencies of words
	len_corpus = 1 + len(corpus)
	len_df = 1 + np.count_nonzero(tf > 0, axis=0)

	# Compute Inverse document frequency
	idf = np.log(len_corpus / len_df) + 1

	# Compute TF-IDF
	return tf * idf

