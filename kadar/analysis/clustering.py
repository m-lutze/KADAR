import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform


class ClusteringAnalysis:
    """
    Class implementing various clustering methods for compositional analysis.

    DBSCAN, K-means, hierarchical clustering, and Gaussian
    """
    
    def __init__(self, profiler, scaler: StandardScaler):
        """
        Initialize clustering analysis.
        
        Args:
            profiler: KmerProfiler instance
            scaler: StandardScaler instance
        """
        self.profiler = profiler
        self.scaler = scaler
    
    def run_clustering(self, seq_ids: Optional[List[str]] = None, 
                      method: str = 'dbscan', **kwargs) -> Dict:
        """
        Run clustering analysis using the specified method.
        
        Args:
            seq_ids: Sequences to include
            method: Clustering method ('dbscan', 'kmeans', 'hierarchical', 'gmm')
            **kwargs: Additional parameters for clustering algorithm
            
        Returns:
            Dictionary with clustering results
        """
        method = method.lower()
        
        if method == 'dbscan':
            return self.dbscan_clustering(seq_ids, **kwargs)
        elif method == 'kmeans':
            return self.kmeans_clustering(seq_ids, **kwargs)
        elif method == 'hierarchical':
            return self.hierarchical_clustering(seq_ids, **kwargs)
        elif method == 'gmm':
            return self.gaussian_mixture_clustering(seq_ids, **kwargs)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
    
    def dbscan_clustering(self, seq_ids: Optional[List[str]] = None, 
                         eps: float = 0.5, min_samples: int = 5, 
                         metric: str = 'euclidean') -> Dict:
        """
        Perform DBSCAN clustering.
        
        Args:
            seq_ids: Sequences to include
            eps: Maximum distance between samples in the same neighborhood
            min_samples: Minimum number of samples in a neighborhood
            metric: Distance metric to use
            
        Returns:
            Dictionary with DBSCAN results
        """
        matrix, seq_ids, kmer_list = self.profiler.get_profile_matrix(seq_ids)
        matrix_scaled = self.scaler.fit_transform(matrix)
        
        # Perform DBSCAN clustering
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = clusterer.fit_predict(matrix_scaled)
        
        # Calculate clustering metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        
        # Calculate silhouette score if more than one cluster and no noise points
        silhouette = None
        if n_clusters > 1 and n_noise < len(labels):
            try:
                silhouette = silhouette_score(matrix_scaled, labels)
            except ValueError:
                silhouette = None
        
        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_statistics(
            matrix_scaled, labels, seq_ids, kmer_list
        )
        
        results = {
            'labels': labels,
            'seq_ids': seq_ids,
            'method': 'dbscan',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': silhouette,
            'clusterer': clusterer,
            'parameters': {'eps': eps, 'min_samples': min_samples, 'metric': metric},
            'cluster_stats': cluster_stats
        }
        
        return results
    
    def kmeans_clustering(self, seq_ids: Optional[List[str]] = None, 
                         n_clusters: int = 3, random_state: int = 42,
                         init: str = 'k-means++', max_iter: int = 300) -> Dict:
        """
        Perform K-means clustering.
        
        Args:
            seq_ids: Sequences to include
            n_clusters: Number of clusters
            random_state: Random state for reproducibility
            init: Initialization method
            max_iter: Maximum number of iterations
            
        Returns:
            Dictionary with K-means results
        """
        matrix, seq_ids, kmer_list = self.profiler.get_profile_matrix(seq_ids)
        matrix_scaled = self.scaler.fit_transform(matrix)
        
        if n_clusters > len(seq_ids):
            n_clusters = len(seq_ids)
            print(f"Warning: Reduced n_clusters to {n_clusters} (number of sequences)")
        
        # Perform K-means clustering
        clusterer = KMeans(
            n_clusters=n_clusters, 
            random_state=random_state,
            init=init,
            max_iter=max_iter
        )
        labels = clusterer.fit_predict(matrix_scaled)
        
        # Calculate clustering metrics
        silhouette = silhouette_score(matrix_scaled, labels) if n_clusters > 1 else None
        calinski_harabasz = calinski_harabasz_score(matrix_scaled, labels) if n_clusters > 1 else None
        davies_bouldin = davies_bouldin_score(matrix_scaled, labels) if n_clusters > 1 else None
        
        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_statistics(
            matrix_scaled, labels, seq_ids, kmer_list
        )
        
        # Calculate within-cluster sum of squares (WCSS)
        wcss = clusterer.inertia_
        
        results = {
            'labels': labels,
            'seq_ids': seq_ids,
            'method': 'kmeans',
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'wcss': wcss,
            'cluster_centers': clusterer.cluster_centers_,
            'clusterer': clusterer,
            'parameters': {
                'n_clusters': n_clusters, 
                'random_state': random_state,
                'init': init, 
                'max_iter': max_iter
            },
            'cluster_stats': cluster_stats
        }
        
        return results
    
    def hierarchical_clustering(self, seq_ids: Optional[List[str]] = None,
                               n_clusters: int = 3, linkage_method: str = 'ward',
                               distance_threshold: Optional[float] = None) -> Dict:
        """
        Perform hierarchical clustering.
        
        Args:
            seq_ids: Sequences to include
            n_clusters: Number of clusters (ignored if distance_threshold is set)
            linkage_method: Linkage method ('ward', 'complete', 'average', 'single')
            distance_threshold: Distance threshold for clustering
            
        Returns:
            Dictionary with hierarchical clustering results
        """
        matrix, seq_ids, kmer_list = self.profiler.get_profile_matrix(seq_ids)
        matrix_scaled = self.scaler.fit_transform(matrix)
        
        # Perform hierarchical clustering
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters if distance_threshold is None else None,
            linkage=linkage_method,
            distance_threshold=distance_threshold
        )
        labels = clusterer.fit_predict(matrix_scaled)
        
        # Calculate linkage matrix for dendrogram
        if linkage_method == 'ward':
            linkage_matrix = linkage(matrix_scaled, method='ward')
        else:
            distances = pdist(matrix_scaled)
            linkage_matrix = linkage(distances, method=linkage_method)
        
        # Calculate clustering metrics
        n_clusters_found = len(set(labels))
        silhouette = silhouette_score(matrix_scaled, labels) if n_clusters_found > 1 else None
        
        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_statistics(
            matrix_scaled, labels, seq_ids, kmer_list
        )
        
        results = {
            'labels': labels,
            'seq_ids': seq_ids,
            'method': 'hierarchical',
            'n_clusters': n_clusters_found,
            'silhouette_score': silhouette,
            'linkage_matrix': linkage_matrix,
            'clusterer': clusterer,
            'parameters': {
                'n_clusters': n_clusters,
                'linkage_method': linkage_method,
                'distance_threshold': distance_threshold
            },
            'cluster_stats': cluster_stats
        }
        
        return results
    
    def gaussian_mixture_clustering(self, seq_ids: Optional[List[str]] = None,
                                   n_components: int = 3, covariance_type: str = 'full',
                                   random_state: int = 42, max_iter: int = 100) -> Dict:
        """
        Perform Gaussian Mixture Model clustering.
        
        Args:
            seq_ids: Sequences to include
            n_components: Number of mixture components
            covariance_type: Type of covariance parameters
            random_state: Random state for reproducibility
            max_iter: Maximum number of iterations
            
        Returns:
            Dictionary with GMM clustering results
        """
        matrix, seq_ids, kmer_list = self.profiler.get_profile_matrix(seq_ids)
        matrix_scaled = self.scaler.fit_transform(matrix)
        
        if n_components > len(seq_ids):
            n_components = len(seq_ids)
            print(f"Warning: Reduced n_components to {n_components} (number of sequences)")
        
        # Perform GMM clustering
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            max_iter=max_iter
        )
        labels = gmm.fit_predict(matrix_scaled)
        probabilities = gmm.predict_proba(matrix_scaled)
        
        # Calculate clustering metrics
        silhouette = silhouette_score(matrix_scaled, labels) if n_components > 1 else None
        
        # Calculate information criteria
        aic = gmm.aic(matrix_scaled)
        bic = gmm.bic(matrix_scaled)
        log_likelihood = gmm.score(matrix_scaled)
        
        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_statistics(
            matrix_scaled, labels, seq_ids, kmer_list
        )
        
        results = {
            'labels': labels,
            'probabilities': probabilities,
            'seq_ids': seq_ids,
            'method': 'gmm',
            'n_components': n_components,
            'silhouette_score': silhouette,
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood,
            'means': gmm.means_,
            'covariances': gmm.covariances_,
            'weights': gmm.weights_,
            'clusterer': gmm,
            'parameters': {
                'n_components': n_components,
                'covariance_type': covariance_type,
                'random_state': random_state,
                'max_iter': max_iter
            },
            'cluster_stats': cluster_stats
        }
        
        return results
    
    def optimal_clusters_analysis(self, seq_ids: Optional[List[str]] = None,
                                 method: str = 'kmeans', max_clusters: int = 10) -> Dict:
        """
        Analyze optimal number of clusters using multiple metrics.
        
        Args:
            seq_ids: Sequences to include
            method: Clustering method to use
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Dictionary with optimal cluster analysis results
        """
        matrix, seq_ids, kmer_list = self.profiler.get_profile_matrix(seq_ids)
        matrix_scaled = self.scaler.fit_transform(matrix)
        
        max_clusters = min(max_clusters, len(seq_ids) - 1)
        
        cluster_range = range(2, max_clusters + 1)
        metrics = {
            'silhouette_scores': [],
            'calinski_harabasz_scores': [],
            'davies_bouldin_scores': [],
            'wcss_values': [] if method == 'kmeans' else None,
            'aic_values': [] if method == 'gmm' else None,
            'bic_values': [] if method == 'gmm' else None
        }
        
        for k in cluster_range:
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=k, random_state=42)
                labels = clusterer.fit_predict(matrix_scaled)
                
                metrics['wcss_values'].append(clusterer.inertia_)
                
            elif method == 'gmm':
                clusterer = GaussianMixture(n_components=k, random_state=42)
                labels = clusterer.fit_predict(matrix_scaled)
                
                metrics['aic_values'].append(clusterer.aic(matrix_scaled))
                metrics['bic_values'].append(clusterer.bic(matrix_scaled))
                
            elif method == 'hierarchical':
                clusterer = AgglomerativeClustering(n_clusters=k)
                labels = clusterer.fit_predict(matrix_scaled)
            
            # Calculate common metrics
            metrics['silhouette_scores'].append(silhouette_score(matrix_scaled, labels))
            metrics['calinski_harabasz_scores'].append(calinski_harabasz_score(matrix_scaled, labels))
            metrics['davies_bouldin_scores'].append(davies_bouldin_score(matrix_scaled, labels))
        
        # Find optimal number of clusters
        optimal_k = {}
        optimal_k['silhouette'] = cluster_range[np.argmax(metrics['silhouette_scores'])]
        optimal_k['calinski_harabasz'] = cluster_range[np.argmax(metrics['calinski_harabasz_scores'])]
        optimal_k['davies_bouldin'] = cluster_range[np.argmin(metrics['davies_bouldin_scores'])]
        
        if method == 'kmeans':
            # Elbow method for WCSS
            wcss_diff = np.diff(metrics['wcss_values'])
            wcss_diff2 = np.diff(wcss_diff)
            if len(wcss_diff2) > 0:
                elbow_k = cluster_range[np.argmax(wcss_diff2) + 2]  # +2 due to double diff
                optimal_k['elbow'] = elbow_k
        
        if method == 'gmm':
            optimal_k['aic'] = cluster_range[np.argmin(metrics['aic_values'])]
            optimal_k['bic'] = cluster_range[np.argmin(metrics['bic_values'])]
        
        results = {
            'cluster_range': list(cluster_range),
            'metrics': metrics,
            'optimal_k': optimal_k,
            'method': method,
            'seq_ids': seq_ids
        }
        
        return results
    
    def _calculate_cluster_statistics(self, matrix: np.ndarray, labels: np.ndarray,
                                    seq_ids: List[str], kmer_list: List[str]) -> Dict:
        """
        Calculate detailed statistics for each cluster.
        
        Args:
            matrix: Scaled profile matrix
            labels: Cluster labels
            seq_ids: Sequence IDs
            kmer_list: K-mer list
            
        Returns:
            Dictionary with cluster statistics
        """
        unique_labels = set(labels)
        cluster_stats = {}
        
        for label in unique_labels:
            mask = labels == label
            cluster_sequences = [seq_ids[i] for i in range(len(seq_ids)) if mask[i]]
            cluster_data = matrix[mask]
            
            if len(cluster_data) > 0:
                stats = {
                    'size': len(cluster_sequences),
                    'sequences': cluster_sequences,
                    'centroid': np.mean(cluster_data, axis=0),
                    'std': np.std(cluster_data, axis=0),
                    'diameter': self._calculate_cluster_diameter(cluster_data),
                    'intra_cluster_distance': self._calculate_intra_cluster_distance(cluster_data),
                    'compactness': self._calculate_cluster_compactness(cluster_data)
                }
                
                # Find most characteristic k-mers for this cluster
                if len(cluster_data) > 1:
                    centroid = stats['centroid']
                    characteristic_kmers = []
                    
                    for i, kmer in enumerate(kmer_list):
                        kmer_mean = centroid[i]
                        kmer_std = stats['std'][i]
                        
                        if kmer_std > 0:  # Avoid division by zero
                            characteristic_kmers.append({
                                'kmer': kmer,
                                'mean_frequency': kmer_mean,
                                'std_frequency': kmer_std,
                                'coefficient_variation': kmer_std / abs(kmer_mean) if kmer_mean != 0 else float('inf')
                            })
                    
                    # Sort by mean frequency (descending)
                    characteristic_kmers.sort(key=lambda x: x['mean_frequency'], reverse=True)
                    stats['top_kmers'] = characteristic_kmers[:10]  # Top 10
                
            else:
                stats = {'size': 0, 'sequences': []}
            
            cluster_stats[label] = stats
        
        return cluster_stats
    
    def _calculate_cluster_diameter(self, cluster_data: np.ndarray) -> float:
        """Calculate the diameter (maximum distance) within a cluster."""
        if len(cluster_data) < 2:
            return 0.0
        
        distances = pdist(cluster_data)
        return np.max(distances)
    
    def _calculate_intra_cluster_distance(self, cluster_data: np.ndarray) -> float:
        """Calculate the average intra-cluster distance."""
        if len(cluster_data) < 2:
            return 0.0
        
        distances = pdist(cluster_data)
        return np.mean(distances)
    
    def _calculate_cluster_compactness(self, cluster_data: np.ndarray) -> float:
        """Calculate cluster compactness (average distance to centroid)."""
        if len(cluster_data) < 2:
            return 0.0
        
        centroid = np.mean(cluster_data, axis=0)
        distances = np.linalg.norm(cluster_data - centroid, axis=1)
        return np.mean(distances)
    
    def consensus_clustering(self, seq_ids: Optional[List[str]] = None,
                           methods: List[str] = None, n_iterations: int = 100) -> Dict:
        """
        Perform consensus clustering across multiple methods and parameters.
        
        Args:
            seq_ids: Sequences to include
            methods: List of methods to use for consensus
            n_iterations: Number of bootstrap iterations
            
        Returns:
            Dictionary with consensus clustering results
        """
        if methods is None:
            methods = ['kmeans', 'hierarchical']
        
        matrix, seq_ids, kmer_list = self.profiler.get_profile_matrix(seq_ids)
        matrix_scaled = self.scaler.fit_transform(matrix)
        n_samples = len(seq_ids)
        
        # Initialize consensus matrix
        consensus_matrix = np.zeros((n_samples, n_samples))
        
        np.random.seed(42)  # For reproducibility
        
        for iteration in range(n_iterations):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_matrix = matrix_scaled[indices]
            
            for method in methods:
                if method == 'kmeans':
                    # Try different numbers of clusters
                    for k in range(2, min(6, n_samples)):
                        clusterer = KMeans(n_clusters=k, random_state=iteration)
                        labels = clusterer.fit_predict(bootstrap_matrix)
                        
                        # Update consensus matrix
                        for i in range(len(indices)):
                            for j in range(i + 1, len(indices)):
                                if labels[i] == labels[j]:
                                    orig_i, orig_j = indices[i], indices[j]
                                    consensus_matrix[orig_i, orig_j] += 1
                                    consensus_matrix[orig_j, orig_i] += 1
                
                elif method == 'hierarchical':
                    for linkage_method in ['ward', 'complete']:
                        for k in range(2, min(6, n_samples)):
                            clusterer = AgglomerativeClustering(
                                n_clusters=k, 
                                linkage=linkage_method
                            )
                            labels = clusterer.fit_predict(bootstrap_matrix)
                            
                            # Update consensus matrix
                            for i in range(len(indices)):
                                for j in range(i + 1, len(indices)):
                                    if labels[i] == labels[j]:
                                        orig_i, orig_j = indices[i], indices[j]
                                        consensus_matrix[orig_i, orig_j] += 1
                                        consensus_matrix[orig_j, orig_i] += 1
        
        # Normalize consensus matrix
        max_count = n_iterations * len(methods) * 4  # 4 different parameter combinations
        consensus_matrix /= max_count
        
        # Diagonal should be 1 (each sample always clusters with itself)
        np.fill_diagonal(consensus_matrix, 1.0)
        
        # Perform final clustering on consensus matrix
        # Convert similarity to distance
        distance_matrix = 1 - consensus_matrix
        
        # Use hierarchical clustering on consensus
        linkage_matrix = linkage(squareform(distance_matrix), method='average')
        
        # Determine optimal number of clusters based on consensus
        best_k = 2
        best_score = -1
        
        for k in range(2, min(8, n_samples)):
            clusterer = AgglomerativeClustering(
                n_clusters=k,
                linkage='precomputed',
                metric='precomputed'
            )
            labels = clusterer.fit_predict(distance_matrix)
            
            # Calculate consensus-based silhouette score
            if len(set(labels)) > 1:
                score = silhouette_score(distance_matrix, labels, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_k = k
        
        # Final clustering with optimal k
        final_clusterer = AgglomerativeClustering(
            n_clusters=best_k,
            linkage='precomputed',
            metric='precomputed'
        )
        final_labels = final_clusterer.fit_predict(distance_matrix)
        
        results = {
            'consensus_matrix': consensus_matrix,
            'distance_matrix': distance_matrix,
            'linkage_matrix': linkage_matrix,
            'final_labels': final_labels,
            'optimal_k': best_k,
            'consensus_score': best_score,
            'seq_ids': seq_ids,
            'methods_used': methods,
            'n_iterations': n_iterations
        }
        
        return results
    
    def stability_analysis(self, seq_ids: Optional[List[str]] = None,
                          method: str = 'kmeans', n_clusters: int = 3,
                          n_bootstrap: int = 100) -> Dict:
        """
        Analyze clustering stability using bootstrap sampling.
        
        Args:
            seq_ids: Sequences to include
            method: Clustering method to test
            n_clusters: Number of clusters
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with stability analysis results
        """
        matrix, seq_ids, kmer_list = self.profiler.get_profile_matrix(seq_ids)
        matrix_scaled = self.scaler.fit_transform(matrix)
        n_samples = len(seq_ids)
        
        # Original clustering
        if method == 'kmeans':
            original_clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'hierarchical':
            original_clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Stability analysis not implemented for {method}")
        
        original_labels = original_clusterer.fit_predict(matrix_scaled)
        
        # Bootstrap stability analysis
        jaccard_scores = []
        adjusted_rand_scores = []
        
        np.random.seed(42)
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_matrix = matrix_scaled[indices]
            
            # Cluster bootstrap sample
            if method == 'kmeans':
                bootstrap_clusterer = KMeans(n_clusters=n_clusters, random_state=i)
            else:  # hierarchical
                bootstrap_clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            
            bootstrap_labels = bootstrap_clusterer.fit_predict(bootstrap_matrix)
            
            # Map back to original indices and calculate similarity
            mapped_labels = np.full(n_samples, -1)
            for j, orig_idx in enumerate(indices):
                mapped_labels[orig_idx] = bootstrap_labels[j]
            
            # Calculate Jaccard coefficient for cluster assignments
            jaccard = self._calculate_jaccard_similarity(original_labels, mapped_labels)
            jaccard_scores.append(jaccard)
            
            # Calculate Adjusted Rand Index
            from sklearn.metrics import adjusted_rand_score
            valid_mask = mapped_labels != -1
            if np.sum(valid_mask) > 0:
                ari = adjusted_rand_score(
                    original_labels[valid_mask], 
                    mapped_labels[valid_mask]
                )
                adjusted_rand_scores.append(ari)
        
        results = {
            'mean_jaccard': np.mean(jaccard_scores),
            'std_jaccard': np.std(jaccard_scores),
            'jaccard_scores': jaccard_scores,
            'mean_ari': np.mean(adjusted_rand_scores),
            'std_ari': np.std(adjusted_rand_scores),
            'ari_scores': adjusted_rand_scores,
            'original_labels': original_labels,
            'method': method,
            'n_clusters': n_clusters,
            'n_bootstrap': n_bootstrap,
            'seq_ids': seq_ids
        }
        
        return results
    
    def _calculate_jaccard_similarity(self, labels1: np.ndarray, labels2: np.ndarray) -> float:
        """Calculate Jaccard similarity between two clusterings."""
        # Create pairwise co-occurrence matrices
        n = len(labels1)
        cooccur1 = np.zeros((n, n))
        cooccur2 = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if labels1[i] == labels1[j]:
                    cooccur1[i, j] = cooccur1[j, i] = 1
                if labels2[i] != -1 and labels2[j] != -1 and labels2[i] == labels2[j]:
                    cooccur2[i, j] = cooccur2[j, i] = 1
        
        # Calculate Jaccard coefficient
        intersection = np.sum(cooccur1 * cooccur2)
        union = np.sum((cooccur1 + cooccur2) > 0)
        
        return intersection / union if union > 0 else 0.0
