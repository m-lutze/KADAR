import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler

from .kmer_profiler import KmerProfiler
from ..analysis.statistical_methods import StatisticalAnalysis
from ..analysis.clustering import ClusteringAnalysis  
from ..analysis.anomaly_detection import AnomalyDetection
from ..analysis.ivom import IVOMAnalysis
from ..visualization.plots import ResultsVisualizer


class GenomeIslandPredictor:
    """
    Main class for predicting genome islands using various techniques.
    """
    
    def __init__(self, profiler: KmerProfiler):
        """
        Initialize predictor with a KmerProfiler instance.
        
        Args:
            profiler: KmerProfiler instance for seq management
            
        Raises:
            ValueError: If profiler is empty or invalid
        """
        if not isinstance(profiler, KmerProfiler):
            raise TypeError("profiler must be a KmerProfiler instance")
            
        if len(profiler) == 0:
            raise ValueError("profiler must contain at least one sequence")
            
        self.profiler = profiler
        self.results = {}
        self.scaler = StandardScaler()
        
        # Initialize analysis modules
        self._statistical = StatisticalAnalysis(profiler, self.scaler)
        self._clustering = ClusteringAnalysis(profiler, self.scaler)
        self._anomaly = AnomalyDetection(profiler, self.scaler)
        # Could maybe move all the vis logic out?
        self._visualizer = ResultsVisualizer()
        
    def sliding_window_analysis(self, seq_id: str, window_size: int = 1000, 
                              step_size: int = 500) -> Dict:
        """
        Perform sliding window analysis on a sequence.
        
        Args:
            seq_id: Sequence identifier
            window_size: Size of sliding window
            step_size: Step size for sliding window
            
        Returns:
            Dictionary with window analysis results
            
        Raises:
            KeyError: If seq_id not found
            ValueError: If window_size or step_size invalid
        """
        if seq_id not in self.profiler:
            raise KeyError(f"Sequence '{seq_id}' not found in profiler")
            
        if window_size <= 0 or step_size <= 0:
            raise ValueError("window_size and step_size must be positive")
            
        sequence = self.profiler.sequences[seq_id]
        
        if len(sequence) < window_size:
            raise ValueError(f"Sequence length ({len(sequence)}) is smaller than window size ({window_size})")
            
        windows = []
        positions = []
        
        # Create temporary profiler for windows
        temp_profiler = KmerProfiler(k=self.profiler.k, normalize=self.profiler.normalize)
        
        for start in range(0, len(sequence) - window_size + 1, step_size):
            end = start + window_size
            window_seq = sequence[start:end]
            window_id = f"{seq_id}_window_{start}_{end}"
            
            temp_profiler.add_sequence(window_id, window_seq)
            windows.append(window_id)
            positions.append((start, end))
        
        return {
            'windows': windows,
            'positions': positions,
            'profiler': temp_profiler,
            'parent_sequence': seq_id,
            'window_size': window_size,
            'step_size': step_size
        }
    
    def z_score_analysis(self, reference_seqs: List[str], target_seqs: List[str]) -> Dict:
        """
        Perform Z-score analysis to identify outliers.
        
        Args:
            reference_seqs: Reference (host) sequences
            target_seqs: Target sequences to analyze
            
        Returns:
            Dictionary with Z-score results
        """
        result = self._statistical.z_score_analysis(reference_seqs, target_seqs)
        self.results['z_score'] = result
        return result
    
    def pca_analysis(self, seq_ids: Optional[List[str]] = None, n_components: int = 2) -> Dict:
        """
        Perform PCA analysis on k-mer profiles.
        
        Args:
            seq_ids: Sequences to include
            n_components: Number of PCA components
            
        Returns:
            Dictionary with PCA results
        """
        result = self._statistical.pca_analysis(seq_ids, n_components)
        self.results['pca'] = result
        return result
    
    def clustering_analysis(self, seq_ids: Optional[List[str]] = None, 
                          method: str = 'dbscan', **kwargs) -> Dict:
        """
        Perform clustering analysis to identify potential genome islands.
        
        Args:
            seq_ids: Sequences to include
            method: Clustering method ('dbscan', 'kmeans')
            **kwargs: Additional parameters for clustering algorithm
            
        Returns:
            Dictionary with clustering results
        """
        result = self._clustering.run_clustering(seq_ids, method, **kwargs)
        self.results['clustering'] = result
        return result
    
    def isolation_forest_analysis(self, seq_ids: Optional[List[str]] = None, 
                                 contamination: float = 0.1) -> Dict:
        """
        Use Isolation Forest for anomaly detection.
        
        Args:
            seq_ids: Sequences to include
            contamination: Expected proportion of outliers
            
        Returns:
            Dictionary with anomaly detection results
        """
        result = self._anomaly.isolation_forest_analysis(seq_ids, contamination)
        self.results['isolation_forest'] = result
        return result
    
    def chi_square_analysis(self, reference_seqs: List[str], target_seqs: List[str]) -> Dict:
        """
        Perform chi-square test for compositional differences.
        
        Args:
            reference_seqs: Reference sequences
            target_seqs: Target sequences to test
            
        Returns:
            Dictionary with chi-square test results
        """
        result = self._statistical.chi_square_analysis(reference_seqs, target_seqs)
        self.results['chi_square'] = result
        return result
    
    def local_outlier_factor_analysis(self, seq_ids: Optional[List[str]] = None,
                                    n_neighbors: int = 20) -> Dict:
        """
        Use Local Outlier Factor for anomaly detection.
        
        Args:
            seq_ids: Sequences to include  
            n_neighbors: Number of neighbors for LOF calculation
            
        Returns:
            Dictionary with LOF results
        """
        result = self._anomaly.local_outlier_factor_analysis(seq_ids, n_neighbors)
        self.results['lof'] = result
        return result
    
    def one_class_svm_analysis(self, seq_ids: Optional[List[str]] = None,
                             nu: float = 0.1) -> Dict:
        """
        Use One-Class SVM for anomaly detection.
        
        Args:
            seq_ids: Sequences to include
            nu: An upper bound on the fraction of training errors
            
        Returns:
            Dictionary with One-Class SVM results
        """
        result = self._anomaly.one_class_svm_analysis(seq_ids, nu)
        self.results['one_class_svm'] = result
        return result
    
    def ivom_analysis(self, reference_seqs: List[str], target_seqs: Optional[List[str]] = None,
                 max_order: int = 8, min_order: int = 1,
                 threshold_percentile: float = 95.0,
                 interpolation_weights: Optional[List[float]] = None) -> Dict:
        """
        Perform IVOM (Interpolated Variable Order Motifs) analysis for HGT detection.
        
        This method implements variable-order motif analysis similar to Alien_hunter,
        exploiting compositional biases to detect horizontal gene transfer events.
        
        Args:
            reference_seqs: Reference (host) sequence IDs for background model
            target_seqs: Target sequence IDs to analyze (all sequences if None)
            max_order: Maximum k-mer order for IVOM analysis
            min_order: Minimum k-mer order for IVOM analysis
            threshold_percentile: Percentile for anomaly detection threshold  
            interpolation_weights: Custom weights for interpolating different orders
            
        Returns:
            Dictionary with IVOM analysis results
            
        Raises:
            ValueError: If reference sequences are invalid or empty
        """
        # Initialize IVOM analyzer if not exists
        if not hasattr(self, '_ivom'):
            self._ivom = IVOMAnalysis(self.profiler, self.scaler)
        
        result = self._ivom.ivom_analysis(
            reference_seqs=reference_seqs,
            target_seqs=target_seqs,
            max_order=max_order,
            min_order=min_order,
            threshold_percentile=threshold_percentile,
            interpolation_weights=interpolation_weights
        )
        
        self.results['ivom'] = result
        return result

    def ivom_sliding_window_analysis(self, seq_id: str, window_size: int = 5000,
                                    step_size: int = 2500, reference_proportion: float = 0.1,
                                    **ivom_kwargs) -> Dict:
        """
        Perform IVOM analysis on sliding windows of a sequence.
        
        Args:
            seq_id: Sequence identifier for sliding window analysis
            window_size: Size of sliding windows
            step_size: Step size for sliding windows
            reference_proportion: Proportion of windows to use as reference
            **ivom_kwargs: Additional parameters for IVOM analysis
            
        Returns:
            Dictionary with sliding window IVOM results
        """
        # Initialize IVOM analyzer if not exists
        if not hasattr(self, '_ivom'):
            self._ivom = IVOMAnalysis(self.profiler, self.scaler)
        
        result = self._ivom.sliding_window_ivom_analysis(
            seq_id=seq_id,
            window_size=window_size,
            step_size=step_size,
            reference_proportion=reference_proportion,
            **ivom_kwargs
        )
        
        self.results['ivom_sliding_window'] = result
        return result
    
    def run_comprehensive_analysis(self, reference_seqs: List[str], 
                                 target_seqs: List[str]) -> Dict[str, Dict]:
        """
        Run a comprehensive analysis using multiple methods.
        
        Args:
            reference_seqs: Reference sequences for comparison
            target_seqs: Target sequences to analyze
            
        Returns:
            Dictionary containing results from all methods
        """
        all_seqs = list(set(reference_seqs + target_seqs))
        
        comprehensive_results = {}
        
        print("Running Z-score analysis...")
        comprehensive_results['z_score'] = self.z_score_analysis(reference_seqs, target_seqs)
        
        print("Running PCA analysis...")
        comprehensive_results['pca'] = self.pca_analysis(all_seqs)
        
        print("Running clustering analysis...")
        comprehensive_results['clustering'] = self.clustering_analysis(all_seqs, method='dbscan')
        
        print("Running Isolation Forest...")
        comprehensive_results['isolation_forest'] = self.isolation_forest_analysis(all_seqs)
        
        print("Running chi-square analysis...")
        comprehensive_results['chi_square'] = self.chi_square_analysis(reference_seqs, target_seqs)
        
        print("Running LOF analysis...")
        comprehensive_results['lof'] = self.local_outlier_factor_analysis(all_seqs)
        
        try:
            comprehensive_results['ivom'] = self.ivom_analysis(reference_seqs, target_seqs)
        except Exception as e:
            print(f"   Warning: IVOM analysis failed: {e}")
            comprehensive_results['ivom'] = {'error': str(e)}
    
        return comprehensive_results
    
    def visualize_results(self, analysis_type: str, **kwargs):
        """
        Visualize analysis results.
        
        Args:
            analysis_type: Type of analysis to visualize
            **kwargs: Additional parameters for visualization
        """
        if analysis_type not in self.results:
            print(f"No results found for {analysis_type}. Run analysis first.")
            return
        
        self._visualizer.plot_results(
            analysis_type=analysis_type,
            results=self.results[analysis_type],
            pca_results=self.results.get('pca'),
            **kwargs
        )
    
    def get_summary_report(self) -> Dict:
        """
        Generate a summary report of all analyses.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_sequences': len(self.profiler.sequences),
            'kmer_size': self.profiler.k,
            'vocabulary_size': len(self.profiler.kmer_vocabulary),
            'analyses_performed': list(self.results.keys()),
            'profiler_stats': self.profiler.get_kmer_statistics()
        }
        
        # Add specific results summaries
        for analysis_type, results in self.results.items():
            try:
                if analysis_type == 'z_score':
                    summary[f'{analysis_type}_outliers'] = np.sum(results['composite_z_scores'] > 2)
                    summary[f'{analysis_type}_high_outliers'] = np.sum(results['composite_z_scores'] > 3)
                    
                elif analysis_type in ['isolation_forest', 'lof', 'one_class_svm']:
                    summary[f'{analysis_type}_anomalies'] = np.sum(results['is_anomaly'])
                    summary[f'{analysis_type}_anomaly_rate'] = np.mean(results['is_anomaly'])
                    
                elif analysis_type == 'clustering':
                    unique_labels = set(results['labels'])
                    summary[f'{analysis_type}_clusters'] = len(unique_labels) - (1 if -1 in unique_labels else 0)
                    summary[f'{analysis_type}_noise_points'] = np.sum(results['labels'] == -1)
                    if results.get('silhouette_score') is not None:
                        summary[f'{analysis_type}_silhouette'] = results['silhouette_score']
                        
                elif analysis_type == 'chi_square':
                    summary[f'{analysis_type}_significant'] = np.sum(results['p_values'] < 0.05)
                    summary[f'{analysis_type}_highly_significant'] = np.sum(results['p_values'] < 0.001)
                    
                elif analysis_type == 'pca':
                    summary[f'{analysis_type}_components'] = len(results['explained_variance_ratio'])
                    summary[f'{analysis_type}_total_variance'] = np.sum(results['explained_variance_ratio'])
                elif analysis_type in ['ivom', 'ivom_sliding_window']:

                    if 'error' not in results:
                        summary[f'{analysis_type}_anomalies'] = results.get('n_anomalies', 0)
                        summary[f'{analysis_type}_anomaly_rate'] = results.get('anomaly_rate', 0.0)
                        summary[f'{analysis_type}_threshold'] = results.get('threshold', 0.0)
                        summary[f'{analysis_type}_max_deviation'] = np.max(results.get('deviation_scores', [0]))
                        summary[f'{analysis_type}_mean_deviation'] = np.mean(results.get('deviation_scores', [0]))
                        
                        if 'ivom_params' in results:
                            params = results['ivom_params']
                            summary[f'{analysis_type}_order_range'] = f"{params['min_order']}-{params['max_order']}"
                else:
                    summary[f'{analysis_type}_error'] = results['error']
                    
            except (KeyError, TypeError, AttributeError) as e:
                summary[f'{analysis_type}_error'] = f"Error summarizing results: {e}"
        
        return summary
    
    def export_results(self, filepath: str, format: str = 'json'):
        """
        Export analysis results to file.
        
        Args:
            filepath: Output file path
            format: Export format ('json', future options?)
        """
        import json
        
        if format.lower() == 'json':
            # Convert numpy arrays to lists for JSON serialization
            export_data = {}
            for key, result in self.results.items():
                export_data[key] = {}
                for sub_key, value in result.items():
                    if isinstance(value, np.ndarray):
                        export_data[key][sub_key] = value.tolist()
                    elif hasattr(value, '__dict__'):  # Skip sklearn models
                        export_data[key][sub_key] = str(value)
                    else:
                        export_data[key][sub_key] = value
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
    
    def compare_methods(self, methods: List[str] = None) -> Dict:
        """
        Compare detection results across different methods.
        
        Args:
            methods: List of method names to compare (all available if None)
            
        Returns:
            Dictionary with comparison statistics
        """
        if methods is None:
            # Include methods that produce is_anomaly results
            methods = [method for method in self.results.keys() 
                    if 'is_anomaly' in self.results.get(method, {})]
        
        comparison = {
            'methods_compared': methods,
            'method_predictions': {},
            'overlap_analysis': {},
            'consensus_analysis': {}
        }
        
        # Collect predictions from each method
        for method in methods:
            if method in self.results and 'is_anomaly' in self.results[method]:
                comparison['method_predictions'][method] = self.results[method]['is_anomaly']
        
        # Calculate pairwise overlaps
        method_names = list(comparison['method_predictions'].keys())
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names[i+1:], i+1):
                pred1 = comparison['method_predictions'][method1]
                pred2 = comparison['method_predictions'][method2]
                
                # Calculate overlap metrics
                both_positive = np.sum(pred1 & pred2)
                either_positive = np.sum(pred1 | pred2)
                jaccard = both_positive / either_positive if either_positive > 0 else 0
                
                overlap_key = f"{method1}_vs_{method2}"
                comparison['overlap_analysis'][overlap_key] = {
                    'both_positive': both_positive,
                    'jaccard_index': jaccard,
                    'method1_only': np.sum(pred1 & ~pred2),
                    'method2_only': np.sum(pred2 & ~pred1)
                }
        
        # Consensus analysis
        if len(method_names) >= 2:
            all_predictions = np.array([comparison['method_predictions'][method] 
                                    for method in method_names])
            
            # Majority vote
            vote_counts = np.sum(all_predictions, axis=0)
            majority_threshold = len(method_names) // 2 + 1
            consensus_predictions = vote_counts >= majority_threshold
            
            # Unanimous agreement
            unanimous_predictions = vote_counts == len(method_names)
            
            comparison['consensus_analysis'] = {
                'majority_consensus': consensus_predictions,
                'unanimous_consensus': unanimous_predictions,
                'n_majority_consensus': np.sum(consensus_predictions),
                'n_unanimous_consensus': np.sum(unanimous_predictions),
                'vote_distribution': np.bincount(vote_counts)
            }
        
        return comparison
