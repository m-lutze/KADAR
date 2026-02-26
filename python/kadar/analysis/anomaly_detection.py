from typing import Dict, List, Optional

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class AnomalyDetection:
    """
    Class implementing various anomaly detection methods for compositional analysis.
    """

    def __init__(self, profiler, scaler: StandardScaler):
        """
        Initialize anomaly detection.

        Args:
            profiler: KmerProfiler instance
            scaler: StandardScaler instance
        """
        self.profiler = profiler
        self.scaler = scaler

    def isolation_forest_analysis(
        self,
        seq_ids: Optional[List[str]] = None,
        contamination: float = 0.1,
        random_state: int = 42,
        n_estimators: int = 100,
    ) -> Dict:
        """
        Use Isolation Forest for anomaly detection.

        Args:
            seq_ids: Sequences to include
            contamination: Expected proportion of outliers
            random_state: Random state for reproducibility
            n_estimators: Number of trees in the forest

        Returns:
            Dictionary with Isolation Forest results
        """
        matrix, seq_ids, kmer_list = self.profiler.get_profile_matrix(seq_ids)
        matrix_scaled = self.scaler.fit_transform(matrix)

        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
        )
        anomaly_labels = iso_forest.fit_predict(matrix_scaled)
        anomaly_scores = iso_forest.score_samples(matrix_scaled)

        # Convert labels: -1 (anomaly) to True, 1 (normal) to False
        is_anomaly = anomaly_labels == -1

        # Calculate additional metrics
        decision_scores = iso_forest.decision_function(matrix_scaled)

        # Calculate anomaly statistics
        anomaly_stats = self._calculate_anomaly_statistics(
            matrix_scaled, is_anomaly, seq_ids, kmer_list
        )

        results = {
            'is_anomaly': is_anomaly,
            'anomaly_scores': anomaly_scores,
            'decision_scores': decision_scores,
            'anomaly_labels': anomaly_labels,
            'seq_ids': seq_ids,
            'model': iso_forest,
            'contamination': contamination,
            'n_anomalies': np.sum(is_anomaly),
            'anomaly_rate': np.mean(is_anomaly),
            'method': 'isolation_forest',
            'anomaly_stats': anomaly_stats,
        }

        return results

    def local_outlier_factor_analysis(
        self,
        seq_ids: Optional[List[str]] = None,
        n_neighbors: int = 20,
        contamination: float = 0.1,
    ) -> Dict:
        """
        Use Local Outlier Factor for anomaly detection.

        Args:
            seq_ids: Sequences to include
            n_neighbors: Number of neighbors for LOF calculation
            contamination: Expected proportion of outliers

        Returns:
            Dictionary with LOF results
        """
        matrix, seq_ids, kmer_list = self.profiler.get_profile_matrix(seq_ids)
        matrix_scaled = self.scaler.fit_transform(matrix)

        # Adjust n_neighbors if necessary
        n_neighbors = min(n_neighbors, len(seq_ids) - 1)

        # Fit Local Outlier Factor
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        anomaly_labels = lof.fit_predict(matrix_scaled)

        # Get LOF scores (negative values indicate outliers)
        lof_scores = lof.negative_outlier_factor_

        # Convert labels: -1 (anomaly) to True, 1 (normal) to False
        is_anomaly = anomaly_labels == -1

        # Calculate additional LOF-specific metrics
        local_densities = self._calculate_local_densities(matrix_scaled, n_neighbors)

        # Calculate anomaly statistics
        anomaly_stats = self._calculate_anomaly_statistics(
            matrix_scaled, is_anomaly, seq_ids, kmer_list
        )

        results = {
            'is_anomaly': is_anomaly,
            'lof_scores': lof_scores,
            'local_densities': local_densities,
            'anomaly_labels': anomaly_labels,
            'seq_ids': seq_ids,
            'model': lof,
            'n_neighbors': n_neighbors,
            'contamination': contamination,
            'n_anomalies': np.sum(is_anomaly),
            'anomaly_rate': np.mean(is_anomaly),
            'method': 'lof',
            'anomaly_stats': anomaly_stats,
        }

        return results

    def one_class_svm_analysis(
        self,
        seq_ids: Optional[List[str]] = None,
        nu: float = 0.1,
        kernel: str = 'rbf',
        gamma: str = 'scale',
    ) -> Dict:
        """
        Use One-Class SVM for anomaly detection.

        Args:
            seq_ids: Sequences to include
            nu: An upper bound on the fraction of training errors
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            gamma: Kernel coefficient

        Returns:
            Dictionary with One-Class SVM results
        """
        matrix, seq_ids, kmer_list = self.profiler.get_profile_matrix(seq_ids)
        matrix_scaled = self.scaler.fit_transform(matrix)

        # Fit One-Class SVM
        oc_svm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        anomaly_labels = oc_svm.fit_predict(matrix_scaled)

        # Get decision scores
        decision_scores = oc_svm.decision_function(matrix_scaled)

        # Convert labels: -1 (anomaly) to True, 1 (normal) to False
        is_anomaly = anomaly_labels == -1

        # Calculate anomaly statistics
        anomaly_stats = self._calculate_anomaly_statistics(
            matrix_scaled, is_anomaly, seq_ids, kmer_list
        )

        results = {
            'is_anomaly': is_anomaly,
            'decision_scores': decision_scores,
            'anomaly_labels': anomaly_labels,
            'seq_ids': seq_ids,
            'model': oc_svm,
            'nu': nu,
            'kernel': kernel,
            'gamma': gamma,
            'n_anomalies': np.sum(is_anomaly),
            'anomaly_rate': np.mean(is_anomaly),
            'method': 'one_class_svm',
            'anomaly_stats': anomaly_stats,
        }

        return results

    def elliptic_envelope_analysis(
        self,
        seq_ids: Optional[List[str]] = None,
        contamination: float = 0.1,
        support_fraction: Optional[float] = None,
    ) -> Dict:
        """
        Use Elliptic Envelope (Robust Covariance) for anomaly detection.

        Args:
            seq_ids: Sequences to include
            contamination: Expected proportion of outliers
            support_fraction: Fraction of points included in support

        Returns:
            Dictionary with Elliptic Envelope results
        """
        matrix, seq_ids, kmer_list = self.profiler.get_profile_matrix(seq_ids)
        matrix_scaled = self.scaler.fit_transform(matrix)

        # Fit Elliptic Envelope
        elliptic_env = EllipticEnvelope(
            contamination=contamination,
            support_fraction=support_fraction,
            random_state=42,
        )
        anomaly_labels = elliptic_env.fit_predict(matrix_scaled)

        # Get Mahalanobis distances
        mahalanobis_distances = elliptic_env.mahalanobis(matrix_scaled)
        decision_scores = elliptic_env.decision_function(matrix_scaled)

        # Convert labels: -1 (anomaly) to True, 1 (normal) to False
        is_anomaly = anomaly_labels == -1

        # Calculate anomaly statistics
        anomaly_stats = self._calculate_anomaly_statistics(
            matrix_scaled, is_anomaly, seq_ids, kmer_list
        )

        results = {
            'is_anomaly': is_anomaly,
            'mahalanobis_distances': mahalanobis_distances,
            'decision_scores': decision_scores,
            'anomaly_labels': anomaly_labels,
            'seq_ids': seq_ids,
            'model': elliptic_env,
            'contamination': contamination,
            'n_anomalies': np.sum(is_anomaly),
            'anomaly_rate': np.mean(is_anomaly),
            'method': 'elliptic_envelope',
            'anomaly_stats': anomaly_stats,
        }

        return results

    def ensemble_anomaly_detection(
        self,
        seq_ids: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
        contamination: float = 0.1,
    ) -> Dict:
        """
        Ensemble anomaly detection using multiple methods.

        Args:
            seq_ids: Sequences to include
            methods: List of methods to ensemble
            contamination: Expected proportion of outliers

        Returns:
            Dictionary with ensemble results
        """
        if methods is None:
            methods = ['isolation_forest', 'lof', 'one_class_svm']

        matrix, seq_ids, kmer_list = self.profiler.get_profile_matrix(seq_ids)

        # Run individual methods
        individual_results = {}
        anomaly_votes = np.zeros(len(seq_ids))
        decision_scores_ensemble = np.zeros(len(seq_ids))

        for method in methods:
            if method == 'isolation_forest':
                result = self.isolation_forest_analysis(seq_ids, contamination)
                individual_results[method] = result
                anomaly_votes += result['is_anomaly'].astype(int)
                decision_scores_ensemble += result['anomaly_scores']

            elif method == 'lof':
                result = self.local_outlier_factor_analysis(
                    seq_ids, contamination=contamination
                )
                individual_results[method] = result
                anomaly_votes += result['is_anomaly'].astype(int)
                decision_scores_ensemble += -result[
                    'lof_scores'
                ]  # LOF scores are negative

            elif method == 'one_class_svm':
                result = self.one_class_svm_analysis(seq_ids, nu=contamination)
                individual_results[method] = result
                anomaly_votes += result['is_anomaly'].astype(int)
                decision_scores_ensemble += -result[
                    'decision_scores'
                ]  # Make consistent

            elif method == 'elliptic_envelope':
                result = self.elliptic_envelope_analysis(seq_ids, contamination)
                individual_results[method] = result
                anomaly_votes += result['is_anomaly'].astype(int)
                decision_scores_ensemble += -result['decision_scores']

        # Ensemble decision: majority vote
        ensemble_anomalies = anomaly_votes >= (len(methods) / 2)

        # Ensemble scores: average of normalized scores
        ensemble_scores = decision_scores_ensemble / len(methods)

        # Calculate consensus strength
        consensus_strength = anomaly_votes / len(methods)

        # Calculate anomaly statistics for ensemble
        anomaly_stats = self._calculate_anomaly_statistics(
            self.scaler.fit_transform(matrix), ensemble_anomalies, seq_ids, kmer_list
        )

        results = {
            'is_anomaly': ensemble_anomalies,
            'anomaly_votes': anomaly_votes,
            'consensus_strength': consensus_strength,
            'ensemble_scores': ensemble_scores,
            'individual_results': individual_results,
            'seq_ids': seq_ids,
            'methods': methods,
            'n_anomalies': np.sum(ensemble_anomalies),
            'anomaly_rate': np.mean(ensemble_anomalies),
            'method': 'ensemble',
            'anomaly_stats': anomaly_stats,
        }

        return results

    def _calculate_local_densities(
        self, matrix: np.ndarray, n_neighbors: int
    ) -> np.ndarray:
        """Calculate local density for each point."""
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 to exclude self
        nbrs.fit(matrix)
        distances, indices = nbrs.kneighbors(matrix)

        # Calculate local density as inverse of mean distance to k-nearest neighbors
        # Exclude the first distance (distance to self = 0)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        local_densities = 1.0 / (
            mean_distances + 1e-10
        )  # Add small epsilon to avoid division by zero

        return local_densities

    def _calculate_anomaly_statistics(
        self,
        matrix: np.ndarray,
        is_anomaly: np.ndarray,
        seq_ids: List[str],
        kmer_list: List[str],
    ) -> Dict:
        """Calculate detailed statistics for anomalies vs normal sequences."""
        normal_mask = ~is_anomaly
        anomaly_mask = is_anomaly

        stats = {}

        if np.sum(anomaly_mask) > 0 and np.sum(normal_mask) > 0:
            normal_data = matrix[normal_mask]
            anomaly_data = matrix[anomaly_mask]

            # Basic statistics
            stats['normal'] = {
                'count': np.sum(normal_mask),
                'mean_profile': np.mean(normal_data, axis=0),
                'std_profile': np.std(normal_data, axis=0),
                'sequences': [
                    seq_ids[i] for i in range(len(seq_ids)) if normal_mask[i]
                ],
            }

            stats['anomaly'] = {
                'count': np.sum(anomaly_mask),
                'mean_profile': np.mean(anomaly_data, axis=0),
                'std_profile': np.std(anomaly_data, axis=0),
                'sequences': [
                    seq_ids[i] for i in range(len(seq_ids)) if anomaly_mask[i]
                ],
            }

            # K-mer differences
            mean_diff = (
                stats['anomaly']['mean_profile'] - stats['normal']['mean_profile']
            )

            # Find most discriminative k-mers
            discriminative_kmers = []
            for i, kmer in enumerate(kmer_list):
                normal_mean = stats['normal']['mean_profile'][i]
                anomaly_mean = stats['anomaly']['mean_profile'][i]

                if stats['normal']['std_profile'][i] > 0:
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        (
                            stats['normal']['std_profile'][i] ** 2
                            + stats['anomaly']['std_profile'][i] ** 2
                        )
                        / 2
                    )
                    cohens_d = (
                        (anomaly_mean - normal_mean) / pooled_std
                        if pooled_std > 0
                        else 0
                    )

                    discriminative_kmers.append(
                        {
                            'kmer': kmer,
                            'normal_mean': normal_mean,
                            'anomaly_mean': anomaly_mean,
                            'difference': mean_diff[i],
                            'effect_size': cohens_d,
                            'fold_change': anomaly_mean / normal_mean
                            if normal_mean > 0
                            else float('inf'),
                        }
                    )

            # Sort by absolute effect size
            discriminative_kmers.sort(key=lambda x: abs(x['effect_size']), reverse=True)
            stats['discriminative_kmers'] = discriminative_kmers[:20]  # Top 20

            # Statistical tests for each k-mer
            p_values = []
            for i in range(len(kmer_list)):
                try:
                    _, p_val = stats.ttest_ind(normal_data[:, i], anomaly_data[:, i])
                    p_values.append(p_val)
                except:
                    p_values.append(1.0)

            stats['kmer_p_values'] = np.array(p_values)
            stats['significant_kmers'] = np.sum(np.array(p_values) < 0.05)

        else:
            # Handle edge cases
            stats['normal'] = {'count': np.sum(normal_mask)}
            stats['anomaly'] = {'count': np.sum(anomaly_mask)}
            stats['discriminative_kmers'] = []

        return stats

    def threshold_analysis(
        self,
        seq_ids: Optional[List[str]] = None,
        method: str = 'isolation_forest',
        thresholds: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Analyze anomaly detection performance across different thresholds.

        Args:
            seq_ids: Sequences to include
            method: Anomaly detection method
            thresholds: Array of thresholds to test

        Returns:
            Dictionary with threshold analysis results
        """
        # Run the specified method to get decision scores
        if method == 'isolation_forest':
            result = self.isolation_forest_analysis(seq_ids, contamination=0.1)
            decision_scores = result['anomaly_scores']
        elif method == 'lof':
            result = self.local_outlier_factor_analysis(seq_ids, contamination=0.1)
            decision_scores = -result['lof_scores']
        elif method == 'one_class_svm':
            result = self.one_class_svm_analysis(seq_ids, nu=0.1)
            decision_scores = -result['decision_scores']
        else:
            raise ValueError(f'Unsupported method for threshold analysis: {method}')

        if thresholds is None:
            # Create thresholds based on score percentiles
            thresholds = np.percentile(decision_scores, np.linspace(5, 95, 20))

        threshold_results = []

        for threshold in thresholds:
            predicted_anomalies = decision_scores >= threshold
            n_anomalies = np.sum(predicted_anomalies)
            anomaly_rate = np.mean(predicted_anomalies)

            threshold_results.append(
                {
                    'threshold': threshold,
                    'n_anomalies': n_anomalies,
                    'anomaly_rate': anomaly_rate,
                    'predicted_anomalies': predicted_anomalies,
                }
            )

        results = {
            'thresholds': thresholds,
            'threshold_results': threshold_results,
            'decision_scores': decision_scores,
            'method': method,
            'seq_ids': seq_ids,
        }

        return results

    def cross_validation_analysis(
        self,
        seq_ids: Optional[List[str]] = None,
        method: str = 'isolation_forest',
        n_folds: int = 5,
    ) -> Dict:
        """
        Perform cross-validation analysis for anomaly detection stability.

        Args:
            seq_ids: Sequences to include
            method: Anomaly detection method
            n_folds: Number of cross-validation folds

        Returns:
            Dictionary with cross-validation results
        """
        matrix, seq_ids, kmer_list = self.profiler.get_profile_matrix(seq_ids)
        n_samples = len(seq_ids)

        # Create fold indices
        np.random.seed(42)
        fold_indices = np.random.permutation(n_samples)
        fold_size = n_samples // n_folds

        cv_results = []
        consistency_matrix = np.zeros((n_samples, n_samples))

        for fold in range(n_folds):
            # Create train/test splits
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_samples

            test_indices = fold_indices[start_idx:end_idx]
            train_indices = np.concatenate(
                [fold_indices[:start_idx], fold_indices[end_idx:]]
            )

            train_seq_ids = [seq_ids[i] for i in train_indices]
            test_seq_ids = [seq_ids[i] for i in test_indices]

            # Train on training set and predict on test set
            if method == 'isolation_forest':
                train_matrix, _, _ = self.profiler.get_profile_matrix(train_seq_ids)
                test_matrix, _, _ = self.profiler.get_profile_matrix(test_seq_ids)

                train_scaled = self.scaler.fit_transform(train_matrix)
                test_scaled = self.scaler.transform(test_matrix)

                model = IsolationForest(contamination=0.1, random_state=42)
                model.fit(train_scaled)

                test_predictions = model.predict(test_scaled) == -1
                test_scores = model.score_samples(test_scaled)

            elif method == 'lof':
                # LOF needs all data together, so we'll use different approach
                all_matrix = self.scaler.fit_transform(matrix)
                model = LocalOutlierFactor(contamination=0.1)
                all_predictions = model.fit_predict(all_matrix) == -1

                test_predictions = all_predictions[test_indices]
                test_scores = model.negative_outlier_factor_[test_indices]

            else:
                raise ValueError(f'Cross-validation not implemented for {method}')

            # Store results
            fold_result = {
                'fold': fold,
                'train_indices': train_indices,
                'test_indices': test_indices,
                'test_predictions': test_predictions,
                'test_scores': test_scores,
                'n_anomalies': np.sum(test_predictions),
            }
            cv_results.append(fold_result)

            # Update consistency matrix
            for i, test_idx in enumerate(test_indices):
                for j, other_test_idx in enumerate(test_indices):
                    if test_predictions[i] == test_predictions[j]:
                        consistency_matrix[test_idx, other_test_idx] = 1

        # Calculate overall consistency metrics
        mean_anomaly_rate = np.mean(
            [
                result['n_anomalies'] / len(result['test_indices'])
                for result in cv_results
            ]
        )
        std_anomaly_rate = np.std(
            [
                result['n_anomalies'] / len(result['test_indices'])
                for result in cv_results
            ]
        )

        results = {
            'cv_results': cv_results,
            'consistency_matrix': consistency_matrix,
            'mean_anomaly_rate': mean_anomaly_rate,
            'std_anomaly_rate': std_anomaly_rate,
            'method': method,
            'n_folds': n_folds,
            'seq_ids': seq_ids,
        }

        return results
