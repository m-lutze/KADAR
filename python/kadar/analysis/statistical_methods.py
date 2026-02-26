from typing import Dict, List, Optional

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class StatisticalAnalysis:
    """
    Class implementing statistical methods for compositional analysis.

    Z-score analysis, PCA, and chi-square tests.
    """

    def __init__(self, profiler, scaler: StandardScaler):
        """
        Initialize statistical analysis.

        Args:
            profiler: KmerProfiler instance
            scaler: StandardScaler instance (from sklearn)
        """
        self.profiler = profiler
        self.scaler = scaler

    def z_score_analysis(
        self, reference_seqs: List[str], target_seqs: List[str]
    ) -> Dict:
        """
        Perform Z-score analysis to identify outliers.

        reference_seqs: Reference (host) sequences
        target_seqs: Target sequences to analyze
        """
        if not reference_seqs:
            raise ValueError('reference_seqs cannot be empty')
        if not target_seqs:
            raise ValueError('target_seqs cannot be empty')

        # Validate sequence IDs
        all_seq_ids = set(self.profiler.sequences.keys())
        invalid_refs = [
            seq_id for seq_id in reference_seqs if seq_id not in all_seq_ids
        ]
        invalid_targets = [
            seq_id for seq_id in target_seqs if seq_id not in all_seq_ids
        ]

        if invalid_refs:
            raise ValueError(f'Invalid reference sequence IDs: {invalid_refs}')
        if invalid_targets:
            raise ValueError(f'Invalid target sequence IDs: {invalid_targets}')

        # Get profiles for reference and target sequences
        ref_matrix, _, kmer_list = self.profiler.get_profile_matrix(reference_seqs)
        target_matrix, _, _ = self.profiler.get_profile_matrix(target_seqs)

        # Calculate reference statistics
        ref_means = np.mean(ref_matrix, axis=0)
        ref_stds = np.std(ref_matrix, axis=0, ddof=1)

        # Avoid division by zero
        ref_stds[ref_stds == 0] = 1e-10

        # Calculate Z-scores for targets
        z_scores = (target_matrix - ref_means) / ref_stds

        # Calculate composite Z-score (Euclidean norm)
        composite_z_scores = np.linalg.norm(z_scores, axis=1)

        # Calculate Mahalanobis-like distance
        try:
            cov_matrix = np.cov(ref_matrix.T)
            cov_inv = np.linalg.pinv(cov_matrix)
            mahalanobis_distances = []

            for target_profile in target_matrix:
                diff = target_profile - ref_means
                maha_dist = np.sqrt(diff.T @ cov_inv @ diff)
                mahalanobis_distances.append(maha_dist)

            mahalanobis_distances = np.array(mahalanobis_distances)
        except (np.linalg.LinAlgError, ValueError):
            # Fallback if covariance matrix is singular
            mahalanobis_distances = composite_z_scores.copy()

        results = {
            'z_scores': z_scores,
            'composite_z_scores': composite_z_scores,
            'mahalanobis_distances': mahalanobis_distances,
            'target_seqs': target_seqs,
            'reference_seqs': reference_seqs,
            'kmer_list': kmer_list,
            'reference_means': ref_means,
            'reference_stds': ref_stds,
        }

        return results

    def pca_analysis(
        self, seq_ids: Optional[List[str]] = None, n_components: int = 2
    ) -> Dict:
        """
        Perform PCA analysis on k-mer profiles.

        Args:
            seq_ids: Sequences to include
            n_components: Number of PCA components

        Returns:
            Dictionary with PCA results
        """
        if n_components < 1:
            raise ValueError('n_components must be at least 1')

        matrix, seq_ids, kmer_list = self.profiler.get_profile_matrix(seq_ids)

        if matrix.shape[0] < n_components:
            n_components = matrix.shape[0]
            print(
                f'Warning: Reduced n_components to {n_components} (number of sequences)'
            )

        # Standardize the data
        matrix_scaled = self.scaler.fit_transform(matrix)

        # Perform PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(matrix_scaled)

        # Calculate loadings (contribution of each k-mer to each PC)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        results = {
            'components': components,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'explained_variance': pca.explained_variance_,
            'loadings': loadings,
            'seq_ids': seq_ids,
            'kmer_list': kmer_list,
            'pca_model': pca,
            'n_components': n_components,
        }

        return results

    def chi_square_analysis(
        self, reference_seqs: List[str], target_seqs: List[str]
    ) -> Dict:
        """
        Perform chi-square test for compositional differences.

        reference_seqs: Reference sequences
        target_seqs: Target sequences to test

        """
        if not reference_seqs:
            raise ValueError('reference_seqs cannot be empty')
        if not target_seqs:
            raise ValueError('target_seqs cannot be empty')

        ref_matrix, _, kmer_list = self.profiler.get_profile_matrix(reference_seqs)
        target_matrix, _, _ = self.profiler.get_profile_matrix(target_seqs)

        # Calculate expected frequencies from reference
        ref_expected = np.mean(ref_matrix, axis=0)

        chi2_stats = []
        p_values = []
        individual_contributions = []

        for i, _target_seq in enumerate(target_seqs):
            observed = target_matrix[i]

            # Convert to pseudo-counts to avoid issues with normalized data
            scaling_factor = 10000
            observed_counts = observed * scaling_factor
            expected_counts = ref_expected * scaling_factor

            # Avoid zero expected counts
            mask = expected_counts > 0

            if np.sum(mask) > 1:  # Need at least 2 categories
                try:
                    chi2, p_val = stats.chisquare(
                        observed_counts[mask], expected_counts[mask]
                    )

                    # Calculate individual k-mer contributions
                    contributions = (
                        observed_counts[mask] - expected_counts[mask]
                    ) ** 2 / expected_counts[mask]

                    chi2_stats.append(chi2)
                    p_values.append(p_val)
                    individual_contributions.append(contributions)

                except (ValueError, ZeroDivisionError):
                    chi2_stats.append(0)
                    p_values.append(1)
                    individual_contributions.append(np.zeros(np.sum(mask)))
            else:
                chi2_stats.append(0)
                p_values.append(1)
                individual_contributions.append(np.array([]))

        # Adjust p-values for multiple testing (Benjamini-Hochberg)
        adjusted_p_values = self._benjamini_hochberg_correction(np.array(p_values))

        results = {
            'chi2_stats': np.array(chi2_stats),
            'p_values': np.array(p_values),
            'adjusted_p_values': adjusted_p_values,
            'individual_contributions': individual_contributions,
            'target_seqs': target_seqs,
            'reference_seqs': reference_seqs,
            'kmer_list': kmer_list,
        }

        return results

    def kolmogorov_smirnov_analysis(
        self, reference_seqs: List[str], target_seqs: List[str]
    ) -> Dict:
        """
        Perform Kolmogorov-Smirnov test for each k-mer distribution.

        reference_seqs: Reference sequences
        target_seqs: Target sequences to test

        """
        ref_matrix, _, kmer_list = self.profiler.get_profile_matrix(reference_seqs)
        target_matrix, _, _ = self.profiler.get_profile_matrix(target_seqs)

        ks_stats = []
        p_values = []

        # Test each k-mer distribution
        for j, _kmer in enumerate(kmer_list):
            ref_dist = ref_matrix[:, j]
            target_dist = target_matrix[:, j]

            try:
                ks_stat, p_val = stats.ks_2samp(ref_dist, target_dist)
                ks_stats.append(ks_stat)
                p_values.append(p_val)
            except ValueError:
                ks_stats.append(0)
                p_values.append(1)

        # Adjust for multiple testing
        adjusted_p_values = self._benjamini_hochberg_correction(np.array(p_values))

        results = {
            'ks_stats': np.array(ks_stats),
            'p_values': np.array(p_values),
            'adjusted_p_values': adjusted_p_values,
            'kmer_list': kmer_list,
            'reference_seqs': reference_seqs,
            'target_seqs': target_seqs,
        }

        return results

    def permutation_test(
        self,
        reference_seqs: List[str],
        target_seqs: List[str],
        n_permutations: int = 1000,
        statistic: str = 'mean_difference',
    ) -> Dict:
        """
        Perform permutation test for compositional differences.

        reference_seqs: Reference sequences
        target_seqs: Target sequences
        n_permutations: Number of permutations
        statistic: Test statistic ('mean_difference', 'variance_ratio')

        """
        ref_matrix, _, kmer_list = self.profiler.get_profile_matrix(reference_seqs)
        target_matrix, _, _ = self.profiler.get_profile_matrix(target_seqs)

        # Combine all data
        all_matrix = np.vstack([ref_matrix, target_matrix])
        n_ref = len(reference_seqs)
        n_target = len(target_seqs)
        n_total = n_ref + n_target

        # Calculate observed statistic
        if statistic == 'mean_difference':
            observed_stat = np.mean(target_matrix, axis=0) - np.mean(ref_matrix, axis=0)
        elif statistic == 'variance_ratio':
            observed_stat = np.var(target_matrix, axis=0) / (
                np.var(ref_matrix, axis=0) + 1e-10
            )
        else:
            raise ValueError("statistic must be 'mean_difference' or 'variance_ratio'")

        # Permutation testing
        permuted_stats = []
        np.random.seed(42)  # For reproducibility

        for _ in range(n_permutations):
            # Randomly permute group labels
            indices = np.random.permutation(n_total)
            perm_ref = all_matrix[indices[:n_ref]]
            perm_target = all_matrix[indices[n_ref:]]

            if statistic == 'mean_difference':
                perm_stat = np.mean(perm_target, axis=0) - np.mean(perm_ref, axis=0)
            else:  # variance_ratio
                perm_stat = np.var(perm_target, axis=0) / (
                    np.var(perm_ref, axis=0) + 1e-10
                )

            permuted_stats.append(perm_stat)

        permuted_stats = np.array(permuted_stats)

        # Calculate p-values
        if statistic == 'mean_difference':
            # Two-tailed test
            p_values = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat), axis=0)
        else:  # variance_ratio
            # One-tailed test (assuming we're looking for increased variance)
            p_values = np.mean(permuted_stats >= observed_stat, axis=0)

        results = {
            'observed_statistic': observed_stat,
            'permuted_statistics': permuted_stats,
            'p_values': p_values,
            'n_permutations': n_permutations,
            'statistic_type': statistic,
            'kmer_list': kmer_list,
            'reference_seqs': reference_seqs,
            'target_seqs': target_seqs,
        }

        return results

    def _benjamini_hochberg_correction(
        self, p_values: np.ndarray, alpha: float = 0.05
    ) -> np.ndarray:
        """
        Apply Benjamini-Hochberg multiple testing correction.

        TD: Maybe remove this?

        p_values: Array of p-values
        alpha: Family-wise error rate

        """
        n = len(p_values)
        if n == 0:
            return p_values

        # Sort p-values and keep track of original indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        # Apply BH correction
        adjusted_p = np.zeros(n)
        for i in range(n):
            adjusted_p[i] = min(sorted_p[i] * n / (i + 1), 1.0)

        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            adjusted_p[i] = min(adjusted_p[i], adjusted_p[i + 1])

        # Restore original order
        result = np.zeros(n)
        result[sorted_indices] = adjusted_p

        return result

    def correlation_analysis(self, seq_ids: Optional[List[str]] = None) -> Dict:
        """
        Analyze correlations between k-mers.

        seq_ids: Sequences to include
        """
        matrix, seq_ids, kmer_list = self.profiler.get_profile_matrix(seq_ids)

        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(matrix.T)

        # Find highly correlated k-mer pairs
        high_corr_pairs = []
        n_kmers = len(kmer_list)

        for i in range(n_kmers):
            for j in range(i + 1, n_kmers):
                corr = correlation_matrix[i, j]
                if abs(corr) > 0.8:  # High correlation threshold
                    high_corr_pairs.append(
                        {
                            'kmer1': kmer_list[i],
                            'kmer2': kmer_list[j],
                            'correlation': corr,
                        }
                    )

        # Sort by absolute correlation
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

        results = {
            'correlation_matrix': correlation_matrix,
            'kmer_list': kmer_list,
            'high_correlation_pairs': high_corr_pairs,
            'seq_ids': seq_ids,
        }

        return results
