import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
from scipy.cluster.hierarchy import dendrogram
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings('ignore')


class ResultsVisualizer:
    """
    Class for creating visualizations of analysis results.

    Largely PCA plots, clustering visualizations, and statistical plots
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: tuple = (12, 8)):
        """
        Initialize visualizer with plotting parameters.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
            
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 10)
        
    def plot_results(self, analysis_type: str, results: Dict, 
                    pca_results: Optional[Dict] = None, **kwargs):
        """
        Main plotting function that routes to specific plot types.
        
        Args:
            analysis_type: Type of analysis to visualize
            results: Results dictionary from analysis
            pca_results: PCA results for dimensional reduction plots
            **kwargs: Additional plotting parameters
        """
        plot_method = getattr(self, f'_plot_{analysis_type}', None)
        
        if plot_method:
            plot_method(results, pca_results=pca_results, **kwargs)
        else:
            print(f"No visualization method available for {analysis_type}")
    
    def _plot_pca(self, results: Dict, **kwargs):
        """Plot PCA results."""
        components = results['components']
        explained_var = results['explained_variance_ratio']
        seq_ids = results['seq_ids']
        
        n_components = components.shape[1]
        
        if n_components >= 2:
            fig, axes = plt.subplots(1, 2 if n_components >= 3 else 1, 
                                   figsize=(15 if n_components >= 3 else 8, 6))
            if n_components == 2:
                axes = [axes]
            
            # PC1 vs PC2
            ax = axes[0] if n_components >= 3 else axes
            scatter = ax.scatter(components[:, 0], components[:, 1], 
                               alpha=0.7, s=60, c=range(len(seq_ids)), 
                               cmap='viridis')
            ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
            ax.set_title('PCA: PC1 vs PC2')
            ax.grid(True, alpha=0.3)
            
            # Add labels if requested
            if kwargs.get('show_labels', False):
                for i, seq_id in enumerate(seq_ids):
                    ax.annotate(seq_id, (components[i, 0], components[i, 1]),
                              xytext=(5, 5), textcoords='offset points', 
                              fontsize=8, alpha=0.7)
            
            # PC1 vs PC3 if available
            if n_components >= 3:
                ax2 = axes[1]
                scatter2 = ax2.scatter(components[:, 0], components[:, 2], 
                                     alpha=0.7, s=60, c=range(len(seq_ids)), 
                                     cmap='viridis')
                ax2.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
                ax2.set_ylabel(f'PC3 ({explained_var[2]:.1%} variance)')
                ax2.set_title('PCA: PC1 vs PC3')
                ax2.grid(True, alpha=0.3)
        
        else:
            # 1D PCA
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.scatter(components[:, 0], np.zeros(len(components)), 
                      alpha=0.7, s=60, c=range(len(seq_ids)), cmap='viridis')
            ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
            ax.set_title('PCA: PC1')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Variance explained plot
        if len(explained_var) > 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Individual variance explained
            ax1.bar(range(1, len(explained_var) + 1), explained_var)
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Variance Explained')
            ax1.set_title('Variance Explained by Each PC')
            ax1.set_xticks(range(1, len(explained_var) + 1))
            
            # Cumulative variance explained
            cumulative_var = np.cumsum(explained_var)
            ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'o-')
            ax2.set_xlabel('Number of Components')
            ax2.set_ylabel('Cumulative Variance Explained')
            ax2.set_title('Cumulative Variance Explained')
            ax2.set_xticks(range(1, len(cumulative_var) + 1))
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def _plot_clustering(self, results: Dict, pca_results: Optional[Dict] = None, **kwargs):
        """Plot clustering results."""
        labels = results['labels']
        seq_ids = results['seq_ids']
        method = results['method']
        
        # If PCA results not provided, we can't create scatter plots
        if pca_results is None:
            print("PCA results needed for clustering visualization")
            return
        
        components = pca_results['components']
        explained_var = pca_results['explained_variance_ratio']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Clustering in PC space
        unique_labels = set(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:  # Noise points (for DBSCAN)
                axes[0].scatter(components[mask, 0], components[mask, 1], 
                              c='black', marker='x', s=50, alpha=0.6, label='Noise')
            else:
                axes[0].scatter(components[mask, 0], components[mask, 1], 
                              c=[colors[i]], s=60, alpha=0.7, 
                              label=f'Cluster {label}')
        
        axes[0].set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
        axes[0].set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
        axes[0].set_title(f'{method.upper()} Clustering Results')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cluster size distribution
        cluster_sizes = [np.sum(labels == label) for label in unique_labels if label != -1]
        cluster_labels = [f'Cluster {label}' for label in unique_labels if label != -1]
        
        if cluster_sizes:
            axes[1].bar(cluster_labels, cluster_sizes)
            axes[1].set_xlabel('Cluster')
            axes[1].set_ylabel('Number of Sequences')
            axes[1].set_title('Cluster Size Distribution')
            axes[1].tick_params(axis='x', rotation=45)
        
        # Add silhouette score if available
        if 'silhouette_score' in results and results['silhouette_score'] is not None:
            plt.suptitle(f'Silhouette Score: {results["silhouette_score"]:.3f}')
        
        plt.tight_layout()
        plt.show()
        
        # Additional clustering-specific plots
        if method == 'hierarchical' and 'linkage_matrix' in results:
            self._plot_dendrogram(results)
        elif method == 'kmeans' and 'cluster_centers' in results:
            self._plot_cluster_centers(results, pca_results)
    
    def _plot_z_score(self, results: Dict, **kwargs):
        """Plot Z-score analysis results."""
        composite_z_scores = results['composite_z_scores']
        target_seqs = results['target_seqs']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Composite Z-scores bar plot
        axes[0, 0].bar(range(len(target_seqs)), composite_z_scores)
        axes[0, 0].axhline(y=2, color='red', linestyle='--', label='Z=2 threshold')
        axes[0, 0].axhline(y=3, color='darkred', linestyle='--', label='Z=3 threshold')
        axes[0, 0].set_xlabel('Target Sequences')
        axes[0, 0].set_ylabel('Composite Z-score')
        axes[0, 0].set_title('Composite Z-scores')
        axes[0, 0].legend()
        axes[0, 0].set_xticks(range(len(target_seqs)))
        axes[0, 0].set_xticklabels(target_seqs, rotation=45, ha='right')
        
        # Z-score distribution
        axes[0, 1].hist(composite_z_scores, bins=min(10, len(composite_z_scores)), 
                       alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=2, color='red', linestyle='--', label='Z=2')
        axes[0, 1].axvline(x=3, color='darkred', linestyle='--', label='Z=3')
        axes[0, 1].set_xlabel('Composite Z-score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Z-score Distribution')
        axes[0, 1].legend()
        
        # Individual Z-scores heatmap
        z_scores = results['z_scores']
        if z_scores.shape[1] > 20:  # Show top 20 k-mers if too many
            top_indices = np.argsort(np.var(z_scores, axis=0))[-20:]
            z_scores_plot = z_scores[:, top_indices]
            kmer_labels = [results['kmer_list'][i] for i in top_indices]
        else:
            z_scores_plot = z_scores
            kmer_labels = results['kmer_list']
        
        im = axes[1, 0].imshow(z_scores_plot.T, aspect='auto', cmap='RdBu_r', 
                              vmin=-3, vmax=3)
        axes[1, 0].set_xlabel('Target Sequences')
        axes[1, 0].set_ylabel('K-mers')
        axes[1, 0].set_title('Individual K-mer Z-scores')
        axes[1, 0].set_xticks(range(len(target_seqs)))
        axes[1, 0].set_xticklabels(target_seqs, rotation=45, ha='right')
        axes[1, 0].set_yticks(range(len(kmer_labels)))
        axes[1, 0].set_yticklabels(kmer_labels, fontsize=8)
        
        plt.colorbar(im, ax=axes[1, 0], label='Z-score')
        
        # Mahalanobis distances if available
        if 'mahalanobis_distances' in results:
            mahal_dist = results['mahalanobis_distances']
            axes[1, 1].bar(range(len(target_seqs)), mahal_dist)
            axes[1, 1].set_xlabel('Target Sequences')
            axes[1, 1].set_ylabel('Mahalanobis Distance')
            axes[1, 1].set_title('Mahalanobis Distances')
            axes[1, 1].set_xticks(range(len(target_seqs)))
            axes[1, 1].set_xticklabels(target_seqs, rotation=45, ha='right')
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_isolation_forest(self, results: Dict, pca_results: Optional[Dict] = None, **kwargs):
        """Plot Isolation Forest results."""
        self._plot_anomaly_detection(results, pca_results, method_name='Isolation Forest')
    
    def _plot_lof(self, results: Dict, pca_results: Optional[Dict] = None, **kwargs):
        """Plot LOF results."""
        self._plot_anomaly_detection(results, pca_results, method_name='Local Outlier Factor')
    
    def _plot_one_class_svm(self, results: Dict, pca_results: Optional[Dict] = None, **kwargs):
        """Plot One-Class SVM results."""
        self._plot_anomaly_detection(results, pca_results, method_name='One-Class SVM')
    
    def _plot_anomaly_detection(self, results: Dict, pca_results: Optional[Dict] = None, 
                               method_name: str = 'Anomaly Detection'):
        """Generic anomaly detection plotting."""
        is_anomaly = results['is_anomaly']
        seq_ids = results['seq_ids']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Anomaly detection in PCA space if available
        if pca_results is not None:
            components = pca_results['components']
            explained_var = pca_results['explained_variance_ratio']
            
            colors = ['red' if anomaly else 'blue' for anomaly in is_anomaly]
            axes[0, 0].scatter(components[:, 0], components[:, 1], 
                             c=colors, alpha=0.7, s=60)
            axes[0, 0].set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
            axes[0, 0].set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
            axes[0, 0].set_title(f'{method_name}\n(Red: Anomalies, Blue: Normal)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Legend
            normal_patch = plt.scatter([], [], c='blue', alpha=0.7, label='Normal')
            anomaly_patch = plt.scatter([], [], c='red', alpha=0.7, label='Anomaly')
            axes[0, 0].legend(handles=[normal_patch, anomaly_patch])
        else:
            axes[0, 0].axis('off')
        
        # Anomaly counts
        n_normal = np.sum(~is_anomaly)
        n_anomaly = np.sum(is_anomaly)
        
        axes[0, 1].bar(['Normal', 'Anomaly'], [n_normal, n_anomaly], 
                      color=['blue', 'red'], alpha=0.7)
        axes[0, 1].set_ylabel('Number of Sequences')
        axes[0, 1].set_title('Anomaly Detection Summary')
        
        # Add percentage labels
        total = len(is_anomaly)
        axes[0, 1].text(0, n_normal + total*0.01, f'{n_normal/total:.1%}', 
                       ha='center', va='bottom')
        axes[0, 1].text(1, n_anomaly + total*0.01, f'{n_anomaly/total:.1%}', 
                       ha='center', va='bottom')
        
        # Decision scores/anomaly scores
        score_key = None
        for key in ['anomaly_scores', 'lof_scores', 'decision_scores']:
            if key in results:
                score_key = key
                break
        
        if score_key:
            scores = results[score_key]
            if score_key == 'lof_scores':
                scores = -scores  # LOF scores are negative
            
            # Score distribution
            axes[1, 0].hist(scores[~is_anomaly], bins=20, alpha=0.7, 
                           label='Normal', color='blue', density=True)
            axes[1, 0].hist(scores[is_anomaly], bins=20, alpha=0.7, 
                           label='Anomaly', color='red', density=True)
            axes[1, 0].set_xlabel('Anomaly Score')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Anomaly Score Distribution')
            axes[1, 0].legend()
            
            # Score vs sequence
            axes[1, 1].scatter(range(len(scores)), scores, 
                             c=['red' if a else 'blue' for a in is_anomaly], 
                             alpha=0.7)
            axes[1, 1].set_xlabel('Sequence Index')
            axes[1, 1].set_ylabel('Anomaly Score')
            axes[1, 1].set_title('Anomaly Scores by Sequence')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 0].axis('off')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_chi_square(self, results: Dict, **kwargs):
        """Plot chi-square test results."""
        chi2_stats = results['chi2_stats']
        p_values = results['p_values']
        target_seqs = results['target_seqs']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Chi-square statistics
        axes[0, 0].bar(range(len(target_seqs)), chi2_stats)
        axes[0, 0].set_xlabel('Target Sequences')
        axes[0, 0].set_ylabel('Chi-square Statistic')
        axes[0, 0].set_title('Chi-square Statistics')
        axes[0, 0].set_xticks(range(len(target_seqs)))
        axes[0, 0].set_xticklabels(target_seqs, rotation=45, ha='right')
        
        # P-values (negative log10)
        log_p_values = -np.log10(p_values + 1e-300)  # Add small value to avoid log(0)
        axes[0, 1].bar(range(len(target_seqs)), log_p_values)
        axes[0, 1].axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        axes[0, 1].axhline(y=-np.log10(0.001), color='darkred', linestyle='--', label='p=0.001')
        axes[0, 1].set_xlabel('Target Sequences')
        axes[0, 1].set_ylabel('-log10(p-value)')
        axes[0, 1].set_title('Statistical Significance')
        axes[0, 1].legend()
        axes[0, 1].set_xticks(range(len(target_seqs)))
        axes[0, 1].set_xticklabels(target_seqs, rotation=45, ha='right')
        
        # P-value distribution
        axes[1, 0].hist(p_values, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0.05, color='red', linestyle='--', label='α=0.05')
        axes[1, 0].set_xlabel('P-value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('P-value Distribution')
        axes[1, 0].legend()
        
        # Chi-square vs p-value scatter
        axes[1, 1].scatter(chi2_stats, log_p_values, alpha=0.7)
        axes[1, 1].set_xlabel('Chi-square Statistic')
        axes[1, 1].set_ylabel('-log10(p-value)')
        axes[1, 1].set_title('Chi-square vs Significance')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add sequence labels for significant points
        significant_mask = p_values < 0.05
        for i, (chi2, log_p, seq_id) in enumerate(zip(chi2_stats[significant_mask], 
                                                     log_p_values[significant_mask],
                                                     [target_seqs[j] for j in range(len(target_seqs)) 
                                                      if significant_mask[j]])):
            axes[1, 1].annotate(seq_id, (chi2, log_p), 
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_dendrogram(self, results: Dict):
        """Plot dendrogram for hierarchical clustering."""
        linkage_matrix = results['linkage_matrix']
        seq_ids = results['seq_ids']
        
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, labels=seq_ids, leaf_rotation=90, leaf_font_size=10)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sequences')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.show()
    
    def _plot_cluster_centers(self, results: Dict, pca_results: Dict):
        """Plot cluster centers in PCA space."""
        cluster_centers = results['cluster_centers']
        components = pca_results['components']
        labels = results['labels']
        
        # Transform cluster centers to PCA space
        pca_model = pca_results['pca_model']
        centers_pca = pca_model.transform(cluster_centers)
        
        plt.figure(figsize=(10, 8))
        
        # Plot points colored by cluster
        unique_labels = set(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(components[mask, 0], components[mask, 1], 
                       c=[colors[i]], s=60, alpha=0.7, label=f'Cluster {label}')
        
        # Plot cluster centers
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                   c='black', marker='x', s=200, linewidths=3, label='Centers')
        
        plt.xlabel(f'PC1 ({pca_results["explained_variance_ratio"][0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca_results["explained_variance_ratio"][1]:.1%} variance)')
        plt.title('K-means Clustering with Centers')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def create_summary_dashboard(self, all_results: Dict, pca_results: Dict):
        """Create a comprehensive dashboard of all results."""
        n_methods = len(all_results)
        fig = plt.figure(figsize=(20, 5 * ((n_methods + 1) // 2)))
        
        plot_idx = 1
        
        for method, results in all_results.items():
            ax = plt.subplot((n_methods + 1) // 2, 2, plot_idx)
            
            if method in ['isolation_forest', 'lof', 'one_class_svm']:
                # Anomaly detection summary
                is_anomaly = results['is_anomaly']
                components = pca_results['components']
                
                colors = ['red' if anomaly else 'blue' for anomaly in is_anomaly]
                ax.scatter(components[:, 0], components[:, 1],
                          c=colors, alpha=0.7, s=40)
                ax.set_title(f'{method.replace("_", " ").title()}\n'
                            f'{np.sum(is_anomaly)} anomalies ({np.mean(is_anomaly):.1%})')
                
            elif method == 'clustering':
                # Clustering summary
                labels = results['labels']
                unique_labels = set(labels)
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    if label == -1:  # Noise
                        ax.scatter(components[mask, 0], components[mask, 1],
                                  c='black', marker='x', s=40, alpha=0.6)
                    else:
                        ax.scatter(components[mask, 0], components[mask, 1],
                                  c=[colors[i]], s=40, alpha=0.7)
                
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                ax.set_title(f'{results["method"].title()} Clustering\n'
                            f'{n_clusters} clusters')
                
            elif method == 'z_score':
                # Z-score summary - color by z-score magnitude
                z_scores = results['composite_z_scores']
                # Map z-scores to colors
                scatter = ax.scatter(components[:, 0], components[:, 1],
                                   c=z_scores, cmap='coolwarm', s=40, alpha=0.7)
                plt.colorbar(scatter, ax=ax, label='Z-score')
                ax.set_title(f'Z-score Analysis\n'
                            f'Max Z-score: {np.max(z_scores):.2f}')
            
            ax.set_xlabel(f'PC1 ({pca_results["explained_variance_ratio"][0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca_results["explained_variance_ratio"][1]:.1%})')
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        plt.tight_layout()
        plt.suptitle('Genome Island Detection Summary Dashboard', 
                    fontsize=16, y=1.02)
        plt.show()
    
    def plot_method_comparison(self, results_dict: Dict, ground_truth: Optional[List[bool]] = None):
        """
        Compare results from different methods.
        
        Args:
            results_dict: Dictionary of {method_name: results}
            ground_truth: Optional ground truth labels for evaluation
        """
        methods = list(results_dict.keys())
        n_methods = len(methods)
        
        if n_methods < 2:
            print("Need at least 2 methods for comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Method agreement matrix
        agreement_matrix = np.zeros((n_methods, n_methods))
        method_predictions = {}
        
        for i, method in enumerate(methods):
            if 'is_anomaly' in results_dict[method]:
                method_predictions[method] = results_dict[method]['is_anomaly']
            elif 'labels' in results_dict[method]:
                # For clustering, consider anything not in largest cluster as anomaly
                labels = results_dict[method]['labels']
                largest_cluster = max(set(labels), key=list(labels).count)
                method_predictions[method] = labels != largest_cluster
        
        # Calculate agreement
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if method1 in method_predictions and method2 in method_predictions:
                    pred1 = method_predictions[method1]
                    pred2 = method_predictions[method2]
                    agreement = np.mean(pred1 == pred2)
                    agreement_matrix[i, j] = agreement
        
        # Plot agreement matrix
        im1 = axes[0, 0].imshow(agreement_matrix, cmap='Blues', vmin=0, vmax=1)
        axes[0, 0].set_xticks(range(n_methods))
        axes[0, 0].set_yticks(range(n_methods))
        axes[0, 0].set_xticklabels([m.replace('_', '\n') for m in methods])
        axes[0, 0].set_yticklabels([m.replace('_', '\n') for m in methods])
        axes[0, 0].set_title('Method Agreement Matrix')
        
        # Add text annotations
        for i in range(n_methods):
            for j in range(n_methods):
                axes[0, 0].text(j, i, f'{agreement_matrix[i, j]:.2f}',
                               ha='center', va='center')
        
        plt.colorbar(im1, ax=axes[0, 0], label='Agreement')
        
        # Anomaly detection rates
        anomaly_rates = []
        method_names_clean = []
        for method in methods:
            if method in method_predictions:
                rate = np.mean(method_predictions[method])
                anomaly_rates.append(rate)
                method_names_clean.append(method.replace('_', ' ').title())
        
        axes[0, 1].bar(range(len(method_names_clean)), anomaly_rates)
        axes[0, 1].set_xlabel('Method')
        axes[0, 1].set_ylabel('Anomaly Detection Rate')
        axes[0, 1].set_title('Anomaly Detection Rates by Method')
        axes[0, 1].set_xticks(range(len(method_names_clean)))
        axes[0, 1].set_xticklabels(method_names_clean, rotation=45, ha='right')
        
        # Venn diagram-style overlap (for up to 3 methods)
        if len(method_predictions) <= 3 and len(method_predictions) >= 2:
            self._plot_method_overlap(method_predictions, axes[1, 0])
        else:
            axes[1, 0].axis('off')
        
        # Performance metrics if ground truth available
        if ground_truth is not None:
            self._plot_performance_metrics(method_predictions, ground_truth, axes[1, 1])
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_method_overlap(self, method_predictions: Dict, ax):
        """Plot overlap between methods (simplified Venn diagram)."""
        methods = list(method_predictions.keys())
        n_methods = len(methods)
        
        if n_methods == 2:
            # Two-way overlap
            pred1, pred2 = method_predictions[methods[0]], method_predictions[methods[1]]
            
            only_1 = np.sum(pred1 & ~pred2)
            only_2 = np.sum(~pred1 & pred2)
            both = np.sum(pred1 & pred2)
            neither = np.sum(~pred1 & ~pred2)
            
            categories = ['Only\n' + methods[0].replace('_', ' '), 
                         'Only\n' + methods[1].replace('_', ' '), 
                         'Both', 'Neither']
            counts = [only_1, only_2, both, neither]
            
            bars = ax.bar(categories, counts)
            ax.set_title('Method Overlap')
            ax.set_ylabel('Number of Sequences')
            
            # Color bars
            bars[0].set_color('lightblue')
            bars[1].set_color('lightgreen') 
            bars[2].set_color('orange')
            bars[3].set_color('lightgray')
            
        elif n_methods == 3:
            # Three-way overlap
            pred1 = method_predictions[methods[0]]
            pred2 = method_predictions[methods[1]] 
            pred3 = method_predictions[methods[2]]
            
            only_1 = np.sum(pred1 & ~pred2 & ~pred3)
            only_2 = np.sum(~pred1 & pred2 & ~pred3)
            only_3 = np.sum(~pred1 & ~pred2 & pred3)
            one_two = np.sum(pred1 & pred2 & ~pred3)
            one_three = np.sum(pred1 & ~pred2 & pred3)
            two_three = np.sum(~pred1 & pred2 & pred3)
            all_three = np.sum(pred1 & pred2 & pred3)
            
            categories = ['1 only', '2 only', '3 only', '1&2', '1&3', '2&3', 'All 3']
            counts = [only_1, only_2, only_3, one_two, one_three, two_three, all_three]
            
            ax.bar(categories, counts)
            ax.set_title('Three-Method Overlap')
            ax.set_ylabel('Number of Sequences')
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_performance_metrics(self, method_predictions: Dict, ground_truth: List[bool], ax):
        """Plot performance metrics against ground truth."""
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        methods = list(method_predictions.keys())
        metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
        
        method_metrics = {}
        
        for method in methods:
            pred = method_predictions[method]
            
            precision = precision_score(ground_truth, pred, zero_division=0)
            recall = recall_score(ground_truth, pred, zero_division=0)
            f1 = f1_score(ground_truth, pred, zero_division=0)
            accuracy = accuracy_score(ground_truth, pred)
            
            method_metrics[method] = [precision, recall, f1, accuracy]
        
        # Create grouped bar chart
        x = np.arange(len(metrics))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            ax.bar(x + i * width, method_metrics[method], width, 
                  label=method.replace('_', ' ').title())
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics (vs Ground Truth)')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    def plot_kmer_importance(self, results: Dict, top_n: int = 20):
        """
        Plot k-mer importance from various analyses.
        
        Args:
            results: Analysis results containing k-mer information
            top_n: Number of top k-mers to show
        """
        if 'kmer_list' not in results:
            print("No k-mer information available in results")
            return
        
        kmer_list = results['kmer_list']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Z-score based importance
        if 'z_scores' in results:
            z_scores = results['z_scores']
            mean_abs_z = np.mean(np.abs(z_scores), axis=0)
            top_indices = np.argsort(mean_abs_z)[-top_n:]
            
            axes[0, 0].barh(range(top_n), mean_abs_z[top_indices])
            axes[0, 0].set_yticks(range(top_n))
            axes[0, 0].set_yticklabels([kmer_list[i] for i in top_indices])
            axes[0, 0].set_xlabel('Mean Absolute Z-score')
            axes[0, 0].set_title(f'Top {top_n} K-mers by Z-score')
        
        # PCA loadings based importance
        if 'loadings' in results:
            loadings = results['loadings']
            # Use first PC loadings
            pc1_importance = np.abs(loadings[:, 0])
            top_indices = np.argsort(pc1_importance)[-top_n:]
            
            axes[0, 1].barh(range(top_n), pc1_importance[top_indices])
            axes[0, 1].set_yticks(range(top_n))
            axes[0, 1].set_yticklabels([kmer_list[i] for i in top_indices])
            axes[0, 1].set_xlabel('PC1 Loading (Absolute)')
            axes[0, 1].set_title(f'Top {top_n} K-mers by PCA Loading')
        
        # Variance-based importance
        if 'z_scores' in results or 'components' in results:
            # Use the profile matrix to calculate variance
            # This is a placeholder - you'd need the actual matrix
            axes[1, 0].text(0.5, 0.5, 'K-mer Variance\n(Requires profile matrix)', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('K-mer Variance Importance')
        
        # Correlation-based importance
        axes[1, 1].text(0.5, 0.5, 'K-mer Correlations\n(Requires correlation analysis)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('K-mer Correlation Network')
        
        plt.tight_layout()
        plt.show()


    def _plot_ivom(self, results: Dict, pca_results: Optional[Dict] = None, **kwargs):
        """Plot IVOM analysis results."""
        
        deviation_scores = results['deviation_scores']
        is_anomaly = results['is_anomaly']
        seq_ids = results['seq_ids']
        threshold = results.get('threshold', np.percentile(deviation_scores, 95))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('IVOM (Interpolated Variable Order Motifs) Analysis', 
                    fontsize=16, fontweight='bold')
        
        # 1. Deviation scores distribution
        axes[0, 0].hist(deviation_scores, bins=20, alpha=0.7, color='skyblue', 
                    edgecolor='black', label='All sequences')
        
        # Highlight anomalies
        anomaly_scores = deviation_scores[is_anomaly]
        if len(anomaly_scores) > 0:
            axes[0, 0].hist(anomaly_scores, bins=20, alpha=0.8, color='red', 
                        edgecolor='darkred', label='Anomalies')
        
        # Mark threshold
        axes[0, 0].axvline(threshold, color='red', linestyle='--', linewidth=2, 
                        label=f'Threshold = {threshold:.3f}')
        
        axes[0, 0].set_xlabel('IVOM Deviation Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of IVOM Deviation Scores')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Sequence-wise deviation scores
        x_positions = range(len(seq_ids))
        colors = ['red' if anomaly else 'blue' for anomaly in is_anomaly]
        
        scatter = axes[0, 1].scatter(x_positions, deviation_scores, c=colors, alpha=0.7, s=60)
        axes[0, 1].axhline(threshold, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Sequence Index')
        axes[0, 1].set_ylabel('IVOM Deviation Score')
        axes[0, 1].set_title('IVOM Scores by Sequence')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add legend
        red_patch = plt.scatter([], [], c='red', alpha=0.7, label='Anomaly')
        blue_patch = plt.scatter([], [], c='blue', alpha=0.7, label='Normal')
        axes[0, 1].legend(handles=[red_patch, blue_patch])
        
        # 3. IVOM vs PCA (if available)
        if pca_results is not None:
            components = pca_results['components']
            explained_var = pca_results['explained_variance_ratio']
            
            # Match sequences between IVOM and PCA results
            pca_seq_ids = pca_results['seq_ids']
            matched_indices = []
            matched_scores = []
            matched_anomalies = []
            
            for i, seq_id in enumerate(seq_ids):
                if seq_id in pca_seq_ids:
                    pca_idx = pca_seq_ids.index(seq_id)
                    matched_indices.append(pca_idx)
                    matched_scores.append(deviation_scores[i])
                    matched_anomalies.append(is_anomaly[i])
            
            if matched_indices:
                matched_components = components[matched_indices]
                colors = ['red' if anomaly else 'blue' for anomaly in matched_anomalies]
                
                scatter = axes[1, 0].scatter(matched_components[:, 0], matched_components[:, 1], 
                                        c=colors, s=matched_scores * 100, alpha=0.7)
                axes[1, 0].set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
                axes[1, 0].set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
                axes[1, 0].set_title('IVOM Results in PCA Space\n(Size = Deviation Score)')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No matching sequences\nbetween IVOM and PCA', 
                            ha='center', va='center', transform=axes[1, 0].transAxes)
        else:
            axes[1, 0].text(0.5, 0.5, 'PCA results not available', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 4. IVOM parameters and statistics
        axes[1, 1].axis('off')
        
        # Display IVOM parameters and stats
        ivom_params = results.get('ivom_params', {})
        stats_text = f"""IVOM Analysis Summary:
        
    Method: Interpolated Variable Order Motifs
    Order Range: {ivom_params.get('min_order', 1)}-{ivom_params.get('max_order', 8)}
    Threshold: {threshold:.4f}
    Anomalies Detected: {np.sum(is_anomaly)} / {len(is_anomaly)}
    Anomaly Rate: {np.mean(is_anomaly):.1%}

    Score Statistics:
    Mean Deviation: {np.mean(deviation_scores):.4f}
    Std Deviation: {np.std(deviation_scores):.4f}
    Max Score: {np.max(deviation_scores):.4f}
    Min Score: {np.min(deviation_scores):.4f}

    Reference Sequences: {len(results.get('reference_seqs', []))}
    Background Motifs: {len(results.get('background_distribution', {}))}
    """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontfamily='monospace', fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        plt.show()

    def plot_method_comparison(self, comparison_results: Dict, **kwargs):
        """
        Plot comprehensive comparison between different detection methods.
        
        Args:
            comparison_results: Results from predictor.compare_methods()
            **kwargs: Additional plotting parameters
        """
        
        methods = comparison_results['methods_compared']
        method_predictions = comparison_results['method_predictions']
        overlap_analysis = comparison_results['overlap_analysis']
        
        n_methods = len(methods)
        if n_methods < 2:
            print("Need at least 2 methods for comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Method Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. Detection rates by method
        detection_rates = [np.mean(method_predictions[method]) for method in methods]
        anomaly_counts = [np.sum(method_predictions[method]) for method in methods]
        
        x_pos = range(len(methods))
        bars = axes[0, 0].bar(x_pos, detection_rates, color='skyblue', alpha=0.7, edgecolor='black')
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, anomaly_counts)):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        
        axes[0, 0].set_xlabel('Detection Method')
        axes[0, 0].set_ylabel('Anomaly Detection Rate')
        axes[0, 0].set_title('Detection Rates by Method')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Pairwise overlap heatmap (Jaccard indices)
        if len(methods) >= 2:
            jaccard_matrix = np.zeros((n_methods, n_methods))
            
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if i == j:
                        jaccard_matrix[i, j] = 1.0  # perfect self-similarity
                    else:
                        overlap_key = f"{method1}_vs_{method2}"
                        reverse_key = f"{method2}_vs_{method1}"
                        
                        if overlap_key in overlap_analysis:
                            jaccard_matrix[i, j] = overlap_analysis[overlap_key]['jaccard_index']
                        elif reverse_key in overlap_analysis:
                            jaccard_matrix[i, j] = overlap_analysis[reverse_key]['jaccard_index']
            
            im = axes[0, 1].imshow(jaccard_matrix, cmap='Blues', vmin=0, vmax=1)
            axes[0, 1].set_title('Method Overlap (Jaccard Index)')
            
            # Add text annotations
            for i in range(n_methods):
                for j in range(n_methods):
                    text = axes[0, 1].text(j, i, f'{jaccard_matrix[i, j]:.2f}',
                                        ha='center', va='center', fontweight='bold')
            
            # Set ticks and labels
            axes[0, 1].set_xticks(range(n_methods))
            axes[0, 1].set_yticks(range(n_methods))
            axes[0, 1].set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
            axes[0, 1].set_yticklabels([m.replace('_', ' ').title() for m in methods])
            
            # Add colorbar
            plt.colorbar(im, ax=axes[0, 1], label='Jaccard Index')
        
        # 3. Consensus analysis
        consensus_analysis = comparison_results.get('consensus_analysis', {})
        if consensus_analysis:
            vote_distribution = consensus_analysis.get('vote_distribution', [])
            
            if len(vote_distribution) > 0:
                x_votes = range(len(vote_distribution))
                axes[1, 0].bar(x_votes, vote_distribution, color='lightcoral', alpha=0.7, edgecolor='black')
                axes[1, 0].set_xlabel('Number of Methods Agreeing')
                axes[1, 0].set_ylabel('Number of Sequences')
                axes[1, 0].set_title('Consensus Vote Distribution')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add text annotations
                for i, count in enumerate(vote_distribution):
                    if count > 0:
                        axes[1, 0].text(i, count + max(vote_distribution) * 0.01, 
                                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # 4. Method-specific unique detections
        unique_detections = {}
        for method in methods:
            method_anomalies = set(np.where(method_predictions[method])[0])
            other_methods_anomalies = set()
            
            for other_method in methods:
                if other_method != method:
                    other_methods_anomalies.update(np.where(method_predictions[other_method])[0])
            
            unique_detections[method] = len(method_anomalies - other_methods_anomalies)
        
        method_names_clean = [m.replace('_', ' ').title() for m in methods]
        unique_counts = [unique_detections[method] for method in methods]
        
        axes[1, 1].bar(range(len(methods)), unique_counts, color='gold', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Detection Method')
        axes[1, 1].set_ylabel('Unique Detections')
        axes[1, 1].set_title('Method-Specific Unique Detections')
        axes[1, 1].set_xticks(range(len(methods)))
        axes[1, 1].set_xticklabels(method_names_clean, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add count labels
        for i, count in enumerate(unique_counts):
            if count > 0:
                axes[1, 1].text(i, count + max(unique_counts) * 0.02, 
                            str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

    def plot_ivom_sliding_window(self, results: Dict, genome_data: Optional[Dict] = None, **kwargs):
        """
        Plot IVOM sliding window analysis results along genomic coordinates.
        
        Args:
            results: IVOM sliding window results
            genome_data: Optional genome annotation data
            **kwargs: Additional plotting parameters
        """
        
        if 'window_positions' not in results:
            print("No window position data available for sliding window plot")
            return
        
        positions = results['window_positions'] 
        deviation_scores = results['deviation_scores']
        is_anomaly = results['is_anomaly']
        threshold = results.get('threshold', np.percentile(deviation_scores, 95))
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), height_ratios=[1, 1, 0.3])
        fig.suptitle(f'IVOM Sliding Window Analysis: {results.get("parent_sequence", "Unknown")}', 
                    fontsize=16, fontweight='bold')
        
        # Calculate window centers for plotting
        window_centers = [(pos[0] + pos[1]) / 2 for pos in positions]
        genome_length = max(pos[1] for pos in positions) if positions else 100000
        
        # 1. IVOM deviation scores along genome
        axes[0].plot(window_centers, deviation_scores, 'b-', alpha=0.7, linewidth=1, label='IVOM Score')
        axes[0].axhline(threshold, color='red', linestyle='--', linewidth=2, 
                    label=f'Threshold = {threshold:.3f}')
        
        # Highlight anomalous windows
        anomaly_centers = [center for center, anomaly in zip(window_centers, is_anomaly) if anomaly]
        anomaly_scores = [score for score, anomaly in zip(deviation_scores, is_anomaly) if anomaly]
        
        if anomaly_centers:
            axes[0].scatter(anomaly_centers, anomaly_scores, color='red', s=50, 
                        alpha=0.8, zorder=5, label='Anomalous Windows')
        
        axes[0].set_ylabel('IVOM Deviation Score')
        axes[0].set_title('IVOM Scores Along Genome')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Chromosome view with predicted islands
        chromosome_height = 0.3
        axes[1].add_patch(Rectangle((0, -chromosome_height/2), genome_length, chromosome_height,
                                facecolor='lightgray', edgecolor='black', linewidth=1))
        
        # Plot true annotations if available
        if genome_data and 'annotations' in genome_data:
            for ann in genome_data['annotations']:
                start, end = ann['start'], ann['end']
                island_type = ann.get('type', 'unknown')
                
                axes[1].add_patch(Rectangle((start, chromosome_height/2 + 0.1), end - start, 0.15,
                                        facecolor='green', edgecolor='darkgreen', 
                                        linewidth=2, alpha=0.8))
                
                # Add label
                axes[1].text((start + end)/2, chromosome_height/2 + 0.3, 
                            f"True {island_type}", ha='center', va='bottom', 
                            fontsize=8, rotation=45, fontweight='bold')
        
        # Plot predicted islands (anomalous windows)
        for pos, anomaly in zip(positions, is_anomaly):
            if anomaly:
                start, end = pos
                axes[1].add_patch(Rectangle((start, -chromosome_height/2 - 0.15), end - start, 0.15,
                                        facecolor='red', edgecolor='darkred', 
                                        linewidth=1, alpha=0.7))
        
        axes[1].set_xlim(0, genome_length)
        axes[1].set_ylim(-0.8, 0.8)
        axes[1].set_ylabel('Chromosome')
        axes[1].set_title('Genomic Overview: IVOM Predicted Regions')
        axes[1].set_yticks([])
        
        # Add legend
        legend_elements = [
            patches.Patch(color='lightgray', label='Chromosome'),
            patches.Patch(color='red', label='IVOM Predicted Islands')
        ]
        if genome_data and 'annotations' in genome_data:
            legend_elements.append(patches.Patch(color='green', label='True Islands'))
        
        axes[1].legend(handles=legend_elements, loc='upper right')
        
        # 3. Analysis parameters and statistics
        axes[2].axis('off')
        
        window_size = results.get('window_size', 'Unknown')
        step_size = results.get('step_size', 'Unknown')
        n_windows = len(positions)
        n_anomalies = np.sum(is_anomaly)
        
        stats_text = f"""IVOM Sliding Window Analysis:
            Window Size: {window_size} bp | Step Size: {step_size} bp | Windows: {n_windows} | Anomalies: {n_anomalies} ({n_anomalies/n_windows:.1%})
            Max Score: {np.max(deviation_scores):.4f} | Mean Score: {np.mean(deviation_scores):.4f} | Threshold: {threshold:.4f}"""
                
        axes[2].text(0.02, 0.5, stats_text, transform=axes[2].transAxes, 
                    fontfamily='monospace', fontsize=10, verticalalignment='center')
        
        # Format x-axis with genomic coordinates
        def format_genomic_axis(ax, max_pos):
            """Format x-axis with appropriate genomic coordinate labels."""
            if max_pos > 1e6:
                ax.set_xlabel('Genomic Position (Mb)')
                ticks = ax.get_xticks()
                ax.set_xticklabels([f'{tick/1e6:.1f}' for tick in ticks])
            elif max_pos > 1e3:
                ax.set_xlabel('Genomic Position (kb)')
                ticks = ax.get_xticks()
                ax.set_xticklabels([f'{tick/1e3:.0f}' for tick in ticks])
            else:
                ax.set_xlabel('Genomic Position (bp)')
        
        format_genomic_axis(axes[0], genome_length)
        format_genomic_axis(axes[1], genome_length)
        
        plt.tight_layout()
        plt.show()

    # Update the existing plot_results method to handle IVOM:
    def plot_results(self, analysis_type: str, results: Dict, pca_results: Optional[Dict] = None, **kwargs):
        """
        Plot analysis results with support for IVOM.
        
        Args:
            analysis_type: Type of analysis ('z_score', 'clustering', 'isolation_forest', 'lof', 'ivom', etc.)
            results: Analysis results dictionary
            pca_results: Optional PCA results for overlay
            **kwargs: Additional plotting parameters
        """
        
        if analysis_type == 'z_score':
            self._plot_z_score(results, pca_results, **kwargs)
        elif analysis_type == 'clustering':
            self._plot_clustering(results, pca_results, **kwargs)
        elif analysis_type == 'isolation_forest':
            self._plot_isolation_forest(results, pca_results, **kwargs)
        elif analysis_type == 'lof':
            self._plot_lof(results, pca_results, **kwargs)
        elif analysis_type == 'one_class_svm':
            self._plot_one_class_svm(results, pca_results, **kwargs)
        elif analysis_type == 'ivom':  # NEW: IVOM support
            self._plot_ivom(results, pca_results, **kwargs)
        elif analysis_type == 'ivom_sliding_window':  # NEW: IVOM sliding window support
            self.plot_ivom_sliding_window(results, **kwargs)
        elif analysis_type == 'method_comparison':  # NEW: Method comparison support
            self.plot_method_comparison(results, **kwargs)
        else:
            print(f"Unknown analysis type: {analysis_type}")
            print("Available types: 'z_score', 'clustering', 'isolation_forest', 'lof', 'one_class_svm', 'ivom', 'ivom_sliding_window', 'method_comparison'")