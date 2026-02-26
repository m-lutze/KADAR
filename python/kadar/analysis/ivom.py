import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import math
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class IVOMAnalysis:
    """
    IVOM (Interpolated Variable Order Motifs) analysis for compositional bias detection.
    
    This class implements the IVOM approach similar to Alien_hunter, using variable-order
    k-mer distributions to detect horizontal gene transfer events based on compositional
    signatures that deviate from the host genome background.
    """
    
    def __init__(self, profiler, scaler: StandardScaler):
        """
        Initialize IVOM analysis.
        
        Args:
            profiler: KmerProfiler instance
            scaler: StandardScaler instance (for compatibility with other methods)
        """
        self.profiler = profiler
        self.scaler = scaler
        
        # IVOM-specific parameters
        self.max_order = 8
        self.min_order = 1
        self.interpolation_weights = None
        self.background_distribution = None
        
    def set_ivom_parameters(self, max_order: int = 8, min_order: int = 1,
                           interpolation_weights: Optional[List[float]] = None):
        """
        Set IVOM analysis parameters.
        
        Args:
            max_order: Maximum k-mer order to consider
            min_order: Minimum k-mer order to consider
            interpolation_weights: Custom weights for interpolating different orders
        """
        self.max_order = max_order
        self.min_order = min_order
        
        if interpolation_weights is None:
            self.interpolation_weights = self._calculate_default_weights()
        else:
            self.interpolation_weights = interpolation_weights
    
    def _calculate_default_weights(self) -> List[float]:
        """
        Calculate default interpolation weights favoring intermediate orders.
        
        Based on Alien_hunter strategy - higher weights for 3-6 mers.
        """
        n_orders = self.max_order - self.min_order + 1
        weights = []
        
        for i in range(n_orders):
            order = self.min_order + i
            # Peak weight around order 4-6, similar to Alien_hunter
            if order <= 2:
                weight = 0.3 + 0.2 * order
            elif order <= 6:
                weight = 1.0
            else:
                weight = 1.0 / (order - 5)  # decreasing for higher orders
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        return [w / total_weight for w in weights]
    
    def extract_variable_order_motifs(self, sequence: str) -> Dict[int, Dict[str, float]]:
        """
        Extract variable-order motif distributions from sequence.

        TD: swap this over to use profiler.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Dictionary mapping order -> motif -> frequency
        """
        sequence = sequence.upper().replace('N', '')  # clean sequence
        seq_len = len(sequence)
        
        order_distributions = {}
        
        for order in range(self.min_order, self.max_order + 1):
            if seq_len < order:
                continue
                
            motif_counts = Counter()
            total_motifs = 0
            
            # Extract all k-mers of current order
            for i in range(seq_len - order + 1):
                motif = sequence[i:i+order]
                
                # Skip motifs containing invalid characters
                if all(base in 'ATGC' for base in motif):
                    motif_counts[motif] += 1
                    total_motifs += 1
            
            # Convert to frequencies
            if total_motifs > 0:
                order_distributions[order] = {
                    motif: count / total_motifs 
                    for motif, count in motif_counts.items()
                }
            else:
                order_distributions[order] = {}
        
        return order_distributions
    
    def calculate_interpolated_distribution(self, order_distributions: Dict[int, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate interpolated variable-order motif distribution.
        
        Args:
            order_distributions: Distributions for each order
            
        Returns:
            Interpolated motif distribution
        """
        interpolated_dist = defaultdict(float)
        
        for i, order in enumerate(range(self.min_order, self.max_order + 1)):
            if order not in order_distributions:
                continue
                
            weight = self.interpolation_weights[i] if i < len(self.interpolation_weights) else 0.1
            
            for motif, freq in order_distributions[order].items():
                interpolated_dist[motif] += weight * freq
        
        return dict(interpolated_dist)
    
    def build_background_model(self, reference_seqs: List[str]) -> Dict[str, float]:
        """
        Build background IVOM distribution from reference sequences.
        
        Args:
            reference_seqs: List of reference sequence IDs
            
        Returns:
            Background motif distribution
        """
        # Validate reference sequences
        missing_seqs = [seq_id for seq_id in reference_seqs if seq_id not in self.profiler.sequences]
        if missing_seqs:
            raise ValueError(f"Reference sequences not found: {missing_seqs}")
        
        # Aggregate motif distributions from all reference sequences
        aggregated_distributions = defaultdict(list)
        
        for seq_id in reference_seqs:
            sequence = self.profiler.sequences[seq_id]
            order_dists = self.extract_variable_order_motifs(sequence)
            interpolated_dist = self.calculate_interpolated_distribution(order_dists)
            
            for motif, freq in interpolated_dist.items():
                aggregated_distributions[motif].append(freq)
        
        # Calculate mean frequencies across reference sequences
        background_dist = {}
        for motif, freq_list in aggregated_distributions.items():
            background_dist[motif] = np.mean(freq_list)
        
        # Normalize to ensure proper probability distribution
        total_freq = sum(background_dist.values())
        if total_freq > 0:
            background_dist = {motif: freq / total_freq 
                             for motif, freq in background_dist.items()}
        
        self.background_distribution = background_dist
        return background_dist
    
    def calculate_compositional_deviation(self, sequence: str, 
                                        background_dist: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate compositional deviation using IVOM approach.
        
        Args:
            sequence: Target sequence to analyze
            background_dist: Background distribution (uses stored if None)
            
        Returns:
            Compositional deviation score
        """
        if background_dist is None:
            background_dist = self.background_distribution
            
        if background_dist is None:
            raise ValueError("No background distribution available. Run build_background_model first.")
        
        # Extract IVOM distribution for target sequence
        order_dists = self.extract_variable_order_motifs(sequence)
        target_dist = self.calculate_interpolated_distribution(order_dists)
        
        # Calculate various deviation metrics
        deviation_scores = []
        
        # 1. Kullback-Leibler divergence (primary metric in Alien_hunter)
        kl_divergence = self._calculate_kl_divergence(target_dist, background_dist)
        deviation_scores.append(kl_divergence)
        
        # 2. Jensen-Shannon divergence (more symmetric alternative)
        js_divergence = self._calculate_js_divergence(target_dist, background_dist)
        deviation_scores.append(js_divergence)
        
        # 3. Cosine distance in motif space
        cosine_dist = self._calculate_cosine_distance(target_dist, background_dist)
        deviation_scores.append(cosine_dist)
        
        # Combine scores with weights (KL gets highest weight like Alien_hunter)
        weights = [0.5, 0.3, 0.2]
        combined_score = sum(w * score for w, score in zip(weights, deviation_scores))
        
        return combined_score
    
    def _calculate_kl_divergence(self, p_dist: Dict[str, float], 
                               q_dist: Dict[str, float]) -> float:
        """Calculate Kullback-Leibler divergence between distributions."""
        
        all_motifs = set(p_dist.keys()) | set(q_dist.keys())
        kl_div = 0.0
        epsilon = 1e-10  # small value to avoid log(0)
        
        for motif in all_motifs:
            p_freq = p_dist.get(motif, epsilon)
            q_freq = q_dist.get(motif, epsilon)
            
            if p_freq > epsilon:  # only include if motif present in target
                kl_div += p_freq * math.log(p_freq / q_freq)
        
        return kl_div
    
    def _calculate_js_divergence(self, p_dist: Dict[str, float], 
                               q_dist: Dict[str, float]) -> float:
        """Calculate Jensen-Shannon divergence between distributions."""
        
        all_motifs = set(p_dist.keys()) | set(q_dist.keys())
        epsilon = 1e-10
        
        # Calculate average distribution M = (P + Q) / 2
        m_dist = {}
        for motif in all_motifs:
            p_freq = p_dist.get(motif, epsilon)
            q_freq = q_dist.get(motif, epsilon)
            m_dist[motif] = (p_freq + q_freq) / 2.0
        
        # JS divergence = (KL(P||M) + KL(Q||M)) / 2
        kl_pm = self._calculate_kl_divergence(p_dist, m_dist)
        kl_qm = self._calculate_kl_divergence(q_dist, m_dist)
        
        return (kl_pm + kl_qm) / 2.0
    
    def _calculate_cosine_distance(self, p_dist: Dict[str, float], 
                                 q_dist: Dict[str, float]) -> float:
        """Calculate cosine distance between motif distributions."""
        
        all_motifs = sorted(set(p_dist.keys()) | set(q_dist.keys()))
        
        if len(all_motifs) == 0:
            return 0.0
        
        # Convert to vectors
        p_vector = np.array([p_dist.get(motif, 0.0) for motif in all_motifs])
        q_vector = np.array([q_dist.get(motif, 0.0) for motif in all_motifs])
        
        # Calculate cosine similarity and convert to distance
        if np.linalg.norm(p_vector) == 0 or np.linalg.norm(q_vector) == 0:
            return 1.0  # maximum distance if either vector is zero
            
        similarity = cosine_similarity([p_vector], [q_vector])[0][0]
        distance = 1.0 - similarity
        
        return distance
    
    def ivom_analysis(self, reference_seqs: List[str], target_seqs: Optional[List[str]] = None,
                     max_order: int = 8, min_order: int = 1,
                     threshold_percentile: float = 95.0,
                     interpolation_weights: Optional[List[float]] = None) -> Dict:
        """
        Perform IVOM analysis for horizontal gene transfer detection.
        
        Args:
            reference_seqs: Reference sequence IDs for background model
            target_seqs: Target sequence IDs to analyze (all if None)
            max_order: Maximum k-mer order for IVOM analysis
            min_order: Minimum k-mer order for IVOM analysis
            threshold_percentile: Percentile for anomaly detection threshold
            interpolation_weights: Custom weights for interpolating different orders
            
        Returns:
            Dictionary with IVOM analysis results
        """
        # Set IVOM parameters
        self.set_ivom_parameters(max_order, min_order, interpolation_weights)
        
        # Validate inputs
        if not reference_seqs:
            raise ValueError("reference_seqs cannot be empty")
        
        # Set target sequences
        if target_seqs is None:
            target_seqs = list(self.profiler.sequences.keys())
        
        # Build background model
        reference_sequences = [self.profiler.sequences[seq_id] for seq_id in reference_seqs]
        background_dist = self.build_background_model(reference_seqs)
        
        # Calculate deviation scores for target sequences
        deviation_scores = []
        sequence_ids = []
        motif_distributions = {}
        
        for seq_id in target_seqs:
            if seq_id not in self.profiler.sequences:
                continue
                
            sequence = self.profiler.sequences[seq_id]
            
            try:
                deviation_score = self.calculate_compositional_deviation(sequence, background_dist)
                deviation_scores.append(deviation_score)
                sequence_ids.append(seq_id)
                
                # Store individual motif distribution for detailed analysis
                order_dists = self.extract_variable_order_motifs(sequence)
                motif_distributions[seq_id] = self.calculate_interpolated_distribution(order_dists)
                
            except Exception as e:
                print(f"Warning: Failed to analyze sequence {seq_id}: {e}")
                continue
        
        if not deviation_scores:
            raise ValueError("No sequences could be analyzed successfully")
        
        deviation_scores = np.array(deviation_scores)
        
        # Determine anomaly threshold
        threshold = np.percentile(deviation_scores, threshold_percentile)
        is_anomaly = deviation_scores > threshold
        
        # Calculate additional statistics
        z_scores = stats.zscore(deviation_scores)
        
        # Prepare results in standard format
        results = {
            'method': 'IVOM',
            'deviation_scores': deviation_scores,
            'anomaly_scores': deviation_scores,  # alias for compatibility
            'z_scores': z_scores,
            'is_anomaly': is_anomaly,
            'seq_ids': sequence_ids,
            'threshold': threshold,
            'threshold_percentile': threshold_percentile,
            'background_distribution': background_dist,
            'motif_distributions': motif_distributions,
            'reference_seqs': reference_seqs,
            'target_seqs': target_seqs,
            'ivom_params': {
                'max_order': max_order,
                'min_order': min_order,
                'interpolation_weights': self.interpolation_weights
            },
            'n_anomalies': np.sum(is_anomaly),
            'anomaly_rate': np.mean(is_anomaly)
        }
        
        return results
    
    def sliding_window_ivom_analysis(self, seq_id: str, window_size: int = 5000,
                                   step_size: int = 2500, reference_proportion: float = 0.1,
                                   **ivom_kwargs) -> Dict:
        """
        Perform IVOM analysis on sliding windows of a sequence.
        
        Args:
            seq_id: Sequence identifier for analysis
            window_size: Size of sliding windows
            step_size: Step size for sliding windows
            reference_proportion: Proportion of windows to use as reference
            **ivom_kwargs: Additional parameters for IVOM analysis
            
        Returns:
            Dictionary with sliding window IVOM results
        """
        if seq_id not in self.profiler.sequences:
            raise KeyError(f"Sequence '{seq_id}' not found in profiler")
        
        sequence = self.profiler.sequences[seq_id]
        
        if len(sequence) < window_size:
            raise ValueError(f"Sequence length smaller than window size")
        
        # Create sliding windows
        windows = []
        positions = []
        window_sequences = {}
        
        for start in range(0, len(sequence) - window_size + 1, step_size):
            end = start + window_size
            window_seq = sequence[start:end]
            window_id = f"window_{start}_{end}"
            
            windows.append(window_id)
            positions.append((start, end))
            window_sequences[window_id] = window_seq
        
        # Select reference windows (every nth window to get representative sample)
        n_ref = max(1, int(len(windows) * reference_proportion))
        step = len(windows) // n_ref
        reference_windows = [windows[i] for i in range(0, len(windows), max(1, step))][:n_ref]
        
        # Create temporary profiler for windows
        from ..core.kmer_profiler import KmerProfiler
        temp_profiler = KmerProfiler(k=self.profiler.k, normalize=self.profiler.normalize)
        
        for window_id, window_seq in window_sequences.items():
            temp_profiler.add_sequence(window_id, window_seq)
        
        # Create temporary IVOM analyzer
        temp_ivom = IVOMAnalysis(temp_profiler, self.scaler)
        
        # Run IVOM analysis on windows
        ivom_results = temp_ivom.ivom_analysis(
            reference_seqs=reference_windows,
            target_seqs=windows,
            **ivom_kwargs
        )
        
        # Add window position information
        ivom_results.update({
            'parent_sequence': seq_id,
            'window_size': window_size,
            'step_size': step_size,
            'window_positions': positions,
            'window_ids': windows,
            'reference_windows': reference_windows
        })
        
        return ivom_results