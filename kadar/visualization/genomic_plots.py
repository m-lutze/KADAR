import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd


class GenomicLocationVisualizer:
    """
    Visualizer for showing island locations along genomic sequences.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """Initialize the genomic visualizer."""
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Color scheme for different features
        self.colors = {
            'chromosome': '#E8E8E8',
            'island_predicted': '#FF6B6B',
            'island_confirmed': '#4ECDC4',
            'gene': '#45B7D1', 
            'prophage': '#FFA07A',
            'pathogenicity': '#DDA0DD',
            'antibiotic_resistance': '#F0E68C',
            'transposon': '#98FB98',
            'integron': '#FFB6C1'
        }
    
    def plot_genomic_overview(self, genome_data: Dict, window_results: Dict, 
                            predictions: Dict, figsize: Tuple[int, int] = (16, 10)):
        """
        Create a comprehensive genomic overview showing island locations.
        
        Args:
            genome_data: Dictionary with genome information and annotations
            window_results: Results from sliding window analysis
            predictions: Prediction results from various methods
            figsize: Figure size
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize, height_ratios=[1, 1, 1, 0.3])
        fig.suptitle('Genomic Island Detection Overview', fontsize=16, fontweight='bold')
        
        genome_length = len(genome_data['genome'])
        positions = [pos[0] + window_results['window_size']//2 
                    for pos in window_results['positions']]
        
        # 1. Chromosome overview with predicted islands
        self._plot_chromosome_overview(axes[0], genome_data, window_results, 
                                     predictions, genome_length)
        
        # 2. Detection scores along genome
        self._plot_detection_scores(axes[1], positions, predictions, genome_length)
        
        # 3. GC content and compositional features
        self._plot_compositional_features(axes[2], genome_data, window_results, 
                                        positions, genome_length)
        
        # 4. Gene density (if available) or sequence features
        self._plot_sequence_features(axes[3], genome_data, genome_length)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_chromosome_overview(self, ax, genome_data: Dict, window_results: Dict,
                                predictions: Dict, genome_length: int):
        """Plot chromosome overview with island predictions."""
        # Draw chromosome backbone
        chromosome_height = 0.3
        ax.add_patch(Rectangle((0, -chromosome_height/2), genome_length, chromosome_height,
                              facecolor=self.colors['chromosome'], 
                              edgecolor='black', linewidth=1))
        
        # Add true annotations if available
        if 'annotations' in genome_data:
            for ann in genome_data['annotations']:
                start, end = ann['start'], ann['end']
                island_type = ann.get('type', 'unknown')
                color = self.colors.get(island_type, self.colors['island_confirmed'])
                
                ax.add_patch(Rectangle((start, -chromosome_height/2), end - start, 
                                     chromosome_height, facecolor=color, 
                                     edgecolor='darkred', linewidth=2, alpha=0.8))
                
                # Add label
                ax.text((start + end)/2, chromosome_height/2 + 0.1, 
                       f"True {island_type}", ha='center', va='bottom', 
                       fontsize=8, rotation=45, fontweight='bold')
        
        # Add predicted islands
        if 'is_anomaly' in predictions:
            window_size = window_results['window_size']
            step_size = window_results['step_size']
            
            for i, (is_anomaly, pos) in enumerate(zip(predictions['is_anomaly'], 
                                                    window_results['positions'])):
                if is_anomaly:
                    start, end = pos
                    ax.add_patch(Rectangle((start, -chromosome_height/2 - 0.15), 
                                         end - start, 0.15,
                                         facecolor=self.colors['island_predicted'], 
                                         edgecolor='red', linewidth=1, alpha=0.7))
        
        ax.set_xlim(0, genome_length)
        ax.set_ylim(-0.8, 0.8)
        ax.set_ylabel('Chromosome')
        ax.set_title('Genomic Overview: True vs Predicted Islands')
        
        # Add legend
        legend_elements = [
            patches.Patch(color=self.colors['chromosome'], label='Chromosome'),
            patches.Patch(color=self.colors['island_confirmed'], label='True Islands'),
            patches.Patch(color=self.colors['island_predicted'], label='Predicted Islands')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # Format x-axis
        self._format_genomic_axis(ax, genome_length)
    
    def _plot_detection_scores(self, ax, positions: List[int], predictions: Dict, 
                             genome_length: int):
        """Plot detection scores along the genome."""
        # Plot anomaly scores
        if 'anomaly_scores' in predictions:
            scores = predictions['anomaly_scores']
            is_anomaly = predictions['is_anomaly']
            
            # Color points by anomaly status
            colors = ['red' if anomaly else 'blue' for anomaly in is_anomaly]
            scatter = ax.scatter(positions, scores, c=colors, alpha=0.7, s=30)
            
            # Add threshold line if available
            threshold = np.percentile(scores, 90)  # Top 10% as threshold
            ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                      label=f'90th percentile: {threshold:.3f}')
            
            ax.set_ylabel('Anomaly Score')
            ax.set_title('Anomaly Detection Scores Along Genome')
            ax.legend()
        
        ax.set_xlim(0, genome_length)
        ax.grid(True, alpha=0.3)
        self._format_genomic_axis(ax, genome_length, show_labels=False)
    
    def _plot_compositional_features(self, ax, genome_data: Dict, window_results: Dict,
                                   positions: List[int], genome_length: int):
        """Plot GC content and other compositional features."""
        # Calculate GC content for windows
        window_profiler = window_results['profiler']
        gc_contents = []
        
        for window_id in window_results['windows']:
            gc_content = window_profiler.calculate_gc_content(window_id)
            gc_contents.append(gc_content)
        
        # Plot GC content
        ax.plot(positions, gc_contents, 'g-', linewidth=2, alpha=0.8, label='GC Content')
        ax.fill_between(positions, gc_contents, alpha=0.3, color='green')
        
        # Add overall genome GC content as reference line
        overall_gc = (genome_data['genome'].count('G') + genome_data['genome'].count('C')) / len(genome_data['genome'])
        ax.axhline(y=overall_gc, color='darkgreen', linestyle='--', alpha=0.8,
                  label=f'Genome average: {overall_gc:.3f}')
        
        # Highlight regions with unusual GC content
        gc_mean = np.mean(gc_contents)
        gc_std = np.std(gc_contents)
        threshold = gc_mean + 2 * gc_std
        
        unusual_regions = np.array(gc_contents) > threshold
        if np.any(unusual_regions):
            unusual_positions = np.array(positions)[unusual_regions]
            unusual_gc = np.array(gc_contents)[unusual_regions]
            ax.scatter(unusual_positions, unusual_gc, color='orange', s=50, 
                      alpha=0.8, label='Unusual GC content', zorder=5)
        
        ax.set_ylabel('GC Content')
        ax.set_title('GC Content Variation Along Genome')
        ax.legend()
        ax.set_xlim(0, genome_length)
        ax.grid(True, alpha=0.3)
        self._format_genomic_axis(ax, genome_length, show_labels=False)
    
    def _plot_sequence_features(self, ax, genome_data: Dict, genome_length: int):
        """Plot sequence features like gene density or repeats."""
        # Create a simple representation of sequence complexity
        window_size = 5000
        complexities = []
        positions = []
        
        sequence = genome_data['genome']
        for i in range(0, len(sequence) - window_size + 1, window_size // 2):
            window = sequence[i:i + window_size]
            # Calculate sequence complexity (Shannon entropy)
            complexity = self._calculate_sequence_complexity(window)
            complexities.append(complexity)
            positions.append(i + window_size // 2)
        
        ax.plot(positions, complexities, 'purple', linewidth=1.5, alpha=0.8)
        ax.fill_between(positions, complexities, alpha=0.3, color='purple')
        
        ax.set_ylabel('Sequence\nComplexity')
        ax.set_xlabel('Genomic Position (bp)')
        ax.set_title('Sequence Complexity Along Genome')
        ax.set_xlim(0, genome_length)
        ax.grid(True, alpha=0.3)
        self._format_genomic_axis(ax, genome_length)
    
    def _calculate_sequence_complexity(self, sequence: str) -> float:
        """Calculate Shannon entropy as a measure of sequence complexity."""
        from collections import Counter
        import math
        
        if len(sequence) == 0:
            return 0
        
        # Count nucleotides
        counts = Counter(sequence)
        # Calculate Shannon entropy
        entropy = 0
        for count in counts.values():
            p = count / len(sequence)
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _format_genomic_axis(self, ax, genome_length: int, show_labels: bool = True):
        """Format genomic position axis with appropriate units."""
        # Determine appropriate units
        if genome_length >= 1000000:
            scale = 1000000
            unit = 'Mb'
        elif genome_length >= 1000:
            scale = 1000
            unit = 'kb'
        else:
            scale = 1
            unit = 'bp'
        
        if show_labels:
            # Set tick positions
            n_ticks = 6
            tick_positions = np.linspace(0, genome_length, n_ticks)
            tick_labels = [f'{pos/scale:.1f}' for pos in tick_positions]
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel(f'Position ({unit})')
        else:
            ax.set_xticks([])
    
    def plot_circular_genome(self, genome_data: Dict, predictions: Dict,
                           window_results: Dict, figsize: Tuple[int, int] = (12, 12)):
        """
        Create a circular genome plot showing island locations.
        
        Args:
            genome_data: Genome data dictionary
            predictions: Prediction results
            window_results: Sliding window results
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        genome_length = len(genome_data['genome'])
        
        # Convert positions to angles (radians)
        def pos_to_angle(pos):
            return 2 * np.pi * pos / genome_length
        
        # Draw genome circle
        angles = np.linspace(0, 2*np.pi, 1000)
        radius = 1.0
        ax.plot(angles, [radius] * len(angles), 'k-', linewidth=3, alpha=0.3)
        
        # Plot true islands
        if 'annotations' in genome_data:
            for ann in genome_data['annotations']:
                start_angle = pos_to_angle(ann['start'])
                end_angle = pos_to_angle(ann['end'])
                
                # Create arc for true island
                arc_angles = np.linspace(start_angle, end_angle, 50)
                arc_radius = radius + 0.1
                ax.plot(arc_angles, [arc_radius] * len(arc_angles), 
                       color=self.colors.get(ann['type'], 'red'), 
                       linewidth=8, alpha=0.8, label=f"True {ann['type']}")
        
        # Plot predicted islands
        if 'is_anomaly' in predictions:
            for i, (is_anomaly, pos) in enumerate(zip(predictions['is_anomaly'], 
                                                    window_results['positions'])):
                if is_anomaly:
                    start_angle = pos_to_angle(pos[0])
                    end_angle = pos_to_angle(pos[1])
                    
                    arc_angles = np.linspace(start_angle, end_angle, 20)
                    arc_radius = radius - 0.1
                    ax.plot(arc_angles, [arc_radius] * len(arc_angles),
                           color=self.colors['island_predicted'], 
                           linewidth=6, alpha=0.7)
        
        # Plot GC content as inner circle
        positions = [pos[0] + window_results['window_size']//2 
                    for pos in window_results['positions']]
        window_profiler = window_results['profiler']
        gc_contents = [window_profiler.calculate_gc_content(window_id) 
                      for window_id in window_results['windows']]
        
        # Normalize GC content for radius
        gc_min, gc_max = min(gc_contents), max(gc_contents)
        gc_normalized = [(gc - gc_min) / (gc_max - gc_min) * 0.3 + 0.4 
                        for gc in gc_contents]
        
        angles_gc = [pos_to_angle(pos) for pos in positions]
        ax.scatter(angles_gc, gc_normalized, c=gc_contents, cmap='RdYlGn', 
                  s=20, alpha=0.7)
        
        # Formatting
        ax.set_ylim(0, 1.5)
        ax.set_rticks([])  # Remove radial ticks
        ax.set_title('Circular Genome View\nInner: GC Content, Outer: True Islands, Middle: Predictions',
                    pad=20, fontsize=14, fontweight='bold')
        
        # Add position markers (every 10% of genome)
        for i in range(0, 11):
            angle = 2 * np.pi * i / 10
            pos = int(genome_length * i / 10)
            ax.text(angle, 1.3, f'{pos/1000:.0f}kb', ha='center', va='center', 
                   fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_island_details(self, genome_data: Dict, island_regions: List[Dict],
                          predictions: Dict, figsize: Tuple[int, int] = (15, 8)):
        """
        Create detailed view of specific island regions.
        
        Args:
            genome_data: Genome data dictionary
            island_regions: List of island region dictionaries
            predictions: Prediction results
            figsize: Figure size
        """
        n_islands = len(island_regions)
        fig, axes = plt.subplots(n_islands, 1, figsize=figsize)
        if n_islands == 1:
            axes = [axes]
        
        fig.suptitle('Detailed Island Region Analysis', fontsize=16, fontweight='bold')
        
        for i, (island, ax) in enumerate(zip(island_regions, axes)):
            self._plot_single_island_detail(ax, genome_data, island, predictions)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_single_island_detail(self, ax, genome_data: Dict, island: Dict, 
                                 predictions: Dict):
        """Plot detailed view of a single island region."""
        start, end = island['start'], island['end']
        island_length = end - start
        buffer = int(island_length * 0.2)  # 20% buffer on each side
        
        plot_start = max(0, start - buffer)
        plot_end = min(len(genome_data['genome']), end + buffer)
        
        # Draw region
        ax.add_patch(Rectangle((plot_start, -0.5), plot_end - plot_start, 1,
                              facecolor='lightgray', alpha=0.3))
        
        # Highlight island region
        ax.add_patch(Rectangle((start, -0.5), end - start, 1,
                              facecolor=self.colors.get(island['type'], 'red'), 
                              alpha=0.6, edgecolor='black', linewidth=2))
        
        # Add compositional analysis
        window_size = min(1000, island_length // 5)
        if window_size > 0:
            positions = []
            gc_contents = []
            
            sequence = genome_data['genome']
            for pos in range(plot_start, plot_end, window_size):
                window_end = min(pos + window_size, plot_end)
                window_seq = sequence[pos:window_end]
                if len(window_seq) > 0:
                    gc = (window_seq.count('G') + window_seq.count('C')) / len(window_seq)
                    positions.append(pos + window_size // 2)
                    gc_contents.append(gc)
            
            # Plot GC content
            ax2 = ax.twinx()
            ax2.plot(positions, gc_contents, 'g-', linewidth=2, alpha=0.8)
            ax2.set_ylabel('GC Content', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
        
        # Formatting
        ax.set_xlim(plot_start, plot_end)
        ax.set_ylim(-0.6, 0.6)
        ax.set_ylabel('Island Region')
        ax.set_title(f"{island['type'].title()} Island: {start:,} - {end:,} bp ({island_length:,} bp)")
        
        # Format x-axis
        self._format_genomic_axis(ax, plot_end - plot_start)
        ax.grid(True, alpha=0.3)
    
    def create_summary_map(self, multiple_genomes: Dict[str, Dict], 
                          all_predictions: Dict[str, Dict], 
                          figsize: Tuple[int, int] = (16, 12)):
        """
        Create a summary map comparing islands across multiple genomes.
        
        Args:
            multiple_genomes: Dictionary of genome_id -> genome_data
            all_predictions: Dictionary of genome_id -> predictions
            figsize: Figure size
        """
        n_genomes = len(multiple_genomes)
        fig, axes = plt.subplots(n_genomes, 1, figsize=figsize, sharex=True)
        if n_genomes == 1:
            axes = [axes]
        
        fig.suptitle('Multi-Genome Island Comparison', fontsize=16, fontweight='bold')
        
        # Find maximum genome length for consistent scaling
        max_length = max(len(genome_data['genome']) for genome_data in multiple_genomes.values())
        
        for i, (genome_id, genome_data) in enumerate(multiple_genomes.items()):
            ax = axes[i]
            predictions = all_predictions.get(genome_id, {})
            
            genome_length = len(genome_data['genome'])
            
            # Draw genome
            ax.add_patch(Rectangle((0, -0.3), genome_length, 0.6,
                                  facecolor=self.colors['chromosome'], 
                                  edgecolor='black', alpha=0.7))
            
            # Add true islands if available
            if 'annotations' in genome_data:
                for ann in genome_data['annotations']:
                    start, end = ann['start'], ann['end']
                    ax.add_patch(Rectangle((start, -0.3), end - start, 0.6,
                                         facecolor=self.colors.get(ann['type'], 'red'), 
                                         alpha=0.8, edgecolor='darkred'))
            
            # Add predicted islands
            if 'positions' in predictions:
                for pos, is_anomaly in zip(predictions['positions'], predictions['is_anomaly']):
                    if is_anomaly:
                        start, end = pos
                        ax.add_patch(Rectangle((start, -0.4), end - start, 0.2,
                                             facecolor=self.colors['island_predicted'], 
                                             alpha=0.8))
            
            ax.set_xlim(0, max_length)
            ax.set_ylim(-0.5, 0.5)
            ax.set_ylabel(f'{genome_id}\n({genome_length/1000:.0f}kb)', rotation=0, 
                         ha='right', va='center')
            ax.set_yticks([])
        
        # Format final axis
        self._format_genomic_axis(axes[-1], max_length)
        axes[-1].set_xlabel('Genomic Position')
        
        # Add legend
        legend_elements = [
            patches.Patch(color=self.colors['chromosome'], label='Genome'),
            patches.Patch(color=self.colors['island_confirmed'], label='True Islands'),
            patches.Patch(color=self.colors['island_predicted'], label='Predicted Islands')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        plt.show()


# Add this method to the existing ResultsVisualizer class
def plot_genomic_locations(self, genome_data: Dict, window_results: Dict, 
                         predictions: Dict, **kwargs):
    """
    Plot genomic locations of predicted islands.
    
    This method should be added to the ResultsVisualizer class in plots.py
    """
    genomic_viz = GenomicLocationVisualizer()
    
    plot_type = kwargs.get('plot_type', 'overview')
    
    if plot_type == 'overview':
        genomic_viz.plot_genomic_overview(genome_data, window_results, predictions)
    elif plot_type == 'circular':
        genomic_viz.plot_circular_genome(genome_data, predictions, window_results)
    elif plot_type == 'details' and 'island_regions' in kwargs:
        genomic_viz.plot_island_details(genome_data, kwargs['island_regions'], predictions)
    else:
        genomic_viz.plot_genomic_overview(genome_data, window_results, predictions)


# Extension for the main predictor class
def visualize_genomic_locations(self, genome_data: Dict, window_results: Dict,
                              plot_type: str = 'overview', **kwargs):
    """
    Visualize genomic locations of predicted islands.
    
    This method should be added to the GenomeIslandPredictor class.
    
    Args:
        genome_data: Dictionary containing genome sequence and annotations
        window_results: Results from sliding_window_analysis
        plot_type: Type of plot ('overview', 'circular', 'details')
        **kwargs: Additional plotting parameters
    """
    from .visualization.genomic_plots import GenomicLocationVisualizer
    
    # Get predictions from isolation forest results
    if 'isolation_forest' in self.results:
        predictions = self.results['isolation_forest'].copy()
        # Map window predictions back to genomic positions
        predictions['positions'] = window_results['positions']
    else:
        print("No isolation forest results available. Run isolation_forest_analysis first.")
        return
    
    genomic_viz = GenomicLocationVisualizer()
    
    if plot_type == 'overview':
        genomic_viz.plot_genomic_overview(genome_data, window_results, predictions)
    elif plot_type == 'circular':
        genomic_viz.plot_circular_genome(genome_data, predictions, window_results)
    elif plot_type == 'details':
        island_regions = kwargs.get('island_regions', genome_data.get('annotations', []))
        genomic_viz.plot_island_details(genome_data, island_regions, predictions)
    else:
        print(f"Unknown plot type: {plot_type}")
        print("Available types: 'overview', 'circular', 'details'")


# Example usage function
def example_genomic_visualization():
    """
    Example of how to use the genomic visualization features.
    """
    from genome_island_predictor import KmerProfiler, GenomeIslandPredictor
    from genome_island_predictor.utils.synthetic_data import generate_realistic_genome
    
    # Generate realistic genome with islands
    genome_data = generate_realistic_genome(
        genome_size=100000,  # 100kb
        island_regions=[
            (20000, 25000, 'prophage'),
            (60000, 67000, 'pathogenicity'),
            (85000, 90000, 'antibiotic_resistance')
        ]
    )
    
    # Set up analysis
    profiler = KmerProfiler(k=4, normalize=True)
    profiler.add_sequence('synthetic_genome', genome_data['genome'])
    predictor = GenomeIslandPredictor(profiler)
    
    # Sliding window analysis
    window_results = predictor.sliding_window_analysis(
        'synthetic_genome', 
        window_size=5000, 
        step_size=2500
    )
    
    # Anomaly detection on windows
    window_profiler = window_results['profiler']
    window_predictor = GenomeIslandPredictor(window_profiler)
    predictions = window_predictor.isolation_forest_analysis(contamination=0.15)
    
    # Visualize results
    genomic_viz = GenomicLocationVisualizer()
    
    # 1. Overview plot
    genomic_viz.plot_genomic_overview(genome_data, window_results, predictions)
    
    # 2. Circular plot
    genomic_viz.plot_circular_genome(genome_data, predictions, window_results)
    
    # 3. Detailed island views
    genomic_viz.plot_island_details(genome_data, genome_data['annotations'], predictions)
    
    print("✅ Genomic visualization example completed!")


if __name__ == "__main__":
    example_genomic_visualization()