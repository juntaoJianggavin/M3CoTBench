#!/usr/bin/env python3
"""
LCS (Longest Common Subsequence) Reasoning Path Similarity Analyzer
Finds the mode path first, then calculates LCS similarity with each path.

Usage:
    1. Run with default settings:
       python lcs_analyzer.py
    2. Run with custom paths (hyperparameters):
       python lcs_analyzer.py --type_file "path/to/type.xlsx" --csv_dir "path/to/csvs"
"""

import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ========== Configuration Defaults ==========
# ============================================================

# Default paths (used if no command line arguments are provided)
DEFAULT_TYPE_FILE = "example_data/input_data/type.xlsx"
DEFAULT_CSV_DIR = "example_data/processed_output"
DEFAULT_OUTPUT_DIR = "example_data/analysis_results"

# CSV report filename (without path, filename only)
OUTPUT_CSV_FILENAME = "lcs_analysis_summary.csv"

# Plots directory name (without path, directory name only)
OUTPUT_PLOTS_DIRNAME = "lcs_analysis_plots"

# Field mapping configuration (CSV fields -> reasoning types)
FIELD_MAPPING = {
    'modality_order': 'modality',
    'feature_order': 'observation',  # feature maps to observation
    'conclusion_order': 'conclusion',
    'others_order': 'knowledge'  # others maps to knowledge
}

# ============================================================
# Matplotlib Configuration
# ============================================================

def setup_matplotlib():
    """
    Configure matplotlib basic settings
    Since all chart text uses English, default matplotlib fonts are sufficient
    """
    # Fix minus sign display issue
    plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Analyzer Class
# ============================================================

class LCSAnalyzer:
    def __init__(self, type_file: str = None, csv_dir: str = None, output_dir: str = None, 
                 output_csv_filename: str = None, output_plots_dirname: str = None):
        """
        Initialize analyzer
        
        Args:
            type_file: Question type Excel file path
            csv_dir: CSV file directory path
            output_dir: Output directory path
            output_csv_filename: CSV report filename
            output_plots_dirname: Plots directory name
        """
        project_root = Path(__file__).parent
        
        # Use provided parameters or defaults
        self.type_file = Path(type_file) if type_file else project_root / DEFAULT_TYPE_FILE
        self.csv_dir = Path(csv_dir) if csv_dir else project_root / DEFAULT_CSV_DIR
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = project_root / DEFAULT_OUTPUT_DIR
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set output filename and directory name
        self.output_csv_filename = output_csv_filename or OUTPUT_CSV_FILENAME
        self.output_plots_dirname = output_plots_dirname or OUTPUT_PLOTS_DIRNAME
        
        self.analysis_types = None
        self.model_data = {}
        self.field_mapping = FIELD_MAPPING
        
    def load_analysis_types(self):
        """Load question type data"""
        print("📖 Loading question type data...")
        print(f"   File path: {self.type_file}")
        
        if not self.type_file.exists():
            print(f"❌ type.xlsx file does not exist: {self.type_file}")
            return False
            
        try:
            df = pd.read_excel(self.type_file)
            print(f"✅ Successfully loaded type.xlsx, contains {len(df)} records")
            
            # Check required fields
            if 'index' not in df.columns or 'analysis_type' not in df.columns:
                print(f"❌ type.xlsx missing required fields: index or analysis_type")
                print(f"   Existing fields: {list(df.columns)}")
                return False
            
            self.analysis_types = df.set_index('index')['analysis_type'].to_dict()
            print(f"📊 Question type statistics:")
            type_counts = df['analysis_type'].value_counts()
            for type_name, count in type_counts.items():
                print(f"  {type_name}: {count} records")
                
            return True
            
        except Exception as e:
            print(f"❌ Error loading type.xlsx: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_reasoning_path(self, row):
        """
        Extract reasoning path
        Convert reasoning order to reasoning path list
        Uses configured field mapping: modality_order, feature_order, conclusion_order, others_order
        """
        try:
            # Read order values using configured field mapping
            modality_order = int(row['modality_order']) if pd.notna(row.get('modality_order')) else 0
            feature_order = int(row['feature_order']) if pd.notna(row.get('feature_order')) else 0
            conclusion_order = int(row['conclusion_order']) if pd.notna(row.get('conclusion_order')) else 0
            others_order = int(row['others_order']) if pd.notna(row.get('others_order')) else 0
            
            # Build (order, type) pairs, then sort by order
            order_type_pairs = []
            if modality_order > 0:
                order_type_pairs.append((modality_order, 'modality'))
            if feature_order > 0:
                order_type_pairs.append((feature_order, 'observation'))  # feature maps to observation
            if conclusion_order > 0:
                order_type_pairs.append((conclusion_order, 'conclusion'))
            if others_order > 0:
                order_type_pairs.append((others_order, 'knowledge'))  # others maps to knowledge
            
            # Sort by order, extract reasoning path
            order_type_pairs.sort(key=lambda x: x[0])
            path = [pair[1] for pair in order_type_pairs]
            
            return path
        except Exception as e:
            # Debug info
            print(f"⚠️  Failed to extract reasoning path: {e}")
            return []
    
    def lcs_length(self, path1, path2):
        """
        Calculate the longest common subsequence length of two reasoning paths
        Uses dynamic programming algorithm
        """
        m, n = len(path1), len(path2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if path1[i-1] == path2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def calculate_path_similarity(self, path1, path2):
        """
        Calculate similarity between two reasoning paths
        Uses LCS method: sim(P1, P2) = LCS(P1, P2) / max(len(P1), len(P2))
        """
        if not path1 and not path2:
            return 1.0  # Two empty paths considered completely similar
        if not path1 or not path2:
            return 0.0  # One empty, one not empty, considered dissimilar
        
        lcs_len = self.lcs_length(path1, path2)
        max_len = max(len(path1), len(path2))
        
        return lcs_len / max_len
    
    def analyze_single_model_consistency(self, csv_file):
        """Analyze single model's reasoning consistency (based on mode path)"""
        model_name = csv_file.stem
        print(f"\n🔍 Analyzing model reasoning consistency: {model_name}")
        
        try:
            # Read CSV file (supports multiple encodings, handles Windows-edited files)
            encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
            df = None
            encoding_used = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    encoding_used = encoding
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if df is None:
                print(f"  ❌ Unable to read CSV file (encoding issue)")
                return
            
            # If non-utf-8-sig encoding used, note it
            if encoding_used and encoding_used != 'utf-8-sig':
                print(f"  ⚠️  Read using {encoding_used} encoding (file may have been edited in Windows)")
            
            print(f"  📊 Number of records: {len(df)}")
            
            # Check required fields (using configured field mapping)
            required_fields = ['index', 'modality_order', 'feature_order', 
                             'conclusion_order', 'others_order']
            
            missing_fields = [field for field in required_fields if field not in df.columns]
            if missing_fields:
                print(f"  ❌ Missing fields: {missing_fields}")
                print(f"  📋 Existing fields: {list(df.columns)}")
                return
            
            # Add question type
            df['analysis_type'] = df['index'].map(self.analysis_types)
            
            # Extract reasoning paths
            df['reasoning_path'] = df.apply(self.extract_reasoning_path, axis=1)
            
            # Analyze by question type group
            model_consistency = {}
            total_records = len(df)  # Total records for this model
            
            for analysis_type in df['analysis_type'].dropna().unique():
                type_data = df[df['analysis_type'] == analysis_type]
                
                if len(type_data) == 0:
                    continue
                
                print(f"  📈 {analysis_type}: {len(type_data)} records")
                
                # Count reasoning paths
                path_counts = Counter([tuple(path) for path in type_data['reasoning_path']])
                
                if not path_counts:
                    continue
                
                # Find mode path (most common path)
                most_common_path = list(path_counts.most_common(1)[0][0])
                most_common_count = path_counts.most_common(1)[0][1]
                most_common_ratio = most_common_count / len(type_data)
                
                print(f"    Mode path: {most_common_path} (ratio: {most_common_ratio:.1%})")
                
                # Calculate similarity of each path with mode path
                similarities = []
                path_similarity_details = []
                
                for path_tuple, count in path_counts.items():
                    path = list(path_tuple)
                    similarity = self.calculate_path_similarity(most_common_path, path)
                    similarities.append(similarity)
                    
                    path_similarity_details.append({
                        'path': path,
                        'count': count,
                        'similarity': similarity
                    })
                    
                    print(f"      Path {path}: similarity {similarity:.3f} (appeared {count} times)")
                
                # Calculate weighted average similarity
                total_records = len(type_data)
                weighted_similarity = sum(
                    detail['similarity'] * detail['count'] / total_records
                    for detail in path_similarity_details
                )
                
                print(f"    Weighted average similarity: {weighted_similarity:.3f}")
                
                model_consistency[analysis_type] = {
                    'total_records': len(type_data),
                    'mode_path': most_common_path,
                    'mode_ratio': most_common_ratio,
                    'weighted_similarity': weighted_similarity,
                    'path_distribution': dict(path_counts),
                    'path_similarity_details': path_similarity_details
                }
            
            # Calculate overall average similarity
            type_similarities = [data['weighted_similarity'] for data in model_consistency.values()]
            overall_average_similarity = np.mean(type_similarities) if type_similarities else 0
            
            print(f"  🎯 Overall average similarity: {overall_average_similarity:.3f} (based on {len(type_similarities)} types)")
            
            self.model_data[model_name] = {
                'consistency_by_type': model_consistency,
                'overall_average_similarity': overall_average_similarity,
                'total_records': total_records,
                'type_count': len(type_similarities)
            }
            
        except Exception as e:
            print(f"  ❌ Error analyzing model {model_name}: {e}")
    
    def analyze_all_models(self):
        """Analyze reasoning consistency of all models"""
        print("🚀 Starting reasoning path consistency analysis (based on mode path)")
        print("=" * 60)
        
        # Load question type data
        if not self.load_analysis_types():
            return
        
        # Find CSV files
        csv_files = list(self.csv_dir.glob("*.csv"))
        # Exclude type.xlsx (if there's a CSV with same name)
        csv_files = [f for f in csv_files if f.name != 'type.csv']
        if not csv_files:
            print(f"❌ No CSV files found in: {self.csv_dir}")
            return
        
        print(f"✅ Found {len(csv_files)} CSV files")
        
        # Analyze each model
        for csv_file in csv_files:
            self.analyze_single_model_consistency(csv_file)
    
    def analyze_cross_model_comparison(self):
        """Analyze cross-model reasoning path comparison"""
        print("\n🔍 Analyzing cross-model reasoning path comparison")
        print("=" * 60)
        
        cross_model_comparison = {}
        
        # Analyze by question type
        for analysis_type in self.analysis_types.values():
            print(f"\n📈 Question type: {analysis_type}")
            
            # Collect mode paths of all models for this type
            model_mode_paths = {}
            for model_name, model_data in self.model_data.items():
                if analysis_type in model_data['consistency_by_type']:
                    type_data = model_data['consistency_by_type'][analysis_type]
                    model_mode_paths[model_name] = type_data['mode_path']
            
            if len(model_mode_paths) < 2:
                continue
            
            print(f"  Mode paths of each model:")
            for model_name, mode_path in model_mode_paths.items():
                print(f"    {model_name}: {mode_path}")
            
            # Calculate cross-model similarity
            similarities = []
            model_pairs = []
            
            model_names = list(model_mode_paths.keys())
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], i+1):
                    similarity = self.calculate_path_similarity(
                        model_mode_paths[model1], 
                        model_mode_paths[model2]
                    )
                    similarities.append(similarity)
                    model_pairs.append((model1, model2))
                    
                    print(f"  {model1} vs {model2}: {similarity:.3f}")
                    print(f"    Path 1: {model_mode_paths[model1]}")
                    print(f"    Path 2: {model_mode_paths[model2]}")
            
            avg_cross_similarity = np.mean(similarities) if similarities else 0
            print(f"  Average cross-model similarity: {avg_cross_similarity:.3f}")
            
            cross_model_comparison[analysis_type] = {
                'model_mode_paths': model_mode_paths,
                'similarities': similarities,
                'model_pairs': model_pairs,
                'avg_similarity': avg_cross_similarity
            }
        
        return cross_model_comparison
    
    def generate_consistency_report(self, cross_model_comparison):
        """Generate consistency report"""
        print("\n📊 Generating consistency report")
        print("=" * 60)
        
        # Create summary data
        summary_data = []
        
        for model_name, model_data in self.model_data.items():
            for analysis_type, type_data in model_data['consistency_by_type'].items():
                summary_data.append({
                    'model': model_name,
                    'analysis_type': analysis_type,
                    'mode_path': ' -> '.join(type_data['mode_path']),
                    'mode_ratio': type_data['mode_ratio'],
                    'weighted_similarity': type_data['weighted_similarity'],
                    'overall_average_similarity': model_data['overall_average_similarity'],
                    'total_records': type_data['total_records']
                })
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary report
        report_file = self.output_dir / self.output_csv_filename
        summary_df.to_csv(report_file, index=False)
        print(f"💾 Consistency report saved: {report_file}")
        
        # Display summary statistics
        print(f"\n📈 Consistency analysis summary:")
        print(f"  Number of models: {len(self.model_data)}")
        print(f"  Number of question types: {summary_df['analysis_type'].nunique()}")
        
        # Display model consistency ranking
        model_similarities = summary_df.groupby('model')['overall_average_similarity'].first().sort_values(ascending=False)
        print(f"\n🎯 Model consistency ranking:")
        for i, (model_name, similarity) in enumerate(model_similarities.items(), 1):
            print(f"  {i}. {model_name}: {similarity:.3f}")
        
        return summary_df
    
    def create_consistency_visualizations(self, summary_df, cross_model_comparison):
        """Create consistency visualization charts"""
        print("\n🎨 Creating consistency visualization charts")
        print("=" * 60)
        
        # Configure matplotlib basic settings
        setup_matplotlib()
        
        # Create plots directory
        plots_dir = self.output_dir / self.output_plots_dirname
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Model consistency ranking
        self.create_consistency_ranking(summary_df, plots_dir)
        
        # 2. Cross-model similarity heatmap
        self.create_cross_model_heatmap(cross_model_comparison, plots_dir)
        
        # 3. Mode path distribution plot
        self.create_mode_path_distribution_plot(summary_df, plots_dir)
        
        print(f"💾 Charts saved to: {plots_dir}")
    
    def create_consistency_ranking(self, summary_df, plots_dir):
        """Create consistency ranking chart"""
        # Calculate average consistency score for each model
        model_avg_consistency = summary_df.groupby('model')['overall_average_similarity'].first().sort_values(ascending=False)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(model_avg_consistency)), model_avg_consistency.values)
        plt.xticks(range(len(model_avg_consistency)), model_avg_consistency.index, rotation=45)
        plt.title('Model Reasoning Consistency Ranking (Based on Mode Path)')
        plt.xlabel('Model')
        plt.ylabel('Average Consistency Score')
        
        # Add value labels
        for i, (bar, similarity) in enumerate(zip(bars, model_avg_consistency.values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                     f'{similarity:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'consistency_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  📊 Consistency ranking chart saved")
    
    def create_cross_model_heatmap(self, cross_model_comparison, plots_dir):
        """Create cross-model similarity heatmap"""
        # Prepare heatmap data
        all_models = set()
        for analysis_type, data in cross_model_comparison.items():
            all_models.update(data['model_mode_paths'].keys())
        
        all_models = sorted(list(all_models))
        
        # Create heatmap for each question type
        for analysis_type, data in cross_model_comparison.items():
            model_mode_paths = data['model_mode_paths']
            
            if len(model_mode_paths) < 2:
                continue
            
            # Create similarity matrix
            similarity_matrix = np.zeros((len(all_models), len(all_models)))
            
            for i, model1 in enumerate(all_models):
                for j, model2 in enumerate(all_models):
                    if model1 in model_mode_paths and model2 in model_mode_paths:
                        similarity = self.calculate_path_similarity(
                            model_mode_paths[model1], model_mode_paths[model2]
                        )
                        similarity_matrix[i][j] = similarity
                    elif i == j:
                        similarity_matrix[i][j] = 1.0
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                        xticklabels=all_models, yticklabels=all_models)
            plt.title(f'{analysis_type} - Cross-Model Mode Path Similarity')
            plt.tight_layout()
            plt.savefig(plots_dir / f'{analysis_type}_cross_model_similarity.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()
        
        print("  📊 Cross-model similarity heatmap saved")
    
    def create_mode_path_distribution_plot(self, summary_df, plots_dir):
        """Create mode path distribution plot"""
        # Count mode paths by question type
        path_stats = summary_df.groupby(['analysis_type', 'mode_path']).size().reset_index(name='count')
        
        # Create path distribution plot for each question type
        for analysis_type in summary_df['analysis_type'].unique():
            type_data = path_stats[path_stats['analysis_type'] == analysis_type]
            
            if len(type_data) == 0:
                continue
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(type_data)), type_data['count'])
            plt.xticks(range(len(type_data)), type_data['mode_path'], rotation=45, ha='right')
            plt.title(f'{analysis_type} - Mode Reasoning Path Distribution')
            plt.xlabel('Reasoning Path')
            plt.ylabel('Number of Models')
            
            # Add value labels
            for bar, count in zip(bars, type_data['count']):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                         str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'{analysis_type}_mode_path_distribution.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()
        
        print("  📊 Mode path distribution plot saved")
    
    def run_analysis(self):
        """Run complete analysis"""
        print("🚀 LCS Reasoning Path Consistency Analysis Started")
        print("=" * 60)
        
        # Analyze all models
        self.analyze_all_models()
        
        # Analyze cross-model comparison
        cross_model_comparison = self.analyze_cross_model_comparison()
        
        # Generate report
        summary_df = self.generate_consistency_report(cross_model_comparison)
        
        if summary_df is not None and not summary_df.empty:
            # Create visualizations
            self.create_consistency_visualizations(summary_df, cross_model_comparison)
            
            print("\n✅ LCS analysis completed!")
            print(f"📁 Output directory: {self.output_dir}")
            print("📁 Output files:")
            print(f"  - {self.output_dir / self.output_csv_filename} (consistency report)")
            print(f"  - {self.output_dir / self.output_plots_dirname / ''} (visualization charts)")
        else:
            print("\n❌ Analysis failed, no valid data generated")

def main():
    """Main function handling argument parsing"""
    parser = argparse.ArgumentParser(description="LCS Reasoning Path Similarity Analyzer")
    
    parser.add_argument('--type_file', type=str, default=DEFAULT_TYPE_FILE,
                        help=f'Path to the question type Excel file (default: {DEFAULT_TYPE_FILE})')
    
    parser.add_argument('--csv_dir', type=str, default=DEFAULT_CSV_DIR,
                        help=f'Directory containing model CSV files (default: {DEFAULT_CSV_DIR})')
    
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save analysis results (default: {DEFAULT_OUTPUT_DIR})')

    args = parser.parse_args()
    
    analyzer = LCSAnalyzer(
        type_file=args.type_file,
        csv_dir=args.csv_dir,
        output_dir=args.output_dir
    )
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
