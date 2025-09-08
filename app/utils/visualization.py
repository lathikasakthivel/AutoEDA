#!/usr/bin/env python3
"""
Visualization Generator for AutoEDA

This module provides utilities for generating interactive visualizations
including trend analysis, anomaly detection plots, feature importance,
and synthetic data quality assessment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Tuple, Optional, Any
import os
import json
from datetime import datetime

class VisualizationGenerator:
    """Generates interactive visualizations for AutoEDA results"""
    
    def __init__(self, output_dir="static/plots"):
        self.output_dir = output_dir
        self._create_output_dir()
        self.color_palette = px.colors.qualitative.Set3
        
    def _create_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_data_overview(self, df: pd.DataFrame, filename: str = "data_overview.html") -> str:
        """
        Generate comprehensive data overview visualization
        
        Args:
            df: Input DataFrame
            filename: Output filename
            
        Returns:
            str: Filepath to generated visualization
        """
        print("Generating data overview visualization...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Data Shape & Types', 'Missing Values', 'Numeric Distribution', 
                          'Categorical Distribution', 'Correlation Matrix', 'Time Series Overview'),
            specs=[[{"type": "table"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # 1. Data info table
        info_data = {
            'Metric': ['Shape', 'Memory Usage', 'Numeric Columns', 'Categorical Columns', 'Missing Values'],
            'Value': [
                f"{df.shape[0]} rows × {df.shape[1]} columns",
                f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
                len(df.select_dtypes(include=[np.number]).columns),
                len(df.select_dtypes(include=['object', 'category']).columns),
                df.isnull().sum().sum()
            ]
        }
        
        fig.add_trace(
            go.Table(header=dict(values=list(info_data.keys())),
                    cells=dict(values=list(info_data.values()))),
            row=1, col=1
        )
        
        # 2. Missing values
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        if len(missing_counts) > 0:
            fig.add_trace(
                go.Bar(x=missing_counts.index, y=missing_counts.values,
                      name='Missing Values', marker_color='red'),
                row=1, col=2
            )
        
        # 3. Numeric distribution
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for i, col in enumerate(numeric_cols[:3]):  # Show first 3 numeric columns
                fig.add_trace(
                    go.Histogram(x=df[col].dropna(), name=col, opacity=0.7),
                    row=2, col=1
                )
        
        # 4. Categorical distribution
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for i, col in enumerate(categorical_cols[:3]):  # Show first 3 categorical columns
                value_counts = df[col].value_counts().head(10)
                fig.add_trace(
                    go.Bar(x=value_counts.index, y=value_counts.values, name=col),
                    row=2, col=2
                )
        
        # 5. Correlation matrix
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, x=corr_matrix.index, y=corr_matrix.columns,
                          colorscale='RdBu', zmid=0),
                row=3, col=1
            )
        
        # 6. Time series overview (if datetime index exists)
        if isinstance(df.index, pd.DatetimeIndex):
            sample_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]
            fig.add_trace(
                go.Scatter(x=df.index, y=df[sample_col], mode='lines', name=sample_col),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Data Overview Dashboard",
            height=1200,
            showlegend=True
        )
        
        # Save and return
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)
        return filepath
    
    def generate_trend_analysis(self, df: pd.DataFrame, feature_cols: List[str], 
                               filename: str = "trend_analysis.html") -> str:
        """
        Generate trend analysis visualization
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature columns to analyze
            filename: Output filename
            
        Returns:
            str: Filepath to generated visualization
        """
        print("Generating trend analysis visualization...")
        
        # Determine if data is time series
        is_time_series = isinstance(df.index, pd.DatetimeIndex)
        
        if is_time_series:
            fig = make_subplots(
                rows=len(feature_cols), cols=1,
                subplot_titles=[f'Trend Analysis: {col}' for col in feature_cols],
                shared_xaxes=True
            )
            
            for i, col in enumerate(feature_cols):
                if col in df.columns:
                    # Original data
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df[col], mode='lines', 
                                 name=f'{col} (Original)', line=dict(color='blue')),
                        row=i+1, col=1
                    )
                    
                    # Rolling mean
                    rolling_mean = df[col].rolling(window=min(30, len(df)//10)).mean()
                    fig.add_trace(
                        go.Scatter(x=df.index, y=rolling_mean, mode='lines',
                                 name=f'{col} (Rolling Mean)', line=dict(color='red', dash='dash')),
                        row=i+1, col=1
                    )
                    
                    # Trend line
                    if len(df) > 10:
                        x_trend = np.arange(len(df))
                        z = np.polyfit(x_trend, df[col].fillna(method='ffill'), 1)
                        p = np.poly1d(z)
                        trend_line = p(x_trend)
                        fig.add_trace(
                            go.Scatter(x=df.index, y=trend_line, mode='lines',
                                     name=f'{col} (Trend)', line=dict(color='green', dash='dot')),
                            row=i+1, col=1
                        )
        else:
            # For non-time series data, show distribution and basic statistics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Distribution', 'Box Plot', 'Statistics', 'Scatter Matrix']
            )
            
            for i, col in enumerate(feature_cols[:2]):
                if col in df.columns:
                    # Distribution
                    fig.add_trace(
                        go.Histogram(x=df[col].dropna(), name=col, opacity=0.7),
                        row=1, col=1
                    )
                    
                    # Box plot
                    fig.add_trace(
                        go.Box(y=df[col].dropna(), name=col),
                        row=1, col=2
                    )
            
            # Statistics table
            stats_data = []
            for col in feature_cols:
                if col in df.columns and df[col].dtype in [np.number, 'float64', 'int64']:
                    stats_data.append({
                        'Feature': col,
                        'Mean': f"{df[col].mean():.4f}",
                        'Std': f"{df[col].std():.4f}",
                        'Min': f"{df[col].min():.4f}",
                        'Max': f"{df[col].max():.4f}"
                    })
            
            if stats_data:
                fig.add_trace(
                    go.Table(header=dict(values=list(stats_data[0].keys())),
                            cells=dict(values=[[row[k] for row in stats_data] for k in stats_data[0].keys()])),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title="Trend Analysis Dashboard",
            height=800,
            showlegend=True
        )
        
        # Save and return
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)
        return filepath
    
    def generate_anomaly_visualization(self, df: pd.DataFrame, anomaly_results: Dict[str, Any],
                                      filename: str = "anomaly_analysis.html") -> str:
        """
        Generate anomaly detection visualization
        
        Args:
            df: Input DataFrame
            anomaly_results: Results from anomaly detection
            filename: Output filename
            
        Returns:
            str: Filepath to generated visualization
        """
        print("Generating anomaly visualization...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Anomaly Detection Results', 'Reconstruction Error', 
                          'Anomaly Distribution', 'Feature-wise Anomalies'],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # 1. Anomaly detection results
        if 'anomaly_scores' in anomaly_results:
            scores = anomaly_results['anomaly_scores']
            if isinstance(df.index, pd.DatetimeIndex):
                x_axis = df.index
            else:
                x_axis = list(range(len(scores)))
            
            fig.add_trace(
                go.Scatter(x=x_axis, y=scores, mode='lines', name='Anomaly Score',
                          line=dict(color='blue')),
                row=1, col=1
            )
            
            # Add threshold line
            if 'threshold' in anomaly_results:
                threshold = anomaly_results['threshold']
                fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                            annotation_text=f"Threshold: {threshold:.4f}",
                            row=1, col=1)
        
        # 2. Reconstruction error
        if 'reconstruction_error' in anomaly_results:
            error = anomaly_results['reconstruction_error']
            if isinstance(df.index, pd.DatetimeIndex):
                x_axis = df.index
            else:
                x_axis = list(range(len(error)))
            
            fig.add_trace(
                go.Scatter(x=x_axis, y=error, mode='lines', name='Reconstruction Error',
                          line=dict(color='orange')),
                row=1, col=2
            )
        
        # 3. Anomaly distribution
        if 'anomaly_labels' in anomaly_results:
            labels = anomaly_results['anomaly_labels']
            anomaly_counts = pd.Series(labels).value_counts()
            
            fig.add_trace(
                go.Bar(x=['Normal', 'Anomaly'], y=[anomaly_counts.get(0, 0), anomaly_counts.get(1, 0)],
                      marker_color=['green', 'red']),
                row=2, col=1
            )
        
        # 4. Feature-wise anomalies
        if 'feature_anomalies' in anomaly_results:
            feature_anomalies = anomaly_results['feature_anomalies']
            if feature_anomalies:
                features = list(feature_anomalies.keys())
                anomaly_counts = list(feature_anomalies.values())
                
                fig.add_trace(
                    go.Bar(x=features, y=anomaly_counts, name='Feature Anomalies'),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title="Anomaly Detection Dashboard",
            height=800,
            showlegend=True
        )
        
        # Save and return
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)
        return filepath
    
    def generate_feature_importance(self, feature_importance: Dict[str, float],
                                   attention_weights: Optional[np.ndarray] = None,
                                   filename: str = "feature_importance.html") -> str:
        """
        Generate feature importance visualization
        
        Args:
            feature_importance: Dictionary of feature importance scores
            attention_weights: Optional attention weights array
            filename: Output filename
            
        Returns:
            str: Filepath to generated visualization
        """
        print("Generating feature importance visualization...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Feature Importance', 'Attention Weights', 
                          'Importance Distribution', 'Top Features'],
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        # 1. Feature importance bar chart
        if feature_importance:
            features = list(feature_importance.keys())
            scores = list(feature_importance.values())
            
            fig.add_trace(
                go.Bar(x=features, y=scores, name='Importance Score',
                      marker_color='lightblue'),
                row=1, col=1
            )
        
        # 2. Attention weights heatmap
        if attention_weights is not None:
            fig.add_trace(
                go.Heatmap(z=attention_weights, colorscale='Viridis',
                          name='Attention Weights'),
                row=1, col=2
            )
        
        # 3. Importance distribution
        if feature_importance:
            scores = list(feature_importance.values())
            fig.add_trace(
                go.Histogram(x=scores, name='Importance Distribution',
                           nbinsx=20, opacity=0.7),
                row=2, col=1
            )
        
        # 4. Top features pie chart
        if feature_importance:
            # Get top 10 features
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            top_features = [item[0] for item in sorted_features]
            top_scores = [item[1] for item in sorted_features]
            
            fig.add_trace(
                go.Pie(labels=top_features, values=top_scores, name='Top Features'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Feature Importance Dashboard",
            height=800,
            showlegend=True
        )
        
        # Save and return
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)
        return filepath
    
    def generate_synthetic_data_analysis(self, original_df: pd.DataFrame, 
                                        synthetic_df: pd.DataFrame,
                                        quality_metrics: Dict[str, float],
                                        filename: str = "synthetic_analysis.html") -> str:
        """
        Generate synthetic data analysis visualization
        
        Args:
            original_df: Original dataset
            synthetic_df: Synthetic dataset
            quality_metrics: Quality assessment metrics
            filename: Output filename
            
        Returns:
            str: Filepath to generated visualization
        """
        print("Generating synthetic data analysis visualization...")
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Original vs Synthetic Distribution', 'Quality Metrics',
                          'Feature Comparison', 'Correlation Comparison',
                          'Temporal Patterns', 'Bias Analysis'],
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Distribution comparison
        numeric_cols = original_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            sample_col = numeric_cols[0]
            fig.add_trace(
                go.Histogram(x=original_df[sample_col].dropna(), name='Original', opacity=0.7),
                row=1, col=1
            )
            fig.add_trace(
                go.Histogram(x=synthetic_df[sample_col].dropna(), name='Synthetic', opacity=0.7),
                row=1, col=1
            )
        
        # 2. Quality metrics
        if quality_metrics:
            metrics = list(quality_metrics.keys())
            values = list(quality_metrics.values())
            
            fig.add_trace(
                go.Bar(x=metrics, y=values, name='Quality Score',
                      marker_color='lightgreen'),
                row=1, col=2
            )
        
        # 3. Feature comparison
        if len(numeric_cols) > 1:
            for i, col in enumerate(numeric_cols[:3]):
                fig.add_trace(
                    go.Scatter(x=original_df[col].dropna(), y=synthetic_df[col].dropna(),
                              mode='markers', name=f'{col} Comparison', opacity=0.6),
                    row=2, col=1
                )
        
        # 4. Correlation comparison
        if len(numeric_cols) > 1:
            orig_corr = original_df[numeric_cols].corr()
            synth_corr = synthetic_df[numeric_cols].corr()
            
            fig.add_trace(
                go.Heatmap(z=orig_corr.values, x=orig_corr.index, y=orig_corr.columns,
                          colorscale='RdBu', name='Original Correlation'),
                row=2, col=2
            )
        
        # 5. Temporal patterns (if time series)
        if isinstance(original_df.index, pd.DatetimeIndex):
            sample_col = numeric_cols[0] if len(numeric_cols) > 0 else original_df.columns[0]
            fig.add_trace(
                go.Scatter(x=original_df.index, y=original_df[sample_col], 
                          mode='lines', name='Original', line=dict(color='blue')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=synthetic_df.index, y=synthetic_df[sample_col],
                          mode='lines', name='Synthetic', line=dict(color='red')),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="Synthetic Data Analysis Dashboard",
            height=1200,
            showlegend=True
        )
        
        # Save and return
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)
        return filepath
    
    def generate_summary_report(self, analysis_results: Dict[str, Any],
                               filename: str = "summary_report.html") -> str:
        """
        Generate comprehensive summary report
        
        Args:
            analysis_results: Complete analysis results
            filename: Output filename
            
        Returns:
            str: Filepath to generated visualization
        """
        print("Generating summary report...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Analysis Summary', 'Key Findings', 'Model Performance', 'Recommendations'],
            specs=[[{"type": "table"}, {"type": "bar"}],
                   [{"type": "indicator"}, {"type": "table"}]]
        )
        
        # 1. Analysis summary table
        summary_data = {
            'Metric': ['Dataset Size', 'Features Analyzed', 'Anomalies Detected', 'Model Accuracy'],
            'Value': [
                f"{analysis_results.get('dataset_size', 'N/A')}",
                f"{analysis_results.get('n_features', 'N/A')}",
                f"{analysis_results.get('n_anomalies', 'N/A')}",
                f"{analysis_results.get('model_accuracy', 'N/A'):.4f}" if 'model_accuracy' in analysis_results else 'N/A'
            ]
        }
        
        fig.add_trace(
            go.Table(header=dict(values=list(summary_data.keys())),
                    cells=dict(values=list(summary_data.values()))),
            row=1, col=1
        )
        
        # 2. Key findings
        findings = analysis_results.get('key_findings', [])
        if findings:
            fig.add_trace(
                go.Bar(x=['Finding 1', 'Finding 2', 'Finding 3'], y=findings[:3],
                      name='Key Findings'),
                row=1, col=2
            )
        
        # 3. Model performance indicator
        if 'model_accuracy' in analysis_results:
            accuracy = analysis_results['model_accuracy']
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=accuracy * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Model Accuracy (%)"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                   {'range': [50, 80], 'color': "yellow"},
                                   {'range': [80, 100], 'color': "green"}]}
                ),
                row=2, col=1
            )
        
        # 4. Recommendations table
        recommendations = analysis_results.get('recommendations', [])
        if recommendations:
            rec_data = {
                'Priority': ['High', 'Medium', 'Low'],
                'Recommendation': recommendations[:3]
            }
            
            fig.add_trace(
                go.Table(header=dict(values=list(rec_data.keys())),
                        cells=dict(values=list(rec_data.values()))),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="AutoEDA Analysis Summary Report",
            height=800,
            showlegend=True
        )
        
        # Save and return
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)
        return filepath
    
    def generate_all_visualizations(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate all visualizations for the analysis
        
        Args:
            df: Input DataFrame
            analysis_results: Complete analysis results
            
        Returns:
            Dict[str, str]: Mapping of visualization type to filepath
        """
        print("Generating all visualizations...")
        
        visualizations = {}
        
        try:
            # Data overview
            visualizations['data_overview'] = self.generate_data_overview(df)
            
            # Trend analysis
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(feature_cols) > 0:
                visualizations['trend_analysis'] = self.generate_trend_analysis(df, feature_cols[:5])
            
            # Anomaly visualization
            if 'anomaly_results' in analysis_results:
                visualizations['anomaly_analysis'] = self.generate_anomaly_visualization(
                    df, analysis_results['anomaly_results'])
            
            # Feature importance
            if 'feature_importance' in analysis_results:
                attention_weights = analysis_results.get('attention_weights')
                visualizations['feature_importance'] = self.generate_feature_importance(
                    analysis_results['feature_importance'], attention_weights)
            
            # Synthetic data analysis
            if 'synthetic_data' in analysis_results and 'quality_metrics' in analysis_results:
                visualizations['synthetic_analysis'] = self.generate_synthetic_data_analysis(
                    df, analysis_results['synthetic_data'], analysis_results['quality_metrics'])
            
            # Summary report
            visualizations['summary_report'] = self.generate_summary_report(analysis_results)
            
            print(f"Generated {len(visualizations)} visualizations successfully!")
            
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
        
        return visualizations
