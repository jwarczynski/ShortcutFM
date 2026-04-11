"""Analysis package for ShortcutFM."""

from .metrics_analysis import (
    assign_colors_to_pairs,
    build_best_df,
    calculate_nfe,
    clean_metric_name,
    create_baseline_comparison_plots,
    create_correlation_heatmap,
    create_hierarchical_legend_mapping,
    flatten_dictionary_columns,
    generate_baseline_comparison_summary,
    get_metric_columns,
    load_dataset_metrics,
    map_markers,
    map_paired_visual_properties,
    plot_experiment_performance,
    plot_metric_over_training_steps,
    setup_plot_style,
)

__all__ = [
    "assign_colors_to_pairs",
    "build_best_df",
    "calculate_nfe",
    "clean_metric_name",
    "create_baseline_comparison_plots",
    "create_correlation_heatmap",
    "create_hierarchical_legend_mapping",
    "flatten_dictionary_columns",
    "generate_baseline_comparison_summary",
    "get_metric_columns",
    "load_dataset_metrics",
    "map_markers",
    "map_paired_visual_properties",
    "plot_experiment_performance",
    "plot_metric_over_training_steps",
    "setup_plot_style",
]
