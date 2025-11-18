#!/usr/bin/env python3
"""
Script to compute summary statistics from genetic layer benchmark results.
"""

import pandas as pd


def compute_summary_stats(csv_path):
    """Compute average statistics for each architecture/network type combination."""
    df = pd.read_csv(csv_path)

    # Group by Architecture and Network_Type, compute averages
    summary = (
        df.groupby(["Architecture", "Network_Type"])
        .agg(
            {
                "Best_Bal_Acc": "mean",
                "Final_Bal_Acc": "mean",
                "Training_Time": "mean",
                "Num_Params": "mean",
            }
        )
        .round(3)
    )

    # Format the results
    summary = summary.reset_index()

    # Format numbers nicely
    summary["Avg Best Bal Acc"] = summary["Best_Bal_Acc"]
    summary["Avg Final Bal Acc"] = summary["Final_Bal_Acc"]
    summary["Avg Training Time"] = summary["Training_Time"].apply(lambda x: f"{x:.2f}s")
    summary["Avg Params"] = summary["Num_Params"].apply(lambda x: f"{int(x):,}")

    # Reorder columns
    summary = summary[
        [
            "Architecture",
            "Network_Type",
            "Avg Best Bal Acc",
            "Avg Final Bal Acc",
            "Avg Training Time",
            "Avg Params",
        ]
    ]

    return summary


if __name__ == "__main__":
    csv_path = "artefacts/summary_table.csv"
    summary = compute_summary_stats(csv_path)
    print("Summary Statistics:")
    print(summary.to_string(index=False))
