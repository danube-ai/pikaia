import pandas as pd


def create_summary_table(results, processed_datasets):
    """
    Create a summary table of results across datasets, architectures, network types, and parameters.

    Args:
        results: Dictionary with results keyed as f"{dataset_name}_{arch}_{net_type}_{param}"
        processed_datasets: Dictionary of processed datasets

    Returns:
        pd.DataFrame: Summary table with columns: Dataset, Architecture, Network_Type, Parameter, Final_Bal_Acc, Best_Bal_Acc, Epochs_Trained
    """
    summary_data = []

    for key, metrics in results.items():
        parts = key.rsplit("_", 3)
        dataset_name = "_".join(parts[:-3])
        arch = parts[-3]
        net_type = parts[-2]
        param = int(parts[-1])  # Assuming param is int

        final_bal_acc = (
            metrics["val_bal_acc_history"][-1]
            if metrics["val_bal_acc_history"]
            else None
        )
        best_bal_acc = (
            max(metrics["val_bal_acc_history"])
            if metrics["val_bal_acc_history"]
            else None
        )
        epochs_trained = len(metrics["val_bal_acc_history"])
        num_params = metrics.get("num_params", None)
        training_time = metrics.get("training_time", None)
        param_time_ratio = (
            num_params / training_time if training_time and training_time > 0 else None
        )
        bal_acc_param_ratio = (
            final_bal_acc / num_params
            if final_bal_acc and num_params and num_params > 0
            else None
        )

        summary_data.append(
            {
                "Dataset": dataset_name,
                "Architecture": arch,
                "Network_Type": net_type,
                "Parameter": param,
                "Final_Bal_Acc": final_bal_acc,
                "Best_Bal_Acc": best_bal_acc,
                "Epochs_Trained": epochs_trained,
                "Num_Params": num_params,
                "Training_Time": training_time,
                "Param_Time_Ratio": param_time_ratio,
                "Bal_Acc_Param_Ratio": bal_acc_param_ratio,
            }
        )

    df = pd.DataFrame(summary_data)
    df = df.sort_values(by=["Dataset", "Architecture", "Network_Type", "Parameter"])
    return df
