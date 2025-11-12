import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_architecture_comparison(results, dataset_name, architectures, depths_or_heads):
    """
    Plot balanced accuracy comparison for different architectures and depths/heads.

    Args:
        results: Dictionary with results keyed as f"{dataset}_{arch}_{depth_or_heads}"
        dataset_name: Name of the dataset
        architectures: List of architecture names (e.g., ['feedforward_classical', 'heads_classical', ...])
        depths_or_heads: List of depth/n_heads values
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.set_title(f"{dataset_name.capitalize()} - Balanced Accuracy Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks(np.arange(0, 1.05, 0.05))

    colors = {
        "feedforward_classical": "green",
        "heads_classical": "blue",
        "feedforward_genetic": "red",
        "heads_genetic": "purple",
    }

    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]

    for arch_idx, arch in enumerate(architectures):
        for depth_idx, depth in enumerate(depths_or_heads):
            key = f"{dataset_name}_{arch}_{depth}"
            if key not in results:
                continue

            metrics = results[key]
            epochs = range(1, len(metrics["val_bal_acc_history"]) + 1)

            color = colors.get(arch, "black")
            linestyle = linestyles[arch_idx % len(linestyles)]
            marker = markers[depth_idx % len(markers)]

            ax.plot(
                epochs,
                metrics["val_bal_acc_history"],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=5,
                label=f"{arch.replace('_', ' ').title()} ({depth})",
            )

    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_performance_summary(results, datasets, architectures, depths_or_heads):
    """
    Plot summary of final performance metrics across all datasets.

    Args:
        results: Dictionary with results
        datasets: List of dataset names
        architectures: List of architecture names
        depths_or_heads: List of depth/n_heads values
    """
    # Collect final balanced accuracies
    summary_data = {}
    for arch in architectures:
        summary_data[arch] = {}
        for depth in depths_or_heads:
            summary_data[arch][depth] = []

    for dataset in datasets:
        for arch in architectures:
            for depth in depths_or_heads:
                key = f"{dataset}_{arch}_{depth}"
                if key in results:
                    final_bal_acc = results[key]["val_bal_acc_history"][-1]
                    summary_data[arch][depth].append(final_bal_acc)

    # Plot box plots for each architecture and depth
    fig, axes = plt.subplots(
        len(architectures), 1, figsize=(12, 4 * len(architectures))
    )
    if len(architectures) == 1:
        axes = [axes]

    for idx, arch in enumerate(architectures):
        ax = axes[idx]
        data_to_plot = [summary_data[arch][depth] for depth in depths_or_heads]
        ax.boxplot(data_to_plot, labels=[f"Depth/Heads: {d}" for d in depths_or_heads])
        ax.set_title(
            f"{arch.replace('_', ' ').title()} - Final Balanced Accuracy Across Datasets"
        )
        ax.set_ylabel("Balanced Accuracy")
        ax.set_ylim(0, 1)

    plt.tight_layout()
    return fig


def plot_balanced_accuracy_over_epochs(
    results, dataset, arch_type, models, save_path=None
):
    """
    Plot balanced accuracy over epochs for different network types and parameters.

    Args:
        results: Dictionary with results keyed as f"{dataset}_{arch_type}_{net_type}_{param}"
        dataset: Name of the dataset
        arch_type: Architecture type ('feedforward' or 'heads')
        models: List of tuples (net_type, param) where net_type is 'classical' or 'genetic', param is depth or n_heads
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Balanced Accuracy
    ax.set_title(f"{dataset.capitalize()} {arch_type.capitalize()} - Balanced Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks(np.arange(0, 1.05, 0.05))

    colors = {"classical": "green", "genetic": "blue"}
    # Different linestyles for each model
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    # Same marker for all to avoid confusion
    markers = ["o"] * len(models)

    for i, (net_type, param) in enumerate(models):
        key = f"{dataset}_{arch_type}_{net_type}_{param}"
        metrics = results[key]

        epochs = range(1, len(metrics["train_loss_history"]) + 1)

        color = colors[net_type]
        linestyle = linestyles[i]
        marker = markers[i]

        # Plot balanced accuracy
        ax.plot(
            epochs,
            metrics["val_bal_acc_history"][: len(epochs)],
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=5,
            label=f"{net_type.capitalize()} {param}",
        )

    ax.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    return fig


def plot_bal_acc_vs_params(results, dataset, arch_type, save_path=None):
    """
    Plot final balanced accuracy vs number of parameters for a dataset and architecture type.

    Args:
        results: Dictionary with results
        dataset: Name of the dataset
        arch_type: Architecture type ('feedforward' or 'heads')
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.set_title(
        f"{dataset.capitalize()} {arch_type.capitalize()} - Balanced Accuracy vs Number of Parameters"
    )
    ax.set_xlabel("Number of Parameters")
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, pos: f"{x / 1e6:.0f}M"
            if x >= 1e6
            else (f"{x / 1e3:.0f}K" if x >= 1e3 else f"{x:.0f}")
        )
    )
    ax.set_ylabel("Final Balanced Accuracy")
    ax.set_ylim(0.4, 1.1)
    ax.set_yticks(np.arange(0.5, 1.05, 0.05))

    net_types = ["classical", "genetic"]
    architecture_params = {
        "feedforward": [2, 4],
        "heads": [1, 2],
    }

    colors = {
        "classical": "green",
        "genetic": "blue",
    }

    markers = {1: "^", 2: "o", 4: "s"}

    for net_type in net_types:
        for param in architecture_params[arch_type]:
            key = f"{dataset}_{arch_type}_{net_type}_{param}"
            if key not in results:
                continue

            metrics = results[key]
            num_params = metrics["num_params"]
            final_bal_acc = metrics["val_bal_acc_history"][-1]

            color = colors[net_type]
            marker = markers[param]

            ax.scatter(
                num_params,
                final_bal_acc,
                color=color,
                marker=marker,
                s=100,
                label=f"{net_type} ({param})",
            )

    ax.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    return fig
