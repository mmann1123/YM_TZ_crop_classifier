# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm

data = {
    "Accuracy": {
        "ASAP": "0.83 ± 0.01",
        "Copernicus": "0.86 ± 0.01",
        "DEA": "0.84 ± 0.01",
        "Dynamic World": "0.74 ± 0.01",
        "ESA-CCI": "0.78 ± 0.01",
        "Esri": "0.74 ± 0.01",
        "GFSAD": "0.88 ± 0.01",
        "GLAD": "0.86 ± 0.01",
        "GlobCover": "0.70 ± 0.01",
        "Majority Vote": "0.88 ± 0.01",
        "Mean": "0.82 ± 0.01",
        "Nabil et al.": "0.88 ± 0.01",
        "WorldCover": "0.88 ± 0.01",
    },
    "F1": {
        "ASAP": "0.65 ± 0.05",
        "Copernicus": "0.72 ± 0.06",
        "DEA": "0.69 ± 0.04",
        "Dynamic World": "0.32 ± 0.02",
        "ESA-CCI": "0.60 ± 0.06",
        "Esri": "0.29 ± 0.01",
        "GFSAD": "0.76 ± 0.06",
        "GLAD": "0.69 ± 0.05",
        "GlobCover": "0.51 ± 0.06",
        "Majority Vote": "0.71 ± 0.03",
        "Mean": "0.62 ± 0.04",
        "Nabil et al.": "0.77 ± 0.06",
        "WorldCover": "0.73 ± 0.03",
    },
    "Precision (UA)": {
        "ASAP": "0.92 ± 0.02",
        "Copernicus": "0.86 ± 0.02",
        "DEA": "0.84 ± 0.02",
        "Dynamic World": "0.93 ± 0.02",
        "ESA-CCI": "0.73 ± 0.02",
        "Esri": "0.99 ± 0.01",
        "GFSAD": "0.89 ± 0.02",
        "GLAD": "0.95 ± 0.01",
        "GlobCover": "0.60 ± 0.03",
        "Majority Vote": "0.97 ± 0.01",
        "Mean": "0.88 ± 0.02",
        "Nabil et al.": "0.90 ± 0.02",
        "WorldCover": "0.94 ± 0.01",
    },
    "Recall (PA)": {
        "ASAP": "0.50 ± 0.02",
        "Copernicus": "0.61 ± 0.02",
        "DEA": "0.59 ± 0.01",
        "Dynamic World": "0.19 ± 0.00",
        "ESA-CCI": "0.51 ± 0.02",
        "Esri": "0.17 ± 0.00",
        "GFSAD": "0.67 ± 0.02",
        "GLAD": "0.54 ± 0.02",
        "GlobCover": "0.45 ± 0.02",
        "Majority Vote": "0.56 ± 0.01",
        "Mean": "0.51 ± 0.01",
        "Nabil et al.": "0.67 ± 0.02",
        "WorldCover": "0.60 ± 0.01",
    },
}


# Convert the data to a DataFrame
rows = []
for metric, models in data.items():
    for model, value in models.items():
        estimate, error = value.split(" ± ")
        rows.append(
            {
                "Metric": metric,
                "Model": model,
                "Estimate": float(estimate),
                "Error": float(error),
            }
        )

df = pd.DataFrame(rows)

# Get unique models and metrics
models = df["Model"].unique()
metrics = df["Metric"].unique()

# Map models to x positions
model_indices = np.arange(len(models))

# Generate a color map with a unique color for each model
color_map = cm.get_cmap("viridis", len(models))
colors = {model: color_map(i) for i, model in enumerate(models)}

# Target lines for each metric
target_lines = {
    "Balanced Accuracy": 0.79,
    "Kappa Accuracy": 0.80,
    "Accuracy": 0.82,
    "F1": 0.82,
}

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]
    data_metric = df[df["Metric"] == metric]

    # Ensure consistent model order
    data_metric = data_metric.set_index("Model").loc[models].reset_index()

    estimates = data_metric["Estimate"]
    errors = data_metric["Error"]

    # Plot the data with unique colors for each model
    for j, model in enumerate(models):
        ax.errorbar(
            j,
            estimates[j],
            yerr=errors[j],
            fmt="o",
            capsize=3,
            color=colors[model],
            label=model if i == 0 else "",
        )  # Label only once for legend

    # Add a dashed line if the metric has a target value
    if metric in target_lines:
        ax.axhline(
            target_lines[metric],
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Target for {metric}",
        )

    ax.set_xticks(model_indices)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_title(metric, fontsize=14)
    ax.set_ylabel("Estimate", fontsize=12)

    # Only add x-axis labels to the bottom plots
    if i in [2, 3]:
        ax.set_xlabel("Model", fontsize=12)
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

# Adjust layout to make room for x-axis labels
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.subplots_adjust(hspace=0.3)

# Add a single x-axis label for the entire figure
# fig.text(0.5, 0.01, "Model", ha="center", fontsize=16)

# Add a legend for the models
# fig.legend(models, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05), fontsize=10)

plt.savefig("./writeup/figures/model_performance.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
