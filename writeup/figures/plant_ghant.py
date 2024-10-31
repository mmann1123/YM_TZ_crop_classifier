# %%
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from datetime import datetime
import calendar

# Data from the crop calendar in the image
data = {
    "Crop": [
        "Corn (Msimu)",
        "Corn (Masika)",
        "Corn (Vuli)",
        "Cotton",
        "Millet (Long, Masika)",
        "Millet (Short, Vuli)",
        "Millet (Unimodal, Msimu)",
        "Peanut",
        "Rice",
        "Sorghum (Long, Masika)",
        "Sorghum (Unimodal, Msimu)",
    ],
    "Plant": [
        ("Nov", "Dec"),  # corn msimu
        ("Mar", "Apr"),  # corn masika
        ("Sep", "Nov"),  # corn vuli
        [("Nov", "Dec"), ("Jan", "Feb")],  # cotton
        ("Mar", "Apr"),  # millet long masika
        ("Oct", "Nov"),  # millet short vuli
        [("Nov", "Dec"), ("Jan", "Jan")],  # millet unimodal msimu
        ("Dec", "Dec"),  # peanut
        [("Dec", "Dec"), ("Jan", "Jan")],  # rice
        ("Mar", "Mar"),  # sorghum long masika
        ("Nov", "Dec"),  # sorghum unimodal msimu
    ],
    "Mid-Season": [
        ("Jan", "Jun"),  # corn msimu
        ("Apr", "Jul"),  # corn masika
        ("Nov", "Dec"),  # corn vuli
        ("Feb", "Jul"),  # cotton
        ("Apr", "Jul"),  # millet long masika
        ("Dec", "Dec"),  # millet short vuli
        ("Feb", "May"),  # millet unimodal msimu
        ("Jan", "Feb"),  # peanut
        ("Feb", "Apr"),  # rice
        ("Apr", "Jun"),  # sorghum long masika
        ("Jan", "Apr"),  # sorghum unimodal msimu
    ],
    "Harvest": [
        ("May", "Jun"),  # corn msimu
        ("Jul", "Aug"),  # corn masika
        ("Jan", "Feb"),  # corn vuli
        ("Jul", "Aug"),  # cotton
        ("Jul", "Aug"),  # millet long masika
        ("Jan", "Mar"),  # millet short vuli
        ("May", "Jul"),  # millet unimodal msimu
        ("Mar", "May"),  # peanut
        ("May", "Jul"),  # rice
        ("Jul", "Aug"),  # sorghum long masika
        ("May", "Jun"),  # sorghum unimodal msimu
    ],
}

# Convert months to datetime objects for plotting
months = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def to_date(month):
    return datetime(2022, months[month], 1)


def to_end_date(month):
    year = 2022
    month_num = months[month]
    last_day = calendar.monthrange(year, month_num)[1]
    return datetime(year, month_num, last_day)


# Prepare figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each phase as horizontal bars
for i, crop in enumerate(data["Crop"]):
    # Plotting Plant phase
    if type(data["Plant"][i]) == list:
        for start, end in data["Plant"][i]:
            ax.barh(
                crop,
                (to_end_date(end) - to_date(start)).days,
                left=to_date(start),
                color="green",
                edgecolor="black",
                height=0.4,
            )
    else:
        start, end = data["Plant"][i]
        ax.barh(
            crop,
            (to_end_date(end) - to_date(start)).days,
            left=to_date(start),
            color="green",
            edgecolor="black",
            height=0.4,
        )

    # Plotting Mid-Season phase
    start, end = data["Mid-Season"][i]
    ax.barh(
        crop,
        (to_end_date(end) - to_date(start)).days,
        left=to_date(start),
        color="grey",
        edgecolor="black",
        height=0.4,
    )

    # Plotting Harvest phase
    start, end = data["Harvest"][i]
    ax.barh(
        crop,
        (to_end_date(end) - to_date(start)).days,
        left=to_date(start),
        color="orange",
        edgecolor="black",
        height=0.4,
    )

# Set date formatting on x-axis
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

# Remove the last tick label
ticks = ax.get_xticks()
tick_labels = [mdates.num2date(tick).strftime("%b") for tick in ticks]
ax.set_xticks(ticks[:-1])
ax.set_xticklabels(tick_labels[:-1])


# Add labels and legend
ax.set_xlabel("Months")
ax.set_title("Tanzania Crop Calendar")
legend_elements = [
    Patch(facecolor="green", edgecolor="black", label="Plant"),
    Patch(facecolor="grey", edgecolor="black", label="Mid-Season"),
    Patch(facecolor="orange", edgecolor="black", label="Harvest"),
]
ax.legend(
    handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3
)
plt.tight_layout()
plt.savefig("./figures/plant_ghant.png")

plt.show()


# %%
