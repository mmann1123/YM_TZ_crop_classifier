# %%
# create seaborn histogram of secondary_land_cover with vertical names on x-axis


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import urllib
import PIL
import seaborn as sns
import geopandas as gpd

os.chdir(
    "/mnt/bigdrive/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data/"
)
drop = [
    "Don't know",
    "Other (later, specify in optional notes)",
    "water_body",
    "large_building",
    "could be maize.",
    "no",
    "don_t_know",
    "fallow_barren",  # only two examples
    "forest_shrubland",  # only two examples
]
data = gpd.read_file("outputs/lu_complete_inputs.geojson")
data

data.drop(data[data["lc_name"].isin(drop)].index, inplace=True)

secondary_count = data.groupby(["lc_name"])["lc_name"].count()
secondary_count = secondary_count.reset_index(name="count")

# Calculate the total number of observations
total_observations = secondary_count["count"].sum()

# Calculate the percentage for each category
secondary_count["percentage"] = (secondary_count["count"] / total_observations) * 100

# Sort the DataFrame by percentage in descending order
secondary_count.sort_values(by="percentage", ascending=False, inplace=True)


secondary_count.sort_values(by="percentage", inplace=True, ascending=False)
# Create the histogram plot
ax = sns.barplot(data=secondary_count, x="lc_name", y="percentage")

# Rotate the x-axis labels vertically
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_xlabel("Primary Land Cover", fontsize=15)

# increas font size for x and y axis and axis labels
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


plt.ylabel("% of Total Observations", fontsize=15)

# Use tight_layout to automatically adjust subplot parameters
plt.tight_layout()

plt.savefig(
    f"/home/mmann1123/Documents/github/YM_TZ_crop_classifier/writeup/figures/primary_land_cover.png",
)

# Display the plot
plt.show()


# %%

# NOTES: Millet, Sorghum, maize very similar especially when young,
# maybe combine millet and sorghum into one category
# young plants typically labeled as maize


os.chdir(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover"
)
# Read the CSV file
df = pd.read_csv("../kobo_field_collections/TZ_ground_truth_cleaned_ls.csv")
df = df.reset_index()
df_filtered = df.dropna(subset=["Picture_of_the_field_or_feature_URL", "primar"])
df_filtered = df_filtered[
    [
        "index",
        "Picture_of_the_field_or_feature_URL",
        "primar",
        "Secondary_land_cover",
        "Quality_Drop_Low",
    ]
]
df_filtered
import seaborn as sns

df_filtered["Secondary_land_cover"].replace(
    {
        "vegetables_and_pulses__examples__eggplan": "vegetables_and_pulses",
        "maize__mahindi": "maize",
        "sunflower__alizeti": "sunflower",
        "other__later__specify_in_optional_notes": "other",
        "tree_crops__examples__banana__coconut__g": "tree_crops",
        "other_grains__examples__wheat__barley__o": "other_grains",
        "sorghum__mtama": "sorghum",
        "cotton__pamba": "cotton",
        "peanuts_ground_nuts__karanga": "ground_nuts",
        "rice__mpunga": "rice",
        "millet__ulezi": "millet",
        "specialty_crops__cocoa__coffee__tea__sug": "specialty_crops",
        "grassland_savanna": "grassland_savanna",
        "don_t_know": "don't_know",
    },
    inplace=True,
)


import seaborn as sns

secondary_count = df_filtered.groupby(["Secondary_land_cover"])[
    "Secondary_land_cover"
].count()
secondary_count = secondary_count.reset_index(name="count")

# Calculate the total number of observations
total_observations = secondary_count["count"].sum()

# Calculate the percentage for each category
secondary_count["percentage"] = (secondary_count["count"] / total_observations) * 100

# Sort the DataFrame by percentage in descending order
secondary_count.sort_values(by="percentage", ascending=False, inplace=True)


secondary_count.sort_values(by="percentage", inplace=True, ascending=False)
# Create the histogram plot
ax = sns.barplot(data=secondary_count, x="Secondary_land_cover", y="percentage")

# Rotate the x-axis labels vertically
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_xlabel("Secondary Land Cover", fontsize=15)

# increas font size for x and y axis and axis labels
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


plt.ylabel("Percentage of Total Observations", fontsize=15)


plt.savefig(f"outputs/secondary_land_cover.png", dpi=300)

# Display the plot
plt.show()


# (
#     df_filtered.groupby(["Secondary_land_cover"])["Secondary_land_cover"].count()
#     / df["Secondary_land_cover"].count()
# )


# %% use plant_ghant.py  

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt


# # Create a DataFrame with the start and end months of each growing season
# data = {
#     "Season": ["Masika", "Vuli", "Msimu"],
#     "Start_Month": ["March", "September", "November"],
#     "End_Month": ["August", "February", "June"],
# }
# seasons_df = pd.DataFrame(data)

# # Convert month names to numerical values for sorting
# month_order = [
#     "January",
#     "February",
#     "March",
#     "April",
#     "May",
#     "June",
#     "July",
#     "August",
#     "September",
#     "October",
#     "November",
#     "December",
# ]
# seasons_df["Start_Month"] = pd.Categorical(
#     seasons_df["Start_Month"], categories=month_order, ordered=True
# )
# seasons_df["End_Month"] = pd.Categorical(
#     seasons_df["End_Month"], categories=month_order, ordered=True
# )

# # Sort the DataFrame by start month
# seasons_df = seasons_df.sort_values(by="Start_Month")

# # Define the colors for each growing season
# season_colors = {"Masika": "blue", "Vuli": "green", "Msimu": "orange"}

# # Create the Gantt chart using hlines with customizations
# plt.figure(figsize=(10, 3))
# for index, row in seasons_df.iterrows():
#     plt.hlines(
#         y=row["Season"],
#         xmin=row["Start_Month"],
#         xmax=row["End_Month"],
#         color=season_colors[row["Season"]],
#         lw=3,  # Thicker lines
#     )

# # Customize the plot
# plt.xlabel("Month")
# plt.ylabel("Growing Season")
# plt.title("Monthly Timing of Growing Seasons in Tanzania")
# plt.yticks(ticks=seasons_df.index, labels=seasons_df["Season"])
# plt.xticks(
#     ticks=range(len(month_order)), labels=[month[:3] for month in month_order]
# )  # Abbreviate month names
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.1)  # Reduce space between y-axis ticks
# plt.show()

# %%
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
    ax.tick_params(axis="y", labelsize=16)
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
ax.set_xticklabels(tick_labels[:-1], rotation=45, ha="right", fontsize=16)


# Add labels and legend
# ax.set_xlabel("Months")

ax.set_title("Tanzania Crop Calendar", fontsize=16)
legend_elements = [
    Patch(facecolor="green", edgecolor="black", label="Plant"),
    Patch(facecolor="grey", edgecolor="black", label="Mid-Season"),
    Patch(facecolor="orange", edgecolor="black", label="Harvest"),
]
ax.legend(
    handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,fontsize=16
)
plt.tight_layout()
plt.savefig("./writeup/figures/plant_ghant.png")

plt.show()


# %%
# Flow diagram for the 3-step methodology
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def create_methodology_flowchart():
    """
    Creates a flow diagram showing the 3-step data collection methodology
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Define colors
    color_step1 = '#e8f4f8'
    color_step2 = '#d4e6f1'
    color_step3 = '#aed6f1'
    color_arrow = '#2c3e50'

    # Step 1: Development and Training
    step1_box = FancyBboxPatch(
        (1, 9), 8, 2,
        boxstyle="round,pad=0.1",
        facecolor=color_step1,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(step1_box)
    ax.text(5, 10.3, 'Step 1: Development and Training',
            ha='center', va='center', fontsize=20, weight='bold')
    ax.text(5, 9.7, 'Training of all intended student participants',
            ha='center', va='center', fontsize=16)

    # Arrow 1 -> 2
    arrow1 = FancyArrowPatch(
        (5, 9), (5, 7),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        color=color_arrow,
        linewidth=4
    )
    ax.add_patch(arrow1)

    # Step 2: Data Collection (main box)
    step2_box = FancyBboxPatch(
        (1, 5), 8, 2,
        boxstyle="round,pad=0.1",
        facecolor=color_step2,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(step2_box)
    ax.text(5, 6.3, 'Step 2: Data Collection',
            ha='center', va='center', fontsize=20, weight='bold')
    ax.text(5, 5.7, 'KoboToolbox hosting well-developed data model',
            ha='center', va='center', fontsize=16)
    ax.text(5, 5.3, 'Duration: 14 days total',
            ha='center', va='center', fontsize=14, style='italic')

    # # Sub-box for pilot testing
    # pilot_box = FancyBboxPatch(
    #     (1.5, 5), 7, 0.8,
    #     boxstyle="round,pad=0.05",
    #     facecolor='white',
    #     edgecolor='gray',
    #     linewidth=1,
    #     linestyle='--'
    # )
    # ax.add_patch(pilot_box)
    # ax.text(5, 5.4, '7 days: Iterative pilot testing on different farms, crops, and landscapes',
    #         ha='center', va='center', fontsize=9.5)

    # Arrow 2 -> 3
    arrow2 = FancyArrowPatch(
        (5, 5), (5, 3),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        color=color_arrow,
        linewidth=4
    )
    ax.add_patch(arrow2)

    # Step 3: Data Review and Cleaning
    step3_box = FancyBboxPatch(
        (1, 1), 8, 2,
        boxstyle="round,pad=0.1",
        facecolor=color_step3,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(step3_box)
    ax.text(5, 2.3, 'Step 3: Data Review and Cleaning',
            ha='center', va='center', fontsize=20, weight='bold')
    ax.text(5, 1.7, 'Generate training sample for model development',
            ha='center', va='center', fontsize=16)

    # Title
    ax.text(5, 11.5, 'Data Collection Methodology',
            ha='center', va='center', fontsize=22, weight='bold')

    plt.tight_layout()

    # Save the figure
    output_path = './writeup/figures/methodology_flowchart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Flowchart saved to: {output_path}")

    plt.show()

# Create the flowchart
create_methodology_flowchart()

# %%
