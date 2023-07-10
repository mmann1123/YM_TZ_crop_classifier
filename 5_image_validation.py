# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

os.chdir(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover"
)
# Read the CSV file
df = pd.read_csv("../kobo_field_collections/TZ_ground_truth_cleaned.csv")
df_filtered = df.dropna(subset=["Picture_of_the_field_or_feature_URL", "primar"])
df_filtered = df_filtered[["Picture_of_the_field_or_feature_URL", "primar"]]


# %%
# Filter rows with valid URLs and assigned crop labels
import urllib
import numpy as np
import PIL

for crop in df_filtered["primar"].unique():
    try:
        df_sample = df_filtered[df_filtered["primar"] == crop].sample(
            n=9, random_state=1
        )
    except:
        df_sample = df_filtered[df_filtered["primar"] == crop]

    df_sample = df_sample.reset_index(drop=True)
    # Define the grid size
    num_rows = 3
    num_cols = 3

    # Create the grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

    # Iterate over the filtered rows and plot the images
    for idx, row in df_sample.iterrows():
        if idx >= num_rows * num_cols:
            break

        # Get the image URL and crop label
        img_url = row["Picture_of_the_field_or_feature_URL"]
        crop_label = row["primar"]

        # Load the image from URL
        # img = mpimg.imread(img_url)
        img = PIL.Image.open(urllib.request.urlopen(img_url))

        # Get the corresponding subplot
        ax = axes[idx // num_cols, idx % num_cols]

        # Plot the image
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Exampeles of {crop}")
        # Add the crop label as a text
        ax.text(
            0.5,
            -0.1,
            crop_label,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=12,
            color="white",
            backgroundcolor="black",
        )

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

    plt.savefig("outputs/crop_type_examples_{crop}.png", bbox_inches="tight")

# %%
