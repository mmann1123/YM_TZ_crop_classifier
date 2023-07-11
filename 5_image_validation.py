# %% create grids of images of training data

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import urllib
import PIL

os.chdir(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover"
)
# Read the CSV file
df = pd.read_csv("../kobo_field_collections/TZ_ground_truth_cleaned.csv")
df = df.reset_index()
df_filtered = df.dropna(subset=["Picture_of_the_field_or_feature_URL", "primar"])
df_filtered = df_filtered[["index", "Picture_of_the_field_or_feature_URL", "primar"]]
df_filtered

# %% Subset of images
# Filter rows with valid URLs and assigned crop labels


for crop in df_filtered["primar"].unique():
    try:
        df_sample = df_filtered[df_filtered["primar"] == crop].sample(
            n=12, random_state=1
        )
    except:
        df_sample = df_filtered[df_filtered["primar"] == crop]

    df_sample = df_sample.reset_index(drop=True)
    # Define the grid size
    # Calculate the number of rows based on the number of images
    num_cols = 3
    num_images = len(df_sample)
    try:
        num_rows = (num_images - 1) // num_cols + 1
    except:
        num_rows = 1

    # Create the grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 16))

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
        if num_rows == 1:
            ax = axes[idx % num_cols]
        else:
            ax = axes[idx // num_cols, idx % num_cols]

        # Plot the image
        ax.imshow(img)
        ax.axis("off")
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
    plt.savefig(f"outputs/crop_type_examples_{crop}.png", dpi=300)

    # Show the plot
    plt.show()


# %%
# %% Plot 100  images
# Filter rows with valid URLs and assigned crop labels

for crop in df_filtered["primar"].unique():
    try:
        df_sample = df_filtered[df_filtered["primar"] == crop].sample(
            n=102, random_state=1
        )
    except:
        df_sample = df_filtered[df_filtered["primar"] == crop]

    df_sample = df_sample.reset_index(drop=True)

    # Define the grid size
    # Calculate the number of rows based on the number of images
    num_cols = 3
    num_images = len(df_sample)
    try:
        num_rows = (num_images - 1) // num_cols + 1
    except:
        num_rows = 1

    # Create the grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))

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
        if num_rows == 1:
            ax = axes[idx % num_cols]
        else:
            ax = axes[idx // num_cols, idx % num_cols]

        # Plot the image
        ax.imshow(img)
        ax.axis("off")
        # Add the crop label as a text
        ax.text(
            0.5,
            -0.1,
            f"{crop_label}, row {row['index']}",
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=12,
            color="white",
            backgroundcolor="black",
        )

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.savefig(f"outputs/crop_type_100_{crop}.png", dpi=300)

    # Show the plot
    plt.show()

# %% plot ALL images

keep = [
    "rice",
    "maize",
    "cassava",
    "sunflower",
    "sorghum",
    "cotton",
    "soybeans",
    "millet",
]
drop = [
    "Don't know",
    "Other (later, specify in optional notes)",
    "vegetables",
    "other",
    "speciality_crops",
    "eggplant",
    "okra ",
    "tree_crops",
    "other_grain",
]

# apply keep/drop
df_filtered.drop(df_filtered[df_filtered["primar"].isin(drop)].index, inplace=True)


for crop in df_filtered["primar"].unique():
    print("working on", crop)
    df_sample = df_filtered[df_filtered["primar"] == crop]

    df_sample = df_sample.reset_index(drop=True)

    # Define the grid size
    # Calculate the number of rows based on the number of images
    num_cols = 3
    num_images = len(df_sample)
    try:
        num_rows = (num_images - 1) // num_cols + 1
    except:
        num_rows = 1

    # Create the grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))

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
        if num_rows == 1:
            ax = axes[idx % num_cols]
        else:
            ax = axes[idx // num_cols, idx % num_cols]

        # Plot the image
        ax.imshow(img)
        ax.axis("off")
        # Add the crop label as a text
        ax.text(
            0.5,
            -0.1,
            f"{crop_label}, row {row['index']}",
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=12,
            color="white",
            backgroundcolor="black",
        )

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.savefig(f"outputs/crop_type_ALL_{crop}.png", dpi=300)


# %%
