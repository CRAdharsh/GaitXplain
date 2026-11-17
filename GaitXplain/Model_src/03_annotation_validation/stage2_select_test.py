import pandas as pd
import os

BASE = "/Final Year Project/GAVD-main"

INPUT = os.path.join(
    BASE,
    "analysis",
    "GAVD_combined_annotations.csv"
)

OUTPUT = os.path.join(
    BASE,
    "stage2_test",
    "test_videos.csv"
)

df = pd.read_csv(INPUT)

# --------------------------------------------------
# Basic cleanup
# --------------------------------------------------

df["id"] = df["id"].astype(str)
df["gait_pat"] = df["gait_pat"].astype(str)

# One row per video + gait class + view
video_info = (
    df.groupby("id")
      .agg(
          gait_pat=("gait_pat", "first"),
          cam_view=("cam_view", "first"),
          url=("url", "first"),
          frame_count=("frame_num", "count"),
          sequence_count=("seq", "nunique")
      )
      .reset_index()
)

print("Total unique videos:", len(video_info))

# --------------------------------------------------
# Pick representative videos
# --------------------------------------------------

selected = []

# We want several different gait classes.
classes = sorted(video_info["gait_pat"].dropna().unique())

print("\nAvailable classes:")
for c in classes:
    print(" -", c)

# Pick up to 2 videos from each class.
for gait_class in classes:

    subset = video_info[
        video_info["gait_pat"] == gait_class
    ].copy()

    if len(subset) == 0:
        continue

    # Sort by frame count so we get different sequence sizes
    subset = subset.sort_values("frame_count")

    # Shortest
    selected.append(subset.iloc[0])

    # Longest, if different
    if len(subset) > 1:
        selected.append(subset.iloc[-1])

# Convert to DataFrame
selected_df = pd.DataFrame(selected)

# Remove duplicate videos
selected_df = selected_df.drop_duplicates(subset=["id"])

# Limit test set to 20 videos
selected_df = selected_df.head(20)

# Save
selected_df.to_csv(OUTPUT, index=False)

print("\nSelected videos:")
print(
    selected_df[
        [
            "id",
            "gait_pat",
            "cam_view",
            "frame_count",
            "sequence_count"
        ]
    ].to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT)

print("\nNumber selected:", len(selected_df))