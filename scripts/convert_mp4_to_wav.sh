#!/bin/bash

# Directory containing mp4 files
input_dir="/Users/bkoo/Downloads/gdrive_mp4"
# Output directory for wav files
output_dir="/Users/bkoo/Downloads/gdrive_mp4/wav"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through all mp4 files in the input directory
for mp4_file in "$input_dir"/*.mp4; do
    # Get the base name of the file (without extension)
    base_name=$(basename "$mp4_file" .mp4)
    # Convert mp4 to wav
    ffmpeg -i "$mp4_file" "$output_dir/$base_name.wav"
done
