import pygame as pg

data = []

with open("asu_data.csv", "r") as f:
    file_content = f.readlines()
    for i in range(1, len(file_content)):
        line = file_content[i].strip().split(",")
        star_number = line[0]
        right_ascension = line[1]
        declination = line[2]
        visual_magnitude = line[3]
        name = line[4]
        # Visualization logic goes here
        data.append({
            "star_number": star_number,
            "right_ascension": right_ascension,
            "declination": declination,
            "visual_magnitude": visual_magnitude,
            "name": name,
        })

print(f"Loaded {len(data)} stars into memory.")
print(data[:5])  # Print first 5 entries for verification