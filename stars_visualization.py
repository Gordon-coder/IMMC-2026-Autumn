import pygame as pg

data = []

class Star:
    def __init__(self, star_number, right_ascension, declination, visual_magnitude, name):
        self.star_number = star_number
        self.right_ascension = right_ascension
        self.declination = declination
        self.visual_magnitude = visual_magnitude
        self.name = name

    def __repr__(self):
        return f"Star({self.star_number}, {self.name}, RA: {self.right_ascension}, Dec: {self.declination}, Mag: {self.visual_magnitude})"

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
        data.append(
            Star(star_number, right_ascension, declination, visual_magnitude, name)
        )

print(f"Loaded {len(data)} stars into memory.")