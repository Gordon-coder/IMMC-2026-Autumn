constellations = {}

with open("asu_constellations.tsv", "r") as f:
    file_content = f.readlines()
    for i in range(39, len(file_content)-1):
        line = file_content[i].strip().split(";")
        ra = line[0]
        dec = line[1]
        name = line[4]
        if name not in constellations:
            constellations[name] = []
        constellations[name].append((ra, dec))

with open("constellations.csv", "w") as f:
    for name, points in constellations.items():
        f.write(f"{name},")
        for ra, dec in points:
            f.write(f"{ra};{dec},")
        f.write("\n")