data = {}

with open("asu.tsv", "r") as f:
    file_content = f.readlines()
    for i in range(72,len(file_content)-1):
        line = file_content[i].strip().split(";")
        right_ascension = line[0]
        declination = line[1]
        star_number = line[2]
        visual_magnitude = line[18]
        if right_ascension == "" or declination == "" or visual_magnitude == "":
            continue
        print(f"{star_number}\t{right_ascension}\t{declination}\t{visual_magnitude}")
        data[int(star_number)] = {
            "star_number": star_number,
            "right_ascension": right_ascension,
            "declination": declination,
            "visual_magnitude": visual_magnitude,
            "name": "",
        }

with open("asu_names.tsv", "r") as f:
    file_content = f.readlines()
    for i in range(33,len(file_content)-1):
        line = file_content[i].strip().split(";")
        star_number = int(line[0])
        name = line[1]
        data[star_number]["name"] = name

with open("asu_data.csv", "w") as f:
    f.write("star_number,right_ascension,declination,visual_magnitude,name\n")
    for _, entry in data.items():
        f.write(f"{entry['star_number']},{entry['right_ascension']},{entry['declination']},{entry['visual_magnitude']},{entry['name']}\n")