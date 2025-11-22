data = {}

with open("asu_clusters.csv", "r") as f:
    file_content = f.readlines()
    for i in range(1, len(file_content)):
        line = file_content[i].strip().split(",")
        star_number = line[0]
        right_ascension = line[1]
        declination = line[2]   
        visual_magnitude = line[3]
        name = line[4]
        cluster_id = int(line[5])
        if cluster_id == -1:
            continue
        if cluster_id not in data:
            data[cluster_id] = []
        data[cluster_id].append(
            {
                "ra": round(float(right_ascension), 2),
                "dec": round(float(declination), 2),
                "vm": round(float(visual_magnitude), 2),
            }
        )

no_of_clusters = len(data.keys())
print(f"Total clusters: {no_of_clusters}")

api_key = None
with open(".env", "r") as f:
    file_content = f.readlines()
    for line in file_content:
        api_key = line.strip().split("=")[1]

import os
from openai import OpenAI
import json

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

prompt = """
You are a helpful assistant that analyzes star cluster data.
The data is organized in clusters, each containing multiple stars with attributes like right ascension (in degrees), declination (in degrees), visual magnitude.
Your task is to identify patterns, notable features, or any interesting insights from the clusters.
Provide a suitable name for each cluster based on its characteristics. You may consider factors such as density, brightness, spatial distribution, or any unique configurations of stars within the cluster.
Please use concise, creative, unique and descriptive names that reflect the cluster's features, preferably referencing objects. Please only suggest one name per cluster.
Output your analysis in a json format as such:
{
    "name": "Cluster Name",
    "reasoning": "Explanation of why this name was chosen based on the cluster's features."
}
"""

def recognise_cluster_patterns(cluster_data):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": str(cluster_data)},
        ],
        stream=False,
        response_format={
            'type': 'json_object'
        },
        temperature=1.5,
        max_tokens=500,
    )

    return json.loads(response.choices[0].message.content)

cluster_names = {}

i = 0
for cluster_id, stars in data.items():
    analysis = recognise_cluster_patterns(stars)
    
    print(f"{i}/{no_of_clusters}% done, generated {i} cluster names")

    cluster_names[cluster_id] = analysis
    i = i + 1

with open("cluster_names.json", "w") as f:
    json.dump(cluster_names, f, indent=4)

