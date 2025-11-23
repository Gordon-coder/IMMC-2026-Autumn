import matplotlib.pyplot as plt
import numpy as np
from dbscan_constellations import *

"""
Effect of varying DBSCAN parameters on clustering results.
Independent variables:
- eps_deg: Angular distance threshold in degrees.
- min_samples: Minimum number of samples to form a core point.
- min_total_weight: Minimum total flux of neighboring points to form a core point.
Dependent variables to measure:
- n: Number of clusters formed (excluding noise).
- noise: Percentage of points classified as noise.
- std_dev: Standard deviation of cluster sizes.

A graph will be created for each independent variable showing its effect on each dependent variable. (total of 9 graphs)
"""

# Define ranges for independent variables
delta_eps = 0.1
eps_deg_range = np.arange(1, 5, delta_eps)
delta_min_samples = 1
min_samples_range = np.arange(1, 41, delta_min_samples)
delta_min_total_weights = 0.01
min_total_weights_range = np.arange(0.0, 1, delta_min_total_weights)

def evaluate_dbscan_parameters(eps_deg, min_samples, min_total_weights):
    stars, vectors, fluxes = load_star_vectors_from_csv("asu_clusters.csv")
    eps_radians = np.deg2rad(eps_deg)
    n = {"Epsilon":[], "Minimum Sample Size":[], "Minimum Total Weight":[]}
    noise = {"Epsilon":[], "Minimum Sample Size":[], "Minimum Total Weight":[]}
    std_dev = {"Epsilon":[], "Minimum Sample Size":[], "Minimum Total Weight":[]}

    default_eps = np.deg2rad(3.0)
    default_min_samples = 10
    default_min_total_weight = 0.05

    for eps in eps_radians:
        print(f"Evaluating eps: {np.rad2deg(eps):.2f} degrees")
        labels = dbscan_on_sphere(
            vectors,
            eps,
            default_min_samples,
            fluxes,
            default_min_total_weight,
        )
        n["Epsilon"].append(len(set(labels)) - (1 if -1 in labels else 0))
        noise["Epsilon"].append((labels == -1).sum())
        std_dev["Epsilon"].append(np.std([
            np.sum(labels == cluster_id) for cluster_id in set(labels) if cluster_id != -1
        ]))

    for min_sample in min_samples:
        print(f"Evaluating min_samples: {min_sample}")
        labels = dbscan_on_sphere(
            vectors,
            default_eps,
            min_sample,
            fluxes,
            default_min_total_weight,
        )
        n["Minimum Sample Size"].append(len(set(labels)) - (1 if -1 in labels else 0))
        noise["Minimum Sample Size"].append((labels == -1).sum())
        std_dev["Minimum Sample Size"].append(np.std([
            np.sum(labels == cluster_id) for cluster_id in set(labels) if cluster_id != -1
        ]))

    for min_total_weight in min_total_weights:
        print(f"Evaluating min_total_weight: {min_total_weight}")
        labels = dbscan_on_sphere(
            vectors,
            default_eps,
            default_min_samples,
            fluxes,
            min_total_weight,
        )
        n["Minimum Total Weight"].append(len(set(labels)) - (1 if -1 in labels else 0))
        noise["Minimum Total Weight"].append((labels == -1).sum())
        std_dev["Minimum Total Weight"].append(np.std([
            np.sum(labels == cluster_id) for cluster_id in set(labels) if cluster_id != -1
        ]))

    return n, noise, std_dev

def plot_results(x, y, x_label, y_label, title):
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.savefig(f"./plots/{title.replace(' ', '_').lower()}.png")
    plt.show()

n, noise, std_dev = evaluate_dbscan_parameters(
    eps_deg_range,
    min_samples_range,
    min_total_weights_range
)

for param, x_values in [("Epsilon", eps_deg_range), ("Minimum Sample Size", min_samples_range), ("Minimum Total Weight", min_total_weights_range)]:
    plot_results(
        x_values,
        n[param],
        f"{param}",
        "Number of Clusters",
        f"Effect of {param} on Number of Clusters"
    )
    plot_results(
        x_values,
        noise[param],
        f"{param}",
        "Number of Noise Points",
        f"Effect of {param} on Noise Points"
    )
    plot_results(
        x_values,
        std_dev[param],
        f"{param}",
        "Standard Deviation of Cluster Sizes",
        f"Effect of {param} on Standard Deviation of Cluster Sizes"
    )