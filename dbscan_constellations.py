"""
DBSCAN clustering for ASU star data (create custom constellations).

This script implements a simple DBSCAN that operates on unit direction
vectors on the celestial sphere. It avoids computing explicit angular
values by comparing dot-products against cos(eps).

Usage:
    python3 dbscan_constellations.py --input asu_data.csv --eps-deg 6 --min-samples 4

Outputs:
    - `asu_clusters.csv` : original CSV with an added `cluster_id` column

Notes:
    - RA values are expected in degrees.
    - Dec values are expected in degrees.

This is a standalone, dependency-light implementation using numpy only.
"""

import csv
import math
import argparse
from typing import List, Tuple
import numpy as np


def load_star_vectors_from_csv(csv_path: str) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    """Load star records from a CSV and return star dicts + Nx3 unit vectors.

    The CSV is expected to have header: star_number,right_ascension,declination,visual_magnitude,name
    where right_ascension is in hours (e.g. 001.291250) and declination in degrees.
    """
    stars = []
    vectors = []
    fluxes = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            # trim whitespace and guard against short lines
            row = [c.strip() for c in row]
            # handle missing fields gracefully
            if len(row) < 4:
                continue
            star_number = row[0]
            try:
                ra = float(row[1])
                dec = float(row[2])
                visual_mag = float(row[3])
            except Exception:
                # skip malformed rows
                continue
            name = row[4] if len(row) > 4 else ""

            ra_radians = math.radians(ra)
            dec_radians = math.radians(dec)

            # compute 3D unit vector (z = sin(dec), x/y from RA)
            x = math.cos(dec_radians) * math.cos(ra_radians)
            y = math.cos(dec_radians) * math.sin(ra_radians)
            z = math.sin(dec_radians)
            vec = np.array([x, y, z], dtype=float)
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            unit_vec = vec / norm

            # compute flux from visual magnitude using Pogson relation
            # flux ~ 10^(-0.4 * m)
            flux = 10 ** (-0.4 * visual_mag)

            stars.append({
                "star_number": star_number,
                "ra": ra,
                "dec": dec,
                "visual_magnitude": visual_mag,
                "name": name,
            })
            vectors.append(unit_vec)
            fluxes.append(flux)

    if len(vectors) == 0:
        return stars, np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    vectors_array = np.vstack(vectors)
    fluxes_array = np.array(fluxes, dtype=float)
    return stars, vectors_array, fluxes_array


def dbscan_on_sphere(
    unit_vectors: np.ndarray,
    eps_radians: float,
    min_samples: int,
    fluxes: np.ndarray = None,
    min_total_weight: float = 0.0,
) -> np.ndarray:
    """Run DBSCAN on unit vectors on a sphere using angular radius eps (radians).

    We avoid computing arccos by using the dot-product threshold: vectors
    within angular distance eps have dot >= cos(eps).

    Returns:
        labels: int array of length N with cluster ids (0..k-1) or -1 for noise
    """
    if unit_vectors.shape[0] == 0:
        return np.array([], dtype=int)

    num_points = unit_vectors.shape[0]
    labels = np.full(num_points, -1, dtype=int)  # -1 = unclassified/noise
    visited = np.zeros(num_points, dtype=bool)

    cos_eps = math.cos(eps_radians)

    current_cluster_id = 0

    # For neighbor queries we compute dot products on the fly per point
    for point_idx in range(num_points):
        if visited[point_idx]:
            continue
        visited[point_idx] = True

        # compute dot between this point and all points
        dots = unit_vectors.dot(unit_vectors[point_idx])
        neighbor_indices = np.nonzero(dots >= cos_eps)[0]

        # Decide core-ness: either by count (min_samples) or by neighbor flux sum
        is_core = False
        if fluxes is None or min_total_weight <= 0.0:
            # fall back to classic DBSCAN core test by neighbor count
            if neighbor_indices.size >= min_samples:
                is_core = True
        else:
            # weighted core test: sum fluxes of neighbors
            neighbor_flux_sum = float(fluxes[neighbor_indices].sum())
            if neighbor_flux_sum >= float(min_total_weight):
                is_core = True

        if not is_core:
            # mark as noise (labels already -1)
            continue

        # start new cluster
        labels[neighbor_indices] = current_cluster_id
        # ensure this point is labeled
        labels[point_idx] = current_cluster_id

        # expand cluster
        queue = list(neighbor_indices)
        qi = 0
        while qi < len(queue):
            neighbor_idx = queue[qi]
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                dots_n = unit_vectors.dot(unit_vectors[neighbor_idx])
                neighbor_neighbors = np.nonzero(dots_n >= cos_eps)[0]
                # Determine if this neighbor is itself a core point using the
                # same rule we used above (count or weighted sum).
                neighbor_is_core = False
                if fluxes is None or min_total_weight <= 0.0:
                    if neighbor_neighbors.size >= min_samples:
                        neighbor_is_core = True
                else:
                    neighbor_neighbors_flux_sum = float(fluxes[neighbor_neighbors].sum())
                    if neighbor_neighbors_flux_sum >= float(min_total_weight):
                        neighbor_is_core = True

                if neighbor_is_core:
                    # add newly found neighbors to the queue if they are unlabeled
                    for nn in neighbor_neighbors:
                        if labels[nn] == -1:
                            labels[nn] = current_cluster_id
                        if nn not in queue:
                            queue.append(nn)
            qi += 1

        current_cluster_id += 1

    return labels


def save_clusters_to_csv(input_csv_path: str, output_csv_path: str, cluster_labels: np.ndarray):
    """Write original CSV rows to a new CSV with extra `cluster_id` column.

    Rows that were skipped during loading will be omitted; cluster_labels
    should match the number of rows processed by `load_star_vectors_from_csv`.
    """
    with open(input_csv_path, newline='') as infile, open(output_csv_path, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        header = next(reader, None)
        if header is None:
            return
        new_header = header + ["cluster_id"]
        writer.writerow(new_header)
        i = 0
        for row in reader:
            if not row:
                continue
            row = [c.strip() for c in row]
            # skip malformed rows in the same way loader did
            if len(row) < 4:
                continue
            try:
                float(row[1])
                float(row[2])
                float(row[3])
            except Exception:
                continue
            cluster_id = int(cluster_labels[i]) if i < len(cluster_labels) else -1
            writer.writerow(row + [cluster_id])
            i += 1


def parse_arguments():
    parser = argparse.ArgumentParser(description="DBSCAN clustering for star catalog (asu_data.csv)")
    parser.add_argument("--input", default="asu_data.csv", help="Input CSV path")
    parser.add_argument("--output", default="asu_clusters.csv", help="Output CSV path")
    parser.add_argument("--eps-deg", type=float, default=6.0, help="Angular neighborhood radius in degrees (default 6)")
    parser.add_argument("--min-samples", type=int, default=4, help="Minimum points to form a cluster (default 4)")
    parser.add_argument("--min-total-weight", type=float, default=0.0, help="Minimum total neighbor flux to consider a core point (if >0, uses weighted core test)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(f"Loading stars from {args.input}...")
    stars, unit_vectors, fluxes = load_star_vectors_from_csv(args.input)
    print(f"Loaded {len(stars)} stars; building clusters with eps={args.eps_deg} deg, min_samples={args.min_samples}, min_total_weight={args.min_total_weight}")

    eps_radians = math.radians(args.eps_deg)
    labels = dbscan_on_sphere(unit_vectors, eps_radians, args.min_samples, fluxes=fluxes, min_total_weight=args.min_total_weight)

    num_clusters = len(set(labels.tolist()) - {-1})
    print(f"DBSCAN finished. Found {num_clusters} clusters (noise labeled -1).")

    print(f"Noise percentage: {(labels == -1).sum() / len(labels) * 100:.2f}%")

    print(f"Saving clusters to {args.output}...")
    save_clusters_to_csv(args.input, args.output, labels)
    print("Done.")
