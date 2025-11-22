import pygame as pg
import math
import numpy as np
from random import randint

data = []
constellations = []
cluster_ids = set()
cluster_colors = {}

WIDTH, HEIGHT = 1000, 600

camera_pitch = 0.0
camera_yaw = 0.0

camera_vector = np.array([0.0, 0.0, 0.0])

use_concave_hull = False

# performance / projection parameters
FOCAL_LENGTH = 500.0
FPS_CAP = 60

render_constellations = True
render_custom_constellations = True

class Star:
    def __init__(self, star_number, right_ascension, declination, visual_magnitude, name, cluster_id):
        self.star_number = int(star_number)
        self.right_ascension = math.radians(float(right_ascension))
        self.declination = math.radians(float(declination))
        self.visual_magnitude = float(visual_magnitude)
        self.vector = np.array([math.cos(self.declination) * math.cos(self.right_ascension),
                                math.cos(self.declination) * math.sin(self.right_ascension),
                                math.sin(self.declination)])
        self.name = name

        self.cluster_id = int(cluster_id)
        # precompute unit direction for speed (used in vectorized projection)
        norm = np.linalg.norm(self.vector)
        if norm == 0:
            self.unit_vector = self.vector
        else:
            self.unit_vector = self.vector / norm

    def get_projection(self):
        # unit direction to star
        star_dir = self.vector / (np.linalg.norm(self.vector) + 1e-9)

        # camera forward direction (unit)
        cam_dir = camera_vector / (np.linalg.norm(camera_vector) + 1e-9)

        # world up (z-up). Adjust if your data uses different convention.
        world_up = np.array([0.0, 0.0, 1.0])

        # camera right vector
        right = np.cross(cam_dir, world_up)
        if np.linalg.norm(right) < 1e-6:
            # camera nearly aligned with world_up; pick a fallback right vector
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / np.linalg.norm(right)

        # camera up vector
        up = np.cross(right, cam_dir)
        up = up / (np.linalg.norm(up) + 1e-9)

        # depth in camera space
        forward = np.dot(cam_dir, star_dir)
        if forward <= 0:
            return None  # behind the camera

        # perspective projection parameters (tweak focal to taste)
        focal = 300.0

        x_cam = np.dot(right, star_dir)
        y_cam = np.dot(up, star_dir)

        screen_x = WIDTH / 2 + (x_cam / forward) * focal
        screen_y = HEIGHT / 2 - (y_cam / forward) * focal

        return (int(screen_x), int(screen_y))

    def __repr__(self):
        return f"Star({self.star_number}{self.name}, RA: {self.right_ascension}, Dec: {self.declination}, Mag: {self.visual_magnitude})"
    
class Constellation:
    def __init__(self, name, points):
        self.name = name
        self.points = points  # list of (ra, dec) tuples

    def __repr__(self):
        return f"Constellation({self.name}, Points: {len(self.points)})"

with open("asu_clusters.csv", "r") as f:
    file_content = f.readlines()
    for i in range(1, len(file_content)):
        line = file_content[i].strip().split(",")
        star_number = line[0]
        right_ascension = line[1]
        declination = line[2]   
        visual_magnitude = line[3]
        name = line[4]
        cluster_id = line[5]
        cluster_ids.add(int(cluster_id))
        data.append(
            Star(star_number, right_ascension, declination, visual_magnitude, name, cluster_id)
        )

print(f"Loaded {len(data)} stars into memory.")

no_of_clusters = len(cluster_ids)-1
print(f"Number of clusters: {no_of_clusters}")

for cluster_id in cluster_ids:
    cluster_colors[cluster_id] = (randint(0,255), randint(0,255), randint(0,255))
cluster_colors[ -1 ] = (255, 255, 255)  # color for noise points

with open("constellations.csv", "r") as f:
    file_content = f.readlines()
    for i in range(len(file_content)):
        line = file_content[i].strip().split(",")
        name = line[0].strip()
        points = []
        for point_str in line[1:]:
            if point_str:
                ra_str, dec_str = point_str.split(";")
                points.append((float(ra_str), float(dec_str)))
        constellations.append(Constellation(name, points))

# Build NumPy arrays once for vectorized projection
if len(data) > 0:
    unit_vectors = np.stack([s.unit_vector for s in data])  # shape (N,3)
    visual_magnitudes = np.array([s.visual_magnitude for s in data])
else:
    unit_vectors = np.empty((0, 3))
    visual_magnitudes = np.empty((0,))

# compute flux for each star (Pogson relation) and a normalized flux for display
if visual_magnitudes.size > 0:
    fluxes = 10 ** (-0.4 * visual_magnitudes)
    max_flux = float(fluxes.max()) if fluxes.max() > 0 else 1.0
    norm_fluxes = fluxes / max_flux
else:
    fluxes = np.empty((0,))
    norm_fluxes = np.empty((0,))

# Option: assign noise points (cluster_id == -1) to their nearest detected cluster centroid
# This removes outliers for visualization. Set to False if you prefer to keep noise.
ASSIGN_NOISE_TO_NEAREST = True
if ASSIGN_NOISE_TO_NEAREST and len(data) > 0:
    # build mapping from cluster -> indices (exclude -1)
    cluster_to_indices = {}
    for i, s in enumerate(data):
        if s.cluster_id == -1:
            continue
        cluster_to_indices.setdefault(s.cluster_id, []).append(i)

    # compute centroids (unit vectors) for each cluster
    cluster_centroids = {}
    for cid, indices in cluster_to_indices.items():
        vecs = np.stack([data[i].unit_vector for i in indices])
        c = vecs.sum(axis=0)
        n = np.linalg.norm(c)
        if n > 0:
            cluster_centroids[cid] = c / n

    # assign noise points to nearest centroid (by max dot product)
    if len(cluster_centroids) > 0:
        for i, s in enumerate(data):
            if s.cluster_id != -1:
                continue
            best_cid = None
            best_dot = -2.0
            for cid, cent in cluster_centroids.items():
                d = float(np.dot(s.unit_vector, cent))
                if d > best_dot:
                    best_dot = d
                    best_cid = cid
            if best_cid is not None:
                s.cluster_id = int(best_cid)

pg.init()

screen = pg.display.set_mode((WIDTH, HEIGHT))

# font for star names
font = pg.font.SysFont(None, 16)

# pre-render name surfaces (cache) to avoid per-frame font rendering
name_surfaces = []
for s in data:
    nm = s.name.strip() if hasattr(s, 'name') else ''
    if nm:
        name_surfaces.append(font.render(nm, True, (200, 200, 255)))
    else:
        name_surfaces.append(None)

# clock to cap FPS and reduce CPU usage
clock = pg.time.Clock()

# current focal length (zoom). Use a variable so the mouse wheel can change it.
focal = FOCAL_LENGTH

def render_radial_gradient_backgrounds(screen, cluster_centers_dict, cluster_points_dict, alpha_value=40):
    """Render radial gradient halos around each cluster centroid.
    
    Creates a soft, glowing background that emphasizes cluster centers.
    More visually subtle than polygons.
    
    Args:
        screen: pygame surface
        cluster_centers_dict: dict {cluster_id: (center_x, center_y)}
        cluster_points_dict: dict {cluster_id: [(x, y), ...]}
        alpha_value: max alpha for the gradient
    """
    for cluster_id, center in cluster_centers_dict.items():
        if cluster_id == -1:
            continue  # skip noise
        if cluster_id not in cluster_points_dict or len(cluster_points_dict[cluster_id]) < 1:
            continue
        
        points = cluster_points_dict[cluster_id]
        # compute max distance from center to any point in cluster
        max_dist = 0
        for px, py in points:
            dist = math.hypot(px - center[0], py - center[1])
            max_dist = max(max_dist, dist)
        
        if max_dist < 5:
            max_dist = 50  # minimum halo size
        
        color = cluster_colors.get(cluster_id, (100, 100, 100))
        # create a temporary surface for the gradient
        halo_size = int(max_dist * 2.5)
        halo_surf = pg.Surface((halo_size, halo_size), flags=pg.SRCALPHA)
        
        # draw concentric circles with fading alpha
        for r in range(halo_size // 2, 0, -2):
            # alpha decreases with radius
            alpha = max(0, alpha_value * (1 - (r / (halo_size // 2))))
            c = tuple(int(x) for x in color)
            pg.draw.circle(halo_surf, c + (int(alpha),), (halo_size // 2, halo_size // 2), r)
        
        # blit the halo to screen
        try:
            screen.blit(halo_surf, (int(center[0] - halo_size // 2), int(center[1] - halo_size // 2)))
        except Exception:
            pass


def compute_convex_hull(points):
    """Compute 2D convex hull using Andrew's monotone chain algorithm.

    Returns hull vertices in CCW order. Points may be a list of (x,y) tuples.
    """
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # concatenation of lower and upper gives the hull; omit last element of each
    hull = lower[:-1] + upper[:-1]
    return hull


def compute_concave_hull(points, alpha=40.0):
    """Compute a concave hull (alpha shape) for 2D points.

    This implementation attempts to use scipy.spatial.Delaunay to compute
    a Delaunay triangulation and keeps triangle edges whose circumradius
    is <= alpha. The boundary of the resulting set of triangles is returned
    as one or more polygons; we pick the largest polygon by area.

    If SciPy is not available or the operation fails, fall back to convex hull.
    Alpha is in the same units as point coordinates (pixels here).
    """
    try:
        from scipy.spatial import Delaunay
    except Exception:
        # SciPy not available: fallback
        return compute_convex_hull(points)

    pts = np.array(points, dtype=float)
    if len(pts) <= 3:
        return compute_convex_hull(points)

    try:
        tri = Delaunay(pts)
    except Exception:
        return compute_convex_hull(points)

    edges = {}
    # examine each triangle
    for simplex in tri.simplices:
        ia, ib, ic = simplex
        pa = pts[ia]
        pb = pts[ib]
        pc = pts[ic]
        # side lengths
        a = math.hypot(pb[0] - pc[0], pb[1] - pc[1])
        b = math.hypot(pa[0] - pc[0], pa[1] - pc[1])
        c = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        s = (a + b + c) / 2.0
        # area via Heron's formula; guard against degenerate triangles
        area_sq = max(s * (s - a) * (s - b) * (s - c), 0.0)
        if area_sq <= 0:
            continue
        area = math.sqrt(area_sq)
        # circumradius R = a*b*c / (4*area)
        denom = 4.0 * area
        if denom == 0:
            continue
        R = (a * b * c) / denom
        if R <= alpha:
            # add edges of triangle
            tris = [(ia, ib), (ib, ic), (ic, ia)]
            for u, v in tris:
                if u > v:
                    u, v = v, u
                edges[(u, v)] = edges.get((u, v), 0) + 1

    # boundary edges occur exactly once
    boundary_edges = [e for e, cnt in edges.items() if cnt == 1]
    if not boundary_edges:
        return compute_convex_hull(points)

    # build adjacency map from boundary edges (indices into pts)
    adj = {}
    for u, v in boundary_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    # extract loops by walking adjacency
    loops = []
    visited_edges = set()
    for start in adj:
        for nbr in adj[start]:
            if (start, nbr) in visited_edges or (nbr, start) in visited_edges:
                continue
            loop = [start, nbr]
            visited_edges.add((start, nbr))
            visited_edges.add((nbr, start))
            cur = nbr
            prev = start
            while True:
                neighbors = adj.get(cur, [])
                # pick the neighbor that's not prev
                next_v = None
                for t in neighbors:
                    if t == prev:
                        continue
                    if (cur, t) in visited_edges or (t, cur) in visited_edges:
                        continue
                    next_v = t
                    break
                if next_v is None:
                    break
                loop.append(next_v)
                visited_edges.add((cur, next_v))
                visited_edges.add((next_v, cur))
                prev, cur = cur, next_v
                if cur == start:
                    break
            # convert indices to coordinates
            if len(loop) >= 3:
                loops.append([tuple(pts[i]) for i in loop])

    if not loops:
        return compute_convex_hull(points)

    # pick the largest loop by absolute polygon area
    def polygon_area(poly):
        a = 0.0
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            a += x1 * y2 - x2 * y1
        return abs(a) * 0.5

    best = max(loops, key=polygon_area)
    return best


def render_cluster_boundaries(screen, cluster_points_dict, padding=8, alpha=None):
    """Draw an expanded convex-hull boundary around each cluster.

    padding: pixels to expand the hull outward from its centroid.
    """
    for cid, pts in cluster_points_dict.items():
        if cid == -1:
            continue
        if not pts or len(pts) < 2:
            continue

        if len(pts) == 2:
            # For two points, draw a thick capsule-like rectangle/line with padding
            a = np.array(pts[0], dtype=float)
            b = np.array(pts[1], dtype=float)
            mid = (a + b) / 2.0
            vec = b - a
            length = np.hypot(vec[0], vec[1])
            if length < 1e-6:
                continue
            # perpendicular vector
            perp = np.array([-vec[1], vec[0]]) / length
            p1 = a + perp * padding
            p2 = a - perp * padding
            p3 = b - perp * padding
            p4 = b + perp * padding
            polygon = [tuple(p1), tuple(p2), tuple(p3), tuple(p4)]
            color = cluster_colors.get(cid, (200, 200, 200))
            try:
                pg.draw.polygon(screen, color, polygon, width=2)
            except Exception:
                pass
            continue

        # choose concave hull if alpha specified (or default heuristic), else convex hull
        use_alpha = alpha if alpha is not None else max(12.0, min(100.0, int(sum((math.hypot(p[0]-pts[0][0], p[1]-pts[0][1]) for p in pts))/len(pts))))
        try:
            hull = compute_concave_hull(pts, alpha=use_alpha)
        except Exception:
            hull = compute_convex_hull(pts)
        if not hull:
            continue

        # compute centroid of hull
        cx = sum(p[0] for p in hull) / len(hull)
        cy = sum(p[1] for p in hull) / len(hull)

        # expand polygon outward by padding pixels away from centroid
        expanded = []
        for x, y in hull:
            vx = x - cx
            vy = y - cy
            norm = math.hypot(vx, vy)
            if norm == 0:
                # degenerate: keep the point
                expanded.append((x, y))
            else:
                scale = (norm + padding) / norm
                expanded.append((cx + vx * scale, cy + vy * scale))

        color = cluster_colors.get(cid, (200, 200, 200))
        try:
            pg.draw.polygon(screen, color, expanded, width=2)
        except Exception:
            # fallback: draw hull edges
            for a, b in zip(hull, hull[1:] + hull[:1]):
                try:
                    pg.draw.line(screen, color, a, b, 2)
                except Exception:
                    continue


looking = False

running = True
while running:
    mouse_pos = pg.mouse.get_rel()
    if looking:
        camera_yaw += mouse_pos[0] * 2 * 1/focal
        camera_pitch += mouse_pos[1] * 2 * 1/focal
        if camera_pitch > math.radians(89.0):
            camera_pitch = math.radians(89.0)
        if camera_pitch < math.radians(-89.0):
            camera_pitch = math.radians(-89.0)
    camera_vector = np.array([math.cos(camera_pitch) * math.cos(camera_yaw),
                            math.cos(camera_pitch) * math.sin(camera_yaw),
                            math.sin(camera_pitch)])

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.MOUSEBUTTONDOWN:
            looking = True
        if event.type == pg.MOUSEBUTTONUP:
            looking = False
        if event.type == pg.MOUSEWHEEL:
            # event.y is positive when scrolling up (away from user) -> zoom in
            # use multiplicative zoom for smooth scaling
            focal *= 1.1 ** event.y
            # clamp focal to reasonable range
            focal = max(300.0, min(5000.0, focal))
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_c:
                render_constellations = not render_constellations
            if event.key == pg.K_v:
                render_custom_constellations = not render_custom_constellations
            if event.key == pg.K_x:
                use_concave_hull = not use_concave_hull

    screen.fill((0, 0, 0))

    # Vectorized projection: transform all star direction vectors into camera space
    cam_dir = camera_vector
    cam_norm = np.linalg.norm(cam_dir)
    if cam_norm == 0:
        cam_dir = cam_dir
    else:
        cam_dir = cam_dir / cam_norm

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(cam_dir, world_up)
    if np.linalg.norm(right) < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / np.linalg.norm(right)
    up = np.cross(right, cam_dir)
    up = up / (np.linalg.norm(up) + 1e-9)

    # compute forward (depth) for all stars and select visible ones
    forward = unit_vectors.dot(cam_dir)  # shape (N,)
    vis_mask = forward > 0
    if np.any(vis_mask):
        uv = unit_vectors[vis_mask]
        fwd = forward[vis_mask]
        x_cam = uv.dot(right)
        y_cam = uv.dot(up)

        screen_xs = (WIDTH / 2) + (x_cam / fwd) * focal
        screen_ys = (HEIGHT / 2) - (y_cam / fwd) * focal

        # cull stars that are outside the screen rectangle to avoid extra draws
        on_screen = (screen_xs >= -10) & (screen_xs <= WIDTH + 10) & (screen_ys >= -10) & (screen_ys <= HEIGHT + 10)

        vis_indices = np.nonzero(vis_mask)[0][on_screen]
        xs = screen_xs[on_screen].astype(int)
        ys = screen_ys[on_screen].astype(int)

        for xi, yi, idx in zip(xs, ys, vis_indices):
            # color by cluster base color, modulated by normalized flux
            if render_custom_constellations:
                cid = data[idx].cluster_id
                base_color = cluster_colors.get(cid, (255, 255, 255))
                # normalized flux in [0,1]; fall back to a simple mag-based factor if missing
                # fallback mapping: brighter (smaller mag) -> larger factor
                f = max(0.0, min(1.0, 1.0 - visual_magnitudes[idx] * 40/255))

                # keep colors visible: scale between 0.25 and 1.0
                scale = 0.1 + 0.9 * f
                color = (int(base_color[0] * scale), int(base_color[1] * scale), int(base_color[2] * scale))
            else:
                brightness = min(255, max(0, 255-visual_magnitudes[idx]*40))
                color = (brightness, brightness, brightness)
            # radius scaled slightly by brightness
            pg.draw.circle(screen, color, (int(xi), int(yi)), 2)
            # draw name above the star if present (cached surface)
            text_surf = name_surfaces[idx]
            if text_surf is not None:
                tx = int(xi) - text_surf.get_width() // 2
                ty = int(yi) - text_surf.get_height() - 4
                # simple occlusion: only draw if above the top of the screen
                if ty + text_surf.get_height() >= 0:
                    screen.blit(text_surf, (tx, ty))

        # Render cluster backgrounds using the selected method before drawing stars
        if render_custom_constellations:
            cluster_points_dict = {}
            cluster_centers_dict = {}
            
            # Collect points and centers for each cluster, only from visible (front-facing) stars
            for xi, yi, idx in zip(xs, ys, vis_indices):
                cluster_id = data[idx].cluster_id
                if cluster_id not in cluster_points_dict:
                    cluster_points_dict[cluster_id] = []
                    cluster_centers_dict[cluster_id] = [0, 0]
                cluster_points_dict[cluster_id].append((xi, yi))
            
            # Compute centers
            for cid in cluster_centers_dict:
                if len(cluster_points_dict[cid]) > 0:
                    avg_x = sum(p[0] for p in cluster_points_dict[cid]) / len(cluster_points_dict[cid])
                    avg_y = sum(p[1] for p in cluster_points_dict[cid]) / len(cluster_points_dict[cid])
                    cluster_centers_dict[cid] = (avg_x, avg_y)
            
            # Render overlays
            if use_concave_hull:
                render_cluster_boundaries(screen, cluster_points_dict)
            else:
                render_radial_gradient_backgrounds(screen, cluster_centers_dict, cluster_points_dict, alpha_value=40)

    # render constellations as lines between projected constellation points
    if render_constellations:
        for const in constellations:
            # project each constellation point using the same camera basis
            proj_points = []
            for ra_val, dec_val in const.points:
                ra = math.radians(float(ra_val))
                dec = math.radians(float(dec_val))
                vec = np.array([math.cos(dec) * math.cos(ra),
                                math.cos(dec) * math.sin(ra),
                                math.sin(dec)])
                # unit direction
                vnorm = np.linalg.norm(vec)
                if vnorm == 0:
                    proj_points.append(None)
                    continue
                star_dir = vec / vnorm

                forward_pt = np.dot(cam_dir, star_dir)
                if forward_pt <= 0:
                    proj_points.append(None)
                    continue

                xcam = np.dot(right, star_dir)
                ycam = np.dot(up, star_dir)
                sx = int((WIDTH / 2) + (xcam / forward_pt) * focal)
                sy = int((HEIGHT / 2) - (ycam / forward_pt) * focal)
                proj_points.append((sx, sy))

            # draw lines between consecutive visible projected points
            for a, b in zip(proj_points, proj_points[1:]):
                if a is not None and b is not None:
                    pg.draw.line(screen, (100, 150, 255), a, b, 1)

            avg = [0,0]
            i = 0.00001
            for p in proj_points:
                if p is None:
                    continue
                avg[0] += p[0]
                avg[1] += p[1]
                i += 1

            # draw constellation name near centre of constellation
            text = font.render(const.name, True, (75, 100, 255))
            tx = avg[0] / i - text.get_width() // 2
            ty = avg[1] / i
            if ty + text.get_height() >= 0:
                screen.blit(text, (tx, ty))

    pg.display.flip()

    clock.tick(FPS_CAP)

pg.quit()