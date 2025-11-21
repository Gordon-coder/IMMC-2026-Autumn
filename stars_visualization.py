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

STAR_DISTANCE = 100.0

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
                                math.sin(self.declination)]) * STAR_DISTANCE
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
            if render_custom_constellations:
                color = cluster_colors[data[idx].cluster_id]
            else:
                brightness = min(255, max(0, 255-visual_magnitudes[idx]*40))
                color = (brightness, brightness, brightness)
            pg.draw.circle(screen, color, (int(xi), int(yi)), 2)
            # draw name above the star if present (cached surface)
            text_surf = name_surfaces[idx]
            if text_surf is not None:
                tx = int(xi) - text_surf.get_width() // 2
                ty = int(yi) - text_surf.get_height() - 4
                # simple occlusion: only draw if above the top of the screen
                if ty + text_surf.get_height() >= 0:
                    screen.blit(text_surf, (tx, ty))

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
                                math.sin(dec)]) * STAR_DISTANCE
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