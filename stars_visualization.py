import pygame as pg
import math
import numpy as np

data = []

WIDTH, HEIGHT = 1000, 600

camera_pitch = 0.0
camera_yaw = 0.0

camera_vector = np.array([0.0, 0.0, 0.0])

STAR_DISTANCE = 100.0

# performance / projection parameters
FOCAL_LENGTH = 500.0
FPS_CAP = 60

class Star:
    def __init__(self, star_number, right_ascension, declination, visual_magnitude, name):
        self.star_number = int(star_number)
        # dataset RA is in hours (0..24) — convert hours -> degrees -> radians
        self.right_ascension = math.radians(float(right_ascension))
        self.declination = math.radians(float(declination))
        self.visual_magnitude = float(visual_magnitude)
        self.vector = np.array([math.cos(self.declination) * math.cos(self.right_ascension),
                                math.cos(self.declination) * math.sin(self.right_ascension),
                                math.sin(self.declination)]) * STAR_DISTANCE
        self.name = name
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

with open("asu_data.csv", "r") as f:
    file_content = f.readlines()
    for i in range(1, len(file_content)):
        line = file_content[i].strip().split(",")
        star_number = line[0]
        right_ascension = line[1]
        declination = line[2]   
        visual_magnitude = line[3]
        name = line[4]
        if float(visual_magnitude) > 6.5:
            continue
        data.append(
            Star(star_number, right_ascension, declination, visual_magnitude, name)
        )

print(f"Loaded {len(data)} stars into memory.")

# Build NumPy arrays once for vectorized projection
if len(data) > 0:
    unit_vectors = np.stack([s.unit_vector for s in data])  # shape (N,3)
    visual_magnitudes = np.array([s.visual_magnitude for s in data])
else:
    unit_vectors = np.empty((0, 3))
    visual_magnitudes = np.empty((0,))

pg.init()

pg.mouse.set_visible(False)

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

running = True
while running:
    mouse_pos = pg.mouse.get_rel()
    camera_yaw += mouse_pos[0] * 2 * 1/focal
    camera_pitch += mouse_pos[1] * 2 * 1/focal
    if camera_pitch > math.radians(89.0):
        camera_pitch = math.radians(89.0)
    if camera_pitch < math.radians(-89.0):
        camera_pitch = math.radians(-89.0)
    camera_vector = np.array([math.cos(camera_pitch) * math.cos(camera_yaw),
                              math.cos(camera_pitch) * math.sin(camera_yaw),
                              math.sin(camera_pitch)])

    pg.mouse.set_pos(WIDTH // 2, HEIGHT // 2)

    for event in pg.event.get():
        if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
            running = False
        # mouse wheel (pygame 2+)
        if event.type == pg.MOUSEWHEEL:
            # event.y is positive when scrolling up (away from user) -> zoom in
            # use multiplicative zoom for smooth scaling
            focal *= 1.1 ** event.y
            # clamp focal to reasonable range
            focal = max(300.0, min(5000.0, focal))
        # fallback for older pygame: mouse button 4/5
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 4:  # wheel up
                focal *= 1.1
                focal = min(5000.0, focal)
            elif event.button == 5:  # wheel down
                focal /= 1.1
                focal = max(300.0, focal)

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
            brightness = min(255, max(0, 255-visual_magnitudes[idx]*40))
            pg.draw.circle(screen, (brightness, brightness, brightness), (int(xi), int(yi)), 2)
            # draw name above the star if present (cached surface)
            text_surf = name_surfaces[idx]
            if text_surf is not None:
                tx = int(xi) - text_surf.get_width() // 2
                ty = int(yi) - text_surf.get_height() - 4
                # simple occlusion: only draw if above the top of the screen
                if ty + text_surf.get_height() >= 0:
                    screen.blit(text_surf, (tx, ty))

    pg.display.flip()

    # cap FPS to reduce CPU usage when many stars are drawn
    clock.tick(FPS_CAP)

pg.quit()