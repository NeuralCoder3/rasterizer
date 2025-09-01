# create images
import numpy as np
from PIL import Image
from numpy.typing import NDArray

# use_numba = True
use_numba = False

if use_numba:
    from numba import njit  # pyright: ignore[reportMissingImports]
    from numba_progress import ProgressBar  # pyright: ignore[reportMissingImports]
else:
    from tqdm import tqdm
    type ProgressBar = tqdm
    ProgressBar = tqdm
    njit = lambda nogil=True: lambda x: x

@njit(nogil=True)
def ray_triangle_intersection(
    origin: NDArray[np.float64],
    direction: NDArray[np.float64],
    triangle: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
):
    # Möller-Trumbore
    A, B, C = triangle
    e1 = B - A
    e2 = C - A
    p = np.cross(direction, e2)
    det = np.dot(e1, p)
    # backface culling (otherwise, use abs)
    if det < 1e-6:
        return -1.0
    inv_det = 1.0 / det
    T = origin - A
    u = np.dot(T, p) * inv_det
    if u < 0 or u > 1:
        return -1.0
    q = np.cross(T, e1)
    v = np.dot(direction, q) * inv_det
    if v < 0 or u + v > 1:
        return -1.0
    t = np.dot(e2, q) * inv_det
    if t < 0:
        return -1.0
    w = 1 - u - v
    # t = distance
    # wA+uB+vC = hit point = origin + t*direction
    # UV = w*UV_A + u*UV_B + v*UV_C
    # N = normalize(w*N_A + u*N_B + v*N_C)
    return t

def readObj(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(np.array(line.split()[1:], dtype=np.float64))
            elif line.startswith('f '):
                # Convert face indices to integers and subtract 1 (OBJ files are 1-indexed)
                face_indices = [
                    int(x.split('/')[0]) - 1 
                    for x in line.split()[1:]]
                faces.append(face_indices)
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)



@njit(nogil=True)
def calculate_camera_basis(direction: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Calculate camera basis vectors for debugging"""
    forward = direction / np.linalg.norm(direction)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)
    # Maintain right-handed system
    up = -up
    right = -right  
    return forward, right, up

@njit(nogil=True)
def render(
    width: int,
    height: int,
    objects: list[tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.uint8]]],  # vertices, faces, colors
    origin: NDArray[np.float64],
    direction: NDArray[np.float64],
    fov: float,
    progress_bar: ProgressBar,
):
    depth_map = np.full((height, width), np.inf, dtype=np.float64)
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    
    forward, right, up = calculate_camera_basis(direction)

    aspect = width / height
    half_tan_fov = np.tan(np.deg2rad(fov) * 0.5)
    
    
    for i in range(width):
        for j in range(height):
            progress_bar.update(1)
            u = (i + 0.5) / width * 2.0 - 1.0
            v = (j + 0.5) / height * 2.0 - 1.0
            x = u * aspect * half_tan_fov
            y = v * half_tan_fov
            ray_direction = forward + x * right + y * up
            ray_direction = ray_direction / np.linalg.norm(ray_direction)


            # Check all objects to find the closest intersection
            closest_t = np.inf
            closest_color = np.array([0, 0, 0], dtype=np.uint8)
            
            for object in objects:
                vertices, faces, colors = object
                for face_idx in range(faces.shape[0]):
                    face = faces[face_idx]
                    A = vertices[face[0]]
                    B = vertices[face[1]]
                    C = vertices[face[2]]
                    triangle = (A, B, C)
                    t = ray_triangle_intersection(origin, ray_direction, triangle)
                    if t > 0 and t < closest_t:
                        closest_t = t
                        closest_color = colors[face_idx]
                        # apply phong shading
                        normal = np.cross(B - A, C - A)
                        normal = normal / np.linalg.norm(normal)
                        light_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                        light_direction = light_direction / np.linalg.norm(light_direction)
                        diffuse = np.dot(normal, light_direction)
                        # Ensure diffuse is positive and convert to uint8
                        diffuse = max(0.4, diffuse)  # Minimum lighting to avoid black
                        closest_color = (closest_color.astype(np.float64) * diffuse).astype(np.uint8)
            
            # Update depth map and pixels with the closest intersection
            if closest_t < np.inf:
                depth_map[j, i] = closest_t
                pixels[j, i] = closest_color

            # pixels[i, j] = (i, j, 0)
    return pixels, depth_map

# Mirror object correctly (flip coordinates AND face winding order)
def mirror_object(vertices, faces, axis=0):
    """
    Mirror object along specified axis (0=X, 1=Y, 2=Z)
    Flips coordinates and face winding order to maintain correct normals
    """
    # Flip coordinates
    vertices[:, axis] = -vertices[:, axis]
    
    # Flip face winding order (swap first two vertices of each face)
    faces = faces.copy()
    faces[:, [0, 1]] = faces[:, [1, 0]]
    
    return vertices, faces

def rotate_object(vertices, faces, axis=0, angle=0.0):
    """
    Rotate object around specified axis (0=X, 1=Y, 2=Z)
    """
    # Rotate coordinates around specified axis
    vertices = vertices.copy()
    
    if axis == 0:  # X-axis rotation
        y = vertices[:, 1] * np.cos(angle) - vertices[:, 2] * np.sin(angle)
        z = vertices[:, 1] * np.sin(angle) + vertices[:, 2] * np.cos(angle)
        vertices[:, 1] = y
        vertices[:, 2] = z
    elif axis == 1:  # Y-axis rotation
        x = vertices[:, 0] * np.cos(angle) + vertices[:, 2] * np.sin(angle)
        z = -vertices[:, 0] * np.sin(angle) + vertices[:, 2] * np.cos(angle)
        vertices[:, 0] = x
        vertices[:, 2] = z
    elif axis == 2:  # Z-axis rotation
        x = vertices[:, 0] * np.cos(angle) - vertices[:, 1] * np.sin(angle)
        y = vertices[:, 0] * np.sin(angle) + vertices[:, 1] * np.cos(angle)
        vertices[:, 0] = x
        vertices[:, 1] = y
    
    return vertices, faces

def create_debug_cube(size=1.0):
    """
    Create a colored cube for debugging - each face has a different color
    Returns: vertices, faces, colors
    """
    half_size = size / 2.0
    
    # 8 vertices of a cube
    vertices = np.array([
        [-half_size, -half_size, -half_size],  # 0: back-bottom-left
        [ half_size, -half_size, -half_size],  # 1: back-bottom-right
        [ half_size,  half_size, -half_size],  # 2: back-top-right
        [-half_size,  half_size, -half_size],  # 3: back-top-left
        [-half_size, -half_size,  half_size],  # 4: front-bottom-left
        [ half_size, -half_size,  half_size],  # 5: front-bottom-right
        [ half_size,  half_size,  half_size],  # 6: front-top-right
        [-half_size,  half_size,  half_size],  # 7: front-top-left
    ], dtype=np.float64)
    
    # 12 triangles (2 per face, 6 faces) - FIXED: vertices now match actual coordinates
    faces = np.array([
        # Back face (blue) - Z = -1 - vertices [0,1,2,3] at Z = -1
        [0, 2, 1], [0, 3, 2],
        # Front face (green) - Z = +1 - vertices [4,5,6,7] at Z = +1
        [4, 5, 6], [4, 6, 7],
        # Left face (red) - X = -1 - vertices [0,4,7,3] at X = -1
        [0, 4, 7], [0, 7, 3],
        # Right face (yellow) - X = +1 - vertices [1,2,6,5] at X = +1
        [1, 2, 6], [1, 6, 5],
        # Bottom face (magenta) - Y = -1 - vertices [0,1,5,4] at Y = -1
        [0, 1, 5], [0, 5, 4],
        # Top face (cyan) - Y = +1 - vertices [2,3,7,6] at Y = +1
        [2, 3, 7], [2, 7, 6],
    ], dtype=np.int32)
    
    # Colors for each triangle (RGB) - MUST match the corrected face order above
    colors = np.array([
        [0, 0, 255], [0, 0, 255],      # Back: blue (Z = -1) - vertices [0,1,2,3]
        [0, 255, 0], [0, 255, 0],      # Front: green (Z = +1) - vertices [4,5,6,7]
        [255, 0, 0], [255, 0, 0],      # Left: red (X = -1) - vertices [0,4,7,3]
        [255, 255, 0], [255, 255, 0],  # Right: yellow (X = +1) - vertices [1,2,6,5]
        [255, 0, 255], [255, 0, 255],  # Bottom: magenta (Y = -1) - vertices [0,1,5,4]
        [0, 255, 255], [0, 255, 255],  # Top: cyan (Y = +1) - vertices [2,3,7,6]
    ], dtype=np.uint8)
    
    return vertices, faces, colors


width = 100
height = 100

# Camera position - adjusted to see all faces clearly
origin = (2.0, 10.0, 8.0)  # Moved closer and lower for better visibility
destination = (0.0, 5.0, 0.0)
fov = 90

# Debug camera setup
print(f"Camera origin: {origin}")
print(f"Camera destination: {destination}")

# Create objects list with colors
objects = []

# Create debug cube for testing
# vertices, faces, colors = create_debug_cube(size=8.0)
# objects.append((vertices, faces, colors))

# Add bunny
bunny_vertices, bunny_faces = readObj("data/bunny.obj")
bunny_vertices = bunny_vertices * 50
bunny_colors = np.zeros((bunny_faces.shape[0], 3), dtype=np.uint8)
bunny_colors[:, 0] = 255  # Red
bunny_colors[:, 1] = 0
bunny_colors[:, 2] = 0
objects.append((bunny_vertices, bunny_faces, bunny_colors))


# For debugging, you can also add a colored bottom plane:
# Add bottom plane
# bottom_size = 10.0
# bottom_plane = -5  # Position below the bunny
# vertices_bottom = np.array([
#     [-bottom_size, bottom_plane, -bottom_size],
#     [ bottom_size, bottom_plane, -bottom_size],
#     [ bottom_size, bottom_plane,  bottom_size],
#     [-bottom_size, bottom_plane,  bottom_size],
# ], dtype=np.float64)
# faces_bottom = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
# colors_bottom = np.array([[128, 128, 128], [128, 128, 128]], dtype=np.uint8)  # Gray
# objects.append((vertices_bottom, faces_bottom, colors_bottom))


origin = np.array(origin, dtype=np.float64)
direction = np.array(destination - origin, dtype=np.float64)
direction = direction / np.linalg.norm(direction)

# Calculate and display camera basis
forward, right, up = calculate_camera_basis(direction)
print(f"Camera basis - Forward: {forward.tolist()}")
print(f"Camera basis - Right: {right.tolist()}")
print(f"Camera basis - Up: {up.tolist()}")
print(f"Dot products - F·R: {np.dot(forward, right):.3f}, F·U: {np.dot(forward, up):.3f}, R·U: {np.dot(right, up):.3f}")



with ProgressBar(total=width*height, desc="Rendering") as progress_bar:
    pixels, depth_map = render(width, height, objects, origin, direction, fov, progress_bar)

# Save the colored rendered image
Image.fromarray(pixels).save("output/render.png")

# Save the depth map as a separate image for debugging
depth_pixels = np.zeros((height, width, 3), dtype=np.uint8)
finite_mask = np.isfinite(depth_map)
if np.any(finite_mask):
    min_depth = np.min(depth_map[finite_mask])
    max_depth = np.max(depth_map[finite_mask])
    if max_depth > min_depth:
        depth_norm = 1.0 - (depth_map[finite_mask] - min_depth) / (max_depth - min_depth)
    else:
        depth_norm = np.zeros_like(depth_map[finite_mask])
    depth_u8 = (depth_norm * 255).astype(np.uint8)

    # Write grayscale into RGB where finite
    depth_pixels[finite_mask, 0] = depth_u8
    depth_pixels[finite_mask, 1] = depth_u8
    depth_pixels[finite_mask, 2] = depth_u8

Image.fromarray(depth_pixels).save("output/depth_map.png")







            