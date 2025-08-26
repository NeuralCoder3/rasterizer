# create images with BVH acceleration
import numpy as np
from PIL import Image
from numpy.typing import NDArray
from tqdm import tqdm
from bvh import BVH

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

def render_with_bvh(
    width: int,
    height: int,
    bvh: BVH,
    origin: NDArray[np.float64],
    direction: NDArray[np.float64],
    fov: float,
    progress_bar=None,
):
    depth_map = np.full((height, width), np.inf, dtype=np.float64)
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    
    forward, right, up = calculate_camera_basis(direction)

    aspect = width / height
    half_tan_fov = np.tan(np.deg2rad(fov) * 0.5)
    
    for i in range(width):
        for j in range(height):
            if progress_bar:
                progress_bar.update(1)
            u = (i + 0.5) / width * 2.0 - 1.0
            v = (j + 0.5) / height * 2.0 - 1.0
            x = u * aspect * half_tan_fov
            y = v * half_tan_fov
            ray_direction = forward + x * right + y * up
            ray_direction = ray_direction / np.linalg.norm(ray_direction)

            # Use BVH to find closest intersection
            closest_t, closest_color, closest_normal = bvh.find_closest_intersection(
                origin, ray_direction, ray_triangle_intersection
            )
            
            # Update depth map and pixels with the closest intersection
            if closest_t > 0 and closest_t < np.inf:
                depth_map[j, i] = closest_t
                
                # Apply phong shading if we have a normal
                if closest_normal is not None and closest_color is not None:
                    light_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    light_direction = light_direction / np.linalg.norm(light_direction)
                    diffuse = np.dot(closest_normal, light_direction)
                    # Ensure diffuse is positive and convert to uint8
                    diffuse = max(0.4, diffuse)  # Minimum lighting to avoid black
                    final_color = (closest_color.astype(np.float64) * diffuse).astype(np.uint8)
                    pixels[j, i] = final_color

    return pixels, depth_map

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
    
    # 12 triangles (2 per face, 6 faces)
    faces = np.array([
        # Back face (blue) - Z = -1
        [0, 2, 1], [0, 3, 2],
        # Front face (green) - Z = +1
        [4, 5, 6], [4, 6, 7],
        # Left face (red) - X = -1
        [0, 4, 7], [0, 7, 3],
        # Right face (yellow) - X = +1
        [1, 2, 6], [1, 6, 5],
        # Bottom face (magenta) - Y = -1
        [0, 1, 5], [0, 5, 4],
        # Top face (cyan) - Y = +1
        [2, 3, 7], [2, 7, 6],
    ], dtype=np.int32)
    
    # Colors for each triangle (RGB)
    colors = np.array([
        [0, 0, 255], [0, 0, 255],      # Back: blue
        [0, 255, 0], [0, 255, 0],      # Front: green
        [255, 0, 0], [255, 0, 0],      # Left: red
        [255, 255, 0], [255, 255, 0],  # Right: yellow
        [255, 0, 255], [255, 0, 255],  # Bottom: magenta
        [0, 255, 255], [0, 255, 255],  # Top: cyan
    ], dtype=np.uint8)
    
    return vertices, faces, colors

if __name__ == "__main__":
    width = 100
    height = 100

    # Camera position
    origin = (2.0, 10.0, 8.0)
    destination = (0.0, 5.0, 0.0)
    fov = 90

    # Debug camera setup
    print(f"Camera origin: {origin}")
    print(f"Camera destination: {destination}")

    # Create objects list with colors
    objects = []

    # Add bunny
    bunny_vertices, bunny_faces = readObj("bunny.obj")
    bunny_vertices = bunny_vertices * 50
    bunny_colors = np.zeros((bunny_faces.shape[0], 3), dtype=np.uint8)
    bunny_colors[:, 0] = 255  # Red
    bunny_colors[:, 1] = 0
    bunny_colors[:, 2] = 0
    objects.append((bunny_vertices, bunny_faces, bunny_colors))

    # Add bottom plane
    bottom_size = 10.0
    bottom_plane = -5  # Position below the bunny
    vertices_bottom = np.array([
        [-bottom_size, bottom_plane, -bottom_size],
        [ bottom_size, bottom_plane, -bottom_size],
        [ bottom_size, bottom_plane,  bottom_size],
        [-bottom_size, bottom_plane,  bottom_size],
    ], dtype=np.float64)
    faces_bottom = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    colors_bottom = np.array([[128, 128, 128], [128, 128, 128]], dtype=np.uint8)  # Gray
    objects.append((vertices_bottom, faces_bottom, colors_bottom))

    # Build BVH
    print("Building BVH...")
    bvh = BVH()
    bvh.build_from_objects(objects)
    print(f"BVH built with {len(bvh.triangles)} triangles")

    origin = np.array(origin, dtype=np.float64)
    direction = np.array(destination - origin, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)

    # Calculate and display camera basis
    forward, right, up = calculate_camera_basis(direction)
    print(f"Camera basis - Forward: {forward.tolist()}")
    print(f"Camera basis - Right: {right.tolist()}")
    print(f"Camera basis - Up: {up.tolist()}")
    print(f"Dot products - F·R: {np.dot(forward, right):.3f}, F·U: {np.dot(forward, up):.3f}, R·U: {np.dot(right, up):.3f}")

    # Simple progress tracking without external dependency
    print("Rendering with BVH...")
    pixels, depth_map = render_with_bvh(width, height, bvh, origin, direction, fov, None)

    # Save the colored rendered image
    Image.fromarray(pixels).save("render_bvh.png")

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

    Image.fromarray(depth_pixels).save("depth_map_bvh.png")
    print("Rendering complete! Check render_bvh.png and depth_map_bvh.png")
