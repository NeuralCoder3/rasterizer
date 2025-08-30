import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Optional

# Small leaf size for simplicity and good performance
LEAF_SIZE = 2
# from numba import njit, jitclass


# @njit(nogil=True)
def ray_aabb_intersection(
    origin: NDArray[np.float64],
    direction: NDArray[np.float64],
    min_point: NDArray[np.float64],
    max_point: NDArray[np.float64]
) -> bool:
    """Fast ray-AABB intersection test using slab method"""
    t_min = (min_point - origin) / direction
    t_max = (max_point - origin) / direction
    
    # Handle division by zero
    t_min = np.where(np.isnan(t_min), -np.inf, t_min)
    t_max = np.where(np.isnan(t_max), np.inf, t_max)
    
    # Swap if needed
    t1 = np.minimum(t_min, t_max)
    t2 = np.maximum(t_min, t_max)
    
    t_near = np.max(t1)
    t_far = np.min(t2)
    
    return t_far >= t_near and t_far >= 0

# @njit(nogil=True)
def ray_aabb_entry_t(
    origin: NDArray[np.float64],
    direction: NDArray[np.float64],
    min_point: NDArray[np.float64],
    max_point: NDArray[np.float64]
) -> float:
    """Return entry distance t_near for ray-AABB, or np.inf if no hit.
    Used to order child traversal and enable early-out without extra triangle tests.
    """
    inv_dir = 1.0 / direction
    t1 = (min_point - origin) * inv_dir
    t2 = (max_point - origin) * inv_dir
    tmin = np.maximum.reduce(np.minimum(t1, t2))
    tmax = np.minimum.reduce(np.maximum(t1, t2))
    if tmax >= max(tmin, 0.0):
        return max(tmin, 0.0)
    return np.inf

# @njit(nogil=True)
def compute_triangle_aabb(
    A: NDArray[np.float64],
    B: NDArray[np.float64], 
    C: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute AABB for a single triangle"""
    min_point = np.minimum(np.minimum(A, B), C)
    max_point = np.maximum(np.maximum(A, B), C)
    return min_point, max_point

# NOTE: object-level AABB helper removed – not needed for the simplified BVH

class BVHNode:
    """Single BVH node: either a leaf with triangles or an internal AABB with two children"""

    def __init__(self):
        self.min_point: Optional[NDArray[np.float64]] = None
        self.max_point: Optional[NDArray[np.float64]] = None
        self.left: Optional['BVHNode'] = None
        self.right: Optional['BVHNode'] = None
        self.triangle_indices: List[int] = []
        self.is_leaf: bool = False

class BVH:
    """Simple BVH for ray/triangle intersection acceleration.

    Build: flatten all triangles, compute bounds, split along the longest axis, recurse.
    Traverse: test AABB, then children, test triangles in leaves and keep nearest hit.
    """

    def __init__(self):
        self.root: Optional[BVHNode] = None
        self.triangles: List[Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = []
        self.triangle_colors: List[NDArray[np.uint8]] = []
        self.triangle_normals: List[NDArray[np.float64]] = []
    
    def build_from_objects(self, objects: List[Tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.uint8]]]) -> None:
        """Build BVH from a list of objects (vertices, faces, colors)"""
        self.triangles.clear()
        self.triangle_colors.clear()
        self.triangle_normals.clear()
        
        # 1) Flatten all triangles (and per-triangle data)
        for vertices, faces, colors in objects:
            for face_idx, face in enumerate(faces):
                A = vertices[face[0]]
                B = vertices[face[1]]
                C = vertices[face[2]]
                self.triangles.append((A, B, C))
                self.triangle_colors.append(colors[face_idx])
                # Pre-compute normal for simple shading
                normal = np.cross(B - A, C - A)
                normal = normal / np.linalg.norm(normal)
                self.triangle_normals.append(normal)
        
        if not self.triangles:
            return
        
        # 2) Build BVH tree
        triangle_indices = list(range(len(self.triangles)))
        self.root = self._build_node(triangle_indices)
    
    def _build_node(self, triangle_indices: List[int]) -> BVHNode:
        """Create node: compute its bounds, split if needed, else become a leaf."""
        node = BVHNode()
        
        # Compute AABB for all triangles in this node
        A, B, C = self.triangles[triangle_indices[0]]
        min_point, max_point = compute_triangle_aabb(A, B, C)
        
        for idx in triangle_indices[1:]:
            A, B, C = self.triangles[idx]
            tri_min, tri_max = compute_triangle_aabb(A, B, C)
            min_point = np.minimum(min_point, tri_min)
            max_point = np.maximum(max_point, tri_max)
        
        node.min_point = min_point
        node.max_point = max_point

        # Become leaf if small enough
        if len(triangle_indices) <= LEAF_SIZE:
            node.is_leaf = True
            node.triangle_indices = triangle_indices
            return node

        # Split along longest axis at midpoint of bounds
        extent = max_point - min_point
        split_axis = np.argmax(extent)
        split_value = (min_point[split_axis] + max_point[split_axis]) * 0.5
        
        # Partition by triangle centroid
        left_indices = []
        right_indices = []
        
        for idx in triangle_indices:
            A, B, C = self.triangles[idx]
            triangle_center = (A + B + C) / 3.0
            if triangle_center[split_axis] < split_value:
                left_indices.append(idx)
            else:
                right_indices.append(idx)
        
        # Fallback: balanced split if partition failed
        if not left_indices or not right_indices:
            left_indices = triangle_indices[:len(triangle_indices)//2]
            right_indices = triangle_indices[len(triangle_indices)//2:]
        
        # Recursively build children
        node.left = self._build_node(left_indices)
        node.right = self._build_node(right_indices)
        
        return node
    
    def find_closest_intersection(
        self,
        origin: NDArray[np.float64],
        direction: NDArray[np.float64],
        ray_triangle_intersection_func
    ) -> Tuple[float, Optional[NDArray[np.uint8]], Optional[NDArray[np.float64]]]:
        """Return nearest hit distance, color, and normal using BVH traversal."""
        if self.root is None:
            return -1.0, None, None
        
        closest_t = np.inf
        closest_color = None
        closest_normal = None

        closest_t, closest_color, closest_normal = self._traverse_node(
            self.root,
            origin,
            direction,
            closest_t,
            closest_color,
            closest_normal,
            ray_triangle_intersection_func,
        )
        
        if closest_t == np.inf:
            return -1.0, None, None
        
        return closest_t, closest_color, closest_normal
    
    def _traverse_node(
        self,
        node: BVHNode,
        origin: NDArray[np.float64],
        direction: NDArray[np.float64],
        closest_t: float,
        closest_color: Optional[NDArray[np.uint8]],
        closest_normal: Optional[NDArray[np.float64]],
        ray_triangle_intersection_func
    ) -> Tuple[float, Optional[NDArray[np.uint8]], Optional[NDArray[np.float64]]]:
        """Traverse node: test AABB, then either triangles (leaf) or children (internal)."""
        if node is None:
            return closest_t, closest_color, closest_normal
        
        # AABB cull
        if node.min_point is None or node.max_point is None:
            return closest_t, closest_color, closest_normal
        if not ray_aabb_intersection(origin, direction, node.min_point, node.max_point):
            return closest_t, closest_color, closest_normal
        
        if node.is_leaf:
            # Leaf: test all triangles, keep nearest
            for idx in node.triangle_indices:
                A, B, C = self.triangles[idx]
                t = ray_triangle_intersection_func(origin, direction, (A, B, C))
                if t > 0 and t < closest_t:
                    closest_t = t
                    closest_color = self.triangle_colors[idx]
                    closest_normal = self.triangle_normals[idx]
            return closest_t, closest_color, closest_normal
        
        # Internal: ordered traversal with early-out using child AABB entry distances
        left_t = np.inf
        right_t = np.inf
        if node.left is not None and node.left.min_point is not None and node.left.max_point is not None:
            left_t = ray_aabb_entry_t(origin, direction, node.left.min_point, node.left.max_point)
        if node.right is not None and node.right.min_point is not None and node.right.max_point is not None:
            right_t = ray_aabb_entry_t(origin, direction, node.right.min_point, node.right.max_point)

        # Determine order
        first_child = node.left
        second_child = node.right
        first_t = left_t
        second_t = right_t
        if right_t < left_t:
            first_child, second_child = node.right, node.left
            first_t, second_t = right_t, left_t

        if first_child is not None and first_t < np.inf:
            closest_t, closest_color, closest_normal = self._traverse_node(
                first_child, origin, direction, closest_t, closest_color, closest_normal, ray_triangle_intersection_func
            )

        # Early-out: if current best is closer than second child's AABB entry, skip
        if second_child is not None and second_t < closest_t:
            closest_t, closest_color, closest_normal = self._traverse_node(
                second_child, origin, direction, closest_t, closest_color, closest_normal, ray_triangle_intersection_func
            )
        
        return closest_t, closest_color, closest_normal
