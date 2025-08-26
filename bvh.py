import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Optional



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

def compute_triangle_aabb(
    A: NDArray[np.float64],
    B: NDArray[np.float64], 
    C: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute AABB for a single triangle"""
    min_point = np.minimum(np.minimum(A, B), C)
    max_point = np.maximum(np.maximum(A, B), C)
    return min_point, max_point

def compute_object_aabb(
    vertices: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute AABB for an entire object (all vertices)"""
    if vertices.size == 0:
        return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
    
    min_point = vertices[0].copy()
    max_point = vertices[0].copy()
    
    for i in range(1, vertices.shape[0]):
        min_point = np.minimum(min_point, vertices[i])
        max_point = np.maximum(max_point, vertices[i])
    
    return min_point, max_point

class BVHNode:
    """Node in the BVH tree"""
    
    def __init__(self):
        self.min_point: Optional[NDArray[np.float64]] = None
        self.max_point: Optional[NDArray[np.float64]] = None
        self.left: Optional['BVHNode'] = None
        self.right: Optional['BVHNode'] = None
        self.triangle_indices: List[int] = []
        self.is_leaf: bool = False

class BVH:
    """Bounded Volume Hierarchy for accelerating ray-triangle intersections"""
    
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
        
        # Extract all triangles from objects
        for vertices, faces, colors in objects:
            for face_idx, face in enumerate(faces):
                A = vertices[face[0]]
                B = vertices[face[1]]
                C = vertices[face[2]]
                self.triangles.append((A, B, C))
                self.triangle_colors.append(colors[face_idx])
                
                # Pre-compute normal for shading
                normal = np.cross(B - A, C - A)
                normal = normal / np.linalg.norm(normal)
                self.triangle_normals.append(normal)
        
        if not self.triangles:
            return
        
        # Build BVH tree
        triangle_indices = list(range(len(self.triangles)))
        self.root = self._build_node(triangle_indices)
    
    def _build_node(self, triangle_indices: List[int]) -> BVHNode:
        """Recursively build BVH node"""
        node = BVHNode()
        
        if len(triangle_indices) == 1:
            # Leaf node
            node.is_leaf = True
            node.triangle_indices = triangle_indices
            A, B, C = self.triangles[triangle_indices[0]]
            min_point, max_point = compute_triangle_aabb(A, B, C)
            node.min_point = min_point
            node.max_point = max_point
            return node
        
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
        
        if len(triangle_indices) <= 2:
            # Small leaf node
            node.is_leaf = True
            node.triangle_indices = triangle_indices
            return node
        
        # Split triangles along longest axis
        extent = max_point - min_point
        split_axis = np.argmax(extent)
        split_value = (min_point[split_axis] + max_point[split_axis]) * 0.5
        
        # Partition triangles
        left_indices = []
        right_indices = []
        
        for idx in triangle_indices:
            A, B, C = self.triangles[idx]
            triangle_center = (A + B + C) / 3.0
            if triangle_center[split_axis] < split_value:
                left_indices.append(idx)
            else:
                right_indices.append(idx)
        
        # Handle edge case where all triangles go to one side
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
        """Find closest triangle intersection using BVH traversal"""
        if self.root is None:
            return -1.0, None, None
        
        closest_t = np.inf
        closest_color = None
        closest_normal = None
        
        closest_t, closest_color, closest_normal = self._traverse_node(
            self.root, origin, direction, closest_t, closest_color, closest_normal, ray_triangle_intersection_func
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
        """Recursively traverse BVH node for ray intersection"""
        if node is None:
            return closest_t, closest_color, closest_normal
        
        # Check AABB intersection first
        if node.min_point is None or node.max_point is None:
            return closest_t, closest_color, closest_normal
        if not ray_aabb_intersection(origin, direction, node.min_point, node.max_point):
            return closest_t, closest_color, closest_normal
        
        if node.is_leaf:
            # Check all triangles in leaf node
            for idx in node.triangle_indices:
                A, B, C = self.triangles[idx]
                t = ray_triangle_intersection_func(origin, direction, (A, B, C))
                if t > 0 and t < closest_t:
                    closest_t = t
                    closest_color = self.triangle_colors[idx]
                    closest_normal = self.triangle_normals[idx]
            return closest_t, closest_color, closest_normal
        
        # Traverse children
        if node.left is not None:
            closest_t, closest_color, closest_normal = self._traverse_node(
                node.left, origin, direction, closest_t, closest_color, closest_normal, ray_triangle_intersection_func
            )
        if node.right is not None:
            closest_t, closest_color, closest_normal = self._traverse_node(
                node.right, origin, direction, closest_t, closest_color, closest_normal, ray_triangle_intersection_func
            )
        
        return closest_t, closest_color, closest_normal
