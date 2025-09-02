use std::fs::File;
use std::io::Write;
use std::io::BufReader;
use std::io::BufRead;

#[derive(Debug, Clone)]
struct Color {
    r: u8,
    g: u8,
    b: u8,
}

struct Object {
    vertices: Vec<Vec3>,
    faces: Vec<(usize, usize, usize)>,
    colors: Vec<Color>,
}

struct Camera {
    forward: Vec3,
    right: Vec3,
    up: Vec3,
}

struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn normalize(&self) -> Vec3 {
        let length = self.length();
        Vec3 { x: self.x / length, y: self.y / length, z: self.z / length }
    }

    fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3 { x: self.y * other.z - self.z * other.y, y: self.z * other.x - self.x * other.z, z: self.x * other.y - self.y * other.x }
    }

    fn dot(&self, other: &Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn add(&self, other: &Vec3) -> Vec3 {
        Vec3 { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }

    fn sub(&self, other: &Vec3) -> Vec3 {
        Vec3 { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }

    fn scale(&self, scalar: f32) -> Vec3 {
        Vec3 { x: self.x * scalar, y: self.y * scalar, z: self.z * scalar }
    }

    fn print(&self) {
        println!("({}, {}, {})", self.x, self.y, self.z);
    }
}

struct RayHit {
    t: f32,
    normal: Vec3,
}

fn calculate_camera_basis(direction: Vec3) -> Camera {
    let forward = direction.normalize();
    let mut up = Vec3 { x: 0.0, y: 1.0, z: 0.0 };
    let mut right = up.cross(&forward);
    if right.length() < 1e-6 {
        up = Vec3 { x: 1.0, y: 0.0, z: 0.0 };
        right = up.cross(&forward);
    }
    right = right.normalize();
    up = forward.cross(&right).normalize().scale(-1.0);
    right = right.scale(-1.0);
    Camera {forward, right, up}
}

fn ray_triangle_intersection(origin: &Vec3, direction: &Vec3, triangle: (&Vec3, &Vec3, &Vec3)) -> Option<RayHit> {
    let (a, b, c) = triangle;
    let e1 = b.sub(&a);
    let e2 = c.sub(&a);
    let p = direction.cross(&e2);
    let det = e1.dot(&p);
    if det < 1e-6 {
        return None;
    }
    let inv_det = 1.0 / det;
    let t = origin.sub(&a);
    let u = t.dot(&p) * inv_det;
    if u < 0.0 || u > 1.0 {
        return None;
    }
    let q = t.cross(&e1);
    let v = direction.dot(&q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t2 = e2.dot(&q) * inv_det;
    let normal = e1.cross(&e2).normalize();
    Some(RayHit { t: t2, normal })
}

struct Image {
    width: i32,
    height: i32,
    data: Vec<Color>,
}

fn render(width: i32, height: i32, objects: &Vec<Object>, origin: Vec3, direction: Vec3, fov: f32) -> (Image, Vec<f32>) {
    let light_dir = Vec3 { x: 0.0, y: 0.0, z: 1.0 };
    let mut image = Image { width, height, data: Vec::new() };
    let mut depth_map = Vec::new();
    let Camera {forward, right, up} = calculate_camera_basis(direction);
    let aspect_ratio = width as f32 / height as f32;
    let fov_rad = fov * (std::f32::consts::PI / 180.0);
    let half_tan_fov = (fov_rad * 0.5).tan();
    for j in 0..height {
        for i in 0..width {
            let x = ((i as f32 + 0.5) / width as f32 * 2.0 - 1.0) * aspect_ratio * half_tan_fov;
            let y = ((j as f32 + 0.5) / height as f32 * 2.0 - 1.0) * half_tan_fov;
            let ray_direction = forward.add(&right.scale(x)).add(&up.scale(y)).normalize();
            let mut closest_t = std::f32::INFINITY;
            let mut closest_color = Color { r: 0, g: 0, b: 0 };
            for object in objects {
                for (face_idx, face) in object.faces.iter().enumerate() {
                    let a = &object.vertices[face.0];
                    let b = &object.vertices[face.1];
                    let c = &object.vertices[face.2];
                    let triangle = (a, b, c);
                    let hit = ray_triangle_intersection(&origin, &ray_direction, triangle);
                    if let Some(hit) = hit {
                        if hit.t < closest_t {
                            closest_t = hit.t;
                            let Color { r, g, b } = object.colors[face_idx];
                            let normal = hit.normal;
                            let d = normal.dot(&light_dir).max(0.4);
                            closest_color = Color { r: (r as f32 * d) as u8, g: (g as f32 * d) as u8, b: (b as f32 * d) as u8 };
                        }
                    }
                }
            }
            image.data.push(closest_color);
            depth_map.push(closest_t);
        }
    }
    (image, depth_map)
}

fn save_image_ppm(image: Image, width: i32, height: i32, filename: &str) {
    let mut file = File::create(filename).unwrap();
    write!(file, "P3\n{} {}\n255\n", width, height).unwrap();
    for color in image.data {
        write!(file, "{} {} {}\n", color.r, color.g, color.b).unwrap();
    }
}

fn read_obj(filename: &str, color: Color) -> Object {

    let mut vertices = Vec::new();
    let mut faces = Vec::new();

    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with("v ") {
            let tokens = line.split_whitespace().collect::<Vec<&str>>();
            let x = tokens[1].parse::<f32>().unwrap();
            let y = tokens[2].parse::<f32>().unwrap();
            let z = tokens[3].parse::<f32>().unwrap();
            vertices.push(Vec3 { x, y, z });
        } else if line.starts_with("f ") {
            let tokens = line.split_whitespace().collect::<Vec<&str>>();
            let v1 = tokens[1].parse::<usize>().unwrap() - 1;
            let v2 = tokens[2].parse::<usize>().unwrap() - 1;
            let v3 = tokens[3].parse::<usize>().unwrap() - 1;
            faces.push((v1, v2, v3));
        }
    }
    let len = faces.len();
    Object { vertices, faces, colors: vec![color; len] }
}

fn create_debug_cube(size: f32) -> Object {
    let half_size = size / 2.0;
    let vertices = vec![
        Vec3 { x: -half_size, y: -half_size, z: -half_size },
        Vec3 { x:  half_size, y: -half_size, z: -half_size },
        Vec3 { x:  half_size, y:  half_size, z: -half_size },
        Vec3 { x: -half_size, y:  half_size, z: -half_size },
        Vec3 { x: -half_size, y: -half_size, z:  half_size },
        Vec3 { x:  half_size, y: -half_size, z:  half_size },
        Vec3 { x:  half_size, y:  half_size, z:  half_size },
        Vec3 { x: -half_size, y:  half_size, z:  half_size },
    ];
    let faces = vec![
        (0,2,1), (0,3,2),
        (4,5,6), (4,6,7),
        (0,4,7), (0,7,3),
        (1,2,6), (1,6,5),
        (0,1,5), (0,5,4),
        (2,3,7), (2,7,6),
    ];
    let colors = vec![
        Color { r: 0, g: 0, b: 255 },
        Color { r: 0, g: 0, b: 255 },
        Color { r: 0, g: 255, b: 0 },
        Color { r: 0, g: 255, b: 0 },
        Color { r: 255, g: 0, b: 0 },
        Color { r: 255, g: 0, b: 0 },
        Color { r: 255, g: 255, b: 0 },
        Color { r: 255, g: 255, b: 0 },
        Color { r: 255, g: 0, b: 255 },
        Color { r: 255, g: 0, b: 255 },
        Color { r: 0, g: 255, b: 255 },
        Color { r: 0, g: 255, b: 255 },
    ];
    Object { vertices, faces, colors }
}

// rustc -C opt-level=3 render.rs
fn main() {
    let width = 100;
    let height = 100;

    let origin = Vec3 { x: 2.0, y: 10.0, z: 8.0 };
    let destination = Vec3 { x: 0.0, y: 5.0, z: 0.0 };
    let direction = destination.sub(&origin).normalize();
    let fov = 90.0;

    // let objects = vec![create_debug_cube(8.0)];
    let mut bunny = read_obj("data/bunny.obj", Color { r: 255, g: 0, b: 0 });
    for vertex in bunny.vertices.iter_mut() {
        *vertex = vertex.scale(50.0);
    }
    let objects = vec![bunny];

    let (image, _depth_map) = render(width, height, &objects, origin, direction, fov);
    save_image_ppm(image, width, height, "output/rust.ppm");
}