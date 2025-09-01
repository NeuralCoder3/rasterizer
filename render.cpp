#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <optional>
#include <limits>
#include <tuple>
#include <sstream>
#include <map>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

class Vec3
{
public:
    float x;
    float y;
    float z;

    Vec3(float x, float y, float z)
        : x(x), y(y), z(z) {}

    Vec3 sub(const Vec3& other) const
    {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    Vec3 add(const Vec3& other) const
    {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    Vec3 scale(float scalar) const
    {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    float dot(const Vec3& other) const noexcept
    {
        return x * other.x + y * other.y + z * other.z;
    }

    float length() const noexcept
    {
        return sqrt(x * x + y * y + z * z);
    }

    Vec3 normalize() const
    {
        return Vec3(x / length(), y / length(), z / length());
    }

    Vec3 cross(const Vec3& other) const
    {
        return Vec3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
    }
};

class Color
{
public:
    int r;
    int g;
    int b;

    explicit Color(int r, int g, int b)
        : r(r), g(g), b(b) {}
};

class RayHit
{
public:
    float t;
    Vec3 normal;

    explicit RayHit(float t, Vec3 normal) : t(t), normal(normal) {}
};

class Object
{
public:
    std::vector<Vec3> vertices;
    std::vector<std::tuple<int, int, int>> faces;
    std::vector<Color> colors;

    explicit Object(std::vector<Vec3> vertices, std::vector<std::tuple<int, int, int>> faces, std::vector<Color> colors)
        : vertices(std::move(vertices)), faces(std::move(faces)), colors(std::move(colors)) {}
};

std::optional<RayHit> ray_triangle_intersection(Vec3 origin, Vec3 direction, Vec3 A, Vec3 B, Vec3 C)
{
    Vec3 e1 = B.sub(A);
    Vec3 e2 = C.sub(A);
    Vec3 p = direction.cross(e2);
    float det = e1.dot(p);
    if (det < 1e-6)
    {
        return std::nullopt;
    }
    float inv_det = 1.0 / det;
    Vec3 t = origin.sub(A);
    float u = t.dot(p) * inv_det;
    if (u < 0.0 || u > 1.0)
    {
        return std::nullopt;
    }
    Vec3 q = t.cross(e1);
    float v = direction.dot(q) * inv_det;
    if (v < 0.0 || u + v > 1.0)
    {
        return std::nullopt;
    }
    float dist = e2.dot(q) * inv_det;
    Vec3 normal = e1.cross(e2).normalize();
    return std::optional<RayHit>(RayHit(dist, normal));
}

std::tuple<Vec3, Vec3, Vec3> calculate_camera_basis(const Vec3& direction)
{
    Vec3 forward = direction.normalize();
    Vec3 world_up = Vec3(0.0, 1.0, 0.0);
    Vec3 right = world_up.cross(forward);
    if (right.length() < 1e-6)
    {
        world_up = Vec3(1.0, 0.0, 0.0);
        right = world_up.cross(forward);
    }
    right = right.normalize();
    Vec3 up = forward.cross(right).normalize().scale(-1.0);
    right = right.scale(-1.0);
    return std::tuple<Vec3, Vec3, Vec3>(forward, right, up);
}

std::vector<Color> render(int width, int height, const std::vector<Object>& objects, const Vec3& origin, const Vec3& direction, float fov)
{
    Vec3 light_dir = Vec3(0.0, 0.0, 1.0);
    // alternative: construct with size + default
    // or use pointer/options
    std::vector<Color> image;
    image.reserve(width * height);
    std::vector<float> depth_map(width * height);

    auto [forward, right, up] = calculate_camera_basis(direction);
    float aspect_ratio = (float)width / height;
    float fov_rad = fov * (float)M_PI / 180.0;
    float half_tan_fov = (float)tan(fov_rad * 0.5);
    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            float x = (2.0 * (i + 0.5) / width - 1.0) * aspect_ratio * half_tan_fov;
            float y = (2.0 * (j + 0.5) / height - 1.0) * half_tan_fov;
            Vec3 ray_direction = forward.add(right.scale(x)).add(up.scale(y)).normalize();
            float closest_t = std::numeric_limits<float>::infinity();
            Color closest_color = Color(0, 0, 0);
            for (const Object &object : objects)
            {
                for (int face_index = 0; face_index < object.faces.size(); face_index++)
                {
                    int base_idx = face_index * 3;
                    auto [idx0, idx1, idx2] = object.faces[face_index];
                    Vec3 a = object.vertices[idx0];
                    Vec3 b = object.vertices[idx1];
                    Vec3 c = object.vertices[idx2];
                    std::optional<RayHit> hit = ray_triangle_intersection(origin, ray_direction, a, b, c);
                    if (hit != std::nullopt && hit->t > 0.0 && hit->t < closest_t)
                    {
                        closest_t = hit->t;
                        closest_color = object.colors[face_index];
                        double d = std::max(0.4f, hit->normal.dot(light_dir));
                        closest_color = Color(
                            (int)(closest_color.r * d),
                            (int)(closest_color.g * d),
                            (int)(closest_color.b * d));
                    }
                }
            }
            depth_map[j * width + i] = closest_t;
            image.emplace_back(closest_color.r, closest_color.g, closest_color.b);
        }
    }
    return image;
}

void save_image(const std::vector<Color>& image, int width, int height, const std::string& filename)
{
    std::vector<unsigned char> data(width * height * 3);
    for (size_t i = 0; i < image.size(); ++i) {
        data[i * 3] = image[i].r;
        data[i * 3 + 1] = image[i].g;
        data[i * 3 + 2] = image[i].b;
    }
    stbi_write_png(filename.c_str(), width, height, 3, data.data(), width * 3);
}

void save_image_ppm(const std::vector<Color>& image, int width, int height, const std::string& filename)
{
    std::ofstream file(filename);
    file << "P3" << std::endl << width << " " << height << std::endl << "255" << std::endl;
    for (const Color &color : image)
    {
        file << color.r << " " << color.g << " " << color.b << "\n";
    }
}

Object create_debug_cube(const float size)
{
    float half_size = size / 2.0;
    std::vector<Vec3> vertices = {
        Vec3(-half_size, -half_size, -half_size),
        Vec3(half_size, -half_size, -half_size),
        Vec3(half_size, half_size, -half_size),
        Vec3(-half_size, half_size, -half_size),
        Vec3(-half_size, -half_size, half_size),
        Vec3(half_size, -half_size, half_size),
        Vec3(half_size, half_size, half_size),
        Vec3(-half_size, half_size, half_size),
    };
    std::vector<std::tuple<int, int, int>> faces = {
        {0, 2, 1},
        {0, 3, 2},
        {4, 5, 6},
        {4, 6, 7},
        {0, 4, 7},
        {0, 7, 3},
        {1, 2, 6},
        {1, 6, 5},
        {0, 1, 5},
        {0, 5, 4},
        {2, 3, 7},
        {2, 7, 6},
    };
    std::vector<Color> colors = {
        Color(0, 0, 255),
        Color(0, 0, 255),
        Color(0, 255, 0),
        Color(0, 255, 0),
        Color(255, 0, 0),
        Color(255, 0, 0),
        Color(255, 255, 0),
        Color(255, 255, 0),
        Color(255, 0, 255),
        Color(255, 0, 255),
        Color(0, 255, 255),
        Color(0, 255, 255),
    };
    return Object(vertices, faces, colors);
}

Object read_obj(const std::string& filename, const Color& color)
{
    std::ifstream file(filename);
    std::vector<Vec3> vertices;
    std::vector<std::tuple<int, int, int>> faces;
    std::vector<Color> colors;

    std::string line;
    while (std::getline(file, line)) {
        if (line.substr(0, 2) == "v ") {
            std::istringstream iss(line.substr(2));
            float x, y, z;
            iss >> x >> y >> z;
            if (iss.fail()) {
                std::cerr << "Error parsing vertex line: " << line << std::endl;
                continue;
            }
            vertices.emplace_back(x, y, z);
        } else if (line.substr(0, 2) == "f ") {
            std::istringstream iss(line.substr(2));
            int f1, f2, f3;
            iss >> f1 >> f2 >> f3;
            if (iss.fail()) {
                std::cerr << "Error parsing face line: " << line << std::endl;
                continue;
            }
            faces.emplace_back(f1-1, f2-1, f3-1); // OBJ indices are 1-based
            colors.push_back(color);
        }
    }
    
    // Check if we actually read any data
    if (vertices.empty()) {
        std::cerr << "Error: no vertices found in file " << filename << std::endl;
        exit(1);
    }
    return Object(vertices, faces, colors);
}

// g++ render.cpp -O3 && ./a.out
int main()
{
    int width = 100;
    int height = 100;

    Vec3 origin = Vec3(2.0, 10.0, 8.0);
    Vec3 destination = Vec3(0.0, 5.0, 0.0);
    Vec3 direction = destination.sub(origin).normalize();
    float fov = 90.0;

    // std::vector<Object> objects = {create_debug_cube(8.0)};
    std::vector<Object> objects;

    Object bunny = read_obj("data/bunny.obj", Color(255, 0, 0));
    for (auto& vertex : bunny.vertices) {
        vertex = vertex.scale(50.0);
    }
    objects.push_back(bunny);

    std::vector<Color> image = render(width, height, objects, origin, direction, fov);
    save_image(image, width, height, "output/cpp.png");

    return 0;
}