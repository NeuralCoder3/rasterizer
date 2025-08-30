#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IDX(x, y, width) (y * width + x)
#define PI 3.14159265358979323846

typedef struct {
    int r;
    int g;
    int b;
} Color;

typedef struct {
    float x;
    float y;
    float z;
} Vec3;

typedef struct {
    float dist;
    Vec3 normal;
} RayHit;

typedef struct {
    Vec3* vertices;
    int num_vertices;
    int num_faces;
    int* faces;
    Color* colors;
} Object;

typedef struct {
    Vec3 forward;
    Vec3 right;
    Vec3 up;
} Camera;



Vec3 vec3_add(Vec3 a, Vec3 b) {
    return (Vec3) {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return (Vec3) {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 vec3_scale(Vec3 a, double scalar) {
    return (Vec3) {a.x * scalar, a.y * scalar, a.z * scalar};
}

float vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return (Vec3) {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

float vec3_length(Vec3 a) {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

Vec3 vec3_normalize(Vec3 a) {
    float length = vec3_length(a);
    return (Vec3) {a.x / length, a.y / length, a.z / length};
}

void vec3_print(Vec3 a) {
    printf("(%f, %f, %f)\n", a.x, a.y, a.z);
}

RayHit* ray_triangle_intersection(Vec3 origin, Vec3 direction, Vec3 v0, Vec3 v1, Vec3 v2) {
    Vec3 e1 = vec3_sub(v1, v0);
    Vec3 e2 = vec3_sub(v2, v0);
    Vec3 p = vec3_cross(direction, e2);
    float det = vec3_dot(e1, p);
    if (det < 1e-6) {
        return NULL;
    }
    float inv_det = 1.0 / det;
    Vec3 t = vec3_sub(origin, v0);
    float u = vec3_dot(t, p) * inv_det;
    if (u < 0.0 || u > 1.0) {
        return NULL;
    }
    Vec3 q = vec3_cross(t, e1);
    float v = vec3_dot(direction, q) * inv_det;
    if (v < 0.0 || u + v > 1.0) {
        return NULL;
    }
    float dist = vec3_dot(e2, q) * inv_det;
    Vec3 normal = vec3_cross(e1, e2);
    RayHit* hit = (RayHit*) malloc(sizeof(RayHit));
    hit->dist = dist;
    hit->normal = vec3_normalize(normal);
    return hit;
}

Camera* calculate_camera_basis(Vec3 direction) {
    Vec3 forward = vec3_normalize(direction);
    Vec3 worldUp = (Vec3) {0.0, 1.0, 0.0};
    Vec3 right = vec3_cross(worldUp, forward);
    if (vec3_length(right) < 1e-6) {
        worldUp = (Vec3) {1.0, 0.0, 0.0};
        right = vec3_cross(worldUp, forward);
    }
    right = vec3_normalize(right);
    Vec3 up = vec3_cross(forward, right);
    up = vec3_normalize(up);
    up = vec3_scale(up, -1.0);
    right = vec3_scale(right, -1.0);
    Camera* camera = (Camera*) malloc(sizeof(Camera));
    camera->forward = forward;
    camera->right = right;
    camera->up = up;
    return camera;
}


unsigned char* render(int width, int height, Object** objects, int num_objects, Vec3 origin, Vec3 direction, float fov) {
    Vec3 lightDir = (Vec3) {0.0, 0.0, 1.0};
    unsigned char* image = (unsigned char*) malloc(3 * width * height * sizeof(unsigned char));
    float* depthMap = (float*) malloc(width * height * sizeof(float));

    Camera* camera = calculate_camera_basis(direction);
    Vec3 forward = camera->forward;
    Vec3 right = camera->right;
    Vec3 up = camera->up;

    float aspect_ratio = (float) width / height;
    float fov_rad = fov * (float) PI / 180.0;
    float half_tan_fov = (float) tan(fov_rad * 0.5);

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            float x = (2.0 * (i + 0.5) / width - 1.0) * aspect_ratio * half_tan_fov;
            float y = (2.0 * (j + 0.5) / height - 1.0) * half_tan_fov;
            Vec3 ray_direction = vec3_add(forward, vec3_add(vec3_scale(right, x), vec3_scale(up, y)));
            ray_direction = vec3_normalize(ray_direction);

            float closest_t = INFINITY;
            Color closest_color = (Color) {0, 0, 0};

            for (int object_index = 0; object_index < num_objects; object_index++) {
                Object* object = objects[object_index];

                for (int face_index = 0; face_index < object->num_faces; face_index++) {
                    int base_idx = face_index * 3;
                    int idx0 = object->faces[base_idx];
                    int idx1 = object->faces[base_idx + 1];
                    int idx2 = object->faces[base_idx + 2];
                    Vec3 a = object->vertices[idx0];
                    Vec3 b = object->vertices[idx1];
                    Vec3 c = object->vertices[idx2];
                    RayHit* hit = ray_triangle_intersection(origin, ray_direction, a, b, c);
                    if (hit != NULL && hit->dist > 0.0 && hit->dist < closest_t) {
                        closest_t = hit->dist;
                        Color color = object->colors[face_index];
                        double d = fmax(0.4, vec3_dot(hit->normal, lightDir));
                        closest_color = (Color) {
                            (int) (color.r * d),
                            (int) (color.g * d),
                            (int) (color.b * d)
                        };
                        free(hit);
                    }
                }
            }

            int idx = IDX(i, j, width);
            image[3*idx] = closest_color.r;
            image[3*idx + 1] = closest_color.g;
            image[3*idx + 2] = closest_color.b;
            depthMap[idx] = closest_t;
        }
    }

    free(camera);
    free(depthMap);
    return image;
}

Object* create_debug_cube(float size) {
    float half_size = size / 2.0;
    Vec3* vertices = (Vec3*) malloc(8 * sizeof(Vec3));
    vertices[0] = (Vec3) {-half_size, -half_size, -half_size};
    vertices[1] = (Vec3) { half_size, -half_size, -half_size};
    vertices[2] = (Vec3) { half_size,  half_size, -half_size};
    vertices[3] = (Vec3) {-half_size,  half_size, -half_size};
    vertices[4] = (Vec3) {-half_size, -half_size,  half_size};
    vertices[5] = (Vec3) { half_size, -half_size,  half_size};
    vertices[6] = (Vec3) { half_size,  half_size,  half_size};
    vertices[7] = (Vec3) {-half_size,  half_size,  half_size};

    int* faces = (int*) malloc(3 * 12 * sizeof(int));  // 12 faces * 3 vertices per face
    // Face 0: front face
    faces[0] = 0; faces[1] = 2; faces[2] = 1;
    faces[3] = 0; faces[4] = 3; faces[5] = 2;
    // Face 1: back face  
    faces[6] = 4; faces[7] = 5; faces[8] = 6;
    faces[9] = 4; faces[10] = 6; faces[11] = 7;
    // Face 2: left face
    faces[12] = 0; faces[13] = 4; faces[14] = 7;
    faces[15] = 0; faces[16] = 7; faces[17] = 3;
    // Face 3: right face
    faces[18] = 1; faces[19] = 2; faces[20] = 6;
    faces[21] = 1; faces[22] = 6; faces[23] = 5;
    // Face 4: bottom face
    faces[24] = 0; faces[25] = 1; faces[26] = 5;
    faces[27] = 0; faces[28] = 5; faces[29] = 4;
    // Face 5: top face
    faces[30] = 2; faces[31] = 3; faces[32] = 7;
    faces[33] = 2; faces[34] = 7; faces[35] = 6;

    Color* colors = (Color*) malloc(12 * sizeof(Color));
    colors[0] = (Color) {0, 0, 255};
    colors[1] = (Color) {0, 0, 255};
    colors[2] = (Color) {0, 255, 0};
    colors[3] = (Color) {0, 255, 0};
    colors[4] = (Color) {255, 0, 0};
    colors[5] = (Color) {255, 0, 0};
    colors[6] = (Color) {255, 255, 0};
    colors[7] = (Color) {255, 255, 0};
    colors[8] = (Color) {255, 0, 255};
    colors[9] = (Color) {255, 0, 255};
    colors[10] = (Color) {0, 255, 255};
    colors[11] = (Color) {0, 255, 255};
    
    Object* object = (Object*) malloc(sizeof(Object));
    object->vertices = vertices;
    object->num_faces = 12;
    object->num_vertices = 8;
    object->faces = faces;
    object->colors = colors;
    return object;
}

Object* read_obj(char* filename, Color color) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: could not open file %s\n", filename);
        return NULL;
    }
    Vec3* vertices = (Vec3*) malloc(sizeof(Vec3));
    int num_vertices = 0;
    int vertex_capacity = 1;
    int* faces = (int*) malloc(3*sizeof(int));
    int num_faces = 0;
    int face_capacity = 1;

    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == 'v') {
            int pos = num_vertices;
            num_vertices++;
            if (num_vertices > vertex_capacity) {
                vertex_capacity *= 2;
                vertices = (Vec3*) realloc(vertices, vertex_capacity * sizeof(Vec3));
            }
            sscanf(line, "v %f %f %f", &vertices[pos].x, &vertices[pos].y, &vertices[pos].z);
        } else if (line[0] == 'f') {
            int pos = num_faces*3;
            num_faces++;
            if (num_faces > face_capacity) {
                face_capacity *= 2;
                faces = (int*) realloc(faces, 3 * face_capacity * sizeof(int));
            }
            sscanf(line, "f %d %d %d", &faces[pos], &faces[pos + 1], &faces[pos + 2]);
            for (int i = 0; i < 3; i++) {
                faces[pos + i] -= 1;
            }
        }
    }
    fclose(file);
    Object* object = (Object*) malloc(sizeof(Object));
    object->vertices = vertices;
    object->num_vertices = num_vertices;
    object->num_faces = num_faces;
    object->faces = faces;
    object->colors = (Color*) malloc(num_faces * sizeof(Color));
    for (int i = 0; i < num_faces; i++) {
        object->colors[i] = color;
    }
    return object;
}

void save_image(unsigned char* image, int width, int height, char* filename) {
    stbi_write_png(filename, width, height, 3, image, width * 3);
}

// gcc render.c -lm -O3 && ./a.out
int main() {
    int width = 100;
    int height = 100;

    Vec3 origin = {2.0, 10.0, 8.0};
    Vec3 destination = {0.0, 5.0, 0.0};
    Vec3 direction = vec3_sub(destination, origin);
    direction = vec3_normalize(direction);
    float fov = 90.0;

    Object** objects = (Object**) malloc(1 * sizeof(Object*));
    int num_objects = 1;

    // Object* cube = create_debug_cube(8.0);
    // objects[0] = cube;

    Object* obj = read_obj("data/bunny.obj", (Color) {255, 0, 0});
    for (int i = 0; i < obj->num_vertices; i++) {
        obj->vertices[i] = vec3_scale(obj->vertices[i], 50.0);
    }
    objects[0] = obj;

    unsigned char* image = render(width, height, objects, num_objects, origin, direction, fov);
    save_image(image, width, height, "output/c.png");
    
    // Clean up memory
    free(image);
    free(objects[0]->vertices);
    free(objects[0]->faces);
    free(objects[0]->colors);
    free(objects[0]);
    free(objects);

}
