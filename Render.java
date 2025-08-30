import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import javax.imageio.ImageIO;

public class Render {

    public static class Vec3 {
        public double x, y, z;

        public Vec3(double x, double y, double z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public Vec3 add(Vec3 other) {
            return new Vec3(this.x + other.x, this.y + other.y, this.z + other.z);
        }

        public Vec3 sub(Vec3 other) {
            return new Vec3(this.x - other.x, this.y - other.y, this.z - other.z);
        }

        public Vec3 scale(double scalar) {
            return new Vec3(this.x * scalar, this.y * scalar, this.z * scalar);
        }

        public double dot(Vec3 other) {
            return this.x * other.x + this.y * other.y + this.z * other.z;
        }

        public double length() {
            return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
        }

        public Vec3 normalize() {
            return this.scale(1.0 / this.length());
        }

        public Vec3 cross(Vec3 other) {
            return new Vec3(this.y * other.z - this.z * other.y, this.z * other.x - this.x * other.z,
                    this.x * other.y - this.y * other.x);
        }
    }

    public static class Triangle {
        public Vec3 v0, v1, v2;

        public Triangle(Vec3 v0, Vec3 v1, Vec3 v2) {
            this.v0 = v0;
            this.v1 = v1;
            this.v2 = v2;
        }
    }

    public static class Color {
        public int r, g, b;

        public Color(int r, int g, int b) {
            this.r = r;
            this.g = g;
            this.b = b;
        }
    }

    public static class Triple<T, U, V> {
        public T first;
        public U second;
        public V third;

        public Triple(T first, U second, V third) {
            this.first = first;
            this.second = second;
            this.third = third;
        }
    }

    public static class RayHit {
        public float t;
        public Vec3 normal;

        public RayHit(float t, Vec3 normal) {
            this.t = t;
            this.normal = normal;
        }
    }

    public static class Object {
        public Vec3[] vertices;
        public List<Triple<Integer, Integer, Integer>> faces;
        public Color[] colors;

        public Object(Vec3[] vertices, List<Triple<Integer, Integer, Integer>> faces, Color[] colors) {
            this.vertices = vertices;
            this.faces = faces;
            this.colors = colors;
        }
    }

    public static RayHit rayTriangleIntersection(Vec3 origin, Vec3 direction, Triangle triangle) {
        Vec3 A = triangle.v0;
        Vec3 B = triangle.v1;
        Vec3 C = triangle.v2;

        Vec3 e1 = B.sub(A);
        Vec3 e2 = C.sub(A);
        Vec3 p = direction.cross(e2);

        double det = e1.dot(p);
        if (det < 1e-6) {
            return null;
        }

        double inv_det = 1.0 / det;
        Vec3 t = origin.sub(A);
        double u = t.dot(p) * inv_det;
        if (u < 0.0 || u > 1.0) {
            return null;
        }

        Vec3 q = t.cross(e1);
        double v = direction.dot(q) * inv_det;
        if (v < 0.0 || u + v > 1.0) {
            return null;
        }

        float dist = (float) (e2.dot(q) * inv_det);
        Vec3 normal = e1.cross(e2).normalize();

        return new RayHit(dist, normal);
    }

    public static Triple<Vec3, Vec3, Vec3> calculateCameraBasis(Vec3 direction) {
        Vec3 forward = direction.normalize();
        Vec3 worldUp = new Vec3(0.0, 1.0, 0.0);
        Vec3 right = worldUp.cross(forward);
        if (right.length() < 1e-6) {
            worldUp = new Vec3(1.0, 0.0, 0.0);
            right = worldUp.cross(forward);
        }
        right = right.normalize();
        Vec3 up = forward.cross(right);
        up = up.normalize();
        up = up.scale(-1.0);
        right = right.scale(-1.0);
        return new Triple<>(forward, right, up);
    }

    public static Color[][] render(int width, int height, List<Object> objects, Vec3 origin, Vec3 direction,
            float fov) {

        Vec3 lightDir = new Vec3(0.0, 0.0, 1.0);

        Color[][] image = new Color[width][height];
        float[][] depthMap = new float[width][height];

        Triple<Vec3, Vec3, Vec3> cameraBasis = calculateCameraBasis(direction);
        Vec3 forward = cameraBasis.first;
        Vec3 right = cameraBasis.second;
        Vec3 up = cameraBasis.third;

        float aspectRatio = (float) width / height;
        float fovRad = fov * (float) Math.PI / 180.0f;
        float halfTanFov = (float) Math.tan(fovRad * 0.5f);

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                float x = (2.0f * (i + 0.5f) / width - 1.0f) * aspectRatio * halfTanFov;
                float y = (2.0f * (j + 0.5f) / height - 1.0f) * halfTanFov;
                Vec3 rayDirection = forward.add(right.scale(x)).add(up.scale(y));
                rayDirection = rayDirection.normalize();

                float closestT = Float.POSITIVE_INFINITY;
                Color closestColor = new Color(0, 0, 0);

                for (Object object : objects) {
                    for (int faceIndex = 0; faceIndex < object.faces.size(); faceIndex++) {
                        Triple<Integer, Integer, Integer> face = object.faces.get(faceIndex);
                        int idx0 = face.first;
                        int idx1 = face.second;
                        int idx2 = face.third;

                        Vec3 a = object.vertices[idx0];
                        Vec3 b = object.vertices[idx1];
                        Vec3 c = object.vertices[idx2];

                        Triangle triangle = new Triangle(a, b, c);

                        RayHit hit = rayTriangleIntersection(origin, rayDirection, triangle);
                        if (hit != null && hit.t > 0.0f && hit.t < closestT) {
                            closestT = hit.t;
                            Color color = object.colors[faceIndex];
                            double d = Math.max(0.4, hit.normal.dot(lightDir));
                            closestColor = new Color(
                                (int) (color.r * d),
                                (int) (color.g * d),
                                (int) (color.b * d)
                            );
                        }
                    }
                }

                image[i][j] = closestColor;
                depthMap[i][j] = closestT;
            }
        }

        return image;
    }

    public static void writePPM(Color[][] image, String filename) {
        try (FileOutputStream fos = new FileOutputStream(filename)) {
            fos.write(("P3\n" + image.length + " " + image[0].length + "\n255\n").getBytes());
            int width = image.length;
            int height = image[0].length;
            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    fos.write((image[i][j].r + " " + image[i][j].g + " " + image[i][j].b + "\n").getBytes());
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void saveImage(Color[][] image, String filename) {
        BufferedImage img = new BufferedImage(image.length, image[0].length, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < image.length; i++) {
            for (int j = 0; j < image[i].length; j++) {
                img.setRGB(i, j, image[i][j].r << 16 | image[i][j].g << 8 | image[i][j].b);
            }
        }
        try {
            ImageIO.write(img, "png", new File(filename));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static Object createDebugCube(float size) {
        float halfSize = size / 2.0f;
        Vec3[] vertices = new Vec3[8];
        vertices[0] = new Vec3(-halfSize, -halfSize, -halfSize);
        vertices[1] = new Vec3( halfSize, -halfSize, -halfSize);
        vertices[2] = new Vec3( halfSize,  halfSize, -halfSize);
        vertices[3] = new Vec3(-halfSize,  halfSize, -halfSize);
        vertices[4] = new Vec3(-halfSize, -halfSize,  halfSize);
        vertices[5] = new Vec3( halfSize, -halfSize,  halfSize);
        vertices[6] = new Vec3( halfSize,  halfSize,  halfSize);
        vertices[7] = new Vec3(-halfSize,  halfSize,  halfSize);

        List<Triple<Integer, Integer, Integer>> faces = new ArrayList<>();
        faces.add(new Triple<>(0, 2, 1));
        faces.add(new Triple<>(0, 3, 2));

        faces.add(new Triple<>(4, 5, 6));
        faces.add(new Triple<>(4, 6, 7));

        faces.add(new Triple<>(0, 4, 7));
        faces.add(new Triple<>(0, 7, 3));

        faces.add(new Triple<>(1, 2, 6));
        faces.add(new Triple<>(1, 6, 5));

        faces.add(new Triple<>(0, 1, 5));
        faces.add(new Triple<>(0, 5, 4));

        faces.add(new Triple<>(2, 3, 7));
        faces.add(new Triple<>(2, 7, 6));

        Color[] colors = new Color[12];
        colors[0] = new Color(0, 0, 255);
        colors[1] = new Color(0, 0, 255);
        colors[2] = new Color(0, 255, 0);
        colors[3] = new Color(0, 255, 0);
        colors[4] = new Color(255, 0, 0);
        colors[5] = new Color(255, 0, 0);
        colors[6] = new Color(255, 255, 0);
        colors[7] = new Color(255, 255, 0);
        colors[8] = new Color(255, 0, 255);
        colors[9] = new Color(255, 0, 255);
        colors[10] = new Color(0, 255, 255);
        colors[11] = new Color(0, 255, 255);

        return new Object(vertices, faces, colors);
    }

    public static Object readObj(String filename, Color color) {
        List<Vec3> vertices = new ArrayList<>();
        List<Triple<Integer, Integer, Integer>> faces = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("v ")) {
                    vertices.add(new Vec3(Float.parseFloat(line.split(" ")[1]), Float.parseFloat(line.split(" ")[2]), Float.parseFloat(line.split(" ")[3])));
                }
                if (line.startsWith("f ")) {
                    String[] indices = line.split(" ");
                    int[] intIndices = new int[indices.length];
                    for(int i = 1; i < indices.length; i++) {
                        intIndices[i] = Integer.parseInt(indices[i].split("/")[0]) - 1;
                    }
                    faces.add(new Triple<>(intIndices[1], intIndices[2], intIndices[3]));
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        Color[] colors = new Color[faces.size()];
        for (int i = 0; i < faces.size(); i++) {
            colors[i] = color;
        }

        return new Object(vertices.toArray(new Vec3[0]), faces, colors);
    }

    public static void main(String[] args) {

        int width = 100;
        int height = 100;

        Vec3 origin = new Vec3(2.0f, 10.0f, 8.0f);
        Vec3 destination = new Vec3(0.0f, 5.0f, 0.0f);
        Vec3 direction = destination.sub(origin).normalize();
        float fov = 90.0f;

        ArrayList<Object> objects = new ArrayList<>();
        // objects.add(createDebugCube(8.0f));
        Object bunny = readObj("data/bunny.obj", new Color(255, 0, 0));
        bunny.vertices = Arrays.stream(bunny.vertices).map(v -> v.scale(50.0f)).toArray(Vec3[]::new);
        objects.add(bunny);

        Color[][] image = render(width, height, objects, origin, direction, fov);
        // writePPM(image, "output/java.ppm");
        saveImage(image, "output/java.png");
    }
}
