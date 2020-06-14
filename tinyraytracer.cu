#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include "geometry.h"
//#include "material.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cfloat>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}



struct Ray {
    __device__ Ray() {}
    __device__ Ray(const Vec3f& a, const Vec3f& b) { A = a; B = b; }
    __device__ Vec3f origin() const       { return A; }
    __device__ Vec3f direction() const    { return B; }
    __device__ Vec3f point_at_parameter(float t) const { return A + t*B; }

    Vec3f A;
    Vec3f B;
};



struct Light {
    __host__ __device__ Light(const Vec3f &p, const float i) : position(p), intensity(i) {}
    Vec3f position;
    float intensity;
};

struct Material {
    __host__ __device__ Material(const float r, const Vec4f &a, const Vec3f &color, const float spec) : refractive_index(r), albedo(a), diffuse_color(color), specular_exponent(spec) {}
    __host__ __device__ Material() : refractive_index(1), albedo(1,0,0,0), diffuse_color(), specular_exponent() {}
    float refractive_index;
    Vec4f albedo;
    Vec3f diffuse_color;
    float specular_exponent;
};

struct HitRecord
{
    float t;
    Vec3f p;
    Vec3f normal;
    Material material;
};

struct Sphere {
    Vec3f center;
    float radius;
    Material material;

    __host__ __device__ Sphere(const Vec3f &c, const float r, const Material& m) : center(c), radius(r), material(m) {}

    __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        Vec3f oc = r.origin() - center;
        float a = dot(r.direction(), r.direction());
        float b = dot(oc, r.direction());
        float c = dot(oc, oc) - radius*radius;
        float discriminant = b*b - a*c;
        if (discriminant > 0) {
            float temp = (-b - sqrt(discriminant))/a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.point_at_parameter(rec.t);
                rec.normal = (rec.p - center) * ((float)1.0 / radius);
                rec.material = material;
                return true;
            }
            temp = (-b + sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.point_at_parameter(rec.t);
                rec.normal = (rec.p - center)* ((float)1.0 / radius);
                rec.material = material;
                return true;
            }
        }
        return false;
    }


};

__device__
Vec3f reflect(const Vec3f &I, const Vec3f &N) {
    return I - N*2.f*(I*N);
}

__device__
Vec3f refract(const Vec3f &I, const Vec3f &N, const float eta_t, const float eta_i=1.f) { // Snell's law
    float cosi = - fmaxf(-1.f, fminf(1.f, I*N));
    if (cosi<0) return refract(I, -N, eta_i, eta_t); // if the ray comes from the inside the object, swap the air and the media
    float eta = eta_i / eta_t;
    float k = 1 - eta*eta*(1 - cosi*cosi);
    return k<0 ? Vec3f(1,0,0) : I*eta + N*(eta*cosi - sqrtf(k)); // k<0 = total reflection, no ray to refract. I refract it anyways, this has no physical meaning
}


__device__
bool hit(const Ray &ray, float t_min, float t_max,  const int sphere_cnt, const Sphere* spheres, HitRecord& rec) {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0; i < sphere_cnt; i++) {
        if (spheres[i].hit(ray, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

__device__ int fact(int f)
{
    if (f == 0)
        return 1;
    else
        return f * fact(f - 1);
}


__device__
Vec3f cast_ray(Ray& ray, int sphere_cnt, int light_cnt, const Sphere* spheres, const Light* lights, int depth) {

//    HitRecord rec;
//
//    if (depth>0 || !hit(ray, 0.001f, FLT_MAX, sphere_cnt, spheres, rec)) {
//        return Vec3f(0.2, 0.7, 0.8); // background color
//    }
//
//    Vec3f dir = ray.direction();
//    Vec3f N = rec.normal;
//    Material material = rec.material;
//    Vec3f point = rec.p;
//    Vec3f reflect_dir = reflect(dir, N).normalize();
//    Vec3f refract_dir = refract(dir, N, material.refractive_index).normalize();
//    Vec3f reflect_orig = reflect_dir*N < 0 ? point - N*(float)1e-3 : point + N*(float)1e-3; // offset the original point to avoid occlusion by the object itself
//    Vec3f refract_orig = refract_dir*N < 0 ? point - N*(float)1e-3 : point + N*(float)1e-3;
//    Ray reflect_ray(reflect_orig, reflect_dir);
//    Ray refract_ray(refract_orig, refract_dir);
////    Vec3f reflect_color = cast_ray(reflect_ray, sphere_cnt, light_cnt, spheres, lights, depth + 1);
////    Vec3f refract_color = cast_ray(refract_ray, sphere_cnt, light_cnt, spheres, lights, depth + 1);
//
//    Vec3f reflect_color = Vec3f (0.0f,0.0f,0.0f);//cast_ray(reflect_ray, sphere_cnt, light_cnt, spheres, lights, depth + 1);
//    Vec3f refract_color = Vec3f (0,0,0);//cast_ray(refract_ray, sphere_cnt, light_cnt, spheres, lights, depth + 1);

//    printf("reflect_color*material.albedo[2] %f  %f %f %f\n ", reflect_color.x, reflect_color.y, reflect_color.z, material.albedo[2]);
//    printf("light_dir: %d \n", depth);
//    printf("light_dir: %f %f %f\n", lights[0].position.x, lights[0].position.y, lights[0].position.z);
//    printf("spheres: %f %f %f\n", spheres[0].center.x, spheres[0].center.y, spheres[0].center.z);

    float diffuse_light_intensity = 0, specular_light_intensity = 0;
//    for (int i=0; i<light_cnt; i++) {
////          printf("light_dir: %f %f %f\n", lights[i].position.x, lights[i].position.y, lights[i].position.z);
//
//        Vec3f light_dir      = (lights[i].position - point).normalize();
////        printf("light_dir: %f %f %f\n", light_dir.x, light_dir.y, light_dir.z);
//        float light_distance = (lights[i].position - point).norm();
//
//        Vec3f shadow_orig = light_dir*N < 0 ? point - N*(float)1e-3 : point + N*(float)1e-3; // checking if the point lies in the shadow of the lights[i]
//        Vec3f shadow_pt, shadow_N;
//        Material tmpmaterial;
////        if (hit(shadow_orig, light_dir, spheres, shadow_pt, shadow_N, tmpmaterial) && (shadow_pt-shadow_orig).norm() < light_distance)
////            continue;
//        Ray shadow_ray(shadow_orig, light_dir);
//        HitRecord tem_rec;
//        if (hit(shadow_ray, 0.001f, FLT_MAX, sphere_cnt, spheres, tem_rec)) {
//            if ((tem_rec.p -shadow_orig).norm() < light_distance)
//                continue;
//        }
//
//        diffuse_light_intensity  += lights[i].intensity * fmaxf(0.f, light_dir*N);
//        specular_light_intensity += powf(fmaxf(0.f, -reflect(-light_dir, N)*dir), material.specular_exponent)*lights[i].intensity;
//    }

//    Vec3f temp0 = reflect_color*material.albedo[2];
//    printf("reflect_color*material.albedo[2] %f   %f %f %f\n ",
//           temp0.x, temp0.y, temp0.z, material.albedo[2]);
//    Vec3f color = material.diffuse_color ;//* (diffuse_light_intensity * material.albedo[0]);
////    color = color + Vec3f(1., 1., 1.)*specular_light_intensity * material.albedo[1];
////    color = color + temp0;
////    return material.diffuse_color * diffuse_light_intensity * material.albedo[0]
////            + Vec3f(1., 1., 1.)*specular_light_intensity * material.albedo[1]
////            + temp0
////            /*+ refract_color*material.albedo[3]*/;
//
//    return color;

    //
//    HitRecord rec;
//    if (hit(ray, 0.001f, FLT_MAX, sphere_cnt, spheres, rec)) {
////        return Vec3f(N.x+1.0f,N.y+1.0f,N.z+1.0f) * 0.5f;
////        return Vec3f(0.2, 0.7, 0.8); // background color
//        return 0.5f*Vec3f(rec.normal.x+1.0f, rec.normal.y+1.0f, rec.normal.z+1.0f);
//    } else {
////        return Vec3f(0.2, 0.7, 0.8); // background color
//        Vec3f unit_direction = (ray.direction());
//        float t = 0.5f*(unit_direction.y + 1.0f);
//        return Vec3f(1.0, 1.0, 1.0)*(1.0f-t) + Vec3f(0.5, 0.7, 1.0)*t;
//    }

//    printf("recursive: %d\n", fact(3));
    Ray cur_ray = ray;
    float cur_attenuation = 1.0f;
    for(int i = 0; i < 50; i++) {
        HitRecord rec;
        if (hit(cur_ray, 0.001f, FLT_MAX, sphere_cnt, spheres, rec)) {
            Vec3f target = rec.p + rec.normal /* + random_in_unit_sphere(local_rand_state) */;
            cur_attenuation *= 0.5f;
            cur_ray = Ray(rec.p, target-rec.p);
        }
        else {
            Vec3f unit_direction = (cur_ray.direction()).normalize();
            float t = 0.5f*(unit_direction.y + 1.0f);
            Vec3f c = (1.0f-t)*Vec3f(1.0, 1.0, 1.0) + t*Vec3f(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return Vec3f(0.1,0.3,0.4); // exceeded recursion


}

__device__ Vec3f getcolor(Vec3f orig, Vec3f dir, int sphere_cnt, int light_cnt, Sphere* spheres, Light* lights) {
    return Vec3f(0.2, 0.7, 0.8);
}

__global__ void render_kernel( Sphere* spheres, Light* lights,
                                int sphere_cnt,
                                int light_cnt,
                               int width,
                               int height,
                               float fov,
                               Vec3f* framebuffer) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= width) || (j >= height)) return;
    int pixel_index = j*width + i;

    float dir_x =  (i + 0.5) -  width/2.;
    float dir_y = -(j + 0.5) + height/2.;    // this flips the image at the same time
    float dir_z = -height/(2.*tan(fov/2.));

//    printf("width height %d %d %f %f %f\n", width, height, dir_x, dir_y, dir_z);

    // todo
    Ray ray(Vec3f(0,0,0), (Vec3f(dir_x, dir_y, dir_z)).normalize());
    Vec3f color = cast_ray(ray,
            sphere_cnt, light_cnt, spheres, lights, 0);

    framebuffer[pixel_index] = color;


}


void render(const std::vector<Sphere> &spheres, const std::vector<Light> &lights) {
    const int   width    = 1200;
    const int   height   = 800;
    const float fov      = M_PI/3.;
    std::vector<Vec3f> framebuffer(width*height);

//    #pragma omp parallel for
//    for (size_t j = 0; j<height; j++) { // actual rendering loop
//        for (size_t i = 0; i<width; i++) {
//            float dir_x =  (i + 0.5) -  width/2.;
//            float dir_y = -(j + 0.5) + height/2.;    // this flips the image at the same time
//            float dir_z = -height/(2.*tan(fov/2.));
//            framebuffer[i+j*width] = cast_ray(Vec3f(0,0,0), Vec3f(dir_x, dir_y, dir_z).normalize(), spheres, lights);
//        }
//    }
//
//    std::ofstream ofs; // save the framebuffer to file
//    ofs.open("./out.ppm",std::ios::binary);
//    ofs << "P6\n" << width << " " << height << "\n255\n";
//    for (size_t i = 0; i < height*width; ++i) {
//        Vec3f &c = framebuffer[i];
//        float max = std::max(c[0], std::max(c[1], c[2]));
//        if (max>1) c = c*(1./max);
//        for (size_t j = 0; j<3; j++) {
//            ofs << (char)(255 * std::max(0.f, std::min(1.f, framebuffer[i][j])));
//        }
//    }
//    ofs.close();

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //// CUDA
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    std::cout << "cuda: " << std::endl;
    const int threadsPerBlock = 16;
    const int blocksPerGrid_x =
            (width+threadsPerBlock-1) / threadsPerBlock ;
    const int blocksPerGrid_y =
            (height+threadsPerBlock-1) / threadsPerBlock ;

    Sphere* dev_spheres;
    Light* dev_lights;

    int sphere_cnt = spheres.size();
    int light_cnt = lights.size();

    Vec3f* dev_framebuffer;


    // alloc constant
    checkCudaErrors( cudaMalloc( (void**)&dev_spheres, spheres.size() * sizeof(Sphere) ) );
    checkCudaErrors( cudaMemcpy( dev_spheres, spheres.data(), spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice ) );

    checkCudaErrors( cudaMalloc( (void**)&dev_lights, lights.size() * sizeof(Light) ) );
    checkCudaErrors( cudaMemcpy( dev_lights, lights.data(), lights.size() * sizeof(Light), cudaMemcpyHostToDevice ) );


    // alloc variation
    checkCudaErrors( cudaMalloc( (void**)&dev_framebuffer, width*height * sizeof(Vec3f) ) );


    std::cout << "blocksPerGrid_x: " << blocksPerGrid_x << " " << blocksPerGrid_y << " " << threadsPerBlock << std::endl;
    // launch kernel
    dim3 blocks(blocksPerGrid_x, blocksPerGrid_y);
    dim3 threads(threadsPerBlock,threadsPerBlock);

    render_kernel<<<blocks, threads>>>(dev_spheres, dev_lights,
                                       sphere_cnt, light_cnt,
                                       width, height,
                                       fov,
                                       dev_framebuffer);

    // fetch
    checkCudaErrors( cudaMemcpy( framebuffer.data(), dev_framebuffer,  width*height * sizeof(Vec3f), cudaMemcpyDeviceToHost ) );

    std::ofstream ofs; // save the framebuffer to file
    ofs.open("./out.ppm",std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (size_t i = 0; i < height*width; ++i) {
        Vec3f &c = framebuffer[i];
        float max = std::max(c[0], std::max(c[1], c[2]));
        if (max>1) c = c*(1./max);
        for (size_t j = 0; j<3; j++) {
            ofs << (char)(255 * std::max(0.f, std::min(1.f, framebuffer[i][j])));
        }
    }
    ofs.close();
}

int main() {
    Material      ivory(1.0, Vec4f(0.6,  0.3, 0.1, 0.0), Vec3f(0.4, 0.4, 0.3),   50.);
    Material      glass(1.5, Vec4f(0.0,  0.5, 0.1, 0.8), Vec3f(0.6, 0.7, 0.8),  125.);
    Material red_rubber(1.0, Vec4f(0.9,  0.1, 0.0, 0.0), Vec3f(0.3, 0.1, 0.1),   10.);
    Material     mirror(1.0, Vec4f(0.0, 10.0, 0.8, 0.0), Vec3f(1.0, 1.0, 1.0), 1425.);

    std::vector<Sphere> spheres;
    spheres.push_back(Sphere(Vec3f(-3,    0,   -16), 2,      ivory));
    spheres.push_back(Sphere(Vec3f(-1.0, -1.5, -12), 2,      glass));
    spheres.push_back(Sphere(Vec3f( 1.5, -0.5, -18), 3, red_rubber));
    spheres.push_back(Sphere(Vec3f( 7,    5,   -18), 4,     mirror));
    spheres.push_back(Sphere(Vec3f( 0,    -100,   0), 95,     mirror));

    std::vector<Light>  lights;
    lights.push_back(Light(Vec3f(-20, 20,  20), 1.5));
    lights.push_back(Light(Vec3f( 30, 50, -25), 1.8));
    lights.push_back(Light(Vec3f( 30, 20,  30), 1.7));

    render(spheres, lights);

    return 0;
}

