// main.cpp — single-frame SMPL fit per JSON, intrinsics sampled from an image
#include <array>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <cstdio>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include "Avatar.h"
#include "AvatarOptimizer.h"
#include "optimization.h"  

namespace fs = std::filesystem;
using json = nlohmann::json;

// ---------- MP → SMPL mapping (as before) ----------
static const int MP_MAP[24] = {
    -1, 23, 24, -1, 25, 26, -1, 27, 28, -1,
    31, 32, -1, -1, -1, 0, 11, 12, 13, 14,
    15, 16, -1, -1
};
static const std::array<int,17> USE_SMPL = {
    1, 2, 4, 5, 7, 8, 10, 11, 15, 16, 17, 18, 19, 20, 21
};

static const int BONES[][2] = {
    {1,2},{1,4},{2,5},{4,7},{5,8},
    {16,17},{15,16},{15,17},
    {16,18},{17,19},{18,20},{19,21},
    {1,16},{2,17}
};

// ---------- helpers ----------
static bool has_ext(const fs::path& p, std::initializer_list<std::string> exts) {
    auto e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    for (auto& x : exts) if (e == x) return true;
    return false;
}
static std::vector<fs::path> list_sorted(const fs::path& dir, std::initializer_list<std::string> exts) {
    std::vector<fs::path> v;
    for (auto& p : fs::directory_iterator(dir))
        if (p.is_regular_file() && has_ext(p.path(), exts)) v.push_back(p.path());
    std::sort(v.begin(), v.end());
    return v;
}

static std::vector<PixelKP> load_mp_json(const std::string& path, int W, int H) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; return {}; }
    json j; f >> j;

    auto mid = [&](int a, int b) -> std::array<double,3> {
        return {
            0.5*(j[a]["x"].get<double>() + j[b]["x"].get<double>()),
            0.5*(j[a]["y"].get<double>() + j[b]["y"].get<double>()),
            std::min(j[a].value("visibility", 0.0), j[b].value("visibility", 0.0))
        };
    };
    const auto pelvis = mid(23, 24);
    const auto chest  = mid(11, 12);

    std::vector<PixelKP> out;
    out.reserve(USE_SMPL.size());
    for (int sid : USE_SMPL) {
        double x=0, y=0, vis=0;
        switch (sid) {
        case 0: x=pelvis[0]; y=pelvis[1]; vis=pelvis[2]; break;
        case 6: x=chest[0];  y=chest[1];  vis=chest[2];  break;
        default: {
            int mp = MP_MAP[sid];
            if (mp < 0) continue;
            x   = j[mp]["x"].get<double>();
            y   = j[mp]["y"].get<double>();
            vis = j[mp].value("visibility", 1.0);
        }}
        if (vis < 0.5) continue;
        out.push_back({ sid, x * W, y * H });
    }
    return out;
}

// ---------------WRITES MESH OBJ-------------------
static bool write_ply_ascii(const std::string& path,
                            const Eigen::Matrix<double,3,Eigen::Dynamic>& V,
                            const std::vector<std::array<int,3>>& F)
{
    std::ofstream o(path);
    if (!o) { std::cerr << "Cannot write " << path << "\n"; return false; }
    const size_t nV = (size_t)V.cols();
    const size_t nF = F.size();
    o << "ply\nformat ascii 1.0\n";
    o << "element vertex " << nV << "\n";
    o << "property float x\nproperty float y\nproperty float z\n";
    o << "element face " << nF << "\n";
    o << "property list uchar int vertex_indices\n";
    o << "end_header\n";
    o.setf(std::ios::fixed); o.precision(6);
    for (size_t i = 0; i < nV; ++i)
        o << (float)V(0,(int)i) << " " << (float)V(1,(int)i) << " " << (float)V(2,(int)i) << "\n";
    for (const auto& f : F)
        o << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
    return true;
}

//---------------2D SKELETON PROJECTION--------------------
template <class AvatarT>
void overlay_avatar(const AvatarT& avatar, cv::Mat& img,
                    double fx, double fy, double cx, double cy,
                    double scale,
                    const double* aa_root,
                    const cv::Scalar& color, int thickness,
                    const int (*bones)[2], int num_bones)
{
    const auto& J = avatar.jointPos;
    const Eigen::Vector3d& T = avatar.p;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    if (aa_root) {
        Eigen::Vector3d aa(aa_root[0], aa_root[1], aa_root[2]);
        double theta = aa.norm();
        if (theta > 1e-12) {
            R = Eigen::AngleAxisd(theta, aa / theta).toRotationMatrix();
        }
    }

    std::vector<cv::Point> proj(J.cols(), cv::Point(-9999, -9999));
    for (int i = 0; i < J.cols(); ++i) {
        Eigen::Vector3d X = scale * (R * J.col(i)) + T;
        if (X.z() <= 1e-6) continue;
        double u = fx * X.x() / X.z() + cx;
        double v = fy * X.y() / X.z() + cy;
        proj[i] = { static_cast<int>(std::round(u)), static_cast<int>(std::round(v)) };
    }

    for (int k = 0; k < num_bones; ++k) {
        int a = bones[k][0], b = bones[k][1];
        if (a >= 0 && b >= 0 && a < (int)proj.size() && b < (int)proj.size()) {
            const auto& pa = proj[a];
            const auto& pb = proj[b];
            if (pa.x != -9999 && pb.x != -9999) {
                cv::line(img, pa, pb, color, thickness);
            }
        }
    }
}

//----------------FULL MODEL 3DTO2D RENDERIZATION-----------
void renderSMPLSilhouette( const Eigen::Matrix3Xd& cloud, cv::Mat& img,
    double fx, double fy, double cx, double cy) {

    std::vector<cv::Point> proj(cloud.cols(), cv::Point(-9999, -9999));
    for (int i = 0; i < cloud.cols(); ++i) {
        Eigen::Vector3d X = cloud.col(i);
        if (X.z() <= 1e-6) continue;
        double u = fx * X.x() / X.z() + cx;
        double v = fy * X.y() / X.z() + cy;
        proj[i] = { static_cast<int>(std::round(u)), static_cast<int>(std::round(v)) };
        
    }
    for (int j = 0; j < cloud.cols(); ++j){
        const auto& p = proj[j];
        if (p.x != -9999 && p.y != -9999)
            cv::circle(img, { p.x, p.y }, 3, cv::Scalar(0,0,255), -1);
    }

}


// ---------- main ----------
int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cout << "usage: 3dba <SMPL.npz> <kps_folder> <images_folder> <out_dir> [max_iters=100]\n";
        return 0;
    }
    const std::string smpl_path = argv[1];
    const fs::path kps_folder   = argv[2];
    const fs::path img_folder   = argv[3];
    const fs::path out_dir      = argv[4];
    const int max_iters         = (argc > 5) ? std::atoi(argv[5]) : 100;

    fs::create_directories(out_dir);

    // 1) Sample H/W and intrinsics from the first image in the directory
    auto images = list_sorted(img_folder, {".png",".jpg",".jpeg",".bmp"});
    if (images.empty()) { std::cerr << "No images in " << img_folder << "\n"; return 1; }
    cv::Mat img0 = cv::imread(images.front().string());
    if (img0.empty()) { std::cerr << "Failed to read " << images.front() << "\n"; return 1; }

    const int W = img0.cols, H = img0.rows;
    // Camera intrinsics (same heuristic you showed)
    double fx = 0.9 * W;
    double fy = fx;
    double cx = 0.5 * W;
    double cy = 0.5 * H;

    // 2) List JSON keypoints
    auto jsons = list_sorted(kps_folder, {".json"});
    if (jsons.empty()) { std::cerr << "No JSON files in " << kps_folder << "\n"; return 1; }

    // 3) Load SMPL
    ark::AvatarModel model_av(smpl_path);
    const int nJ = model_av.numJoints();
    std::vector<std::array<int,3>> faces;
    faces.reserve(model_av.numFaces());
    for (int i = 0; i < model_av.mesh.cols(); ++i) {
        faces.push_back({ model_av.mesh(0, i), model_av.mesh(1, i), model_av.mesh(2, i) });
    }

    // 4) Process each frame independently
    for (size_t i = 0; i < jsons.size(); ++i) {
        // --- read the matching image ---
        if (i >= images.size()) { std::cerr << "No image for frame " << i << "\n"; break; }
        cv::Mat img = cv::imread(images[i].string());
        if (img.empty()) { std::cerr << "Failed to read " << images[i] << "\n"; continue; }

        // Load keypoints
        auto kps = load_mp_json(jsons[i].string(), W, H);
        if (kps.empty()) {
            std::cerr << "Frame " << i << " has no valid keypoints; skipping.\n";
            continue;
        }

        // Build default avatar (facing camera)
        ark::Avatar body_av(model_av);
        body_av.w.setZero(model_av.numShapeKeys());
        body_av.p = Eigen::Vector3d(0,0,3.0);
        body_av.r.assign(nJ, Eigen::Matrix3d::Identity());
        Eigen::Matrix3d flipY = Eigen::Matrix3d::Identity(); flipY(1,1) = -1;
        // Eigen::AngleAxisd yaw_pi(M_PI, Eigen::Vector3d::UnitY());
        // body_av.r[0] = yaw_pi.toRotationMatrix() * flipY;
        body_av.update();

        // Sim3 init from joints
        std::vector<int> valid_joint_ids; valid_joint_ids.reserve(kps.size());
        for (const auto& kp : kps) valid_joint_ids.push_back(kp.jid);

        double s_opt = 1.0;
        double Raa_opt[3] = {0,0,0};
        double t_opt[3] = { body_av.p.x(), body_av.p.y(), body_av.p.z() };

        auto [ok, report] = unified_ad::OptimizeAllReprojection_AutoDiff(
            model_av, body_av, kps, fx, fy, cx, cy,
            valid_joint_ids, max_iters,
            &s_opt, Raa_opt, t_opt
        );

       // body_av.update();

        // // Write PLY
        // char name[256]; std::snprintf(name, sizeof(name), "frame_%06zu.ply", i);
        // fs::path ply_path = out_dir / name;
        // if (!write_ply_ascii(ply_path.string(), body_av.cloud, faces)) {
        //     std::cerr << "Failed to write " << ply_path << "\n";
        // } else {
        //     std::cout << "Wrote " << ply_path << (ok ? "" : " (solver warning)") << "\n";
        // }

        cv::Mat img_opt = img.clone();
        overlay_avatar(body_av, img_opt, fx, fy, cx, cy,
                    s_opt, /*aa_root=*/Raa_opt,
                    cv::Scalar(0,0,255), 2, BONES, (int)(sizeof(BONES)/sizeof(BONES[0])));

        // body_av.cloud/body updated from avatar_io.update() above
        cv::Mat color_overlay = img.clone();
        renderSMPLSilhouette( body_av.cloud, color_overlay, fx, fy, cx, cy );

        // Save alongside PLY with a matching name and and 3D projection
        fs::path png_path = out_dir / (std::string("frame_") + std::to_string(i) + "_overlay.png");
        fs::path render2d = out_dir / (std::string("frame_") + std::to_string(i) + "_render.png");
        cv::imwrite(png_path.string(), img_opt);
        cv::imwrite(render2d.string(), color_overlay);

    }

    std::cout << "Done.\n";
    return 0;
}