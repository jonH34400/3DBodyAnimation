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
#include "Sim3BA.h"   
#include "RenderSMPLMesh.h"

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

double mean_pixel_error(const std::vector<PixelKP>& kps,
                        const ark::Avatar& avatar,
                        double fx,double fy,double cx,double cy)
{
    double err = 0.0; int cnt = 0;
    for (const auto& kp : kps) {
        const auto& J = avatar.jointPos.col(kp.jid);
        double u = fx * J.x() / J.z() + cx;
        double v = fy * J.y() / J.z() + cy;
        err += std::hypot(u - kp.u, v - kp.v);
        ++cnt;
    }
    return cnt ? err / cnt : 0.0;
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
    const double beta_pose      = (argc > 6) ? std::atoi(argv[6]) : 20;
    const double beta_shape     = (argc > 7) ? std::atoi(argv[7]) : 30;
    bool opt_shape = false;
    bool use_gmm = false;

    for (int i = 8; i < argc; ++i) {
        if (std::string(argv[i]) == "--opt-shape") {
            opt_shape = true;
            break;
        } else if (std::string(argv[i]) == "--use-gmm"){     
            use_gmm   = true;
        }
    }
    fs::create_directories(out_dir);

    // 1) Sample H/W and intrinsics from the first image in the directory
    auto images = list_sorted(img_folder, {".png",".jpg",".jpeg",".bmp"});
    if (images.empty()) { std::cerr << "No images in " << img_folder << "\n"; return 1; }
    cv::Mat img0 = cv::imread(images.front().string());
    if (img0.empty()) { std::cerr << "Failed to read " << images.front() << "\n"; return 1; }

    const int W = img0.cols, H = img0.rows;
    double f = std::max(H, W);
    double fx = 0.9 * f;
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
        Eigen::AngleAxisd yaw_pi(M_PI, Eigen::Vector3d::UnitY());
        body_av.r[0] = yaw_pi.toRotationMatrix() * flipY;
        body_av.update();

        // Sim3 init from joints
        std::vector<int> valid_joint_ids; valid_joint_ids.reserve(kps.size());
        for (const auto& kp : kps) valid_joint_ids.push_back(kp.jid);

        Sim3Params sim3_id{};
        sim3_id.scale() = 1.0;
        sim3_id.aa_root()[0] = sim3_id.aa_root()[1] = sim3_id.aa_root()[2] = 0.0;
        sim3_id.trans()[0] = body_av.p.x();
        sim3_id.trans()[1] = body_av.p.y();
        sim3_id.trans()[2] = body_av.p.z();

        bool ok;
        std::string report;

        auto* gmmPrior = use_gmm ? &model_av.posePrior : nullptr;
        std::cout << "Pose prior components: " << model_av.posePrior.nComps
                << "  (GMM " << (use_gmm ? "ON" : "OFF") << ")\n";

        // timer start
        auto t0 = std::chrono::steady_clock::now();

        if (opt_shape) {
            std::tie(ok, report) = OptimizePoseShapeReprojection(
                model_av, body_av, kps, fx, fy, cx, cy, valid_joint_ids, sim3_id,
                max_iters, /*betaPose=*/beta_pose, /*betaShape=*/beta_shape,
                /*gmmPosePrior=*/gmmPrior);
        } else {
            std::tie(ok, report) = OptimizePoseReprojection(
                model_av, body_av, kps, fx, fy, cx, cy, valid_joint_ids, sim3_id,
                max_iters, /*betaPose=*/beta_pose, /*betaShape=*/beta_shape,
                /*gmmPosePrior=*/gmmPrior);
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // ------------------------------------------------------------------
        // evaluate 2-D pixel error
        // ------------------------------------------------------------------
        body_av.update();
        double px_err = mean_pixel_error(kps, body_av, fx, fy, cx, cy);

        const std::string log_path = out_dir / "log.csv";
        std::ofstream fout;

        // does the file already exist?
        bool file_exists = static_cast<bool>(std::ifstream(log_path));

        fout.open(log_path, std::ios::app);
        if (!file_exists)               // first run → add header
            fout << "frame,mean_pixel_error_px,time_ms\n";

        fout << i          << ','        // frame index
            << px_err     << ','        // 2-D error
            << ms         << '\n';      // optimisation time (ms)
        fout.close();


        cv::Mat color_overlay = img.clone();
        smpl::render::renderSMPLMesh(body_av.cloud, faces, color_overlay, fx, fy, cx, cy,
               /*fill=*/true, /*backface_cull=*/true, /*wireframe=*/false);
        fs::path render2d = out_dir / (std::string("frame_") + std::to_string(i) + "_render.png");
        cv::imwrite(render2d.string(), color_overlay);

    }

    std::cout << "Done.\n";
    return 0;
}