#include <array>
#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Avatar.h"
#include "AvatarOptimizer.h"
#include "Sim3BA.h"

#include <nlohmann/json.hpp>

// ── MediaPipe-to-SMPL mapping (unchanged) -----------------------------------
static const int MP_MAP[24] = {
    -1, 23, 24, -1, 25, 26, -1, 27, 28, -1,
    31, 32, -1, -1, -1, 0, 11, 12, 13, 14,
    15, 16, -1, -1
};
static const std::array<int,13> USE_SMPL = {
    1,2,4,5,7,8,15,16,17,18,19,20,21
};
static const int BONES[][2] = {
    {1,2},{1,4},{2,5},{4,7},{5,8},
    {16,17},{15,16},{15,17},
    {16,18},{17,19},{18,20},{19,21},
    {1,16},{2,17}
};

std::vector<PixelKP> load_mp(const std::string& json_path, int W, int H)
{
    nlohmann::json j;
    std::ifstream(json_path) >> j;
    std::vector<PixelKP> out;

    auto mid = [&](int a, int b) -> std::array<double,3>
    {
        return { 0.5*(j[a]["x"].get<double>() + j[b]["x"].get<double>()),
                 0.5*(j[a]["y"].get<double>() + j[b]["y"].get<double>()),
                 std::min(j[a]["visibility"].get<double>(),
                          j[b]["visibility"].get<double>()) };
    };

    const auto pelvis   = mid(23,24);
    const auto chest    = mid(11,12);

    for (int sid : USE_SMPL)
    {
        double x=0, y=0, vis=0;

        switch (sid)
        {
        case 0: // pelvis = mid-hip
            x = pelvis[0];  y = pelvis[1];  vis = pelvis[2];  break;
        case 6: // chest = mid-shoulder (not in subset, but kept for ref)
            x = chest[0];   y = chest[1];   vis = chest[2];   break;
        default:
            {
                int mp = MP_MAP[sid];
                if (mp < 0) continue;
                x   = j[mp]["x"].get<double>();
                y   = j[mp]["y"].get<double>();
                vis = j[mp]["visibility"].get<double>();
            }
        }
        if (vis < 0.5) continue;
        out.push_back({ sid, x * static_cast<double>(W), y * static_cast<double>(H) });
    }
    return out;
}

// ── Single overlay (works for both initial and optimized poses) --------------
template <class AvatarT>
void overlay_avatar(const AvatarT& avatar, cv::Mat& img,
                    double fx, double fy, double cx, double cy,
                    double scale,
                    const double* aa_root,             // angle-axis (len=3), may be nullptr for identity
                    const cv::Scalar& color, int thickness,
                    const int (*bones)[2], int num_bones)
{
    const auto& J = avatar.jointPos;      // 3xN
    const Eigen::Vector3d& T = avatar.p;  // translation (already set on avatar)

    
    // Build rotation from angle-axis if provided
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    const int L_SHOULDER = 16;   // your mapping uses 16 for left shoulder
    const int R_SHOULDER = 17;   // and 17 for right shoulder

    // Given: J (3xN joints), scale, R (root/world rotation), T (translation)
    auto X_cam = [&](int jid) {
        return scale * (R * J.col(jid)) + T;  // camera coordinates
    };

    Eigen::Vector3d Ls = X_cam(L_SHOULDER);
    Eigen::Vector3d Rs = X_cam(R_SHOULDER);

    // Be sure both are in front of the camera
    bool shoulders_valid = (Ls.z() > 1e-6 && Rs.z() > 1e-6);
    bool facing_camera = true;   // default

    if (shoulders_valid) {
        Eigen::Vector3d across = (Rs - Ls);                 // left->right
        Eigen::Vector3d up(0, 1, 0);                        // camera/world up
        Eigen::Vector3d forward_est = across.cross(up);     // torso forward
        double s = forward_est.normalized().dot(Eigen::Vector3d::UnitZ());
        facing_camera = (s > 0.0); // +Z means toward camera in your setup
        std::cout << facing_camera;
    }
    if (aa_root) {
        Eigen::Vector3d aa(aa_root[0], aa_root[1], aa_root[2]);
        double theta = aa.norm();
        if (theta > 1e-12) R = Eigen::AngleAxisd(theta, aa / theta).toRotationMatrix();
    }

    std::vector<cv::Point> proj(J.cols(), cv::Point(-9999, -9999));
    for (int i = 0; i < J.cols(); ++i) {
        Eigen::Vector3d X = scale * (R * J.col(i)) + T;
        if (X.z() <= 1e-6) continue; // behind/at camera plane
        double u = fx * X.x() / X.z() + cx;
        double v = fy * X.y() / X.z() + cy;
        proj[i] = { static_cast<int>(std::round(u)),
                    static_cast<int>(std::round(v)) };
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

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cout << "usage: 3dba <SMPL.npz> <mp.json> <image.png> "
          << "[max_iters]\n";
        return 0;
    }

    // Load Avatar model
    ark::AvatarModel model_av(argv[1]);

    // Load image
    cv::Mat img = cv::imread(argv[3]);
    if (img.empty()) { std::cerr << "cannot read image\n"; return 1; }
    int max_iters = (argc > 4) ? std::atoi(argv[4]) : 100;
    const int W = img.cols;
    const int H = img.rows;

    // Camera intrinsics
    double fx = 0.9 * W;
    double fy = fx;
    double cx = 0.5 * W;
    double cy = 0.5 * H;
    ark::CameraIntrin intrin;   // default-construct
    intrin.fx = fx; intrin.fy = fy; intrin.cx = cx; intrin.cy = cy;

    // Load 2D keypoints
    auto kps = load_mp(argv[2], W, H);
    std::cout << "keypoints used: " << kps.size() << '\n';
    for (const auto& kp : kps) {
        std::cout << "jid: " << kp.jid << ", u: " << kp.u << ", v: " << kp.v << '\n';
    }

    // Map SMPL joint id to body part (0..6); default 0
    std::map<int, int> jointToPart = {
        {1, 1}, {2, 1}, {4, 2}, {5, 2}, {7, 3}, {8, 3},
        {15, 0}, {16, 4}, {17, 4}, {18, 5}, {19, 5}, {20, 6}, {21, 6}
    };
    std::vector<int> partMap(model_av.numJoints(), 0);
    for (const auto& [jid, part] : jointToPart) {
        if (jid >= 0 && jid < model_av.numJoints()) partMap[jid] = part;
    }
    int numParts = *std::max_element(partMap.begin(), partMap.end()) + 1;
    (void)numParts; // not used below, but kept for context

    std::vector<int> keep;
    keep.reserve(kps.size());
    for (size_t i = 0; i < kps.size(); ++i) {
        int jid = kps[i].jid;
        int part = (jid >= 0 && jid < (int)partMap.size()) ? partMap[jid] : -1;
        if (part >= 0) keep.push_back(static_cast<int>(i));
    }

    // Build initial avatar (zero pose)
    ark::Avatar body0_av(model_av);
    body0_av.w.setZero(model_av.numShapeKeys());
    body0_av.p = Eigen::Vector3d(0, 0, 3.0);
    body0_av.r.clear();
    for (int i = 0; i < model_av.numJoints(); ++i) {
        body0_av.r.push_back(Eigen::Matrix3d::Identity());
    }
    // root Y flip if your model vs. camera have different handedness
    Eigen::Matrix3d flipY = Eigen::Matrix3d::Identity();
    flipY(1,1) = -1; 
    body0_av.r[0] = flipY;  
    Eigen::AngleAxisd yaw_pi(M_PI, Eigen::Vector3d::UnitY()); // +Y axis
    body0_av.r[0] = yaw_pi.toRotationMatrix() * body0_av.r[0]; // camera-frame yaw
    body0_av.update();

    // Draw initial overlay (identity Sim3)
    cv::Mat img_init_av = img.clone();
    for (auto& kp : kps) {
        cv::circle(img_init_av, {static_cast<int>(kp.u), static_cast<int>(kp.v)}, 3,
                   cv::Scalar(0,255,0), -1);
    }
    double aa_id[3] = {0.0, 0.0, 0.0};
    overlay_avatar(body0_av, img_init_av, fx, fy, cx, cy,
                   /*scale=*/1.0, /*aa_root=*/aa_id,
                   cv::Scalar(255,0,0), 2, BONES, (int)(sizeof(BONES)/sizeof(BONES[0])));
    cv::imwrite("../data/out/out_init_av.png", img_init_av);
    std::cout << "wrote out_init_av.png\n";

    // Whitelist: use all detected joints (or provide your own subset)
    std::vector<int> valid_joint_ids;
    valid_joint_ids.reserve(kps.size());
    for (const auto& kp : kps) valid_joint_ids.push_back(kp.jid);

    Sim3Params init{};
    init.scale() = 1.0;
    init.aa_root()[0] = init.aa_root()[1] = init.aa_root()[2] = 0.0;
    init.trans()[0] = body0_av.p.x();
    init.trans()[1] = body0_av.p.y();
    init.trans()[2] = body0_av.p.z();

    // auto [sol, report] = OptimizeSim3Reprojection_OnePoint(
    //     body0_av.jointPos, kps, fx, fy, cx, cy, 15, &init, max_iters);
    auto [sol, report] = OptimizeSim3Reprojection(
            body0_av.jointPos, kps, fx, fy, cx, cy, valid_joint_ids, &init, max_iters);

    std::cout << report << "\n";
    std::cout << "Solved scale: " << sol.scale()
              << " | root aa: [" << sol.aa_root()[0] << ", "
              << sol.aa_root()[1] << ", " << sol.aa_root()[2] << "]"
              << " | T: [" << sol.trans()[0] << ", "
              << sol.trans()[1] << ", " << sol.trans()[2] << "]\n";

    // Apply optimized translation to avatar (avoid double-adding inside overlay)
    body0_av.p = Eigen::Vector3d(sol.trans()[0], sol.trans()[1], sol.trans()[2]);
    body0_av.update();

    // Draw optimized overlay (pass optimized scale and angle-axis)
    cv::Mat img_opt = img.clone();
    overlay_avatar(body0_av, img_opt, fx, fy, cx, cy,
                   sol.scale(), sol.aa_root(),
                   cv::Scalar(255,0,0), 2, BONES, (int)(sizeof(BONES)/sizeof(BONES[0])));
    cv::imwrite("../data/out/out_opt_av.png", img_opt);
    std::cout << "wrote out_opt_av.png\n";

    return 0;
}
