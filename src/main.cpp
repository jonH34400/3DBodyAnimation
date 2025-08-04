#include <array>
#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Avatar.h"
#include "AvatarOptimizer.h"
#include "AvatarRenderer.h"
#include "Sim3BA.h"
#include <nlohmann/json.hpp>

static const int MP_MAP[24] = {
    -1, 23, 24, -1, 25, 26, -1, 29, 30, -1,
    31, 32, -1, -1, -1, 0, 11, 12, 13, 14,
    15, 16, 19, 20
};
static const std::array<int,17> USE_SMPL = {
    1, 2, 4, 5, 7, 8, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23
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

    auto mid = [&](int a, int b) -> std::array<double,3> {
        return { 0.5*(j[a]["x"].get<double>() + j[b]["x"].get<double>()),
                 0.5*(j[a]["y"].get<double>() + j[b]["y"].get<double>()),
                 std::min(j[a]["visibility"].get<double>(),
                          j[b]["visibility"].get<double>()) };
    };

    const auto pelvis = mid(23, 24);
    const auto chest  = mid(11, 12);

    for (int sid : USE_SMPL) {
        double x=0, y=0, vis=0;
        switch (sid) {
        case 0: // pelvis (mid-hip)
            x = pelvis[0];  y = pelvis[1];  vis = pelvis[2];  break;
        case 6: // chest (mid-shoulder)
            x = chest[0];   y = chest[1];   vis = chest[2];   break;
        default: {
            int mp = MP_MAP[sid];
            if (mp < 0) continue;
            x   = j[mp]["x"].get<double>();
            y   = j[mp]["y"].get<double>();
            vis = j[mp]["visibility"].get<double>();
        }
        }
        if (vis < 0.5) continue;
        out.push_back({ sid, x * W, y * H });
    }
    return out;
}

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

//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
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
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cout << "usage: 3dba <SMPL.npz> <mp.json> <image.png> [max_iters]\n";
        return 0;
    }

    // Load SMPL avatar model
    ark::AvatarModel model_av(argv[1]);

    // Load input image
    cv::Mat img = cv::imread(argv[3]);
    if (img.empty()) {
        std::cerr << "cannot read image\n";
        return 1;
    }
    int max_iters = (argc > 4) ? std::atoi(argv[4]) : 100;
    const int W = img.cols, H = img.rows;

    // Camera intrinsics
    double fx = 0.9 * W;
    double fy = fx;
    double cx = 0.5 * W;
    double cy = 0.5 * H;
    ark::CameraIntrin intrin(cv::Vec4d(fx, fy, cx, cy));
    // Load 2D keypoints from MediaPipe JSON
    auto kps = load_mp(argv[2], W, H);
    std::cout << "keypoints used: " << kps.size() << '\n';
    for (const auto& kp : kps) {
        std::cout << "jid: " << kp.jid << ", u: " << kp.u << ", v: " << kp.v << '\n';
    }

    // Map certain SMPL joints to body part indices (for possible use in part-based algorithms)
    std::map<int, int> jointToPart = {
        {1,1},{2,1},{4,2},{5,2},{7,3},{8,3},
        {15,0},{16,4},{17,4},{18,5},{19,5},{20,6},{21,6}
    };
    std::vector<int> partMap(model_av.numJoints(), 0);
    for (const auto& [jid, part] : jointToPart) {
        if (jid >= 0 && jid < model_av.numJoints()) {
            partMap[jid] = part;
        }
    }

    // Filter keypoints to those with known part mapping
    std::vector<int> keep;
    keep.reserve(kps.size());
    for (size_t i = 0; i < kps.size(); ++i) {
        int jid = kps[i].jid;
        int part = (jid >= 0 && jid < (int)partMap.size()) ? partMap[jid] : -1;
        if (part >= 0) keep.push_back((int)i);
    }

    // Initialize avatar in default pose, facing the camera
    ark::Avatar body0_av(model_av);
    body0_av.w.setZero(model_av.numShapeKeys());
    body0_av.p = Eigen::Vector3d(0, 0, 3.0);
    body0_av.r.clear();
    body0_av.r.resize(model_av.numJoints(), Eigen::Matrix3d::Identity());
    // Apply coordinate system adjustments: flip Y axis and yaw 180° so model faces camera
    Eigen::Matrix3d flipY = Eigen::Matrix3d::Identity();
    flipY(1,1) = -1;
    body0_av.r[0] = flipY;
    Eigen::AngleAxisd yaw_pi(M_PI, Eigen::Vector3d::UnitY());
    body0_av.r[0] = yaw_pi.toRotationMatrix() * body0_av.r[0];
    body0_av.update();

    // Draw initial overlay (avatar in blue, keypoints in green)
    cv::Mat img_init = img.clone();
    for (auto& kp : kps) {
        cv::circle(img_init, { (int)kp.u, (int)kp.v }, 3, cv::Scalar(0,255,0), -1);
    }
    double aa_identity[3] = {0.0, 0.0, 0.0};
    overlay_avatar(body0_av, img_init, fx, fy, cx, cy,
                   1.0, aa_identity,
                   cv::Scalar(255,0,0), 2, BONES, (int)(sizeof(BONES)/sizeof(BONES[0])));
    cv::imwrite("out_init_a3.png", img_init);
    std::cout << "wrote out_init_av.png\n";

    // List of valid joint IDs we have observations for
    std::vector<int> valid_joint_ids;
    valid_joint_ids.reserve(kps.size());
    for (const auto& kp : kps) valid_joint_ids.push_back(kp.jid);

    // First, optimize a global Sim3 (scale, root rotation, translation) to roughly align avatar to keypoints
    Sim3Params initSim3;
    initSim3.scale() = 1.0;
    initSim3.aa_root()[0] = initSim3.aa_root()[1] = initSim3.aa_root()[2] = 0.0;
    initSim3.trans()[0] = body0_av.p.x();
    initSim3.trans()[1] = body0_av.p.y();
    initSim3.trans()[2] = body0_av.p.z();

    auto [sim3sol, sim3report] = OptimizeSim3Reprojection(
        body0_av.jointPos, kps, fx, fy, cx, cy, valid_joint_ids, &initSim3, max_iters
    );
    std::cout << sim3report << "\n";
    std::cout << "Solved scale: " << sim3sol.scale()
              << " | root aa: [" << sim3sol.aa_root()[0] << ", "
              << sim3sol.aa_root()[1] << ", " << sim3sol.aa_root()[2] << "]"
              << " | T: [" << sim3sol.trans()[0] << ", "
              << sim3sol.trans()[1] << ", " << sim3sol.trans()[2] << "]\n";

    // Apply the solved translation to the avatar (so the avatar is roughly at correct position)
    body0_av.p = Eigen::Vector3d(sim3sol.trans()[0], sim3sol.trans()[1], sim3sol.trans()[2]);
    body0_av.update();

    // Save an image of avatar after global alignment (for reference)
    cv::Mat img_sim3 = img.clone();
    overlay_avatar(body0_av, img_sim3, fx, fy, cx, cy,
                   sim3sol.scale(), sim3sol.aa_root(),
                   cv::Scalar(255,0,0), 2, BONES, (int)(sizeof(BONES)/sizeof(BONES[0])));
    cv::imwrite("out_sim3_a3.png", img_sim3);
    std::cout << "wrote out_sim3_av.png\n";

    // Next, optimize full pose (all joint rotations) to align avatar joints to 2D keypoints
    auto [poseSuccess, poseReport] = OptimizePoseReprojection(
        model_av, body0_av, kps, fx, fy, cx, cy, valid_joint_ids, sim3sol, max_iters
    );
    std::cout << poseReport << "\n";
    std::cout << "Solved pose scale: " << sim3sol.scale()
              << " | root aa: [" << sim3sol.aa_root()[0] << ", "
              << sim3sol.aa_root()[1] << ", " << sim3sol.aa_root()[2] << "]"
              << " | T: [" << sim3sol.trans()[0] << ", "
              << sim3sol.trans()[1] << ", " << sim3sol.trans()[2] << "]\n";

    // Update avatar's joint positions after pose optimization
    body0_av.update();
    

    // Draw final overlay (avatar in red aligned to person)
    cv::Mat img_opt = img.clone();
    double aa_final[3] = {0.0, 0.0, 0.0};  // avatar.r[0] already contains final root orientation
    overlay_avatar(body0_av, img_opt, fx, fy, cx, cy,
                   sim3sol.scale()*6000, aa_final,
                   cv::Scalar(0,0,255), 2, BONES, (int)(sizeof(BONES)/sizeof(BONES[0])));
    cv::imwrite("out_opt_a3.png", img_opt);
    std::cout << "wrote out_opt_av.png\n";


    // Full model renderization in 2D
    cv::Mat rgb_img = cv::imread(argv[3]);  // Input image
    cv::Mat color_overlay = rgb_img.clone();

    renderSMPLSilhouette( body0_av.cloud, color_overlay, fx, fy, cx, cy);
    
    //color_overlay2.setTo(cv::Scalar(180, 180, 180), silhouette2);
    cv::imwrite("Silhouette_Overlay.png", color_overlay);
    std::cout << "wrote Silhouette_Overlay.png\n";

    return 0;
}
