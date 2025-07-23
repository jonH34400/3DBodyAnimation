#include <array>
#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include "Avatar.h"
#include "AvatarOptimizer.h"
#include <nlohmann/json.hpp>
#include "OptimizeSMPL.h"

// ── MediaPipe-to-SMPL mapping ------------------------------------------
static const int MP_MAP[24] = {
/* 0  pelvis */  -1,          // mid(23,24)
/* 1  l-hip  */  23,
/* 2  r-hip  */  24,
/* 3  spine01*/  -1,        
/* 4  l-knee */  25,
/* 5  r-knee */  26,
/* 6  spine02  */  -1,          // mid(11,12)
/* 7  l-ank  */  27,
/* 8  r-ank  */  28,
/* 9  spine03   */  -1,        
/*10  l-toe */  31,
/*11  r-toe */  32,
/*12  neck   */  -1,           
/*13  l-clr */  -1,
/*14  r-clr */  -1,
/*15  head   */  0,           
/*16  l-sld  */  11,
/*17  r-sld  */  12,
/*18  l-elb */  13,
/*19  r-elb */  14,
/*20  l-wrst */  15,          
/*21  r-wrst */  16,          
/*22  l-palm  */  -1,          
/*23  r-palm  */  -1           
};

// ---- 13-joint subset we actually keep (ids in SMPL space) -------------
static const std::array<int,13> USE_SMPL = {
      1, 2,            // hips
      4, 5,             // knees
      7, 8,             // ankles
      //10,11,            //toes
      15,                 // head
      16, 17,             // shoulders
      18, 19,             // elbows
      20, 21             // wrists
};

static const int BONES[][2] = {
        {1,2},{1,4},{2,5},{4,7},{5,8},  // legs
        //{7,10},{8,11},       //toes 
        {16,17},{15,16},{15,17},        // head, shoulder
        {16,18},{17,19},{18,20},{19,21}, // arms
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
        case 0:  // pelvis = mid-hip
            x = pelvis[0];  y = pelvis[1];  vis = pelvis[2];  break;

        case 6:  // chest = mid-shoulder (not in subset, but kept for ref)
            x = chest[0];   y = chest[1];   vis = chest[2];   break;

        default:
            {
                int mp = MP_MAP[sid];
                if (mp < 0) continue;                  // no MP landmark
                x   = j[mp]["x"].get<double>();
                y   = j[mp]["y"].get<double>();
                vis = j[mp]["visibility"].get<double>();
            }
        }

        if (vis < 0.5) continue;                       // skip low-vis
        out.push_back({ sid,
                        x * static_cast<double>(W),
                        y * static_cast<double>(H) });
    }
    return out;
}

// Assuming BONES is defined somewhere as an array of joint connections
// Example: const std::array<std::array<int, 2>, 24> BONES = {...};
void overlay_avatar(const ark::Avatar& avatar,
                   cv::Mat& img,
                   double scale,
                   double fx, double fy,
                   double cx, double cy,
                   const cv::Scalar& color,
                   int thickness = 2)
{
    // Get joint positions from avatar (3 x num_joints matrix)
    const ark::CloudType& J = avatar.jointPos;
    int num_joints = J.cols();
    std::vector<cv::Point> proj(num_joints);

    // Avatar's root position is stored separately in p
    const Eigen::Vector3d& T = avatar.p;

    //std::cout << "overlay_avatar: num_joints = " << num_joints << "\n";

    for (int i = 0; i < num_joints; ++i)
    {
        // P_cam = s·P + T
        double Xc = scale * J(0, i) + T(0);
        double Yc = scale * J(1, i) + T(1);
        double Zc = scale * J(2, i) + T(2);

        if (Zc <= 0) {
            std::cout << " → Joint behind camera, skipping.\n";
            continue;
        }

        // Perspective projection with Y flip
        double u = fx * -Xc / Zc + cx;
        double v = fy * -Yc / Zc + cy;  // flip Y once

        proj[i] = {static_cast<int>(u), static_cast<int>(v)};
    }

    // Draw bones
    for (auto& b : BONES) {
        cv::line(img, proj[b[0]], proj[b[1]], color, thickness); 
    }
}

void print_vec(const std::string& name,
               const double*      v,
               int                n,
               int                max_to_show = 12)
{
    std::cout << std::fixed << std::setprecision(4);
    std::cout << name << " [";
    for (int i = 0; i < n; ++i) {
        if (i == max_to_show) { std::cout << " …"; break; }
        std::cout << (i ? ", " : "") << v[i];
    }
    std::cout << "]\n";
}
 


int main(int argc, char** argv)
{
    /*  argv: 0          1           2          3            4   5   6   7
              prog  <SMPL.npz> <mp.json> <image.png>  [fx] [fy] [cx] [cy]   */
    if (argc < 4) {
        std::cout << "usage: 3dba <SMPL.npz> <mp.json> <image.png> "
                     "[fx] [fy] [cx] [cy]\n";
        return 0;
    }

    //------------------ load model -----------------------------------------
    //smplx::Model<smplx::model_config::SMPL_v1> model(argv[1]); //SMPLx
    ark::AvatarModel model_av(argv[1]); //Avatar

    //------------------ load image -----------------------------------------
    cv::Mat img = cv::imread(argv[3]);
    if (img.empty()) { std::cerr << "cannot read image\n"; return 1; }

    const int W = img.cols;
    const int H = img.rows;
    std::cout << "Image size  : " << W << " × " << H << '\n';

    //------------------ camera intrinsics ----------------------------------
    /* defaults:  fx = fy = 0.9*W ; cx,cy = image centre */
    double fx = (argc > 4) ? std::atof(argv[4]) : 0.9 * W;
    double fy = (argc > 5) ? std::atof(argv[5]) : fx;
    double cx = (argc > 6) ? std::atof(argv[6]) : 0.5 * W;
    double cy = (argc > 7) ? std::atof(argv[7]) : 0.5 * H;

    std::cout << "Camera fx,fy: " << fx << ", " << fy << '\n'
              << "Camera cx,cy: " << cx << ", " << cy << '\n';

    ark::CameraIntrin intrin;
    intrin.fx = fx;
    intrin.fy = fy;
    intrin.cx = cx;
    intrin.cy = cy;

    //------------------ key-point loading ---------------------------------
    auto kps = load_mp(argv[2], W, H);     // use real W,H
    std::cout << "keypoints used: " << kps.size() << '\n';

    // Data Cloud estimate
    Eigen::Matrix<double, 3, Eigen::Dynamic> dataCloud(3, kps.size());
    double z = 2.0; // arbitrary depth
    for (size_t i = 0; i < kps.size(); ++i) {
        double x = (kps[i].u - cx) * z / fx;
        double y = (kps[i].v - cy) * z / fy;
        dataCloud(0, i) = x;
        dataCloud(1, i) = -y;
        dataCloud(2, i) = z;
    }

    //------------------ build initial body (all-zero pose) ----------------
    ark::Avatar body0_av (model_av);
    body0_av.w.setZero(model_av.numShapeKeys());
    body0_av.p = Eigen::Vector3d(0, 0, 2.0);
    body0_av.r.clear();
    for (int i = 0; i < model_av.numJoints(); i++) {
        body0_av.r.push_back(Eigen::Matrix3d::Identity());
    }
    body0_av.update();

    //------------------ draw initial overlay ------------------------------
    cv::Mat img_init_av = img.clone();  
    overlay_avatar(body0_av,img_init_av, 1.0, fx, fy, cx, cy, cv::Scalar(255,0,0)); 

    for (auto& kp : kps)                                                   
        cv::circle(img_init_av, {int(kp.u), int(kp.v)}, 3,
                cv::Scalar(0,255,0), -1);

    cv::imwrite("out_init_av.png", img_init_av);
    std::cout << "wrote out_init_av.png\n";


    //------------------ params for optimization ----------------------
    std::map<int, int> jointToPart = {
        {1, 1}, {2, 1},     // hips
        {4, 2}, {5, 2},     // knees
        {7, 3}, {8, 3},     // ankles
        {15, 0},            // head
        {16, 4}, {17, 4},   // shoulders
        {18, 5}, {19, 5},   // elbows
        {20, 6}, {21, 6}    // wrist
    };

    std::vector<int> partMap(model_av.numJoints(), 0);
    for (const auto& [jid, part] : jointToPart) {
        if (jid >= 0 && jid < model_av.numJoints()) {
            partMap[jid] = part;
        }
    }

    Eigen::VectorXi dataPartLabels(kps.size());

    for (size_t i = 0; i < kps.size(); ++i) {
        int smpl_joint_id = kps[i].jid;

        if (smpl_joint_id >= 0 && smpl_joint_id < partMap.size()) {
            dataPartLabels(i) = partMap[smpl_joint_id];
        } else {
            std::cerr << "Warning: joint id " << smpl_joint_id << " out of range\n";
            dataPartLabels(i) = -1; 
        }
    }

    int numParts = *std::max_element(partMap.begin(), partMap.end()) + 1;

    //------------------ some checking ----------------------
    std::cout << "CHECKING...:\n";

    // Check for right sizes
    std::cout << "dataCloud.cols(): " << dataCloud.cols()
          << ", dataPartLabels.size(): " << dataPartLabels.size() << std::endl;
    
    std::cout << "dataCloud.rows(): " << dataCloud.rows() << std::endl;

    // Check for correct values in DataPartLabal
    for (int i = 0; i < dataPartLabels.size(); i++)
    {
        std::cout << "Data Part Label[" << i << "] = " << dataPartLabels[i] << std::endl; 
    }

    // Check for correct values in DatCloud
    for (int i = 0; i < dataCloud.cols(); ++i) {
        for (int r = 0; r < 3; ++r) {
            double val = dataCloud(r,i);
            std::cout << "Data Cloud [" << r << "][" << i << "] = " << dataCloud(r,i) << std::endl; 
            if (!std::isfinite(val)) {
                std::cerr << "ERROR: dataCloud contains invalid value at (" << r << "," << i << "): " << val << std::endl;
            }
        }
    }

    std::cout << "Checks before optimizing:\n";
    std::cout << "- model numJoints: " << model_av.numJoints() << "\n";
    std::cout << "- shape keys: " << model_av.numShapeKeys() << "\n";
    std::cout << "- avatar.r.size(): " << body0_av.r.size() << "\n";
    std::cout << "- avatar.w.size(): " << body0_av.w.size() << "\n";

    //------------------ optimise body  ----------------------------------
    ark::AvatarOptimizer optimizer(body0_av, intrin, cv::Size(W, H), numParts, partMap);

    optimizer.optimize(dataCloud, dataPartLabels,1,1);
    //body0_av.r[0] = Eigen::Matrix3d::Identity();
    //body0_av.update();

    //------------------ draw optimised overlay ----------------------------
    cv::Mat img_opt_av = img.clone();
    overlay_avatar(body0_av,img_opt_av, 1.0, fx, fy, cx, cy, cv::Scalar(0,0,255)); 

    for (auto& kp : kps)                                                   
        cv::circle(img_opt_av, {int(kp.u), int(kp.v)}, 3,
                cv::Scalar(0,255,0), -1);

    cv::imwrite("out_opt_av.png", img_opt_av);
    std::cout << "wrote out_opt_av.png\n";

    return 0;

}

// int main(int argc, char **argv)
// {
//     if (argc < 3) {
//         std::cout << "usage: <SMPL.npz> <mp.json>\n";
//         return 0;
//     }

//     smplx::Model<smplx::model_config::SMPL_v1> model(argv[1]);
    
    
//     cv::Mat img = cv::imread("../data/frames_annotated/video7/frame_0000_annotated.png");
//     if (img.empty()) {
//         std::cerr << "could not read image\n";
//         return 1;
//     }
    
//     constexpr int W=2000,H=720;
//     const double fx = 1000.0, fy = 1000.0;
//     const double cx = W * 0.5, cy = H * 0.5;

//     auto kps = load_mp(argv[2], W, H);
//     std::cout << "keypoints used: " << kps.size() << "\n";
//     smplx::Body<smplx::model_config::SMPL_v1> body0(model);
//     body0.set_zero();
//     body0.trans()(2) = 3.0;   // 3 m in front of camera
//     body0.update();

//     // overlay initial guess in *blue*
//     overlay_smpl(body0, img, fx, fy, cx, cy, cv::Scalar(255,0,0));

//     // overlay detected key-points in *green*
//     for (auto& kp : kps)
//         cv::circle(img, cv::Point((int)kp.u, (int)kp.v), 3,
//                 cv::Scalar(0,255,0), -1);

//     cv::imwrite("out.png", img);
//     cv::waitKey(0);

//     // auto result = optimize_smpl(kps, model);

//     // std::cout << result.summary.BriefReport() << "\n";
//     // std::cout << "Final trans : [" << result.trans[0] << ", " << result.trans[1]
//     //           << ", " << result.trans[2] << "]\n";
//     // std::cout << "Beta[0..4]  : { ";
//     // for (int i = 0; i < 5; ++i) std::cout << result.beta[i] << (i < 4 ? ", " : " ");
//     // std::cout << "}\nTheta[0..4] : { ";
//     // for (int i = 0; i < 5; ++i) std::cout << result.theta[i] << (i < 4 ? ", " : " ");
//     // std::cout << "}\n";

//     // // build SMPL body from optimized params
//     // using S = smplx::Scalar;  // float internal to smplx
//     // smplx::Body<smplx::model_config::SMPL_v1> body(model);

//     // body.shape().head<10>() =
//     //     Eigen::Map<const Eigen::Matrix<double, 10, 1>>(result.beta).template cast<S>();
//     // body.pose() =
//     //     Eigen::Map<const Eigen::Matrix<double, 72, 1>>(result.theta).template cast<S>();
//     // body.trans() =
//     //     Eigen::Map<const Eigen::Matrix<double, 3, 1>>(result.trans).template cast<S>();

//     // body.update();

//     // Eigen::MatrixXd joints3D = body.joints().template cast<double>();

//     return 0;
// }