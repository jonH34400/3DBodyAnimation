#include <array>
#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <nlohmann/json.hpp>
#include <smplx/smplx.hpp>
#include "OptimizeSMPL.hpp"
#include "wandbcpp.hpp"

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
/*10  l-toe */  32,
/*11  r-toe */  31,
/*12  neck   */  -1,           
/*13  l-clr */  -1,
/*14  r-clr */  -1,
/*15  head   */  0,           
/*16  l-sld  */  12,
/*17  r-sld  */  11,
/*18  l-elb */  14,
/*19  r-elb */  13,
/*20  l-wrst */  16,          
/*21  r-wrst */  15,          
/*22  l-palm  */  -1,          
/*23  r-palm  */  -1           
};

// ---- 13-joint subset we actually keep (ids in SMPL space) -------------
static const std::array<int,13> USE_SMPL = {
      1, 2,            // hips
      4, 5,             // knees
      7, 8,             // ankles
      15,                 // head
      16, 17,             // shoulders
      18, 19,             // elbows
      20, 21             // wrists
};

static const int BONES[][2] = {
        {1,2},{1,4},{2,5},{4,7},{5,8},        // legs
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


void overlay_smpl(const smplx::Body<smplx::model_config::SMPL_v1>& body,
                  cv::Mat& img,
                  double  scale,
                  double  fx, double fy,
                  double  cx, double cy,
                  const cv::Scalar& color,
                  int thickness = 2)
{
    // body.trans() already stores the optimised translation (metres)
    const auto& T = body.trans();              // (tx, ty, tz)

    Eigen::MatrixXd J = body.joints().cast<double>();
    std::array<cv::Point,24> proj;

    for (int i = 0; i < 24; ++i)
    {
        // P_cam = s·P + T   -- exactly like the cost function
        double Xc = scale * J(i,0) + T(0);
        double Yc = scale * J(i,1) + T(1);
        double Zc = scale * J(i,2) + T(2);

        double u  = fx *  Xc / Zc + cx;
        double v  = fy * -Yc / Zc + cy;   // flip Y once

        proj[i] = {int(u), int(v)};
    }

    for (auto& b : BONES)
        cv::line(img, proj[b[0]], proj[b[1]], color, thickness);
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
    smplx::Model<smplx::model_config::SMPL_v1> model(argv[1]);

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

    //------------------ key-point loading ---------------------------------
    auto kps = load_mp(argv[2], W, H);     // use real W,H
    std::cout << "keypoints used: " << kps.size() << '\n';

    //------------------ build initial body (all-zero pose) ----------------
    smplx::Body<smplx::model_config::SMPL_v1> body0(model);
    body0.set_zero();
    body0.trans()(2) = 3.0;                // ~3 m from camera
    body0.update();

    //------------------ draw initial overlay ------------------------------
    cv::Mat img_init = img.clone();        // keep original clean
    overlay_smpl(body0, img_init, 1.0, fx, fy, cx, cy, cv::Scalar(255,0,0));   // blue

    for (auto& kp : kps)                                                   // green
        cv::circle(img_init, {int(kp.u), int(kp.v)}, 3,
                cv::Scalar(0,255,0), -1);

    cv::imwrite("out_init.png", img_init);
    std::cout << "wrote out_init.png\n";

    //------------------ optimise SMPL + global scale ----------------------
    SMPLFitResult R = optimize_smpl(kps, model);       // θ,β,trans,scale

    print_vec("β (10)  ", R.beta, 10);
    print_vec("θ (72)  ", R.theta, 72);   // first 12 shown
    std::cout << "trans   [" << R.trans[0] << ", "
                            << R.trans[1] << ", "
                            << R.trans[2] << "]\n"
            << "scale   " << R.scale << "\n";

    //------------------ build final body ----------------------------------
    smplx::Body<smplx::model_config::SMPL_v1> body_opt(model);
    using S = smplx::Scalar;
    body_opt.shape().head<10>() =
        Eigen::Map<const Eigen::Matrix<double,10,1>>(R.beta).template cast<S>();
    body_opt.pose() =
        Eigen::Map<const Eigen::Matrix<double,72,1>>(R.theta).template cast<S>();
    body_opt.trans() =
        Eigen::Map<const Eigen::Matrix<double,3,1>>(R.trans).template cast<S>();
    body_opt.update();

    //------------------ draw optimised overlay ----------------------------
    cv::Mat img_opt = img.clone();
    overlay_smpl(body_opt, img_opt, R.scale,
                R.scale*fx, R.scale*fy, cx, cy,            // scaled intrinsics
                cv::Scalar(0,0,255));                      // red

    for (auto& kp : kps)                                    // green again
        cv::circle(img_opt, {int(kp.u), int(kp.v)}, 3,
                cv::Scalar(0,255,0), -1);

    cv::imwrite("out_opt.png", img_opt);
    cv::waitKey(0);

    wandbcpp::log({{"init_overlay", wandbcpp::Image("out_init.png")},
               {"opt_overlay",  wandbcpp::Image("out_opt.png")}});
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