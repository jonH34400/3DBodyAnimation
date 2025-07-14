#include <array>
#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

// ── 3rd-party ───────────────────────────────────────────────────
#include <ceres/ceres.h>
#include <nlohmann/json.hpp>
#include <smplx/smplx.hpp>

// ── local ──────────────────────────────────────────────────────
#include "OptimizeSMPL.hpp"

// Very small loader for MediaPipe JSON (33 landmarks)
static const int MAP[24] = { -1,23,24, 11,-1,0, 11,12,13,14,15,16,
                             25,26,27,28,29,30,31,32, 1,2,7,8 };

std::vector<PixelKP> load_mp(const std::string &p, int W, int H)
{
    std::ifstream f(p);
    nlohmann::json j;
    f >> j;
    std::vector<PixelKP> out;

    auto mid = [&](int a, int b) {
        return std::array<double, 3>{
            (j[a]["x"].get<double>() + j[b]["x"].get<double>()) * 0.5,
            (j[a]["y"].get<double>() + j[b]["y"].get<double>()) * 0.5,
            std::min(j[a]["visibility"].get<double>(),
                     j[b]["visibility"].get<double>())};
    };

    const auto pelvis = mid(23, 24), spine2 = mid(11, 12);
    for (int sid = 0; sid < 24; ++sid) {
        double x, y, vis;
        if (sid == 0) {
            x = pelvis[0];
            y = pelvis[1];
            vis = pelvis[2];
        } else if (sid == 4) {
            x = spine2[0];
            y = spine2[1];
            vis = spine2[2];
        } else {
            int mp = MAP[sid];
            if (mp < 0) continue;
            x = j[mp]["x"].get<double>();
            y = j[mp]["y"].get<double>();
            vis = j[mp]["visibility"].get<double>();
        }
        if (vis < 0.5) continue;
        out.push_back({sid, x * static_cast<double>(W), y * static_cast<double>(H)});
    }
    return out;
}

void overlay_smpl(const smplx::Body<smplx::model_config::SMPL_v1>& body,
                  cv::Mat& img,
                  double fx, double fy, double cx, double cy,
                  const cv::Scalar& color)
{
    Eigen::MatrixXd J = body.joints().cast<double>();      // Nx3, metres
    for (int i = 0; i < J.rows(); ++i)
    {
        double X = J(i,0),  Y = J(i,1),  Z = J(i,2);

        // one-time Y-flip here keeps draw-code clean
        double u = fx *  X / Z + cx;
        double v = fy * -Y / Z + cy;

        cv::circle(img, cv::Point(static_cast<int>(u), static_cast<int>(v)),
                   3, color, -1);
    }
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

    //------------------ key-point loading ----------------------------------
    auto kps = load_mp(argv[2], W, H);          // use real W,H
    std::cout << "keypoints used: " << kps.size() << '\n';

    //------------------ build + draw initial body --------------------------
    smplx::Body<smplx::model_config::SMPL_v1> body0(model);
    body0.set_zero();
    body0.trans()(2) = 3.0;   // 3 m in front of camera
    body0.update();

    overlay_smpl(body0, img, fx, fy, cx, cy, cv::Scalar(255,0,0)); // blue

    for (auto& kp : kps)                                           // green
        cv::circle(img, {int(kp.u), int(kp.v)}, 3, cv::Scalar(0,255,0), -1);

    cv::imwrite("out.png", img);
    cv::waitKey(0);
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