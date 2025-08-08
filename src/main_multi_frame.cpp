#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "Avatar.h"
#include "RenderSMPLMesh.h"
#include "MultiFrameBA.h"  
#include "Utils.h"

namespace fs = std::filesystem;

int main(int argc,char** argv)
{
    if (argc < 5) {
        std::cout <<
        "usage: 3dba <SMPL.npz> <kps_folder> <image_folder> <out_dir>\n"
        "             [max_iters=120] [max_iters_stage2=120] [anchor_skip=15] [window=30] [overlap=10]\n"
        "             [beta_pose=5.0] [beta_shape=25.0]\n";
        return 0;
    }
    const std::string smpl_path = argv[1];
    const fs::path    kps_folder = argv[2];
    const fs::path    img_folder = argv[3];
    const fs::path    out_dir    = argv[4];

    const int    max_iters_s1 = (argc > 5)  ? std::atoi(argv[5])  : 1000; // stage-1
    const int    max_iters_s2 = (argc > 6)  ? std::atoi(argv[6])  : 500; // stage-2
    const int    skip         = (argc > 7)  ? std::atoi(argv[7])  : 10;  // anchor spacing
    const int    WSIZE        = (argc > 8)  ? std::atoi(argv[8])  : 20;  // window length
    const int    OVERLAP      = (argc > 9)  ? std::atoi(argv[9])  : 5;   // window overlap
    const double betaPose     = (argc > 10) ? std::atof(argv[10]) : 20.0; // pose prior
    const double betaShape    = (argc > 11) ? std::atof(argv[11]) : 30.0;// shape prior
    const double lambdaT      = (argc > 12) ? std::atof(argv[12]) : 3.0; // temporal
    fs::create_directories(out_dir);

    const fs::path log_path = out_dir / "log.csv";
    bool log_exists = fs::exists(log_path);
    std::ofstream log_file(log_path, std::ios::app);
    if (!log_exists) log_file << "frame,mean_pixel_error_px,time_ms\n";

    /* ---------- file lists ---------- */
    auto images = list_sorted(img_folder,{".png",".jpg",".jpeg",".bmp"});
    auto jsons  = list_sorted(kps_folder,{".json"});
    if (images.size() != jsons.size() || images.empty()) {
        std::cerr << "image / json count mismatch\n"; return 1;
    }

    std::cout << "[INFO] frames: " << img_folder << "  = " << std::setw(4) << images.size() << '\n'
            << "[INFO] anchor skip     : " << skip       << '\n'
            << "[INFO] window / overlap: " << WSIZE << " / " << OVERLAP << '\n'
            << "[INFO] β_pose="  << betaPose  << "  β_shape=" << betaShape
            << "  λ_temp=" << lambdaT << std::endl;


    cv::Mat img0 = cv::imread(images.front().string());
    const int W = img0.cols, H = img0.rows;
    const double f  = std::max(W,H);
    const double fx = 0.9 * f, fy = fx, cx = 0.5 * W, cy = 0.5 * H;

    /* ---------- load SMPL ---------- */
    ark::AvatarModel model_av(smpl_path);
    const int nJ = model_av.numJoints();
    std::vector<std::array<int,3>> faces;
    faces.reserve(model_av.numFaces());
    for (int i = 0; i < model_av.mesh.cols(); ++i)
        faces.push_back({ model_av.mesh(0,i), model_av.mesh(1,i), model_av.mesh(2,i) });

    /* ---------- preload keypoints & images ---------- */
    std::vector<cv::Mat>   img_all; img_all.reserve(images.size());
    std::vector<std::vector<PixelKP>> kps_all; kps_all.reserve(jsons.size());
    for (size_t i = 0; i < images.size(); ++i) {
        img_all.emplace_back(cv::imread(images[i].string()));
        kps_all.emplace_back(load_mp_json(jsons[i].string(), W, H));
    }

    /* ---------- create avatars & frame param structs ---------- */
    std::vector<ark::Avatar*>   avatars;
    std::vector<FramePoseParams> poses;
    avatars.reserve(images.size());
    poses.reserve(images.size());

    Eigen::Matrix3d flipY = Eigen::Matrix3d::Identity(); flipY(1,1) = -1;
    Eigen::AngleAxisd yaw_pi(M_PI, Eigen::Vector3d::UnitY());

    for (size_t i=0;i<images.size();++i) {
        avatars.push_back(new ark::Avatar(model_av));
        avatars.back()->w.setZero();                 // shared shape (later locked)
        avatars.back()->p = {0,0,3};
        avatars.back()->r.assign(nJ, Eigen::Matrix3d::Identity());
        avatars.back()->r[0] = yaw_pi.toRotationMatrix()*flipY;
        avatars.back()->update();

        FramePoseParams P;
        P.scale = 1.0;
        P.rootAA[0]=P.rootAA[1]=P.rootAA[2]=0;
        P.rootT[0]=P.rootT[1]=0; P.rootT[2]=3;
        P.jointAA.resize(nJ); for (auto& aa: P.jointAA) aa = {0,0,0};
        poses.push_back(P);
    }

    std::vector<int> valid_ids(USE_SMPL.begin(), USE_SMPL.end());

   /* ===========================================================
   Stage-1 :  use the FIRST 20 frames for shape + pose
   =========================================================== */
    const int SHAPE_FRAMES = 20;
    std::vector<int> anchor_idx;
    {
        const int nshape = std::min<int>(SHAPE_FRAMES, images.size());
        anchor_idx.reserve(nshape);
        for (int i = 0; i < nshape; ++i) anchor_idx.push_back(i);
    }

    std::vector<ark::Avatar*>            anchorAv;
    std::vector<FramePoseParams>         anchorPos;
    std::vector<std::vector<PixelKP>>    kps_anchor;
    anchorAv.reserve(anchor_idx.size());
    anchorPos.reserve(anchor_idx.size());
    kps_anchor.reserve(anchor_idx.size());
    for (int id : anchor_idx) {
        anchorAv.push_back(avatars[id]);
        anchorPos.push_back(poses[id]);
        kps_anchor.push_back(kps_all[id]);
    }

    std::cout << "[INFO] stage-1  (first " << anchor_idx.size() << " frames) for SHAPE+POSE\n";

    auto t0a = std::chrono::steady_clock::now();
    auto [ok1, rep1] = OptimizeMultiFrame(
        model_av,
        anchorAv,
        kps_anchor,
        fx, fy, cx, cy,
        valid_ids,
        anchorPos,
        betaPose, betaShape,   // shape is optimized here
        lambdaT,
        max_iters_s1
    );
    auto t1a = std::chrono::steady_clock::now();
    double ms_anchor = std::chrono::duration<double,std::milli>(t1a - t0a).count();

    std::cout << "[INFO] stage-1 done  (" << (ok1 ? "success" : "fail")
            << ")  in " << ms_anchor << " ms\n"
            << rep1 << std::endl;

    /* share shape among all avatars */
    for (auto* av : avatars) av->w = avatars.front()->w;

    // log anchor frames
    for (size_t k = 0; k < anchor_idx.size(); ++k) {
        int fid = anchor_idx[k];
        anchorAv[k]->update();
        double px = mean_pixel_error(kps_anchor[k], *anchorAv[k], fx, fy, cx, cy);
        // you can log per-frame averaged time:
        log_file << fid << ',' << px << ',' << (ms_anchor / anchor_idx.size()) << '\n';
    }
    log_file.flush();


    /* share shape among all avatars */
    for (auto* av : avatars) av->w = avatars.front()->w;

    /* ---------- initialise render bookkeeping ---------- */
    std::vector<char> rendered(avatars.size(), 0);   // 0 = not yet written

    /* ===================================================
    Stage-2 : sliding window pose refinement + on-the-fly render
    =================================================== */
    const double betaShapeLock = 1e5;
    const int    stride        = WSIZE - OVERLAP;

    for (int s = 0; s < (int)images.size(); s += stride) {
        int e = std::min(s + WSIZE, (int)images.size());

        std::cout << "[INFO] window [" << s << "," << e << ")  solving …" << std::flush;

        /* ---- slice views ---- */
        std::vector<ark::Avatar*>            winAv (avatars.begin()+s, avatars.begin()+e);
        std::vector<FramePoseParams>         winPos(poses.begin()+s  , poses.begin()+e  );
        std::vector<std::vector<PixelKP>>    winKps(kps_all.begin()+s, kps_all.begin()+e);

        /* ---- solve ---- */
        auto t0w = std::chrono::steady_clock::now();
        auto [ok, rep] = OptimizeMultiFrame(
            model_av,
            winAv, winKps,
            fx, fy, cx, cy,
            valid_ids,
            winPos,
            betaPose, betaShapeLock,
            lambdaT,
            max_iters_s2
        );
        auto t1w = std::chrono::steady_clock::now();
        double ms_win = std::chrono::duration<double,std::milli>(t1w - t0w).count();

        std::cout << "  -> " << (ok ? "OK" : "FAIL") << "  (" << ms_win << " ms)\n";

        /* ---- copy results back ---- */
        for (int i = s; i < e; ++i) poses[i] = winPos[i - s];

        for (int i = s; i < e; ++i) {
            winAv[i - s]->update();
            double px = mean_pixel_error(kps_all[i], *winAv[i - s], fx, fy, cx, cy);
            // choose one: either per-frame time as window_avg or the full window time
            log_file << i << ',' << px << ',' << (ms_win / (e - s)) << '\n';
        }
        log_file.flush();


        /* ---- immediate rendering for frames that will never be touched again ----
        Those are exactly the first 'stride' indices of this window.            */
        int last_fixed = std::min(e, s + stride);
        for (int i = s; i < last_fixed; ++i) {
            if (rendered[i]) continue;           // safety
            cv::Mat vis = img_all[i].clone();
            smpl::render::renderSMPLMesh(avatars[i]->cloud, faces,
                                        vis, fx, fy, cx, cy,
                                        /*fill=*/true, /*cull=*/true, /*wire=*/false);
            fs::path out = out_dir / ("frame_" + std::to_string(i) + "_multi.png");
            cv::imwrite(out.string(), vis);
            rendered[i] = 1;
        }
    }

    /* ---------- render the tail frames (the last OVERLAP ones) ---------- */
    for (size_t i = 0; i < avatars.size(); ++i) {
        if (rendered[i]) continue;
        cv::Mat vis = img_all[i].clone();
        smpl::render::renderSMPLMesh(avatars[i]->cloud, faces,
                                    vis, fx, fy, cx, cy,
                                    true, true, false);
        fs::path out = out_dir / ("frame_" + std::to_string(i) + "_multi.png");
        cv::imwrite(out.string(), vis);
        rendered[i] = 1;
    }

    std::cout << "[INFO] rendering finished, saved to  " << out_dir << '\n';
    log_file.close();

    std::cout << "done.\n";
    return 0;
}
