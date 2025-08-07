#pragma once

#include <vector>
#include <array>
#include <algorithm>  // std::sort, std::clamp
#include <cmath>      // std::round
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>

namespace smpl {
namespace render {

// Render SMPL mesh with backface culling + painter's sort (flat shading).
// 'cloud' must be in camera coordinates (apply your root/Sim3 before calling).
inline void renderSMPLMesh(
    const Eigen::Matrix3Xd& cloud,
    const std::vector<std::array<int,3>>& faces,
    cv::Mat& img,
    double fx, double fy, double cx, double cy,
    bool fill = true,
    bool backface_cull = true,
    bool wireframe = false)
{
    struct FaceItem {
        int i0, i1, i2;
        double depth;   // painter's sort (larger = further)
        cv::Point pts[3];
        double shade;   // 0..1 (flat-shaded intensity)
    };

    std::vector<FaceItem> drawlist;
    drawlist.reserve(faces.size());

    // Project all points once
    const int N = static_cast<int>(cloud.cols());
    std::vector<cv::Point2f> proj(N, {-9999.f, -9999.f});
    std::vector<bool> valid(N, false);

    for (int i = 0; i < N; ++i) {
        const double X = cloud(0, i), Y = cloud(1, i), Z = cloud(2, i);
        if (Z <= 1e-6) continue; // behind/at camera
        float u = static_cast<float>(fx * X / Z + cx);
        float v = static_cast<float>(fy * Y / Z + cy);
        proj[i] = {u, v};
        valid[i] = true;
    }

    // Build face list with backface culling + flat shading
    for (const auto& f : faces) {
        int i0 = f[0], i1 = f[1], i2 = f[2];
        if (!valid[i0] || !valid[i1] || !valid[i2]) continue;

        // Camera-space triangle vertices
        const Eigen::Vector3d v0 = cloud.col(i0);
        const Eigen::Vector3d v1 = cloud.col(i1);
        const Eigen::Vector3d v2 = cloud.col(i2);

        // Normal in camera space (camera looks +Z)
        const Eigen::Vector3d e1 = v1 - v0;
        const Eigen::Vector3d e2 = v2 - v0;
        const Eigen::Vector3d n  = e1.cross(e2); // not normalized

        // Backface cull: face camera if n.z < 0
        if (backface_cull && n.z() >= 0.0) continue;

        // Flat shading: dot(view_dir, normal)
        const Eigen::Vector3d c = (v0 + v1 + v2) / 3.0;
        const Eigen::Vector3d view = (-c).normalized();
        double shade = n.normalized().dot(view);
        shade = std::clamp(shade, 0.0, 1.0);

        // Painter depth = average Z (bigger Z is further)
        const double depth = (v0.z() + v1.z() + v2.z()) / 3.0;

        FaceItem item;
        item.i0 = i0; item.i1 = i1; item.i2 = i2;
        item.depth = depth;
        item.shade = shade;
        item.pts[0] = cv::Point(static_cast<int>(std::round(proj[i0].x)),
                                static_cast<int>(std::round(proj[i0].y)));
        item.pts[1] = cv::Point(static_cast<int>(std::round(proj[i1].x)),
                                static_cast<int>(std::round(proj[i1].y)));
        item.pts[2] = cv::Point(static_cast<int>(std::round(proj[i2].x)),
                                static_cast<int>(std::round(proj[i2].y)));
        drawlist.push_back(item);
    }

    // Sort far-to-near (painter's)
    std::sort(drawlist.begin(), drawlist.end(),
              [](const FaceItem& a, const FaceItem& b){ return a.depth > b.depth; });

    // Draw
    for (const auto& it : drawlist) {
        const cv::Point* pts = it.pts;
        const int npts = 3;

        if (fill) {
            // Neutral gray scaled by shade
            const int base = 220;
            const int c = static_cast<int>(std::round(base * it.shade));
            cv::fillConvexPoly(img, pts, npts, cv::Scalar(c, c, c), cv::LINE_AA);
        }

        if (wireframe) {
            std::vector<cv::Point> poly = { pts[0], pts[1], pts[2], pts[0] };
            cv::polylines(img, poly, false, cv::Scalar(40, 40, 40), 1, cv::LINE_AA);
        }
    }
}

} // namespace render
} // namespace smpl
