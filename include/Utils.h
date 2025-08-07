#pragma once
#include <filesystem>        // fs::path, directory_iterator
#include <algorithm>         // std::transform, std::min, std::sort
#include <fstream>           // std::ifstream
#include <iostream>          // std::cerr
#include <string>            // std::string
#include <vector>            // std::vector
#include <initializer_list>  // std::initializer_list

#include <nlohmann/json.hpp> // JSON
using json = nlohmann::json;

#include "Sim3BA.h"          

namespace fs = std::filesystem;

/* ---------- MediaPipe → SMPL joint map ---------- */
static const int MP_MAP[24] = {
    -1, 23, 24, -1, 25, 26, -1, 27, 28, -1,
     31, 32, -1, -1, -1, 0, 11, 12, 13, 14,
     15, 16, -1, -1 };
static const std::array<int,17> USE_SMPL = {
    1, 2, 4, 5, 7, 8, 10, 11, 15, 16, 17, 18, 19, 20, 21 };

static bool has_ext(const fs::path& p, std::initializer_list<std::string> exts)
{
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    for (auto& x : exts) if (e == x) return true;
    return false;
}

static std::vector<fs::path> list_sorted(const fs::path& d,
                                         std::initializer_list<std::string> exts)
{
    std::vector<fs::path> v;
    for (auto& p : fs::directory_iterator(d))
        if (p.is_regular_file() && has_ext(p.path(), exts)) v.push_back(p.path());
    std::sort(v.begin(), v.end());
    return v;
}

/* ---------- json helpers ---------- */
template<typename T>
static T safe_number(const json& j, const char* key, T def = T(0))
{
    return (j.contains(key) && j[key].is_number()) ? j[key].template get<T>()
                                                   : def;
}

static bool safe_coord(const json& j, const char* key, double& out)
{
    if (j.contains(key) && j[key].is_number()) {
        out = j[key].get<double>();
        return true;
    }
    return false;
}

/* ---------- MediaPipe loader ---------- */
static std::vector<PixelKP> load_mp_json(const std::string& path, int W, int H)
{
    std::ifstream f(path);
    if (!f) { std::cerr << "cannot open " << path << '\n'; return {}; }
    json j; f >> j;

    auto mid = [&](int a,int b,double& x,double& y,double& vis)->bool
    {
        double xa,ya,xb,yb;
        if (!safe_coord(j[a],"x",xa) || !safe_coord(j[a],"y",ya) ||
            !safe_coord(j[b],"x",xb) || !safe_coord(j[b],"y",yb))  return false;
        x   = 0.5*(xa+xb);
        y   = 0.5*(ya+yb);
        vis = std::min( safe_number<double>(j[a],"visibility",1.0),
                        safe_number<double>(j[b],"visibility",1.0) );
        return true;
    };

    double pelX,pelY,pelVis, chX,chY,chVis;
    bool havePel = mid(23,24,pelX,pelY,pelVis);
    bool haveCh  = mid(11,12,chX ,chY ,chVis );

    std::vector<PixelKP> out; out.reserve(USE_SMPL.size());
    for (int sid : USE_SMPL) {
        double x=0,y=0,vis=0; bool ok=true;
        switch (sid) {
        case 0: ok = havePel; x=pelX; y=pelY; vis=pelVis; break;
        case 6: ok = haveCh ; x=chX ; y=chY ; vis=chVis ; break;
        default:{
            int mp = MP_MAP[sid];
            if (mp < 0) { ok=false; break; }
            if (!safe_coord(j[mp],"x",x) || !safe_coord(j[mp],"y",y)) ok=false;
            vis = safe_number<double>(j[mp],"visibility",1.0);
        }}
        if (!ok || vis < 0.5) continue;
        out.push_back({ sid, x*W, y*H });
    }
    return out;
}


inline double mean_pixel_error(const std::vector<PixelKP>& kps,
                               const ark::Avatar& avatar,
                               double fx,double fy,double cx,double cy)
{
    if (kps.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& kp : kps) {
        const auto& J = avatar.jointPos.col(kp.jid);
        double u = fx * J.x() / J.z() + cx;
        double v = fy * J.y() / J.z() + cy;
        sum += std::hypot(u - kp.u, v - kp.v);
    }
    return sum / kps.size();
}

