// main.cpp  ─ minimal neutral-pose demo for SMPLpp
// g++ -std=c++17 main.cpp -o neutral_demo  $(pkg-config --cflags --libs xtensor) \
//     -I/path/to/SMPLpp/include  -L/path/to/SMPLpp/lib -lsmplpp -ltorch

#include <iostream>
#include <fstream>
#include <memory>

#include <Eigen/Dense>          // SMPLpp exposes Eigen-compatible getters
#include <smpl/SMPL.h>        // adjust include path if your project layout differs

int main(int argc, char** argv)
{
    // 1. Resolve model file (allow override from CLI)
    std::string model_path = "../assets/models/smpl_female.npz";      // default
    if (argc > 1) model_path = argv[1];

    try
    {
        // 2. Instantiate model
        auto smpl = std::make_shared<smpl::SMPL>(model_path);

        // 3. Neutral parameters
        Eigen::VectorXf pose   = Eigen::VectorXf::Zero(72);   // axis-angle for 24 joints
        Eigen::VectorXf betas  = Eigen::VectorXf::Zero(10);   // shape coefficients
        Eigen::Vector3f trans  = Eigen::Vector3f::Zero();     // global translation

        smpl->set_params(pose, betas, trans);                 // update internal buffers
        smpl->forward();                                      // generate mesh

        // 4. Fetch results
        const auto& V = smpl->get_vertices();   // std::vector<Eigen::Vector3f>
        const auto& F = smpl->get_faces();      // std::vector<Eigen::Vector3i>

        // 5. Write OBJ (1-indexed faces)
        std::ofstream obj("neutral_mesh.obj");
        for (const auto& v : V) obj << "v " << v.x() << ' ' << v.y() << ' ' << v.z() << '\n';
        for (const auto& f : F) obj << "f " << f.x() + 1 << ' ' << f.y() + 1 << ' ' << f.z() + 1 << '\n';
        obj.close();

        std::cout << "✅  Neutral mesh written to neutral_mesh.obj ("
                  << V.size() << " verts).\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "SMPL error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
