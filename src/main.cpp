// minimal_neutral.cpp  – neutral-pose mesh with SMPLpp on CPU only
#include <iostream>
#include <fstream>

#include <torch/torch.h>
#include <smpl/SMPL.h>

int main(int argc, char** argv)
{
    /* 1. model path ---------------------------------------------------- */
    std::string model = "../assets/models/SMPL_FEMALE.json";
    if (argc > 1) model = argv[1];

    try {
        /* 2. build & load model ---------------------------------------- */
        smpl::SMPL smpl;               // default device is CPU
        smpl.setModelPath(model);
        smpl.init();                   // heavy I/O

        /* 3. neutral parameters ---------------------------------------- */
        torch::Tensor betas = torch::zeros({1, 10},      torch::kFloat32);
        torch::Tensor theta = torch::zeros({1, 24, 3},   torch::kFloat32);

        smpl.launch(betas, theta);     // forward pass (CPU)

        /* 4. fetch tensors --------------------------------------------- */
        torch::Tensor V = smpl.getVertex().squeeze(0).cpu();   // (6890, 3)
        torch::Tensor F = smpl.getFaceIndex().cpu();           // (13776, 3)

        /* 5. write OBJ -------------------------------------------------- */
        std::ofstream obj("neutral_mesh.obj");
        auto v = V.accessor<float,2>();
        for (int64_t i = 0; i < V.size(0); ++i)
            obj << "v " << v[i][0] << ' ' << v[i][1] << ' ' << v[i][2] << '\n';

        auto f = F.accessor<int32_t,2>();
        for (int64_t i = 0; i < F.size(0); ++i)
            obj << "f " << f[i][0] + 1 << ' '      
                << f[i][1] + 1 << ' '
                << f[i][2] + 1 << '\n';

        std::cout << "neutral_mesh.obj written (" << V.size(0) << " verts)\n";

        torch::Tensor J = smpl.getRestJoint().squeeze(0).cpu();
        auto j = J.accessor<float,2>();

        const int W = 1280, H = 720;
        const double fx = 1000, fy = 1000, cx = W/2, cy = H/2;

        for (int i = 0; i < 24; ++i) {
            double X = j[i][0], Y = j[i][1], Z = j[i][2];
            double u = fx*X/Z + cx;
            double v = fy*Y/Z + cy;
            std::cout << "joint " << i << ": (" << u << ", " << v << ")\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "SMPL error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
