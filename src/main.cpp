#include <iostream>
#include <fstream>

#include <torch/torch.h>
#include <smpl/SMPL.h>        // SMPLpp public header

int main(int argc, char** argv)
{
    /* ---------- 1. choose model & device ---------- */
    std::string model = "../assets/models/smpl_female.json";   // or .npz
    if (argc > 1) model = argv[1];

    torch::Device dev(torch::kCPU);            // use GPU if you have one

    try
    {
        /* ---------- 2. build & load model ---------- */
        smpl::SMPL smpl;
        smpl.setDevice(dev);
        smpl.setModelPath(model);
        smpl.init();                           // heavy I/O happens here

        /* ---------- 3. neutral parameters ---------- */
        torch::Tensor betas = torch::zeros({1, 10},  torch::kFloat32).to(dev);
        torch::Tensor theta = torch::zeros({1, 24, 3}, torch::kFloat32).to(dev);

        smpl.launch(betas, theta);             // forward pass

        /* ---------- 4. fetch tensors ---------- */
        torch::Tensor V = smpl.getVertex().squeeze(0).cpu();      // (6890,3)
        torch::Tensor F = smpl.getFaceIndex().cpu();              // (13776,3)

        /* ---------- 5. write OBJ ---------- */
        std::ofstream obj("neutral_mesh.obj");
        auto v = V.accessor<float,2>();
        for (int64_t i = 0; i < V.size(0); ++i)
            obj << "v " << v[i][0] << ' ' << v[i][1] << ' ' << v[i][2] << '\n';

        auto f = F.accessor<int64_t,2>();
        for (int64_t i = 0; i < F.size(0); ++i)
            obj << "f " << f[i][0] + 1 << ' '   // +1 for 1-based OBJ indexing
                << f[i][1] + 1 << ' '
                << f[i][2] + 1 << '\n';

        std::cout << "✅  neutral_mesh.obj written (" << V.size(0) << " verts)\n";
    }
    catch (const std::exception& e) {
        std::cerr << "SMPL error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
