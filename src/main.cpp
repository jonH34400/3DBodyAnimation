// minimal_neutral.cpp  – neutral-pose mesh with SMPLpp on CPU only
#include <iostream>
#include <fstream>

#include <torch/torch.h>
#include <smpl/SMPL.h>


// Auxiliar function for loading joints regressor from preprocessed model in JSON
torch::Tensor load_joint_regressor(const std::string& json_path) {
    std::ifstream in(json_path);
    nlohmann::json j;
    in >> j;

    auto jr_data = j["joint_regressor"];
    std::vector<float> flat_data;
    for (const auto& row : jr_data)
        for (const auto& val : row)
            flat_data.push_back(val);

    return torch::from_blob(flat_data.data(), {24, 6890}, torch::kFloat32).clone();
}

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

        /*Meaningful pose for testing ------------------------------------*/
        /******************************************************************/
        theta[0][18][2] = -1.57f; // left arm bent down by elbow
        theta[0][19][2] = -1.57f; // right arm bent upwards by elbow
        theta[0][1][0] = -2.0f;   // Left hip
        theta[0][2][0] = -2.0f;   // Right hip
        theta[0][4][0] = 2.5f;  // Left knee
        theta[0][5][0] = 2.5f;  // Right knee
        /********************************************************************/
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
            obj << "f " << f[i][0] + 1 << ' '      // +1 : OBJ is 1-based
                << f[i][1] + 1 << ' '
                << f[i][2] + 1 << '\n';

        std::cout << "neutral_mesh.obj written (" << V.size(0) << " verts)\n";

        /* 6. get 3D joints and generate 2D projection -------------------------------------------------- */ 
        // Load joints regressor from JSON
        auto JR = load_joint_regressor(model);

        // Get 3D joints positions of deformed pose
        auto joints = torch::matmul(JR, V);

        // Camera parameters (default values)
        const float fx = 800.0f, fy = 800.0f;  // focal lengths
        const float cx = 960.0f, cy = 540.0f;    // principal point 
        const float camera_z = 5.0f;          // camera distance

        std::vector<std::pair<float, float>> joints_2d;
        auto joints_accessor = joints.accessor<float, 2>();

        for (int i = 0; i < joints.size(0); ++i) {
            // Get 3D coordinates
            float x = joints_accessor[i][0];
            float y = -joints_accessor[i][1];
            float z = -joints_accessor[i][2] + camera_z;
            
            if (z < 0.1f) z = 0.1f; // If close to 0 value

            // Project to 2D (perspective projection)
            float u = fx * (x / z) + cx;
            float v = fy * (y / z) + cy;

            joints_2d.emplace_back(u, v);
        }

        // Save results
        std::ofstream out_file("joints_projection.json");
        out_file << "{\n";
        out_file << "  \"camera\": {\n";
        out_file << "    \"fx\": " << fx << ",\n";
        out_file << "    \"fy\": " << fy << ",\n";
        out_file << "    \"cx\": " << cx << ",\n";
        out_file << "    \"cy\": " << cy << ",\n";
        out_file << "    \"z_offset\": " << camera_z << "\n";
        out_file << "  },\n";
        out_file << "  \"joints\": [\n";
        
        for (size_t i = 0; i < joints_2d.size(); ++i) {
            out_file << "    {\"id\": " << i << ", \"x\": " << joints_2d[i].first 
                     << ", \"y\": " << joints_2d[i].second << "}";
            if (i < joints_2d.size() - 1) out_file << ",";
            out_file << "\n";
        }
        
        out_file << "  ]\n";
        out_file << "}\n";
        out_file.close();

        std::cout << "Successfully projected " << joints_2d.size() 
                  << " joints to 2D coordinates.\n";
        std::cout << "Results saved to joints_projection.json\n";


    }
    catch (const std::exception& e) {
        std::cerr << "SMPL error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
