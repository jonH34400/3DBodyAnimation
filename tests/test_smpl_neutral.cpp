TEST(SMPL, NeutralPoseIsT) {
    SMPLEngine<double> smpl("model_m.npz");
    double theta[72] = {0};  double beta[10] = {0};  double T[3] = {0};
    smpl.setPose(theta);  smpl.setShape(beta);  smpl.setTrans(T);

    Eigen::Matrix<double,24,3> J;  smpl.getJoints(J);

    // pelvis at origin, two hip joints roughly ±0.09 m X, –0.02 m Y, –0.88 m Z
    EXPECT_NEAR(J(0,0), 0.0, 1e-5);                 // pelvis x
    EXPECT_NEAR(J(0,2), 0.0, 1e-5);                 // pelvis z
    EXPECT_NEAR(J.row(2).norm(), 0.88, 0.02);       // left hip radius ≈ leg length
}
