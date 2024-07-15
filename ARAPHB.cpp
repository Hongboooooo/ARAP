#include "ARAPHB.h"



void ARAPHB::get_rotation()
{    
    for (int i = 0; i < V.rows(); i++)
    {
        Eigen::MatrixXd P_i(adjacency_list[i].size(), 3);
        Eigen::MatrixXd P_i_(adjacency_list[i].size(), 3);
        Eigen::MatrixXd D = Eigen::MatrixXd::Zero(adjacency_list[i].size(), adjacency_list[i].size());
        for (int j = 0; j < adjacency_list[i].size(); j++)
        {
            P_i.row(j) = V.row(i) - V.row(adjacency_list[i][j]);
            P_i_.row(j) = V_.row(i) - V_.row(adjacency_list[i][j]);
            D(j, j) = W(i, adjacency_list[i][j]);
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(P_i.transpose() * D * P_i_, Eigen::ComputeThinU | Eigen::ComputeThinV);
        R_s[i] = svd.matrixV() * svd.matrixU().transpose();
    }
}

void ARAPHB::get_gradients()
{
    for (int i = 0; i < opt_list.size(); i++)
    {
        Eigen::Vector3d e_i = Eigen::Vector3d(0, 0, 0);
        Eigen::Vector3d e_i_ = Eigen::Vector3d(0, 0, 0);
        for (int j = 0; j < adjacency_list[opt_list[i]].size(); j++)
        {

            e_i += (R_s[opt_list[i]] + R_s[adjacency_list[opt_list[i]][j]]) * (V.row(opt_list[i]) - V.row(adjacency_list[opt_list[i]][j])).transpose();
            e_i_ += V_.row(opt_list[i]) - V_.row(adjacency_list[opt_list[i]][j]);
        }

        G_s.row(i) = (4 * e_i_ - 2 * e_i);

    }
}

void ARAPHB::gradient_descend_step(Eigen::MatrixXd& velocity, float momentum, float learning_rate, Eigen::MatrixXd& G_s)
{
    velocity = momentum * velocity - learning_rate * G_s;
    for (int i = 0; i < opt_list.size(); i++)
    {
        V_.row(opt_list[i]) = V_.row(opt_list[i]) + velocity.row(i);
    }
}

double ARAPHB::get_arap_energy()
{
    double arap_energy = 0.0;

    for (int i = 0; i < V.rows(); i++)
    {
        double rigidity_energy_per_cell = 0.0;
        for (int j : adjacency_list[i]) 
        { 
            Eigen::Vector3d e_ = V_.row(i) - V_.row(j);
            Eigen::Vector3d e = V.row(i) - V.row(j);

            rigidity_energy_per_cell += W(i, j) * (e_ - R_s[i] * e).squaredNorm();
        }
        arap_energy += W(i, i) * rigidity_energy_per_cell; // w_i
    }

    return arap_energy;
}

double ARAPHB::computeChamferDistance(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    double totalDistAtoB = 0.0;
    double totalDistBtoA = 0.0;

    // For each point in A, find the nearest neighbor in B and accumulate the distances
    for (int i = 0; i < A.rows(); ++i) {
        double minDist = std::numeric_limits<double>::max();
        for (int j = 0; j < B.rows(); ++j) {
            double dist = (A.row(i) - B.row(j)).squaredNorm();;
            if (dist < minDist) {
                minDist = dist;
            }
        }
        //std::cout << i << ": " << minDist << std::endl;
        totalDistAtoB += minDist;
    }

    // For each point in B, find the nearest neighbor in A and accumulate the distances
    for (int i = 0; i < B.rows(); ++i) {
        double minDist = std::numeric_limits<double>::max();
        for (int j = 0; j < A.rows(); ++j) {
            double dist = (A.row(i) - B.row(j)).squaredNorm();;
            if (dist < minDist) {
                minDist = dist;
            }
        }
        //std::cout << i << ": " << minDist << std::endl;
        totalDistBtoA += minDist;
    }

    // Average the total distances
    double chamferDistance = (totalDistAtoB / A.rows()) + (totalDistBtoA / B.rows());
    return chamferDistance;
}

void ARAPHB::get_weight(int use_cotangent_weights, Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
    if (use_cotangent_weights <= 0)
    {
        W = Eigen::MatrixXd::Ones(V.rows(), V.rows());
    }
    else
    {
        W = Eigen::MatrixXd::Zero(V.rows(), V.rows());
        std::vector<std::vector<Eigen::Vector2i>> vertexEdges(V.rows());

        for (int i = 0; i < V.rows(); i++)  // Found surround edges of each vertex
        {
            std::vector<Eigen::Vector2i> edges;

            for (int j = 0; j < F.rows(); j++) 
            { 
                Eigen::Vector3i face = F.row(j);

                for (int k = 0; k < 3; k++) 
                { 
                    if (face[k] == i) 
                    {
                        //std::cout << face[(k + 1) % 3]<< ","<< face[(k + 2) % 3] << "  ";
                        edges.push_back(Eigen::Vector2i(face[(k + 1) % 3], face[(k + 2) % 3]));
                    }
                    
                }
            }
            //std::cout << std::endl;
            //std::cout << "-----------" << std::endl;

            vertexEdges[i] = edges; //vertex i's surround edges of each vertex
            
        }



        for (int i = 0; i < V.rows(); i++) 
        {
            for (int neighbor : adjacency_list[i]) // i's neighbors
            { 
                double totalAngle = 0.0;

                for (const Eigen::Vector2i& edge : vertexEdges[i]) // Iterate over the edges
                { 
                    double norm_bc = (V.row(edge[0]) - V.row(edge[1])).norm(); // length between B and C
                    double norm_ab = (V.row(i) - V.row(edge[0])).norm(); // length between A and B
                    double norm_ac = (V.row(i) - V.row(edge[1])).norm(); // length between A and C
                    
                    if (edge[0] == neighbor)
                    {
                        double alpha = acos(((norm_ac * norm_ac) + (norm_bc * norm_bc) - (norm_ab * norm_ab)) / (2 * norm_ac * norm_bc));
                        totalAngle += abs(1 / tan(alpha));
                        //std::cout << "T";
                    }
                    if (edge[1] == neighbor)
                    {
                        double beta = acos(((norm_ab * norm_ab) + (norm_bc * norm_bc) - (norm_ac * norm_ac)) / (2 * norm_ab * norm_bc));
                        totalAngle += abs(1 / tan(beta));
                        //std::cout << "T";
                    }

                }
                //std::cout << "-----------" << std::endl;
                W(i, neighbor) = totalAngle / 2;
            }

            W(i, i) = 1.0; // Override the diagonal entry
        }
        //std::cout << W << std::endl;
    }
}

void ARAPHB::get_Linear_right_side(Eigen::MatrixXd& b, std::vector<int>& selected_vertices_list)
{
    for (int i : opt_list)
    {
        //std::cout << i << ", ";
        Eigen::Vector3d e_i = Eigen::Vector3d(0, 0, 0);
        for (int j : adjacency_list[i])
        {
            e_i += 0.5 * W(i, j) * (V.row(i) - V.row(j)) * (R_s[i].transpose() + R_s[j].transpose());
        }
        b.row(i) = e_i;
    }
    for (int i : selected_vertices_list)
    {
        //std::cout << i << ", ";
        b.row(i) = V_.row(i);
    }
}

//void ARAPHB::get_laplacian_sparse_matrix(Eigen::SparseMatrix<double>& L, std::vector<int>& selected_vertices_list)
void ARAPHB::get_laplacian_sparse_matrix(std::vector<int>& selected_vertices_list)
{
    L.resize(V.rows(), V.rows());
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i : opt_list)
    {
        //std::cout << i << ", ";
        double accum_diagonal = 0;
        for (int j : adjacency_list[i])
        {
            accum_diagonal += W(i, j);
            triplets.push_back(Eigen::Triplet<double>(i, j, -1 * W(i, j)));
        }
        triplets.push_back(Eigen::Triplet<double>(i, i, accum_diagonal));
    }
    //std::cout << std::endl;
    for (int i : selected_vertices_list)
    {
        //std::cout << i << ", ";
        triplets.push_back(Eigen::Triplet<double>(i, i, 1));
    }
    //std::cout << std::endl;
    L.setFromTriplets(triplets.begin(), triplets.end());
}

void ARAPHB::init(int use_cotangent_weights, Eigen::MatrixXd& V_out, Eigen::MatrixXi& F_out)
{
    V = V_out;
    F = F_out;
    igl::adjacency_list(F, adjacency_list);
    get_weight(use_cotangent_weights, V, F);
    R_s.resize(V.rows());
}

void ARAPHB::prep(Eigen::MatrixXd& V_out, Eigen::VectorXi& selected_vertices)
{
    V_ = V_out;
    opt_list.clear();
    selected_vertices_list.clear();
    //std::cout << "V_:" << std::endl;
    for (int i = 0; i < V_.rows(); i++)
    {
        int count = 0;
        for (int j = 0; j < selected_vertices.size(); j++)
        {
            if (i == selected_vertices[j])
            {
                selected_vertices_list.push_back(i);
                break;
                //select_map.push_back(1);
            }
            count++;
        }
        if (count == selected_vertices.size())
        {
            opt_list.push_back(i);
            //select_map.push_back(0);
        }
        //std::cout << V_.row(i) << std::endl;
    }
    //std::cout << std::endl;
    get_laplacian_sparse_matrix(selected_vertices_list);
}

Eigen::MatrixXd ARAPHB::linear_solver(int apply_initial_guess, int step_count)
{
    Eigen::MatrixXd b(V_.rows(), 3);

    if (apply_initial_guess >= 1)
    {
        solver.compute(L);
        V_ = solver.solve(L * V_);
    }

    for (int i = 0; i < step_count; i++)
    {
        // get rotation
        get_rotation();

        // Linear right side
        get_Linear_right_side(b, selected_vertices_list);

        solver.compute(L);
        V_ = solver.solve(b);

    }

    return V_;
}

Eigen::MatrixXd ARAPHB::linear_solver_intermedium(int apply_initial_guess, int step_count)
{
    Eigen::MatrixXd b(V_.rows(), 3);

    if (apply_initial_guess >= 1)
    {
        solver.compute(L);
        V_ = solver.solve(L * V_);
    }


    for (int i = 0; i < step_count; i++)
    {
        get_rotation();

        get_Linear_right_side(b, selected_vertices_list);

        solver.compute(L);
        V_ = solver.solve(b);
        
    }

    return V_;
}

Eigen::MatrixXd ARAPHB::non_linear_solver(float momentum, float learning_rate, int step_count)
{
    Eigen::MatrixXd velocity = Eigen::MatrixXd::Zero(opt_list.size(), 3);
    for (int i = 0; i < step_count; i++)
    {
        // get rotation
        get_rotation();
        // get gradients
        get_gradients();
        // gradient descend
        gradient_descend_step(velocity, momentum, learning_rate, G_s);
    }
    return V_;
}

void ARAPHB::initialize_non_linear_solver(float momentum_out, float learning_rate_out)
{
    velocity = Eigen::MatrixXd::Zero(opt_list.size(), 3);
    G_s.resize(opt_list.size(), 3);
    momentum = momentum_out;
    learning_rate = learning_rate_out;
}

Eigen::MatrixXd ARAPHB::one_iter_non_linear_solver()
{
    get_rotation();
    // get gradients
    get_gradients();
    // gradient descend
    gradient_descend_step(velocity, momentum, learning_rate, G_s);

    return V_;
}

Eigen::MatrixXd ARAPHB::solve(int use_nonlinear_solver, int apply_initial_guess, int display_intermedium, Eigen::MatrixXd& V_change_out)
{
    for (int i = 0; i < V_change_out.rows(); i++)
    {
        V_.row(selected_vertices_list[i]) = V_change_out.row(i);
    }

    if (use_nonlinear_solver > 0)
    {
        return non_linear_solver(0.95, 0.001, 100);
    }
    else
    {
        if (display_intermedium > 0)
        {
            return linear_solver_intermedium(apply_initial_guess, 10);
        }
        else
        {
            return linear_solver(apply_initial_guess, 100);
        }
        
    }
}

void ARAPHB::get_variation(Eigen::MatrixXd& V_change_out)
{
    b_g.resize(V_.rows(), 3);
    for (int i = 0; i < V_change_out.rows(); i++)
    {
        V_.row(selected_vertices_list[i]) = V_change_out.row(i);
    }
}

void ARAPHB::apply_initial_guess()
{
    solver.compute(L);
    V_ = solver.solve(L * V_);
}

Eigen::MatrixXd ARAPHB::one_iter_linear_solver()
{
    get_rotation();

    get_Linear_right_side(b_g, selected_vertices_list);

    solver.compute(L);
    V_ = solver.solve(b_g);

    return V_;
}
