#pragma once
#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOFF.h>
#include <igl/writeOBJ.h>
#include <iostream>
#include <igl/adjacency_list.h>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <igl/arap.h>
#include <chrono>

#include "omp.h"

class ARAPHB
{
public:
	void init(int use_cotangent_weights, Eigen::MatrixXd& V_out, Eigen::MatrixXi& F_out);

	void prep(Eigen::MatrixXd& V_out, Eigen::VectorXi& selected_vertices);

	Eigen::MatrixXd solve(int use_nonlinear_solver, int apply_initial_guess, int display_intermedium, Eigen::MatrixXd& V_change_out);

	double computeChamferDistance(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);

	void get_variation(Eigen::MatrixXd& V_change_out);

	void apply_initial_guess();

	Eigen::MatrixXd one_iter_linear_solver();

	void initialize_non_linear_solver(float momentum, float learning_rate);

	Eigen::MatrixXd one_iter_non_linear_solver();

	double get_arap_energy();


private:
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	Eigen::MatrixXd V_;
	//Eigen::MatrixXi F_;
	std::vector<std::vector<int>> adjacency_list;
	std::vector<int> opt_list;
	std::vector<int> selected_vertices_list;
	Eigen::MatrixXd W;
	Eigen::SparseMatrix<double> L;
	std::vector<Eigen::MatrixXd> R_s;
	Eigen::MatrixXd G_s;
	Eigen::MatrixXd b_g;
	double Pi_half = 1.57079632675;
	Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
	Eigen::MatrixXd velocity;
	float momentum;
	float learning_rate;

	void get_rotation();

	void get_gradients();

	void gradient_descend_step(Eigen::MatrixXd&, float, float, Eigen::MatrixXd&);

	

	void get_weight(int use_cotangent_weights, Eigen::MatrixXd& V, Eigen::MatrixXi& F);

	void get_Linear_right_side(Eigen::MatrixXd& b, std::vector<int>& selected_vertices_list);

	//void get_laplacian_sparse_matrix(Eigen::SparseMatrix<double>&, std::vector<int>&);
	void get_laplacian_sparse_matrix(std::vector<int>&);

	Eigen::MatrixXd linear_solver(int apply_initial_guess, int step_count);

	Eigen::MatrixXd linear_solver_intermedium(int apply_initial_guess, int step_count);

	Eigen::MatrixXd non_linear_solver(float, float, int);


};