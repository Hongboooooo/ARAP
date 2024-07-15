#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/unproject_on_plane.h>
#include <igl/project.h>
#include <igl/readOFF.h>
#include <igl/arap.h>
#include <iostream>
#include "config.h"
#include "ARAPHB.h"
#include <GLFW/glfw3.h>

//V: vertices, U: also vertices, can be used to store the previous pose
//C: colors of the faces
Eigen::MatrixXd V, C;
Eigen::MatrixXi F;
//Store multiple poses for comparison
int modelCount = 3;
int currentModel = 0;
std::vector<Eigen::MatrixXd> Vx;

//Stores the selected constraints in the form of a vector of size = size of V/F
//Every entry contains an int, if 0 then that vertex is not a constraint
//if > 0, then it is a constraint. This way if multiple neighboring faces of a vertex
//are selected we can keep track of that
Eigen::VectorX<int> vertexSelected, faceSelected;
Eigen::VectorX<int> vertexHandles, faceHandles;
//Contains the indices of the constrained vertices
//Needs to be rebuilt when selected faces change
Eigen::VectorXi b, h;
igl::ARAPData arap_data;
int clickMode;
//Index of the current handle vertex
int currentHandle = 0;
enum Mode { Face_Select = 0, Handle_Select, ARAP_Libigl, ARAP_HB, ARAP_HB_Iter, ARAP_Benchmark };
static Mode current_mode = Face_Select;
//How many constrained vertices we have
int vertexSelectedCount = 0;
int vertexHandlesCount = 0;
//Used to signal if b needs to be rebuilt
bool constraintsChanged;
//Signal when the iter mode has started
bool iterativeHB, benchmark = false;
//Count the current iteration
int currentIteration = 0;
std::chrono::duration<double> libiglDur, HBDur;
double chamferDist = 0;

ARAPHB araphb;
char previous_key = '0';

void prepare_araphb(int modelID)
{
	b = Eigen::VectorXi(vertexSelectedCount);
	h = Eigen::VectorXi(vertexHandlesCount);
	int count = 0;
	int countH = 0;
	for (int i = 0; i < vertexSelected.size(); i++) {
		if (vertexSelected[i] > 0) {
			b[count] = i;
			count++;
		}
		if (vertexHandles[i] > 0) {
			h[countH] = i;
			countH++;
		}
	}

	araphb.prep(Vx[modelID], b);
}

//Event for single key down input, just add a new case for new functionality
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier) {
	std::cout << "Key: " << key << " " << (unsigned int)key << std::endl;
	int max = 0;
	switch (key) {
		//For testing
	case 'Q':
		for (int i = 0; i < vertexSelected.size(); i++) {
			if (vertexSelected[i] > max) {
				max = vertexSelected[i];
			}
		}
		std::cout << "Max vertex selected: " << max << std::endl;
		std::cout << "Currently selected vertices: " << vertexSelectedCount << std::endl;
		previous_key = 'Q';
		break;
	case '1':
		current_mode = Face_Select;
		previous_key = '1';
		break;
	case '2':
		current_mode = Handle_Select;
		previous_key = '2';
		break;
	case '3':
		current_mode = ARAP_Libigl;
		previous_key = '3';
		break;
	case '4':
		current_mode = ARAP_HB;
		if (previous_key != '4')
		{
			prepare_araphb(1);
		}
		previous_key = '4';
		break;
	case '5':
		current_mode = ARAP_HB_Iter;
		if (previous_key != '5')
		{
			prepare_araphb(1);
		}
		previous_key = '5';
		break;
	case '6':
		current_mode = ARAP_Benchmark;
		if (previous_key != '6')
		{
			prepare_araphb(1);
		}
		previous_key = '6';
		break;
		
	default:
		break;
	}
	return false;
}

void add_face_to_selection(int fid, bool isHandle) {
	C.row(fid) << 1, 0, 0;
	faceSelected[fid] = 1.0;
	//Add to the vertex selection count
	if (vertexSelected[F(fid, 0)] == 0) {
		vertexSelectedCount++;
	}
	vertexSelected[F(fid, 0)] += 1;
	if (vertexSelected[F(fid, 1)] == 0) {
		vertexSelectedCount++;
	}
	vertexSelected[F(fid, 1)] += 1;
	if (vertexSelected[F(fid, 2)] == 0) {
		vertexSelectedCount++;
	}
	vertexSelected[F(fid, 2)] += 1;

	if (isHandle) {
		C.row(fid) << 0, 1, 0;
		faceHandles[fid] = 1.0;

		//Add to the vertex selection count
		if (vertexHandles[F(fid, 0)] == 0) {
			vertexHandlesCount++;
		}
		vertexHandles[F(fid, 0)] += 1;
		if (vertexHandles[F(fid, 1)] == 0) {
			vertexHandlesCount++;
		}
		vertexHandles[F(fid, 1)] += 1;
		if (vertexHandles[F(fid, 2)] == 0) {
			vertexHandlesCount++;
		}
		vertexHandles[F(fid, 2)] += 1;
	}
}

void remove_face_from_selection(int fid, bool isHandle) {
	C.row(fid) << 1, 1, 1;
	faceSelected[fid] = 0.0;
	vertexSelected[F(fid, 0)] -= 1;
	if (vertexSelected[F(fid, 0)] == 0) {
		vertexSelectedCount--;
	}
	vertexSelected[F(fid, 1)] -= 1;
	if (vertexSelected[F(fid, 1)] == 0) {
		vertexSelectedCount--;
	}
	vertexSelected[F(fid, 2)] -= 1;
	if (vertexSelected[F(fid, 2)] == 0) {
		vertexSelectedCount--;
	}

	if (isHandle) {
		faceHandles[fid] = 0.0;
		vertexHandles[F(fid, 0)] -= 1;
		if (vertexHandles[F(fid, 0)] == 0) {
			vertexHandlesCount--;
		}
		vertexHandles[F(fid, 1)] -= 1;
		if (vertexHandles[F(fid, 1)] == 0) {
			vertexHandlesCount--;
		}
		vertexHandles[F(fid, 2)] -= 1;
		if (vertexHandles[F(fid, 2)] == 0) {
			vertexHandlesCount--;
		}
	}
}

//Select faces to be used for constraints
//Click to set or remove
bool face_selection(igl::opengl::glfw::Viewer& viewer) {
	int fid;
	Eigen::Vector3f bc;
	//Mouse in pixel coordinates
	double x = viewer.current_mouse_x;
	double y = viewer.core().viewport(3) - viewer.current_mouse_y;

	if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core().view, viewer.core().proj, viewer.core().viewport, 
		Vx[currentModel], F, fid, bc)) {
		//paint hit red, undo if already selected
		if (faceSelected[fid] < 1.0) {
			add_face_to_selection(fid, false);
		}
		else {
			remove_face_from_selection(fid, false);
		}

		viewer.data().set_colors(C);
		constraintsChanged = true;
		return true;
	}
	return false;
}

//Change the handle that is moved with mouse input
//Currently no visual confirmation
bool handle_selection(igl::opengl::glfw::Viewer& viewer) {
	int fid;
	Eigen::Vector3f bc;

	double x = viewer.current_mouse_x;
	double y = viewer.core().viewport(3) - viewer.current_mouse_y;

	if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core().view, viewer.core().proj, viewer.core().viewport,
		Vx[currentModel], F, fid, bc)) {
		int maxBC = 0;
		float currentBC = bc[0];
		if (bc[1] > currentBC) {
			currentBC = bc[1];
			maxBC = 1;
		}
		if (bc[2] > currentBC) {
			currentBC = bc[2];
			maxBC = 2;
		}
		vertexSelected[currentHandle]--;
		if (vertexSelected[currentHandle] == 0) {
			vertexSelectedCount--;
		}
		currentHandle = F(fid, maxBC);
		if (vertexSelected[currentHandle] == 0) {
			vertexSelectedCount++;
		}
		vertexSelected[currentHandle]++;
		constraintsChanged = true;
		return true;
	}
	return false;
}

//Select handles based on faces
bool handle_selection_face_based(igl::opengl::glfw::Viewer& viewer) {
	int fid;
	Eigen::Vector3f bc;
	//Mouse in pixel coordinates
	double x = viewer.current_mouse_x;
	double y = viewer.core().viewport(3) - viewer.current_mouse_y;

	if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core().view, viewer.core().proj, viewer.core().viewport,
		Vx[currentModel], F, fid, bc)) {
		if (faceHandles[fid] < 1.0) {
			add_face_to_selection(fid, true);
		}
		else {
			remove_face_from_selection(fid, true);
		}
		viewer.data().set_colors(C);
		constraintsChanged = true;
		return true;

	}
	return false;
}

//try out with V0, generalize
Eigen::MatrixXd move_handle_group(igl::opengl::glfw::Viewer& viewer) {
	Eigen::MatrixXd bc(vertexSelectedCount, V.cols());
	if (vertexHandlesCount <= 0) {
		return bc;
	}
	//Calculate center of mass
	Eigen::Vector3f v(0, 0, 0);
	for (int i = 0; i < vertexHandlesCount; i++) {
		v[0] += Vx[currentModel](h[i], 0);
		v[1] += Vx[currentModel](h[i], 1);
		v[2] += Vx[currentModel](h[i], 2);
	}
	v[0] /= (float)vertexHandlesCount;
	v[1] /= (float)vertexHandlesCount;
	v[2] /= (float)vertexHandlesCount;


	double x = viewer.current_mouse_x;
	double y = viewer.core().viewport(3) - viewer.current_mouse_y;
	//Project vertex handle to screen space, v is now in pixel coordinates and depth buffer
	Eigen::Vector3f vPixel = igl::project(v, viewer.core().view, viewer.core().proj, viewer.core().viewport);
	//Project the current mouse position into world space using v's depth value
	Eigen::Vector3f vNew = igl::unproject(Eigen::Vector3f(x, y, vPixel[2]),
		viewer.core().view, viewer.core().proj, viewer.core().viewport);

	//calculate offset
	Eigen::Vector3f vOffset = vNew - v;

	//Move all selected vertices by offset
	//build bc vector, based on the example for the libigl arap tutorial
	int count = 0;
	for (int i = 0; i < vertexSelectedCount; i++) {
		bc.row(i) = Vx[currentModel].row(b(i));
		//Move only the constraints
		if (count < vertexHandlesCount && b(i) == h(count)) {
			bc(i, 0) += vOffset[0];
			bc(i, 1) += vOffset[1];
			bc(i, 2) += vOffset[2];
			count++;
		}
	}
	


	return bc;
}


//Move the handle with mouse input (using libigl built in arap)
bool move_handle_libigl_arap(igl::opengl::glfw::Viewer& viewer) {
	//Determine b vector, Only do this step if handles have changed in between uses
	//b contains the indices of the constraints

		//Plus one for the handle
	b = Eigen::VectorXi(vertexSelectedCount);
	h = Eigen::VectorXi(vertexHandlesCount);
	int count = 0;
	int countH = 0;
	for (int i = 0; i < vertexSelected.size(); i++) {
		if (vertexSelected[i] > 0) {
			b[count] = i;
			count++;
		}
		if (vertexHandles[i] > 0) {
			h[countH] = i;
			countH++;
		}
	}
	Eigen::MatrixXd bc = move_handle_group(viewer);
	auto timer_start = std::chrono::high_resolution_clock::now();
	//I think this needs to be called every time the model/constraints change
	igl::arap_precomputation(Vx[currentModel], F, V.cols(), b, arap_data);

	//Eigen::MatrixXd bc = move_handle_group(viewer);
	
	
	igl::arap_solve(bc, arap_data, Vx[currentModel]);

	auto timer_end = std::chrono::high_resolution_clock::now();
	libiglDur = timer_end - timer_start;
	std::cout << "Time taken (Libigl): " << libiglDur.count() << "seconds" << std::endl;

	viewer.data().set_vertices(Vx[currentModel]);
	viewer.data().compute_normals();
	return true;
}

bool move_handle_araphb(igl::opengl::glfw::Viewer& viewer) {

	Eigen::MatrixXd bc = move_handle_group(viewer);

	auto timer_start = std::chrono::high_resolution_clock::now();

	araphb.get_variation(bc);
	araphb.apply_initial_guess();

	for (int i = 0; i < 300; i++)
	{
		Vx[currentModel] = araphb.one_iter_linear_solver();
	}

	auto timer_end = std::chrono::high_resolution_clock::now();
	HBDur = timer_end - timer_start;
	std::cout << "Time taken (Custom): " << HBDur.count() << "seconds" << std::endl;

	viewer.data().set_vertices(Vx[currentModel]);
	viewer.data().compute_normals();
	return true;
}

//Apply the first step
bool move_handle_arap_first_step(igl::opengl::glfw::Viewer& viewer) {
	Eigen::MatrixXd bc = move_handle_group(viewer);

	auto timer_start = std::chrono::high_resolution_clock::now();
	//linear
	//araphb.get_variation(bc);
	//araphb.apply_initial_guess();
	//Vx[currentModel] = araphb.one_iter_linear_solver();
	//non-linear
	araphb.get_variation(bc);
	araphb.initialize_non_linear_solver(0.95, 0.001);
	Vx[currentModel] = araphb.one_iter_non_linear_solver();

	auto timer_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> timer_duration = timer_end - timer_start;
	std::cout << "Time taken 1st Iteration: " << timer_duration.count() << "seconds | Energy: " << araphb.get_arap_energy() << std::endl;
	viewer.data().set_vertices(Vx[currentModel]);
	viewer.data().compute_normals();
	return true;
}

void arap_next_iteration(igl::opengl::glfw::Viewer& viewer, int iterations) {
	int prevIter = currentIteration;
	currentIteration += iterations;
	auto timer_start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iterations; i++) {
		// linear
		//Vx[currentModel] = araphb.one_iter_linear_solver();
		// non-linear
		Vx[currentModel] = araphb.one_iter_non_linear_solver();
	}

	auto timer_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> timer_duration = timer_end - timer_start;

	std::cout << "Time taken from " << prevIter << " to " << currentIteration
		<< " Iteration : " << timer_duration.count() << "seconds | Energy: " << araphb.get_arap_energy() <<std::endl;
	viewer.data().set_vertices(Vx[currentModel]);
	viewer.data().compute_normals();

}

//Run all methods using the same model
//Provide benchmark data
//Compare results viually (multiple models, enable and disable)

//What to do on mouse click event
//Changes based on the clickMode variable
bool mouse_down(igl::opengl::glfw::Viewer& viewer, int button, int modifier) {
	if (iterativeHB || benchmark) {
		return false;
	}
	//Reset this when any click is registered
	currentIteration = 0;
	switch (current_mode) {
	case Face_Select:
		return face_selection(viewer);
	case Handle_Select:
		return handle_selection_face_based(viewer);
	case ARAP_Libigl:
		currentModel = 0;
		return move_handle_libigl_arap(viewer);
	case ARAP_HB:
		currentModel = 1;
		return move_handle_araphb(viewer);
	case ARAP_HB_Iter:
		currentModel = 1;
		currentIteration++;
		iterativeHB = true;
		/*prepare_araphb(currentModel);*/
		return move_handle_arap_first_step(viewer);
	case ARAP_Benchmark:
		benchmark = true;
		currentModel = 0;
		move_handle_libigl_arap(viewer);
		currentModel = 1;
		/*prepare_araphb(currentModel);*/
		chamferDist = araphb.computeChamferDistance(Vx[0], Vx[1]);
		return move_handle_araphb(viewer);
	default:
		return false;
	}
}


int main(int argc, char* argv[])
{

	std::vector<int> selected_vertices_list;
	igl::readOFF(modelPath, V, F);
	for (int i = 0; i < modelCount; i++) {
		Vx.emplace_back(V);
	}

	araphb.init(1,V,F);

	//Every time a vertex is selected one is added
	//Because selection is face based, we keep track of how many faces that contain the vertex are selected
	//if index > 0, vertex is considered a constraint
	//Eg. face 1 is added, so its three vertices v1, v2, v3 get +1 in vertexSelected
	vertexSelected = Eigen::VectorX<int>::Constant(V.rows(), 0);
	vertexHandles = Eigen::VectorX<int>::Constant(V.rows(), 0);
	faceSelected = Eigen::VectorX<int>::Constant(F.rows(), 0);
	faceHandles = Eigen::VectorX<int>::Constant(F.rows(), 0);
	//Vertex 0 is set by default as the handle
	//We add 1 to the vertexSelected since the handle is considered a constraint
	//vertexSelected[0] += 1;
	//vertexSelectedCount++;
	C = Eigen::MatrixXd::Constant(F.rows(), 3, 1);
	clickMode = 1;
	//Set by default, we can change this for comparisons
	arap_data.max_iter = 300;
	// Plot the mesh
	igl::opengl::glfw::Viewer viewer;
	//Set events for inputs
	viewer.callback_mouse_down = &mouse_down;
	viewer.callback_key_down = &key_down;

	// Attach a menu plugin
	igl::opengl::glfw::imgui::ImGuiPlugin plugin;
	viewer.plugins.push_back(&plugin);
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	plugin.widgets.push_back(&menu);

	// Customize the menu
	double doubleVariable = 0.1f; // Shared between two menus
	

	// Add content to the default menu window
	menu.callback_draw_viewer_menu = [&]()
	{
		// Draw parent menu content
		//menu.draw_viewer_menu();
		ImGui::PushItemWidth(150);
		
		// Add new group
		if (ImGui::CollapsingHeader("ARAP Options", ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::Combo("Mode Select", (int*)(&current_mode), "Face Select\0Handle Select\0ARAP Libigl\0ARAP HB\0ARAP Iterations\0ARAP Benchmark\0\0");
		}

		// Add a button
		if (ImGui::Button("Reset Models", ImVec2(-1, 0)))
		{
			for (int i = 0; i < modelCount; i++) {
				Vx[i] = V;
			}
			viewer.data().set_vertices(Vx[currentModel]);
		}
		if (iterativeHB) {
			if (ImGui::Button("Next Iteration", ImVec2(-1, 0)))
			{
				arap_next_iteration(viewer, 1);
			}
			if (ImGui::Button("10 Iterations", ImVec2(-1, 0)))
			{
				arap_next_iteration(viewer, 10);
			}

			if (ImGui::Button("End Iterations", ImVec2(-1, 0)))
			{
				iterativeHB = false;
				current_mode = Face_Select;
			}
		}
		if (benchmark) {

			char* str;
			if (currentModel == 1) {
				str = "Custom ARAP";
			}
			else {
				str = "Libigl ARAP";
			}

			if (ImGui::Button(str, ImVec2(-1, 0)))
			{
				currentModel = (currentModel + 1) % 2;
				viewer.data().set_vertices(Vx[currentModel]);
				viewer.data().compute_normals();
			}

			std::string hbdur = "HB ARAP: " + std::to_string(HBDur.count()) + " seconds";
			ImGui::Text(hbdur.c_str());
			std::string libdur = "Libigl ARAP: " + std::to_string(libiglDur.count()) + " seconds";
			ImGui::Text(libdur.c_str());
			std::string chamferStr = "Chamfer Distance: " + std::to_string(chamferDist);
			ImGui::Text(chamferStr.c_str());

			if (ImGui::Button("End Benchmark", ImVec2(-1, 0)))
			{
				benchmark = false;
				chamferDist = 0;
			}

		}


		ImGui::PopItemWidth();
	};
	
	// Draw additional windows
	/*
	menu.callback_draw_custom_window = [&]()
	{
		// Define next window position + size
		ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(200, 160), ImGuiCond_FirstUseEver);
		ImGui::Begin(
			"New Window", nullptr,
			ImGuiWindowFlags_NoSavedSettings
		);

		// Expose the same variable directly ...
		ImGui::PushItemWidth(-80);
		ImGui::DragScalar("double", ImGuiDataType_Double, &doubleVariable, 0.1, 0, 0, "%.4f");
		ImGui::PopItemWidth();

		static std::string str = "bunny";
		ImGui::InputText("Name", str);

		ImGui::End();
	};
	*/



	viewer.data().set_mesh(Vx[0], F);
	viewer.data().set_colors(C);
	viewer.launch();
}
