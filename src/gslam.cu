#include "debug_utils.cuh"
#include "gaussian.cuh"
#include "loss_monitor.cuh"
#include "loss_utils.cuh"
#include "parameters.cuh"
#include "render_utils.cuh"
#include "point_cloud.cuh"
#include "scene.cuh"
#include <args.hxx>
#include <c10/cuda/CUDACachingAllocator.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>


void Write_model_parameters_to_file(const gs::param::ModelParameters& params) {
    std::filesystem::path outputPath = params.output_path; // Get output path
    std::filesystem::create_directories(outputPath); // Make sure the directory exists

    std::ofstream cfg_log_f(outputPath / "cfg_args");
    if (!cfg_log_f.is_open()) {
        std::cerr << "Failed to open file for writing!" << std::endl;
        return;
    }

    // Write basic model parameters
    cfg_log_f << "Namespace(";
    cfg_log_f << "eval=" << (params.eval ? "True" : "False") << ", ";
    cfg_log_f << "images='" << params.images << "', ";
    cfg_log_f << "model_path='" << params.output_path.string() << "', ";
    cfg_log_f << "resolution=" << params.resolution << ", ";
    cfg_log_f << "sh_degree=" << params.sh_degree << ", ";
    cfg_log_f << "source_path='" << params.source_path.string() << "', ";
    cfg_log_f << "white_background=" << (params.white_background ? "True" : "False") << ")";
    cfg_log_f.close();

    std::cout << "Output folder: " << params.output_path.string() << std::endl;
}

/**
 * Generates a random index vector from 0 to max_index - 1 (shuffled and reversed)
 * 
 * @param max_index Maximum index
 * @return Index array
 */
std::vector<int> get_random_indices(int max_index) {
    std::vector<int> indices(max_index); // Create an integer vector with capacity max_index
    std::iota(indices.begin(), indices.end(), 0); // Initialize indices vector with incrementing integers starting at 0
    // Shuffle the vector
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine()); // Shuffle the index vector to create a random sequence of indices
    std::reverse(indices.begin(), indices.end()); // Reverse the index vector to convert the original random sequence into descending order
    return indices;
}

/**
 * Read command line arguments
 * 
 * @param args Command line arguments
 * @param modelParams Model parameters
 * @param optimParams Optimization parameters
 * @return 1 or 0, success or failure
 */
int parse_cmd_line_args(const std::vector<std::string>& args, // Command line arguments
                        gs::param::ModelParameters& modelParams, // Model parameters
                        gs::param::OptimizationParameters& optimParams) // Optimization parameters
{
    if (args.empty()) {
        std::cerr << "No command line arguments provided!" << std::endl;
        return -1;
    }
    args::ArgumentParser parser("3D Gaussian Splatting CUDA Implementation\n",
                                "This program provides a lightning-fast CUDA implementation of the 3D Gaussian Splatting algorithm for real-time radiance field rendering.");
    // Execute specific actions when the following arguments are provided
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<float> convergence_rate(parser, "convergence_rate", "Set convergence rate", {'c', "convergence_rate"});
    args::ValueFlag<int> resolution(parser, "resolution", "Set resolution", {'r', "resolution"});
    args::Flag enable_cr_monitoring(parser, "enable_cr_monitoring", "Enable convergence rate monitoring", {"enable-cr-monitoring"});
    args::Flag force_overwrite_output_path(parser, "force", "Forces to overwrite output folder", {'f', "force"});
    args::Flag empty_gpu_memory(parser, "empty_gpu_cache", "Forces to reset GPU Cache. Should be lighter on VRAM", {"empty-gpu-cache"});
    args::ValueFlag<std::string> data_path(parser, "data_path", "Path to the training data", {'d', "data-path"});
    args::ValueFlag<std::string> output_path(parser, "output_path", "Path to the training output", {'o', "output-path"});
    args::ValueFlag<uint32_t> iterations(parser, "iterations", "Number of iterations to train the model", {'i', "iter"});
    args::CompletionFlag completion(parser, {"complete"});

    try {
        parser.Prog(args.front());
        parser.ParseArgs(std::vector<std::string>(args.begin() + 1, args.end()));
    } catch (const args::Completion& e) {
        std::cout << e.what();
        return 0;
    } catch (const args::Help&) {
        std::cout << parser;
        return -1;
    } catch (const args::ParseError& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return -1;
    }

    // Handle parameters if they exist
    if (data_path) {
        modelParams.source_path = args::get(data_path); // Training data path
    } else {
        std::cerr << "No data path provided!" << std::endl;
        return -1;
    }
    if (output_path) {
        modelParams.output_path = args::get(output_path); // Training output path
    } else {
        std::filesystem::path executablePath = std::filesystem::canonical("/proc/self/exe");
        std::filesystem::path parentDir = executablePath.parent_path().parent_path();
        std::filesystem::path outputDir = parentDir / "output";
        try {
            bool isCreated = std::filesystem::create_directory(outputDir);
            if (!isCreated) {
                if (!force_overwrite_output_path) { // Do not overwrite output folder if not forced
                    std::cout << "Directory already exists! Not overwriting it" << std::endl;
                    return -1;
                } else {
                    std::cout << "Output directory already exists! Overwriting it" << std::endl;
                    std::filesystem::create_directory(outputDir);
                    std::filesystem::remove_all(outputDir);
                }
            }
        } catch (...) {
            std::cerr << "Failed to create output directory!" << std::endl;
            return -1;
        }
        modelParams.output_path = outputDir;
    }

    if (iterations) {
        optimParams.iterations = args::get(iterations);
    }
    optimParams.early_stopping = args::get(enable_cr_monitoring);
    if (optimParams.early_stopping && convergence_rate) {
        optimParams.convergence_threshold = args::get(convergence_rate);
    }

    if (resolution) {
        modelParams.resolution = args::get(resolution);
    }

    optimParams.empty_gpu_cache = args::get(empty_gpu_memory);
    return 0;
}

/**
 * Calculate peak signal-to-noise ratio (PSNR) of images
 * 
 * @param rendered_img Rendered image
 * @param gt_img Ground truth image
 * @return PSNR value
 */
float psnr_metric(const torch::Tensor& rendered_img, const torch::Tensor& gt_img) {

    torch::Tensor squared_diff = (rendered_img - gt_img).pow(2);
    torch::Tensor mse_val = squared_diff.view({rendered_img.size(0), -1}).mean(1, true);
    return (20.f * torch::log10(1.0 / mse_val.sqrt())).mean().item<float>();
}

// Convert torch::Tensor to cv::Mat
cv::Mat tensor_to_mat(const torch::Tensor& tensor) {
    // Clone the tensor to ensure data is contiguous in CPU memory
    torch::Tensor tensor_cpu = tensor.to(torch::kCPU).clone();

    // Get dimensions and data pointer of tensor
    // Tensor is created as C × H × W, normalized in the range [0, 1]
    int height = tensor_cpu.size(1);
    int width = tensor_cpu.size(2);
    int channels = tensor_cpu.size(0);
    const float* data_ptr = tensor_cpu.data_ptr<float>();

    // Create cv::Mat of corresponding dimensions and deep copy data
    cv::Mat mat(height, width, CV_MAKETYPE(CV_32F, channels));
    memcpy(mat.data, data_ptr, sizeof(float) * height * width * channels);

    // Convert image data type to 8-bit unsigned integer type (range [0,255])
    mat.convertTo(mat, CV_8U, 255.0);

    return mat;
}

torch::Tensor mat_to_tensor(const cv::Mat& mat, bool use_cuda = true) {
    // Determine the number of channels
    int channels = mat.channels();

    // Set tensor shape
    std::vector<int64_t> dims = {1, mat.rows, mat.cols, channels};
    torch::TensorOptions options = torch::TensorOptions()
                                    .dtype(torch::kUInt8) // OpenCV uses uint8 by default for images
                                    .device(torch::kCPU); // Initialize as CPU tensor

    // Create tensor and copy data
    torch::Tensor tensor = torch::from_blob(mat.data, dims, options);

    // Convert tensor to [C, H, W] format and normalize to float32 for further processing
    tensor = tensor.permute({0, 3, 1, 2}).to(torch::kFloat32) / 255.0;

    // Move to CUDA if specified and available
    // if (use_cuda && torch::cuda::is_available()) {
    //     tensor = tensor.to(torch::kCUDA);
    // }

    return tensor.clone(); // Clone to ensure it owns its own memory
}

int main(int argc, char* argv[]) {
    std::vector<std::string> args;
    args.reserve(argc);

    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    // TODO: read parameters from JSON file or command line
    auto modelParams = gs::param::ModelParameters();
    auto optimParams = gs::param::read_optim_params_from_json();
    // Command Line
    if (parse_cmd_line_args(args, modelParams, optimParams) < 0) {
        return -1;
    };
    Write_model_parameters_to_file(modelParams);

    int N = 79;
    std::vector<cv::Mat> colorImgs, depthImgs;
    std::vector<Eigen::Isometry3d> poses;

    // Michael: Read TUM for test
    std::ifstream fin("/home/michael/PCLtest/associations.txt");
    if (!fin) {
        std::cerr << "Cannot find pose file" << std::endl;
        return 1;
    }

    for (int i = 0; i < N; i++) {
        std::string data[12];
        for (int j = 0; j < 12; j++) {
            fin >> data[j];
        }
        colorImgs.push_back(cv::imread("/home/michael/Documents/rgbd_dataset_freiburg1_xyz/" + data[1]));
        depthImgs.push_back(cv::imread("/home/michael/Documents/rgbd_dataset_freiburg1_xyz/" + data[3], -1)); // Reading Raw (-1)

        double double_data[7];
        for (int j = 5; j < 12; j++)
            double_data[j - 5] = std::atof(data[j].c_str());

        Eigen::Quaterniond q(double_data[6], double_data[3], double_data[4], double_data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(double_data[0], double_data[1], double_data[2]));
        poses.push_back(T);
    }

    cv::Mat color1 = colorImgs[0];
    cv::Mat depth1 = depthImgs[0];
    Eigen::Isometry3d T = poses[0];
    double cx = 318.6;
    double cy = 255.3;
    double fx = 517.3;
    double fy = 516.5;
    double depthScale = 5000.0;
    int ic = 0;
    PointCloud p;

    for (int v = 0; v < color1.rows; v++) {
        for (int u = 0; u < color1.cols; u++) {
            unsigned int d = depth1.ptr<unsigned short>(v)[u]; // Depth
            if (d == 0) continue; // Not observed
            Eigen::Vector3d point;
            point[2] = double(d) / depthScale;
            point[0] = (u - cx) * point[2] / fx; // Camera Position
            point[1] = (v - cy) * point[2] / fy;
            Eigen::Vector3d pointWorld = T * point; // World Position

            // vertices
            p._points[ic].x = pointWorld[0];
            p._points[ic].y = pointWorld[1];
            p._points[ic].z = pointWorld[2];

            // colors
            p._colors[ic].r = color1.data[v * color1.step + u * color1.channels() + 2];
            p._colors[ic].g = color1.data[v * color1.step + u * color1.channels() + 1];
            p._colors[ic].b = color1.data[v * color1.step + u * color1.channels()];

            ic++;
        }
    }

    torch::Tensor tensor_w2c = torch::from_blob(poses[0].data(), {poses[0].rows(), poses[0].cols()}, torch::kFloat).clone();

    // Check if CUDA is available
    if (!torch::cuda::is_available()) {
        // At the moment, I want to make sure that my GPU is utilized.
        std::cout << "CUDA is not available! Training on CPU." << std::endl;
        exit(-1);
    }
    auto pointType = torch::TensorOptions().dtype(torch::kFloat32);
    auto background = modelParams.white_background ? torch::tensor({1.f, 1.f, 1.f}) : torch::tensor({0.f, 0.f, 0.f}, pointType).to(torch::kCUDA);

    //Create Conv Window for SSIM Loss
    // const int window_size = 11;
    // const int channel = 3;
    // const auto conv_window = gaussian_splatting::create_window(window_size, channel).to(torch::kFloat32).to(torch::kCUDA, true);
    //const int camera_count = scene.Get_camera_count();

    std::vector<int> indices;
    int last_status_len = 0;
    auto start_time = std::chrono::steady_clock::now();//Time start
    float loss_add = 0.f;

    LossMonitor loss_monitor(200);//buffersize=200
    float avg_converging_rate = 0.f;

    float psnr_value = 0.f;//Init PSNR

    //Michael: Modify here to init with TUM
    auto gaussians = GaussianModel(modelParams.sh_degree);//Create Gaussian
    auto scene = Scene(gaussians, p, modelParams);//Initialize Gaussians with sce(image)
    gaussians.Training_setup(optimParams);//Set Training

    //For single image: iterations=100 * 2 for init
    int iterations = 200;

    for (int iter = 1; iter < iterations + 1; ++iter) {
        // if (indices.empty()) {
        //     indices = get_random_indices(camera_count);
        // }
        // const int camera_index = indices.back();//Random index now
        // auto& cam = scene.Get_training_camera(camera_index);
        // auto gt_image = cam.Get_original_image().to(torch::kCUDA, true);
        // auto gt_depth = cam.Get_original_image().to(torch::kCUDA, true);
        // indices.pop_back(); //remove last element to iterate over all cameras randomly

        // Add sh_degree to 1000 _max_sh_degree
        // if (iter % 1000 == 0) {
        //     gaussians.One_up_sh_degree();
        // }

        auto gt_image = mat_to_tensor(colorImgs[0]);
        auto gt_depth = mat_to_tensor(depthImgs[0]);
        //Render
        //Michael: Rasterizer: depth, alpha: No need for cam info from input now, specified for TUM
        auto [image, viewspace_point_tensor, visibility_filter, radii, depth, alpha] = render(gaussians, tensor_w2c, background);

        // Redifine Loss Here! 
        
        /*
        auto tracking_mask = torch::ones({alpha.size(0),alpha.size(1)});
        torch::Tensor depth_mask = gt_depth > 0.0; 
        torch::Tensor alpha_mask = alpha > 0.0; 
        tracking_mask &= depth_mask;
        tracking_mask &= alpha_mask;
        */
        
        auto l1color = gaussian_splatting::l1_loss(image, gt_image) /* *tracking_mask*/;
        auto l1depth = gaussian_splatting::l1_loss(depth, gt_depth) /* *tracking_mask*/;
        auto loss = 0.6 * l1color + 0.4 * l1depth;
        //auto ssim_loss = gaussian_splatting::ssim(image, gt_image, conv_window, window_size, channel);
        //auto loss = (1.f - optimParams.lambda_dssim) * l1l + optimParams.lambda_dssim * (1.f - ssim_loss);

        // Update status line
        // Output at command
        if (iter % 100 == 0) {
            auto cur_time = std::chrono::steady_clock::now();//Time end
            std::chrono::duration<double> time_elapsed = cur_time - start_time;
            // XXX shouldn't have to create a new stringstream, but resetting takes multiple calls
            std::stringstream status_line;
            // XXX Use thousand separators, but doesn't work for some reason
            status_line.imbue(std::locale(""));//Format in Timezone
            status_line
                << "\rIter: " << std::setw(6) << iter
                << "  Loss: " << std::fixed << std::setw(9) << std::setprecision(6) << loss.item<float>();
            if (optimParams.early_stopping) {
                status_line
                    << "  ACR: " << std::fixed << std::setw(9) << std::setprecision(6) << avg_converging_rate;
            }
            status_line
                << "  Splats: " << std::setw(10) << (int)gaussians.Get_xyz().size(0)
                << "  Time: " << std::fixed << std::setw(8) << std::setprecision(3) << time_elapsed.count() << "s"
                << "  Avg iter/s: " << std::fixed << std::setw(5) << std::setprecision(1) << 1.0 * iter / time_elapsed.count()
                << "  " // Some extra whitespace, in case a "Pruning ... points" message gets printed after
                ;
            const int curlen = status_line.str().length();
            const int ws = last_status_len - curlen;
            if (ws > 0)
                status_line << std::string(ws, ' ');
            std::cout << status_line.str() << std::flush;
            last_status_len = curlen;
        }

        if (optimParams.early_stopping) {
            avg_converging_rate = loss_monitor.Update(loss.item<float>());
        }
        loss_add += loss.item<float>();

        loss.backward();//Backward here

        {//Update param
            torch::NoGradGuard no_grad;
            auto visible_max_radii = gaussians._max_radii2D.masked_select(visibility_filter);
            auto visible_radii = radii.masked_select(visibility_filter);
            auto max_radii = torch::max(visible_max_radii, visible_radii);
            gaussians._max_radii2D.masked_scatter_(visibility_filter, max_radii);//Update _max_radii2D

            //Save and Calc PSNR
            if (iter == optimParams.iterations) {
                std::cout << std::endl;
                gaussians.Save_ply(modelParams.output_path, iter, true);
                psnr_value = psnr_metric(image, gt_image);
                //Save Image
                cv::Mat render_image = tensor_to_mat(image);
                cv::Mat gt_image_mat = tensor_to_mat(gt_image);
                cv::imshow("img for 3dgs", render_image);
                cv::imshow("gt img", gt_image_mat);
                // cv::waitKey(1); 
                cv::waitKey(0); 
                break;
            }

            //Save
            if (iter % 2'00 == 0) {
                gaussians.Save_ply(modelParams.output_path, iter, false);
            }

            // Densification
            if (iter < optimParams.densify_until_iter) {
                gaussians.Add_densification_stats(viewspace_point_tensor, visibility_filter);
                if (iter > optimParams.densify_from_iter && iter % optimParams.densification_interval == 0) 
                {
                    // @TODO: Not sure about type
                    float size_threshold = iter > optimParams.opacity_reset_interval ? 20.f : -1.f;
                    gaussians.Densify_and_prune(optimParams.densify_grad_threshold, optimParams.min_opacity, scene.Get_cameras_extent(), size_threshold);
                }

                if (iter % optimParams.opacity_reset_interval == 0 || (modelParams.white_background && iter == optimParams.densify_from_iter)) {
                    gaussians.Reset_opacity();
                }
            }

            if (iter >= optimParams.densify_until_iter && loss_monitor.IsConverging(optimParams.convergence_threshold)) {
                std::cout << "Converged after " << iter << " iterations!" << std::endl;
                gaussians.Save_ply(modelParams.output_path, iter, true);
                break;
            }

            //  Optimizer step
            if (iter < optimParams.iterations) {
                gaussians._optimizer->step();
                gaussians._optimizer->zero_grad(true);
                // @TODO: Not sure about type
                gaussians.Update_learning_rate(iter);
            }

            if (optimParams.empty_gpu_cache && iter % 100) {
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
        }
    }

    auto cur_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_elapsed = cur_time - start_time;//Time Used in Total

    std::cout << std::endl
              << "The training of the 3DGS is done in "
              << std::fixed << std::setw(7) << std::setprecision(3) << time_elapsed.count() << "sec, avg "
              << std::fixed << std::setw(4) << std::setprecision(1) << 1.0 * optimParams.iterations / time_elapsed.count() << " iter/sec, "
              << gaussians.Get_xyz().size(0) << " splats, "
              << std::fixed << std::setw(7) << std::setprecision(6) << " psrn: " << psnr_value << std::endl
              << std::endl
              << std::endl;

    return 0;
}