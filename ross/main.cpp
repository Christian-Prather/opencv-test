// stdlib
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

// Librealsense
#include <librealsense2/rs.hpp>
#include <librealsense2/rs_advanced_mode.hpp>

// OpenCV
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>

std::pair<bool, std::string> parseArguments(int argc, char **argv);
rs2::config createConfig(const std::string &rsConfigPath, bool fromFile);
void initWindows(const std::vector<std::string> &windows);
void displayWindows(const std::map<std::string, const cv::Mat &> &images);
rs2_intrinsics getIntrinsics(const rs2::pipeline_profile &profile);
cv::Mat toCvMat(const rs2::depth_frame &rsDepthFrame);
cv::Mat toCvMat(const rs2::video_frame &rsIrFrame);
cv::Mat processDepthImage(const cv::Mat &initDepthImage);
cv::Mat processCoordinateImage(const cv::Mat &depthImage,
                               const rs2_intrinsics &intrinsics,
                               cv::ocl::Kernel &kernel);
cv::Mat findEdges2(const cv::Mat &zImage);
cv::Mat findEdges(const cv::Mat &zImage);
cv::Mat removeNonEdges(const cv::Mat &edges, const cv::Mat &depths,
                       const cv::Mat &zImage);

const auto DEPTH = "Depth";
const auto COORD = "Coordinates";
const auto COORD_X = "X";
const auto COORD_Y = "Y";
const auto COORD_Z = "Z";
const auto GRAD = "Z Gradients";
const auto CONTOURS = "Contours";
const auto IR = "Infrared";

class Profiler {
public:
  explicit Profiler(std::string tag)
      : start(std::chrono::steady_clock::now()), tag(tag) {}

  ~Profiler() {
    auto end = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << tag << " Took " << elapsed.count() << "ms" << std::endl;
  }

private:
  std::chrono::time_point<std::chrono::steady_clock> start;
  std::string tag;
};

static auto transformKernelSrc = R"""(
__kernel void pixel_to_chair_point(
    __global const ushort* src, int src_step, int src_offset, int rows, int cols, /* depth image buffer */
    __global float* dst_x,  /* output coordinate image buffer x channel */
    __global float* dst_y,  /* output coordinate image buffer y channel */
    __global float* dst_z,  /* output coordinate image buffer z channel */
    float ppx, float ppy, float fx, float fy, /* Frame intrinsics */
    __constant float* tm /* Size 16 transformation matrix */)
{
    /* We use row and col here instead of x and y, reserving those instead for coordinate space. */
    int col = get_global_id(0);
    int row = get_global_id(1);

    /* Validate we are within bounds. The example app has this, but I'm not sure how necessary it is. */
    if (col < cols && row < rows) {
        /* Calculate primes from frame intrinsics. This could be precalculated elsewhere and also passed
         * in as another input array if that would speed things up. */
        float x_prime = ((float)col - ppx) / fx;
        float y_prime = ((float)row - ppy) / fy;

        /* This should be the index for each of our buffers for the data at point. */
        int idx = cols * row + col;

        /* Deproject the depth values to a camera based cartesian coordinate space. */
        int x_tmp = src[idx] * x_prime;
        int y_tmp = src[idx] * y_prime;
        int z_tmp = src[idx];

        /* Run the transformations to put the points into the chair coordinate space. This is just a
         * basic matrix multiplication. */
        dst_x[idx] = tm[0] * x_tmp + tm[1] * y_tmp + tm[2] * z_tmp + tm[3];
        dst_y[idx] = tm[4] * x_tmp + tm[5] * y_tmp + tm[6] * z_tmp + tm[7];
        dst_z[idx] = tm[8] * x_tmp + tm[9] * y_tmp + tm[10] * z_tmp + tm[11];
    }
};
)""";

int main(int argc, char **argv) {
  // Parse arguments.
  const auto [fromFile, filePath] = parseArguments(argc, argv);

  std::cout << "Available? " << (cv::ocl::haveOpenCL() ? "Ya": "Na") << std::endl;
  exit(1);

  cv::ocl::Context ctx = cv::ocl::Context::getDefault();
  if (!ctx.ptr()) {
    std::cerr << "OpenCL is not available" << std::endl;
    return 1;
  }
  cv::ocl::Device device = cv::ocl::Device::getDefault();
  if (!device.compilerAvailable()) {
    std::cerr << "OpenCL compiler is not available" << std::endl;
    return 1;
  }
  auto source =
      cv::ocl::ProgramSource(cv::String(), "simple", transformKernelSrc, "");
  auto errmsg = cv::String();
  auto program = cv::ocl::Program(source, "", errmsg);
  if (program.ptr() == NULL) {
    std::cerr << "Can't compile OpenCL program:" << std::endl
              << errmsg << std::endl;
    exit(1);
  }
  if (!errmsg.empty()) {
    std::cout << "OpenCL program build log:" << std::endl
              << errmsg << std::endl;
  }
  auto kernel = cv::ocl::Kernel("pixel_to_chair_point", program);
  if (kernel.empty()) {
    std::cerr << "Can't get OpenCL Kernel" << std::endl;
    exit(1);
  }

  // Initialize visualizations
  // initWindows({DEPTH, COORD_X, COORD_Y, COORD_Z, GRAD, CONTOURS, IR});

  // Create camera config.
  const auto config = createConfig(filePath, fromFile);

  // Create and start pipeline.
  auto pipeline = rs2::pipeline();
  const auto profile = pipeline.start(config);
  const auto intrinsics = getIntrinsics(profile);

  while (cv::waitKey(1) < 0) {
    // Block on frameset from camera.
    auto frameset = pipeline.wait_for_frames();

    // Time starts now!
    auto start = std::chrono::steady_clock::now();

    // Retrieve depth frame.
    auto rsDepthFrame = frameset.get_depth_frame();
    // auto irFrame = frameset.get_infrared_frame();

    // Convert to OpenCV Matrix.
    cv::Mat initDepthImage;
    {
      auto p = Profiler("RS -> CV depth");
      initDepthImage = toCvMat(rsDepthFrame);
    }
    // auto irImage = toCvMat(irFrame);

    // Process the depth matrix.
    cv::Mat depthImage;
    {
      auto p = Profiler("depth processing");
      depthImage = processDepthImage(initDepthImage);
    }

    // Process the coordinate matrix.
    cv::Mat coordinateImage;
    {
      auto p = Profiler("coordinate processing");
      coordinateImage = processCoordinateImage(depthImage, intrinsics, kernel);
    }

    // Clean and split for visualization
    auto coordinateData = reinterpret_cast<float *>(coordinateImage.data);
    auto coordinateChannelNum = coordinateImage.channels();
    for (int row = 0; row < coordinateImage.rows; ++row) {
      for (int col = 0; col < coordinateImage.cols; ++col) {
        const auto xP = coordinateData +
                        (coordinateChannelNum * coordinateImage.cols * row) +
                        (coordinateChannelNum * col);
        const auto yP = xP + 1;
        const auto zP = yP + 1;

        if (*xP > 50000) {
          *xP = 50000;
        }
        if (*xP < -50000) {
          *xP = -50000;
        }
        if (*yP > 50000) {
          *yP = 50000;
        }
        if (*zP < -5000) {
          *zP = -5000;
        }
      }
    }

    auto coordinateChannels = std::vector<cv::Mat>(coordinateImage.channels());
    cv::split(coordinateImage, coordinateChannels.data());

    auto edges = findEdges2(coordinateChannels.at(2));

    auto contourImage =
        removeNonEdges(edges, depthImage, coordinateChannels.at(2));

    for (auto &mat : coordinateChannels) {
      cv::normalize(mat, mat, 1.0, 0.0, cv::NORM_MINMAX);
    }
    // cv::normalize(edges, edges, 1.0, 0.0, cv::NORM_MINMAX);

    // Processing is all done
    auto end = std::chrono::steady_clock::now();

    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Took: " << elapsed.count() << "ms" << std::endl;

    // Visualize
    // displayWindows({
    //     {DEPTH, depthImage},
    //     {COORD_X, coordinateChannels.at(0)},
    //     {COORD_Y, coordinateChannels.at(1)},
    //     {COORD_Z, coordinateChannels.at(2)},
    //     {GRAD, edges},
    //     {CONTOURS, contourImage},
    //     // {IR, irImage},
    // });
  }

  return 0;
}

void displayWindows(const std::map<std::string, const cv::Mat &> &images) {
  for (const auto &[name, image] : images) {
    cv::imshow(name, image);
  }
}

// cv::Mat removeNonEdges(const cv::Mat &edges, const cv::Mat &depths,
//                        const cv::Mat &zImage) {
//   assert(edges.type() == CV_8U);
//   assert(depths.type() == CV_16U);
//   assert(zImage.type() == CV_32F);

//   auto contours = std::vector<std::vector<cv::Point>>{};
//   auto hierarchies = std::vector<cv::Vec4i>{};
//   cv::findContours(edges, contours, hierarchies, cv::RETR_CCOMP,
//                    cv::CHAIN_APPROX_NONE);

//   auto contourImage = cv::Mat(cv::Mat::zeros(edges.rows, edges.cols, CV_8U));

//   // cv::drawContours(contourImage, contours, cv::FILLED, cv::Scalar(255));

//   // Hierarchies are organized as [Next, Previous, Child, Parent]

//   // Because we have to iterate using the index of the contours for hierarchy
//   // nonsense, we cannot use remove/erase to do our filtering yet. We must
//   // convert it to a form that we can actually iterate over first. The first
//   // element is a reference to the contour in the original vector. Because we
//   // use references, we cannot modify that first contour vector, and it
//   cannot
//   // go out of scope while we are using this one. The second element is
//   whether
//   // or not the contour is closed or not. Untelated, hate this name. I'll
//   think
//   // of something better.
//   auto myContours = std::vector<
//       std::tuple<std::vector<cv::Point> &, bool, std::stringstream>>{};

//   // Populating the collection
//   const auto size = contours.size();
//   for (auto i = 0; i < size; ++i) {
//     const auto childIndex = hierarchies.at(i)[2];
//     const auto isClosed = childIndex != -1;
//     myContours.emplace_back(contours.at(i), isClosed, std::stringstream{});
//   }

//   const auto begin = std::begin(myContours);
//   auto end = std::end(myContours);
//   // Remove very short contours.
//   end = std::remove_if(begin, end, [](auto &value) -> bool {
//     auto &[contour, isClosed, ss] = value;
//     const auto length = cv::arcLength(contour, isClosed);
//     ss << "L: " << length << ",";
//     // NOTE: This value is configurable.
//     const auto LENGTH_THRESHOLD = 20;
//     return length < LENGTH_THRESHOLD;
//   });

//   // Remove contours that are closed and contain a significan percentage of 0
//   // depth data.
//   end = std::remove_if(begin, end, [&depths](auto &value) -> bool {
//     auto &[contour, isClosed, ss] = value;

//     // Only filtering on closed contours. Still up for debate though...
//     // if (!isClosed) {
//     //   ss << "O,";
//     //   return false;
//     // }

//     // Create a mask image
//     cv::Mat mask = cv::Mat::zeros(depths.cols, depths.rows, CV_8U);

//     // Draw the contour
//     cv::fillConvexPoly(mask, contour, cv::Scalar(255));

//     // Find a rectangle around the contour.
//     const auto rectangle = cv::boundingRect(contour);

//     // Iterate over the rectangle region and count the pixels!
//     auto numPixels = 0;
//     auto numZeroPixels = 0;
//     for (auto row = rectangle.tl().y; row < rectangle.br().y; ++row) {
//       for (auto col = rectangle.tl().x; col < rectangle.br().x; ++col) {
//         const auto maskVal = mask.at<uint8_t>(row, col);
//         if (maskVal == 255) {
//           ++numPixels;

//           const auto depthVal = depths.at<uint16_t>(row, col);
//           if (depthVal == 0 ||
//               depthVal == std::numeric_limits<uint16_t>::max()) {
//             ++numZeroPixels;
//           }
//         }
//       }
//     }

//     double zeroPercent = static_cast<double>(numZeroPixels) / numPixels;
//     ss << "C: " << zeroPercent << ",";
//     const auto ZERO_PERCENT_THRESHOLD = 0.3;
//     return zeroPercent > ZERO_PERCENT_THRESHOLD;
//   });

//   // Remove contours that are surrounded only by z values of the same height.
//   end = std::remove_if(begin, end, [&zImage](auto &value) -> bool {
//     auto &[contour, isClosed, ss] = value;
//     // if (!isClosed) {
//     //   return false;
//     // }
//     cv::Mat mask = cv::Mat::zeros(zImage.rows, zImage.cols, CV_8U);
//     cv::polylines(mask, contour, isClosed, cv::Scalar(255));
//     auto meanArr = cv::Scalar(0);
//     auto stddevArr = cv::Scalar(0);
//     cv::meanStdDev(zImage, meanArr, stddevArr, mask);
//     const auto mean = meanArr[0];
//     const auto stddev = stddevArr[0];
//     // NOTE: This value is configurable.
//     const auto STDDEV_THRESHOLD = 20;
//     ss << "SD: " << stddev;
//     return stddev < STDDEV_THRESHOLD;
//   });

//   // myContours.erase(end, std::end(myContours));

//   // std::for_each(begin, end, [&contourImage](const auto &value) {
//   //   const auto &[contour, isClosed, ss] = value;
//   //   cv::polylines(contourImage, contour, isClosed, cv::Scalar(255));
//   //   auto center = cv::Point2f();
//   //   auto radius = float();
//   //   cv::minEnclosingCircle(contour, center, radius);
//   //   cv::putText(contourImage, ss.str(), center,
//   cv::FONT_HERSHEY_PLAIN, 1.0,
//   //               cv::Scalar(255));
//   // });

//   std::for_each(end, std::end(myContours), [&contourImage](const auto &value)
//   {
//     const auto &[contour, isClosed, ss] = value;
//     cv::polylines(contourImage, contour, isClosed, cv::Scalar(127));
//     auto center = cv::Point2f();
//     auto radius = float();
//     cv::minEnclosingCircle(contour, center, radius);
//     cv::putText(contourImage, ss.str(), center, cv::FONT_HERSHEY_PLAIN, 1.0,
//                 cv::Scalar(127));
//   });

//   return contourImage;
// }

cv::Mat findEdges2(const cv::Mat &zImage) {
  // These are still in the same units as the 16 bit depth image
  // is... 10th of a mil? maybe?
  assert(zImage.type() == CV_32F);

  // Make an 8 bit Z height copy
  auto z8bit = cv::Mat();

  // Bounds and shit. This is now in wheelchair coordinate land!
  const auto minCareAboutHeight = -0.5;
  const auto maxCareAboutHeight = 0.5;

  // y = 0.0255 * x + 127.5
  zImage.convertTo(z8bit, CV_8U, 0.0255, 127.5);
  // cv::imshow("FUCK ME", z8bit);

  auto edges = cv::Mat();
  cv::Canny(z8bit, edges, 15, 45);

  // TODO! Do either a dilation or a closing to try to connect lines that are
  // close to eachother

  return edges;
}

cv::Mat findEdges(const cv::Mat &zImage) {
  assert(zImage.type() == CV_32F);

  // NOTE: Might also try the laplace one as well
  auto xGradient = cv::Mat();
  cv::Sobel(zImage, xGradient, CV_32F, 1, 0, cv::FILTER_SCHARR);
  xGradient = cv::abs(xGradient);

  auto yGradient = cv::Mat();
  cv::Sobel(zImage, yGradient, CV_32F, 0, 1, cv::FILTER_SCHARR);
  yGradient = cv::abs(yGradient);

  auto edges = cv::Mat();
  cv::addWeighted(xGradient, 0.5, yGradient, 0.5, 0, edges);

  // auto edges = cv::Mat();
  // cv::Laplacian(zImage, edges, CV_32F);
  // edges = cv::abs(edges);

  // Iterate over both matricies
  auto rows = zImage.rows;
  auto cols = zImage.cols;
  auto zData = reinterpret_cast<float *>(zImage.data);
  auto edgeData = reinterpret_cast<float *>(edges.data);
  auto zChannels = zImage.channels();
  auto edgeChannels = edges.channels();
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const auto edgeP =
          edgeData + (edgeChannels * cols * row) + (edgeChannels * col);
      const auto zP = zData + (zChannels * cols * row) + (zChannels * col);

      // See if the edge is above the threshold.
      // NOTE: This is configurable.
      const auto EDGE_THRESHOLD = 3000;
      auto aboveThreshold = *edgeP > EDGE_THRESHOLD;
      // auto aboveThreshold = true;

      // See if the z height at that point is near the ground plane.
      // NOTE: this stuff is configurable and can be pre calculated
      auto offset = -5000;
      auto slop = 1000;
      auto lowerBound = offset - slop;
      auto upperBound = offset + slop;
      auto withinBounds = *zP > lowerBound && *zP < upperBound;
      withinBounds = true;

      if (!aboveThreshold || !withinBounds) {
        *edgeP = 0;
      } else {
        *edgeP = 255;
      }
    }
  }

  edges.convertTo(edges, CV_8U);

  return edges;
}

// WOOHOO OpenCL here we come!
using std::cos;
using std::sin;
float gam = 0.0;
float beta = -1.9199;
float alpha = -1.8326;
float xTrans = 0;
float yTrans = 0;
float zTrans = 0;
auto transformationMatrix = std::array<float, 16>{
    cos(alpha) * cos(beta),
    cos(alpha) * sin(beta) * sin(gam) - sin(alpha) * cos(gam),
    cos(alpha) * sin(beta) * cos(gam) + sin(alpha) * sin(gam),
    xTrans,
    sin(alpha) * cos(beta),
    sin(alpha) * sin(beta) * sin(gam) + cos(alpha) * cos(gam),
    sin(alpha) * sin(beta) * cos(gam) - cos(alpha) * sin(gam),
    yTrans,
    -sin(beta),
    cos(beta) * sin(gam),
    cos(beta) * cos(gam),
    zTrans,
    // NOTE: now looking, not sure why these are here...
    0,
    0,
    0,
    1,
};
auto transMat = cv::Mat(1, 16, CV_32F,
                        reinterpret_cast<void *>(transformationMatrix.data()));

cv::Mat processCoordinateImage(const cv::Mat &depthImage,
                               const rs2_intrinsics &intrinsics,
                               cv::ocl::Kernel &kernel) {
  // Validate image.
  assert(depthImage.type() == CV_16U);

  auto x = 0;
  // Convert input matrix to OpenCL compatible UMat
  auto clDepthImage = depthImage.getUMat(cv::ACCESS_READ);

  // Create output matricies for each dimension.
  auto clXImage = cv::UMat(depthImage.rows, depthImage.cols, CV_32F);
  auto clYImage = cv::UMat(depthImage.rows, depthImage.cols, CV_32F);
  auto clZImage = cv::UMat(depthImage.rows, depthImage.cols, CV_32F);

  // Put our trans matrix into a UMat... Hopefully can get rid of this.
  auto clTransMat = transMat.getUMat(cv::ACCESS_READ);

  // Define kernel parameters
  size_t globalSize[] = {static_cast<size_t>(depthImage.cols),
                         static_cast<size_t>(depthImage.rows)};
  // Not sure what to pick for this one... The example picked 8.
  size_t localSize[] = {8, 8};

  // Run!
  bool executionResult =
      kernel
          .args(cv::ocl::KernelArg::ReadOnly(clDepthImage),
                cv::ocl::KernelArg::PtrWriteOnly(clXImage),
                cv::ocl::KernelArg::PtrWriteOnly(clYImage),
                cv::ocl::KernelArg::PtrWriteOnly(clZImage), intrinsics.ppx,
                intrinsics.ppy, intrinsics.fx, intrinsics.fy,
                cv::ocl::KernelArg::PtrReadWrite(clTransMat))
          .run(2, globalSize, localSize, true);
  if (!executionResult) {
    std::cerr << "OpenCL Kernel launch failed" << std::endl;
    exit(1);
  }

  auto clCoordImages = std::vector<cv::Mat>{clXImage.getMat(cv::ACCESS_READ),
                                            clYImage.getMat(cv::ACCESS_READ),
                                            clZImage.getMat(cv::ACCESS_READ)};
  auto coordinateImage = cv::Mat();
  cv::merge(clCoordImages, coordinateImage);
  return coordinateImage.clone();

  //   // Dimensions are the same for each image, pull them out here.
  //   auto rows = depthImage.rows;
  //   auto cols = depthImage.cols;

  //   // Create a new matrix to hold the coordinate data. The three channels
  //   are
  //   // (X, Y, Z)
  //   auto coordinateImage = cv::Mat(rows, cols, CV_32FC3);

  //   // Get access to the underlying data of both images.
  //   auto depthData = reinterpret_cast<uint16_t *>(depthImage.data);
  //   auto coordinateData = reinterpret_cast<float *>(coordinateImage.data);

  //   // Get the number of channels of both images.
  //   const auto depthChannels = depthImage.channels();
  //   const auto coordinateChannels = coordinateImage.channels();

  //   // Iterate over each pixel of the image.
  //   for (int row = 0; row < rows; ++row) {
  //     for (int col = 0; col < cols; ++col) {
  //       // NOTE: These values for each (row, col) can be precomputed from the
  //       // intrinsics.
  //       const float xPrime =
  //           (static_cast<float>(col) - intrinsics.ppx) / intrinsics.fx;
  //       const float yPrime =
  //           (static_cast<float>(row) - intrinsics.ppy) / intrinsics.fy;

  //       // Calculate pointer to each channel of both images at the current
  //       // pixel.
  //       const auto depthP =
  //           depthData + (depthChannels * cols * row) + (depthChannels * col);

  //       const auto xP = coordinateData + (coordinateChannels * cols * row) +
  //                       (coordinateChannels * col);
  //       const auto yP = xP + 1;
  //       const auto zP = xP + 2;

  //       // Calculate pre-transform coordinate values
  //       // *xP = *depthP * xPrime;
  //       // *yP = *depthP * yPrime;
  //       // *zP = *depthP;

  //       // Transform into chair coordinates
  //       const auto &tm = transformationMatrix;

  //       const auto xTmp = *depthP * xPrime;
  //       const auto yTmp = *depthP * yPrime;
  //       const auto zTmp = *depthP;

  //       *xP = tm[0] * xTmp + tm[1] * yTmp + tm[2] * zTmp + tm[3];
  //       *yP = tm[4] * xTmp + tm[5] * yTmp + tm[6] * zTmp + tm[7];
  //       *zP = tm[8] * xTmp + tm[9] * yTmp + tm[10] * zTmp + tm[11];
  //     }
  //   }
  //   // TODO: consider padding the left side here
  //   return coordinateImage;
  // }
}

cv::Mat processDepthImage(const cv::Mat &depthImage) {
  // Validate image.
  assert(depthImage.type() == CV_16U);

  // Dilate to fill some holes.
  // NOTE: This kernel is configurable, and can be precalculated.
  const auto dilateStructuringElement =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::dilate(depthImage, depthImage, dilateStructuringElement);

  // Median blur filter to remove large impulse pixels.
  // NOTE: The kernel size is configurable. With u16 pixels, it cannot go
  // above 5.
  const auto medianKernelSize = 5;
  cv::medianBlur(depthImage, depthImage, medianKernelSize);

  return depthImage;
}

cv::Mat toCvMat(const rs2::depth_frame &rsDepthFrame) {
  const auto width = rsDepthFrame.get_width();
  const auto height = rsDepthFrame.get_height();
  const auto size = cv::Size(width, height);

  auto data = const_cast<void *>(rsDepthFrame.get_data());

  return cv::Mat(size, CV_16U, data);
}

cv::Mat toCvMat(const rs2::video_frame &rsIrFrame) {
  const auto width = rsIrFrame.get_width();
  const auto height = rsIrFrame.get_height();
  const auto size = cv::Size(width, height);

  auto data = const_cast<void *>(rsIrFrame.get_data());

  return cv::Mat(size, CV_8U, data);
}

rs2_intrinsics getIntrinsics(const rs2::pipeline_profile &profile) {
  return profile.get_stream(RS2_STREAM_DEPTH)
      .as<rs2::video_stream_profile>()
      .get_intrinsics();
}

std::pair<bool, std::string> parseArguments(int argc, char **argv) {
  std::stringstream usage;
  usage << "Usage: " << argv[0] << " <file/live> <.bag/realsense.json>"
        << std::endl;
  if (argc != 3) {
    std::cerr << usage.str();
    exit(1);
  }
  bool fromFile;
  if (std::string(argv[1]) == "file") {
    fromFile = true;
  } else if (std::string(argv[1]) == "live") {
    fromFile = false;
  } else {
    std::cerr << usage.str();
    exit(1);
  }

  return std::make_pair(fromFile, argv[2]);
}

void initWindows(const std::vector<std::string> &windows) {
  for (const auto &window : windows) {
    cv::namedWindow(window);
  }
}

rs2::config createConfig(const std::string &filePath, bool fromFile) {
  if (fromFile) {
    auto config = rs2::config();
    config.enable_device_from_file(filePath);
    return config;
  }

  // Get context.
  const auto context = rs2::context();

  // Get single device.
  auto devices = context.query_devices();
  if (devices.size() != 1) {
    std::cerr << "Only supports one device" << std::endl;
    exit(1);
  }
  auto &&device = devices[0];

  // Query device information.
  const auto serialNumber =
      std::string(device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));

  // Parse advanced configuration.
  auto rsConfigFile = std::fstream(filePath);
  if (!rsConfigFile) {
    std::cerr << "Could not parse rs configuration at: " << filePath
              << std::endl;
    exit(1);
  }
  auto rsConfig = std::string(std::istreambuf_iterator<char>(rsConfigFile),
                              std::istreambuf_iterator<char>());

  // Program with advanced configuration.
  auto advancedDevice = device.as<rs400::advanced_mode>();
  if (!advancedDevice.is_enabled()) {
    advancedDevice.toggle_advanced_mode(true);
  }
  advancedDevice.load_json(rsConfig);

  // Create rs config.
  auto config = rs2::config();
  config.enable_device(serialNumber);
  config.enable_stream(RS2_STREAM_DEPTH, 480, 270, RS2_FORMAT_Z16, 15);
  // config.enable_stream(RS2_STREAM_INFRARED, 480, 270, RS2_FORMAT_Y8, 15);

  return config;
}

cv::Mat removeNonEdges(const cv::Mat &edges, const cv::Mat &depths,
                       const cv::Mat &zImage) {
  assert(edges.type() == CV_8U);
  assert(depths.type() == CV_16U);
  assert(zImage.type() == CV_32F);

  auto contours = std::vector<std::vector<cv::Point>>{};
  cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

  auto contourImage = cv::Mat(cv::Mat::zeros(edges.rows, edges.cols, CV_8U));

  cv::drawContours(contourImage, contours, -1, cv::Scalar(100));

  const auto first = std::begin(contours);
  auto last = std::end(contours);

  // last = std::remove_if(first, last, [](const auto& contour) {
  std::for_each(first, last, [&](const auto &contour) {
    // Make a box!
    const auto rectangle = cv::boundingRect(contour);

    // How big is the box?
    const auto perimeter =
        2 * (rectangle.size().width + rectangle.size().height);
    const auto area = rectangle.area();

    // What is the percentage of zero depth values at the box?
    auto numZeroPixels = 0;
    for (int row = rectangle.tl().y; row < rectangle.br().y; ++row) {
      for (int col = rectangle.tl().x; col < rectangle.br().x; ++col) {
        const auto &depth = depths.at<uint16_t>(row, col);
        if (depth == 0 || depth == std::numeric_limits<uint16_t>::max()) {
          numZeroPixels++;
        }
      }
    }
    // Cast to a float to force float division
    const auto numZeroPixelsF = static_cast<float>(numZeroPixels);
    const auto percentZeroPixels = area != 0 ? numZeroPixelsF / area : 0;

    // How far apart are the Zs at the corners of the box?
    const auto topLeft = rectangle.tl();
    const auto bottomRight = rectangle.br();
    const auto topRight = cv::Point(bottomRight.x, topLeft.y);
    const auto bottomLeft = cv::Point(topLeft.x, bottomRight.y);

    const auto cornerValues = std::vector<float>{
        zImage.at<float>(topLeft), zImage.at<float>(bottomRight),
        zImage.at<float>(topRight), zImage.at<float>(bottomLeft)};
    const auto [min, max] =
        std::minmax_element(std::begin(cornerValues), std::end(cornerValues));
    const auto difference = *max - *min;

    // Print the info!
    auto orig = cv::Point(rectangle.tl().x + 10, rectangle.tl().y + 10);
    auto ss = std::stringstream();
    // ss << "A: " << area << " P: " << perimeter << " Z: " <<
    // percentZeroPixels
    // << " D: " << difference;
    ss << area << " " << perimeter << " " << percentZeroPixels << " "
       << difference;
    bool keep = true;
    if (area < 700) {
      keep = false;
    }
    if (perimeter < 180) {
      keep = false;
    }
    // if (!keep) {
    //   return;
    // }
    // Really big edges faced with the "grand canyon problem" will get caught
    // here if we don't limit this check to smaller stuff.
    if (percentZeroPixels > 0.25 && perimeter < 400) {
      keep = false;
    }
    if (difference < 500) {
      keep = false;
    }
    if (!keep) {
      return;
    }

    auto color = cv::Scalar(keep ? 255 : 127);
    cv::putText(contourImage, ss.str(), orig, cv::FONT_HERSHEY_PLAIN, 0.8,
                color);
    // Draw the box!
    cv::rectangle(contourImage, rectangle, color);
  });

  return contourImage;
}
