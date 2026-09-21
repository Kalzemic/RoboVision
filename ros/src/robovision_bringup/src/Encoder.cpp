#include <rclcpp/rclcpp.hpp>
#include <isaac_ros_tensor_list_interfaces/msg/tensor.hpp>
#include <isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <functional>
#include <vector>
#include <cstring>
#include <utility>


namespace robovision{

class Encoder : public rclcpp::Node
{
public: 
    explicit Encoder(const rclcpp::NodeOptions& options): Node("encoder", options)
    {
        this->sub_ = this->create_subscription<sensor_msgs::msg::Image>
        ("image_sub",10, std::bind(&Encoder::imageCallback, this, std::placeholders::_1));
        
        this->pub_ = this->create_publisher<TensorList>("tensor_pub",10);

        RCLCPP_INFO(get_logger(),"Encoder Initialized");
    }
private:
    using Tensor = isaac_ros_tensor_list_interfaces::msg::Tensor;
    using TensorList = isaac_ros_tensor_list_interfaces::msg::TensorList;

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        
        constexpr int H = 320; constexpr int W = 320; constexpr int C = 3;
        cv::Mat frame(msg->height, msg->width, CV_8UC3, msg->data.data(),msg->step);

        cv::Mat resized; 
        cv::resize(frame, resized, cv::Size(W, H));

        cv::Mat rgb; 
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        
        cv::Mat rgb_float;
        rgb.convertTo( rgb_float, CV_32FC3, 1.0 / 255.0 );

        std::vector<float> chw(C * H * W);


        for(int y = 0; y < H; y++)
        {
            for(int x = 0; x < W; x++)
            {
                const auto pixel = rgb_float.at<cv::Vec3f>(y,x);
                for(int c = 0; c < C; c++)
                {
                    chw[c * H * W + y * W + x] = pixel[c];
                }
            }
        }

        Tensor tensor;

        tensor.name = "images";

        tensor.shape.rank = 4;
        tensor.shape.dims = {1, C, H, W};
        tensor.data_type = 9; // float32

        tensor.data.resize(chw.size() * sizeof(float));
        std::memcpy(tensor.data.data(),chw.data(),tensor.data.size());

        tensor.strides = { C * H * W * sizeof(float), H * W * sizeof(float), W * sizeof(float), sizeof(float)};

        TensorList tensor_list;
        tensor_list.tensors.push_back(std::move(tensor));

        this->pub_->publish(tensor_list);

    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<TensorList>::SharedPtr pub_;

};
}

RCLCPP_COMPONENTS_REGISTER_NODE(robovision::Encoder)