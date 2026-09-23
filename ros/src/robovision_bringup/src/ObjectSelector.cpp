#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <isaac_ros_tensor_list_interfaces/msg/tensor.hpp>
#include <isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace robovision{
class ObjectSelector : public rclcpp::Node {

public:
    explicit ObjectSelector(const rclcpp::NodeOptions& options): Node("object_selector", options)
    {
        this->tensor_sub_ = this->create_subscription<isaac_ros_tensor_list_interfaces::msg::TensorList>(
            "tensor_sub",10, std::bind(&ObjectSelector::tensorCallback, this, std::placeholders::_1));
        
        this->object_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("object_pub",10);

        RCLCPP_INFO(rclcpp::get_logger("rclcpp"),"Object Selector Initialized");
    }

private:
    using Tensor = isaac_ros_tensor_list_interfaces::msg::Tensor;
    using TensorList = isaac_ros_tensor_list_interfaces::msg::TensorList;

    rclcpp::Subscription<TensorList>::SharedPtr tensor_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr object_pub_;

    const Tensor* find_tensor(const TensorList& msg, const std::string& name) const
    {
        for (const auto& tens : msg.tensors)
        {
            if( tens.name == name)
                return &tens;
        }
        return nullptr;
    }

    template<typename T>
    std::vector<T> tensor_to_vector(const Tensor& tensor) const
    {
        if (tensor.data.size() % sizeof(T) != 0)
            throw std::runtime_error("Tensor byte count is incompatible with the requested type");

        const std::size_t count = tensor.data.size() / sizeof(T);

        std::vector<T> result(count);
        std::memcpy(result.data(), tensor.data.data(),tensor.data.size());

        return result;
    }

    void tensorCallback(const TensorList::SharedPtr msg)
    {
        const Tensor * num_tensor = find_tensor(*msg, "num_detections");
        const Tensor * boxes_tensor = find_tensor(*msg, "detection_boxes");
        const Tensor * scores_tensor = find_tensor(*msg, "detection_scores");
        const Tensor * classes_tensor = find_tensor(*msg, "detection_classes");
        

        if (num_tensor == nullptr || boxes_tensor == nullptr || scores_tensor == nullptr || classes_tensor == nullptr)
        {
            RCLCPP_ERROR( rclcpp::get_logger("rclcpp"), "TensorRT output is missing one or more expected tensors");
            return;
        }
        
        const auto num_detections_data = tensor_to_vector<int32_t>(*num_tensor);
        const auto boxes = tensor_to_vector<float>(*boxes_tensor);
        const auto scores = tensor_to_vector<float>(*scores_tensor);
        const auto classes = tensor_to_vector<int32_t>(*classes_tensor);
        
        const int32_t num_detections = num_detections_data.empty() ? 0 : num_detections_data[0];

        if (num_detections <= 0)
        {
            RCLCPP_DEBUG( get_logger(), "No objects detected" );
            return;
        }

        const std::size_t count = std::min<std::size_t>(static_cast<std::size_t>(num_detections), scores.size());
        if (count == 0) return;

        const auto max_score = std::max_element(scores.begin(), scores.begin() + count);
        const std::size_t max_index = std::distance(scores.begin(),max_score);

        const std::size_t box_offset = max_index * 4;

        const float x1 = boxes[box_offset + 0];
        const float y1 = boxes[box_offset + 1];
        const float x2 = boxes[box_offset + 2];
        const float y2 = boxes[box_offset + 3];

        const float score = scores[max_index];
        const int32_t class_id = classes[max_index];

        std_msgs::msg::Float32MultiArray output;
        output.data = {x1, y1, x2, y2, score, static_cast<float>(class_id) };

        this->object_pub_->publish(output);
        RCLCPP_DEBUG( get_logger(), "Main object: class=%d score=%.3f box=[%.1f %.1f %.1f %.1f]", class_id, score, x1, y1, x2, y2); 
    }
};
}


RCLCPP_COMPONENTS_REGISTER_NODE(robovision::ObjectSelector)