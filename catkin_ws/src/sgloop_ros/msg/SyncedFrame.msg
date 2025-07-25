Header header                # 使用原图时间戳
sensor_msgs/Image rgb        # 原图(/camera/color/image_raw)
sensor_msgs/Image depth      # 深度图(/camera/depth/image_rect_raw)
sensor_msgs/Image mask       # SAM掩码(/mask_image)
float64[16] transform_matrix # 4x4变换矩阵
string json                  # SAM的JSON(/mask_data)