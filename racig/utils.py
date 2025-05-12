import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
def map_segmap_to_color(seg_map):
    # 定义整数到颜色的映射
    unique_vals = np.unique(seg_map)
    color_list = plt.get_cmap('tab20', len(unique_vals)).colors
    color_map = {val: color_list[i] for i, val in enumerate(unique_vals)}

    # 创建一个RGB图像
    color_seg_map = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for val, color in color_map.items():
        color_seg_map[seg_map == val] = (np.array(color[:3]) * 255).astype(np.uint8)
    
    return Image.fromarray(color_seg_map)

def resize_and_crop_image(image, target_width, target_height):
    """Resizes and crops an image to the target width and height."""
    img_aspect_ratio = image.width / image.height
    target_aspect_ratio = target_width / target_height

    # if img_aspect_ratio > target_aspect_ratio:
    #     # Image is wider than target: crop width
    #     new_width = int(target_aspect_ratio * image.height)
    #     left = (image.width - new_width) // 2
    #     right = left + new_width
    #     image = image.crop((left, 0, right, image.height))
    # else:
    #     # Image is taller than target: crop height
    #     new_height = int(image.width / target_aspect_ratio)
    #     top = (image.height - new_height) // 2
    #     bottom = top + new_height
    #     image = image.crop((0, top, image.width, bottom))

    return image.resize((target_width, target_height))

def pil_images_to_matplotlib_figure(pil_images, orientation='Horizontal', title_list=None):
    # 获取每个图像的宽度和高度（像素）
    widths, heights = zip(*(img.size for img in pil_images))
    # 获取每个图像的dpi
    dpis = [img.info.get('dpi', (72, 72)) for img in pil_images]
    
    # 确保所有图像的DPI一致
    assert all(dpi == dpis[0] for dpi in dpis), "All images must have the same DPI"
    dpi = dpis[-1][0]  # 使用最后一个图像的DPI

    # 检查标题列表是否提供，并且长度是否与图像列表匹配
    if title_list is not None:
        assert len(title_list) == len(pil_images), "Length of title_list must match number of images"

    # 将图像大小转换为英寸
    inch_widths = [w / dpi for w in widths]
    inch_heights = [h / dpi for h in heights]
    
    max_images_per_row = 5
    
    if orientation == 'Horizontal':
        # 计算行数和列数
        num_images = len(pil_images)
        num_rows = (num_images + max_images_per_row - 1) // max_images_per_row
        num_cols = min(max_images_per_row, num_images)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(sum(inch_widths[:num_cols]), num_rows * max(inch_heights)), dpi=dpi)
        
        if num_rows == 1:
            axes = [axes]  # 如果只有一行，确保axes是一个列表
        
        axes = np.array(axes).reshape(num_rows, num_cols)
        
    elif orientation == 'Vertical':
        # 竖向排列时，将两张图片裁剪并拼接为指定尺寸 1344*768
        assert len(pil_images) == 2, "Vertical orientation requires exactly 2 images."

        target_width = 768
        target_height = 1344

        # Resize and crop images to desired sizes
        img1 = resize_and_crop_image(pil_images[0], target_width, int(target_height * 0.45))
        img2 = resize_and_crop_image(pil_images[1], target_width, int(target_height * 0.55))

        # Create a new image with the target dimensions
        concatenated_image = Image.new('RGB', (target_width, target_height))

        # Paste the two images into the new image
        concatenated_image.paste(img1, (0, 0))
        concatenated_image.paste(img2, (0, int(target_height * 0.45)))

        # 返回拼接后的图片
        return concatenated_image
    
    else:
        raise NotImplementedError("Unsupported orientation provided.")
    
    # 绘制每张图像
    for idx, img in enumerate(pil_images):
        row = idx // max_images_per_row
        col = idx % max_images_per_row
        ax = axes[row, col] if num_rows > 1 else axes[col]
        
        img_array = np.array(img)
        unique_values = np.unique(img_array)
        
        # if len(unique_values) <= 4 and len(unique_values) >= 2:  # 判断是否为分割图像
        #     ax.imshow(map_segmap_to_color(img_array))  # 使用不同的颜色图
        # else:
        ax.imshow(img_array)
        
        ax.axis('off')
        
        # 如果提供了标题列表，添加标题
        if title_list is not None:
            ax.set_title(title_list[idx])

    # 隐藏未使用的轴
    for r in range(num_rows):
        for c in range(num_cols):
            if r * max_images_per_row + c >= len(pil_images):
                axes[r, c].axis('off')

    plt.tight_layout()
    plt.close()
    return fig

def pil_images_concatenate(pil_images, orientation='Horizontal'):
    # 获取每个图像的宽度和高度
    widths, heights = zip(*(img.size for img in pil_images))
    
    # 检查图像是否可能是分割图像，并应用颜色映射
    processed_images = []
    for img in pil_images:
        img_array = np.array(img)
        unique_values = np.unique(img_array)
        
        # 判断是否为分割图像（例如只有少数几个不同的值）
        if len(unique_values) <= 4 and len(unique_values) >= 2:
            # 使用颜色映射将分割图像转换为彩色图像
            processed_images.append(map_segmap_to_color(img_array))
        else:
            # 非分割图像保持不变
            processed_images.append(img)
    
    # 根据选择的方向拼接图像
    if orientation == 'Horizontal':
        # 计算拼接后的宽度和最大高度
        total_width = sum(img.width for img in processed_images)
        max_height = max(img.height for img in processed_images)
        # 创建新的空白图像
        new_image = Image.new('RGB', (total_width, max_height))
        
        # 将每个图像依次粘贴到新图像中
        current_x = 0
        for img in processed_images:
            new_image.paste(img, (current_x, 0))
            current_x += img.width
    else:
        # 计算拼接后的总高度和最大宽度
        total_height = sum(img.height for img in processed_images)
        max_width = max(img.width for img in processed_images)
        # 创建新的空白图像
        new_image = Image.new('RGB', (max_width, total_height))
        
        # 将每个图像依次粘贴到新图像中
        current_y = 0
        for img in processed_images:
            new_image.paste(img, (0, current_y))
            current_y += img.height

    return new_image


def get_center(mask):
    """计算mask的中心点"""
    coords = torch.nonzero(mask)
    center = coords.float().mean(dim=0)
    return center

def assign_by_nearest(mask0, mask1, intersection):
    """根据最近邻方法将交集部分分配给mask0或mask1"""
    center0 = get_center(mask0)
    center1 = get_center(mask1)
    
    coords_intersection = torch.nonzero(intersection)
    dist0 = torch.norm(coords_intersection.float() - center0, dim=1)
    dist1 = torch.norm(coords_intersection.float() - center1, dim=1)
    
    mask0_assign = dist0 < dist1
    mask1_assign = ~mask0_assign
    
    mask0[coords_intersection[mask0_assign][:, 0], coords_intersection[mask0_assign][:, 1], coords_intersection[mask0_assign][:, 2]] = 1
    mask1[coords_intersection[mask1_assign][:, 0], coords_intersection[mask1_assign][:, 1], coords_intersection[mask1_assign][:, 2]] = 1
    
    return mask0, mask1

def draw_bodypose_openpose(canvas: np.ndarray, keypoints: np.ndarray, min_conf: float) -> np.ndarray:
    H, W, C = canvas.shape
    # automatically adjust the thickness of the skeletons
    if max(W, H) < 500:
        ratio = 1.0
    elif max(W, H) >= 500 and max(W, H) < 1000:
        ratio = 2.0
    elif max(W, H) >= 1000 and max(W, H) < 2000:
        ratio = 3.0
    elif max(W, H) >= 2000 and max(W, H) < 3000:
        ratio = 4.0
    elif max(W, H) >= 3000 and max(W, H) < 4000:
        ratio = 5.0
    elif max(W, H) >= 4000 and max(W, H) < 5000:
        ratio = 6.0
    else:
        ratio = 7.0
    stickwidth = 4
    # connections and colors
    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
        [1, 16], [16, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],[85, 255, 0], 
            [0, 255, 0],[0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],[0, 85, 255], 
            [0, 0, 255], [85, 0, 255], [170, 0,255], [255, 0, 255],[255, 0, 170], [255, 0, 85]]

    # draw the links
    for (k1, k2), color in zip(limbSeq, colors):
        cur_canvas = canvas.copy()
        keypoint1 = keypoints[k1-1, :]
        keypoint2 = keypoints[k2-1, :]

        if keypoint1[-1] < min_conf or keypoint2[-1] < min_conf:
            continue

        Y = np.array([keypoint1[0], keypoint2[0]]) * float(W)
        X = np.array([keypoint1[1], keypoint2[1]]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        import math
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), int(stickwidth * ratio)), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, [int(float(c)) for c in color])
        canvas = cv2.addWeighted(canvas, 0.3, cur_canvas, 0.7, 0)
    # draw the points
    for id, color in zip(range(keypoints.shape[0]), colors):
        keypoint = keypoints[id, :]
        if keypoint[-1]<min_conf:
            continue

        x, y = keypoint[0], keypoint[1]
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), int(4 * ratio), color, thickness=-1)

    return canvas


def resize_pose(pose_img, height, width):
    """
    Resize the pose image to the given height and width, maintaining aspect ratio,
    and pad it to fit the target size.
    """
    H, W, _ = pose_img.shape
    ratio = min(height / H, width / W)
    new_size = (int(W * ratio), int(H * ratio))

    pose_img = Image.fromarray(pose_img.astype(np.uint8))
    pose_img = pose_img.resize(new_size, Image.BILINEAR)
    pose_img = np.array(pose_img)

    pad_h = (height - new_size[1]) // 2
    pad_w = (width - new_size[0]) // 2

    pose_img = np.pad(pose_img, ((pad_h, height - new_size[1] - pad_h), (pad_w, width - new_size[0] - pad_w), (0, 0)), mode='constant', constant_values=0)
    
    return pose_img

def resize_segmap(seg_map, height, width):
    """
    Resize the segmentation map to the given height and width, maintaining aspect ratio,
    and pad it to fit the target size.
    """
    H, W = seg_map.shape
    ratio = min(height / H, width / W)
    new_size = (int(W * ratio), int(H * ratio))

    seg_map = Image.fromarray(seg_map)
    seg_map = seg_map.resize(new_size, Image.NEAREST)
    seg_map = np.array(seg_map)

    pad_h = (height - new_size[1]) // 2
    pad_w = (width - new_size[0]) // 2

    seg_map = np.pad(seg_map, ((pad_h, height - new_size[1] - pad_h), (pad_w, width - new_size[0] - pad_w)), mode='constant', constant_values=255)
    
    return seg_map

def expand_bbox(img, bbox, expansion_factor=0.2):
    """
    Expand the bounding box by a given factor while ensuring it stays within the image boundaries.
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Calculate width and height of the original bbox
    width = x2 - x1
    height = y2 - y1

    # Increase width and height by the expansion factor
    new_width = int(width * (1 + expansion_factor))
    new_height = int(height * (1 + expansion_factor))

    # Calculate new x1, y1, x2, y2 ensuring the bbox stays within image boundaries
    new_x1 = max(0, x1 - (new_width - width) // 2)
    new_y1 = max(0, y1 - (new_height - height) // 2)
    new_x2 = min(img.width, x2 + (new_width - width) // 2)
    new_y2 = min(img.height, y2 + (new_height - height) // 2)

    # Ensure new_x2 and new_y2 are not smaller than new_x1 and new_y1
    new_x2 = max(new_x1 + 1, new_x2)
    new_y2 = max(new_y1 + 1, new_y2)

    return [new_x1, new_y1, new_x2, new_y2]

def draw_face_mask(keypoints, H, W, ratio=0.6):
    """
    Draw a face mask based on the keypoints. 
    keypoints: 0-1 shape: (18,3) openpose
    H, W: int
    """
    # 0: 鼻子 1: 脖子 2: 左肩 3: 左肘 4: 左手 5: 右肩 6: 右肘 7: 右手
    # 16: 左耳 17: 右耳

    keypoints_denormalized = keypoints[:,:2] * np.array([W, H])
    confidence = keypoints[:,2]
    neck_coords = keypoints_denormalized[1]
    nose_coords = keypoints_denormalized[0]
    left_eye_coords = keypoints_denormalized[14]
    right_eye_coords = keypoints_denormalized[15]
    left_ear_coords = keypoints_denormalized[16]
    right_ear_coords = keypoints_denormalized[17]

    # 计算脸部的半径
    relevant_points = [nose_coords, left_ear_coords, right_ear_coords, left_eye_coords, right_eye_coords]
    radius = 0
    for p1 in relevant_points:
        for p2 in relevant_points:
            radius = max(radius, np.linalg.norm(p1-p2)/2)

    # 检测是正脸还是侧脸：
    is_side_face = False
    if nose_coords[0] <= min(left_ear_coords[0], right_ear_coords[0]) or nose_coords[0] >= max(left_ear_coords[0], right_ear_coords[0]): 
        is_side_face = True
    if np.linalg.norm(left_ear_coords - right_ear_coords) < np.linalg.norm((left_ear_coords+right_ear_coords)/2 - nose_coords):
        is_side_face = True
    if confidence[16] < 0.1 or confidence[17] < 0.1:
        is_side_face = True
    
    # 确定圆心位置
    if  is_side_face: # 侧脸
        ear_coords = left_ear_coords if confidence[16] > confidence[17] else right_ear_coords
        circle_center = (nose_coords + ear_coords) / 2
    else: # 正脸
        circle_center = nose_coords

    # 创建一个空白的mask
    face_mask = np.zeros((H, W), dtype=np.uint8)

    # 画圆填充为1
    cv2.circle(face_mask, (int(circle_center[0]), int(circle_center[1])), int(1.2*radius), 1, -1)
    # 再以眼睛为中心，眼睛到鼻子为半径画圆
    cv2.circle(face_mask, (int(left_eye_coords[0]), int(left_eye_coords[1])), int(np.linalg.norm(left_eye_coords - nose_coords)), 1, -1)
    cv2.circle(face_mask, (int(right_eye_coords[0]), int(right_eye_coords[1])), int(np.linalg.norm(right_eye_coords - nose_coords)), 1, -1)
    
    return torch.tensor(face_mask, dtype=torch.float)

def draw_mask_coarse(keypoints, H, W, face_radius=1.5):
    mask = np.zeros((H, W))
    points = np.array([(int(kp[0]*W), int(kp[1]*H)) for kp in keypoints], dtype=np.int32)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 1)
    head_face_mask = draw_face_mask(keypoints, H, W, radius=face_radius)
    mask = (np.array(mask) + np.array(head_face_mask)) > 0
    return mask

def draw_mask_fine(keypoints, H, W, face_ratio=0.6):
    """
    keypoints: (18, 3) openpose format
    """
    mask = np.zeros((H, W))
    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
        [1, 16], [16, 18]]

    for k1, k2 in limbSeq:
        cur_canvas = mask.copy()
        keypoint1 = keypoints[k1-1, :]
        keypoint2 = keypoints[k2-1, :]

        Y = np.array([keypoint1[0], keypoint2[0]]) * float(W)
        X = np.array([keypoint1[1], keypoint2[1]]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        import math
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 1.3), int(35)), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, 1)
        mask = cv2.addWeighted(mask, 0.3, cur_canvas, 0.7, 0)
    
    # fill body: link index 2,8,5,11 points
    # 画一个四边形连接这几个点作为身体
    body_points = np.array([
        (int(keypoints[8, 0] * W), int(keypoints[8, 1] * H)),
        (int(keypoints[11, 0] * W), int(keypoints[11, 1] * H)),
        (int(keypoints[5, 0] * W), int(keypoints[5, 1] * H)),
        (int(keypoints[2, 0] * W), int(keypoints[2, 1] * H))
    ], dtype=np.int32)
    cv2.fillConvexPoly(mask, body_points, 1)

    # fill thigh: link index mid, 8 , 9 and mid, 11 ,12
    mid_points = (keypoints[8] + keypoints[11]) / 2
    thigh_points_1 = np.array([
        (int(mid_points[0] * W), int(mid_points[1] * H)),
        (int(keypoints[8, 0] * W), int(keypoints[8, 1] * H)),
        (int(keypoints[9, 0] * W), int(keypoints[9, 1] * H))
    ], dtype=np.int32)
    cv2.fillConvexPoly(mask, thigh_points_1, 1)

    thigh_points_2 = np.array([
        (int(mid_points[0] * W), int(mid_points[1] * H)),
        (int(keypoints[11, 0] * W), int(keypoints[11, 1] * H)),
        (int(keypoints[12, 0] * W), int(keypoints[12, 1] * H))
    ], dtype=np.int32)
    cv2.fillConvexPoly(mask, thigh_points_2, 1)


    head_face_mask = draw_face_mask(keypoints, H, W, ratio=face_ratio)
    mask = (np.array(mask) + np.array(head_face_mask)) > 0

    return mask