import opencv


def get_mask_texture(src_image_path, seg_image_path):
    seg_image = imread(seg_image_path)
    src_image = imread(src_image_path)

    # 在PRN模型的帮助下，使用src_image获取位置和顶点
    lms_info = read_info.read_landmark_106_array(src_face_lms)
    pos = prn_model.process(src_image, lms_info)
    vertices = prn_model.get_vertices(pos)

    # 将遮罩纹理映射到UV纹理贴图
    mask_image = mask_image / 255.
    mask_UV = cv2.remap(mask_image, pos[:, :, :2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST,
                        boarderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    mask_UV = mask_UV.astype(np.float32)
    mask_UV = cv2.cvtColor(mask_UV, cv2.COLOR_RGB2RGBA)
    mask_UV[background_pos, 3] = 0
    return mask_UV
