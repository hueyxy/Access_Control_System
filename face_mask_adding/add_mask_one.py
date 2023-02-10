from face_masker import FaceMasker

if __name__ == '__main__':
    is_aug = False
    image_path = r'E:\PycharmProjects\FaceTools\face_mask_adding\FMA-3D\hei_11_y_n.jpg'
    face_lms_file = r'E:\PycharmProjects\FaceTools\face_mask_adding\Data\test-data\hei_11_y_n.jpg.txt'
    template_name = '0.png'
    masked_face_path = 'FMA-3D/hei_n.jpg'
    face_lms_str = open(face_lms_file).readline().strip().split(' ')
    face_lms = [float(num) for num in face_lms_str]
    face_masker = FaceMasker(is_aug)
    face_masker.add_mask_one(image_path, face_lms, template_name, masked_face_path)
