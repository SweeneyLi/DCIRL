# -*- coding: utf-8 -*-
# add the probability of emotion
# change the method of align from cv to face alignment
import urllib3
import base64
from urllib.parse import urlencode
import time
import numpy as np
import cv2
import os
from multiprocessing.pool import ThreadPool
import pandas as pd
import logging
import argparse
import requests
import face_alignment
from skimage import transform as trans

urllib3.disable_warnings()

env_linux = True

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--dataset_name', type=str, required=True)
args = parser.parse_args()
dataset_name = args.dataset_name

print(dataset_name)

if env_linux:
    dataset_loc = '/home/lixy/My_Dataset/Data/%s' % dataset_name
    dataset_img_loc = '/home/lixy/My_Dataset/Data/%s/%s_aligned' % (dataset_name, dataset_name)
    root_dir = dataset_loc
    aligned_dir = os.path.join(root_dir, '%s_aligned_new' % dataset_name)
else:
    dataset_loc = 'E:\Workspace\Datasets\%s' % dataset_name
    dataset_img_loc = 'E:\Workspace\Datasets\%s\images' % dataset_name
    root_dir = 'E:\Workspace\Data\%s' % dataset_name
    aligned_dir = os.path.join(root_dir, '%s_aligned' % dataset_name)

########################################################################################

http = urllib3.PoolManager()
########################################################################################

emotion_dict = {
    "angry": 0,
    "disgust": 0,
    "fear": 0,
    "happy": 0,
    "sad": 0,
    "surprise": 0,
    "neutral": 0,
    # "pouty": 0,
    # "grimace": 0,
}

emotion_dict_num = {
    "angry": -1,
    "disgust": -1,
    "fear": -1,
    "happy": -1,
    "sad": -1,
    "surprise": -1,
    "neutral": -1,
    # "pouty": 0,
    # "grimace": 0,
}

COLUMNS = [
    "img_name",
    "angle_pitch",
    "angle_yaw",
    "age",
    "gender_type",
    "eye_status_left_eye",
    "eye_status_right_eye",
    "emotion_type",
    "emotion_probability",
    'cheek_left_1_x',
    'cheek_left_1_y',
    'cheek_left_10_x',
    'cheek_left_10_y',
    'cheek_left_11_x',
    'cheek_left_11_y',
    'cheek_left_2_x',
    'cheek_left_2_y',
    'cheek_left_3_x',
    'cheek_left_3_y',
    'cheek_left_4_x',
    'cheek_left_4_y',
    'cheek_left_5_x',
    'cheek_left_5_y',
    'cheek_left_6_x',
    'cheek_left_6_y',
    'cheek_left_7_x',
    'cheek_left_7_y',
    'cheek_left_8_x',
    'cheek_left_8_y',
    'cheek_left_9_x',
    'cheek_left_9_y',
    'cheek_right_1_x',
    'cheek_right_1_y',
    'cheek_right_10_x',
    'cheek_right_10_y',
    'cheek_right_11_x',
    'cheek_right_11_y',
    'cheek_right_2_x',
    'cheek_right_2_y',
    'cheek_right_3_x',
    'cheek_right_3_y',
    'cheek_right_4_x',
    'cheek_right_4_y',
    'cheek_right_5_x',
    'cheek_right_5_y',
    'cheek_right_6_x',
    'cheek_right_6_y',
    'cheek_right_7_x',
    'cheek_right_7_y',
    'cheek_right_8_x',
    'cheek_right_8_y',
    'cheek_right_9_x',
    'cheek_right_9_y',
    'chin_1_x',
    'chin_1_y',
    'chin_2_x',
    'chin_2_y',
    'chin_3_x',
    'chin_3_y',
    'eye_left_corner_left_x',
    'eye_left_corner_left_y',
    'eye_left_corner_right_x',
    'eye_left_corner_right_y',
    'eye_left_eyeball_center_x',
    'eye_left_eyeball_center_y',
    'eye_left_eyeball_left_x',
    'eye_left_eyeball_left_y',
    'eye_left_eyeball_right_x',
    'eye_left_eyeball_right_y',
    'eye_left_eyelid_lower_1_x',
    'eye_left_eyelid_lower_1_y',
    'eye_left_eyelid_lower_2_x',
    'eye_left_eyelid_lower_2_y',
    'eye_left_eyelid_lower_3_x',
    'eye_left_eyelid_lower_3_y',
    'eye_left_eyelid_lower_4_x',
    'eye_left_eyelid_lower_4_y',
    'eye_left_eyelid_lower_5_x',
    'eye_left_eyelid_lower_5_y',
    'eye_left_eyelid_lower_6_x',
    'eye_left_eyelid_lower_6_y',
    'eye_left_eyelid_lower_7_x',
    'eye_left_eyelid_lower_7_y',
    'eye_left_eyelid_upper_1_x',
    'eye_left_eyelid_upper_1_y',
    'eye_left_eyelid_upper_2_x',
    'eye_left_eyelid_upper_2_y',
    'eye_left_eyelid_upper_3_x',
    'eye_left_eyelid_upper_3_y',
    'eye_left_eyelid_upper_4_x',
    'eye_left_eyelid_upper_4_y',
    'eye_left_eyelid_upper_5_x',
    'eye_left_eyelid_upper_5_y',
    'eye_left_eyelid_upper_6_x',
    'eye_left_eyelid_upper_6_y',
    'eye_left_eyelid_upper_7_x',
    'eye_left_eyelid_upper_7_y',
    'eye_right_corner_left_x',
    'eye_right_corner_left_y',
    'eye_right_corner_right_x',
    'eye_right_corner_right_y',
    'eye_right_eyeball_center_x',
    'eye_right_eyeball_center_y',
    'eye_right_eyeball_left_x',
    'eye_right_eyeball_left_y',
    'eye_right_eyeball_right_x',
    'eye_right_eyeball_right_y',
    'eye_right_eyelid_lower_1_x',
    'eye_right_eyelid_lower_1_y',
    'eye_right_eyelid_lower_2_x',
    'eye_right_eyelid_lower_2_y',
    'eye_right_eyelid_lower_3_x',
    'eye_right_eyelid_lower_3_y',
    'eye_right_eyelid_lower_4_x',
    'eye_right_eyelid_lower_4_y',
    'eye_right_eyelid_lower_5_x',
    'eye_right_eyelid_lower_5_y',
    'eye_right_eyelid_lower_6_x',
    'eye_right_eyelid_lower_6_y',
    'eye_right_eyelid_lower_7_x',
    'eye_right_eyelid_lower_7_y',
    'eye_right_eyelid_upper_1_x',
    'eye_right_eyelid_upper_1_y',
    'eye_right_eyelid_upper_2_x',
    'eye_right_eyelid_upper_2_y',
    'eye_right_eyelid_upper_3_x',
    'eye_right_eyelid_upper_3_y',
    'eye_right_eyelid_upper_4_x',
    'eye_right_eyelid_upper_4_y',
    'eye_right_eyelid_upper_5_x',
    'eye_right_eyelid_upper_5_y',
    'eye_right_eyelid_upper_6_x',
    'eye_right_eyelid_upper_6_y',
    'eye_right_eyelid_upper_7_x',
    'eye_right_eyelid_upper_7_y',
    'eyebrow_left_corner_left_x',
    'eyebrow_left_corner_left_y',
    'eyebrow_left_corner_right_x',
    'eyebrow_left_corner_right_y',
    'eyebrow_left_lower_1_x',
    'eyebrow_left_lower_1_y',
    'eyebrow_left_lower_2_x',
    'eyebrow_left_lower_2_y',
    'eyebrow_left_lower_3_x',
    'eyebrow_left_lower_3_y',
    'eyebrow_left_upper_1_x',
    'eyebrow_left_upper_1_y',
    'eyebrow_left_upper_2_x',
    'eyebrow_left_upper_2_y',
    'eyebrow_left_upper_3_x',
    'eyebrow_left_upper_3_y',
    'eyebrow_left_upper_4_x',
    'eyebrow_left_upper_4_y',
    'eyebrow_left_upper_5_x',
    'eyebrow_left_upper_5_y',
    'eyebrow_right_corner_left_x',
    'eyebrow_right_corner_left_y',
    'eyebrow_right_corner_right_x',
    'eyebrow_right_corner_right_y',
    'eyebrow_right_lower_1_x',
    'eyebrow_right_lower_1_y',
    'eyebrow_right_lower_2_x',
    'eyebrow_right_lower_2_y',
    'eyebrow_right_lower_3_x',
    'eyebrow_right_lower_3_y',
    'eyebrow_right_upper_1_x',
    'eyebrow_right_upper_1_y',
    'eyebrow_right_upper_2_x',
    'eyebrow_right_upper_2_y',
    'eyebrow_right_upper_3_x',
    'eyebrow_right_upper_3_y',
    'eyebrow_right_upper_4_x',
    'eyebrow_right_upper_4_y',
    'eyebrow_right_upper_5_x',
    'eyebrow_right_upper_5_y',
    'mouth_corner_left_inner_x',
    'mouth_corner_left_inner_y',
    'mouth_corner_left_outer_x',
    'mouth_corner_left_outer_y',
    'mouth_corner_right_inner_x',
    'mouth_corner_right_inner_y',
    'mouth_corner_right_outer_x',
    'mouth_corner_right_outer_y',
    'mouth_lip_lower_inner_1_x',
    'mouth_lip_lower_inner_1_y',
    'mouth_lip_lower_inner_10_x',
    'mouth_lip_lower_inner_10_y',
    'mouth_lip_lower_inner_11_x',
    'mouth_lip_lower_inner_11_y',
    'mouth_lip_lower_inner_2_x',
    'mouth_lip_lower_inner_2_y',
    'mouth_lip_lower_inner_3_x',
    'mouth_lip_lower_inner_3_y',
    'mouth_lip_lower_inner_4_x',
    'mouth_lip_lower_inner_4_y',
    'mouth_lip_lower_inner_5_x',
    'mouth_lip_lower_inner_5_y',
    'mouth_lip_lower_inner_6_x',
    'mouth_lip_lower_inner_6_y',
    'mouth_lip_lower_inner_7_x',
    'mouth_lip_lower_inner_7_y',
    'mouth_lip_lower_inner_8_x',
    'mouth_lip_lower_inner_8_y',
    'mouth_lip_lower_inner_9_x',
    'mouth_lip_lower_inner_9_y',
    'mouth_lip_lower_outer_1_x',
    'mouth_lip_lower_outer_1_y',
    'mouth_lip_lower_outer_10_x',
    'mouth_lip_lower_outer_10_y',
    'mouth_lip_lower_outer_11_x',
    'mouth_lip_lower_outer_11_y',
    'mouth_lip_lower_outer_2_x',
    'mouth_lip_lower_outer_2_y',
    'mouth_lip_lower_outer_3_x',
    'mouth_lip_lower_outer_3_y',
    'mouth_lip_lower_outer_4_x',
    'mouth_lip_lower_outer_4_y',
    'mouth_lip_lower_outer_5_x',
    'mouth_lip_lower_outer_5_y',
    'mouth_lip_lower_outer_6_x',
    'mouth_lip_lower_outer_6_y',
    'mouth_lip_lower_outer_7_x',
    'mouth_lip_lower_outer_7_y',
    'mouth_lip_lower_outer_8_x',
    'mouth_lip_lower_outer_8_y',
    'mouth_lip_lower_outer_9_x',
    'mouth_lip_lower_outer_9_y',
    'mouth_lip_upper_inner_1_x',
    'mouth_lip_upper_inner_1_y',
    'mouth_lip_upper_inner_10_x',
    'mouth_lip_upper_inner_10_y',
    'mouth_lip_upper_inner_11_x',
    'mouth_lip_upper_inner_11_y',
    'mouth_lip_upper_inner_2_x',
    'mouth_lip_upper_inner_2_y',
    'mouth_lip_upper_inner_3_x',
    'mouth_lip_upper_inner_3_y',
    'mouth_lip_upper_inner_4_x',
    'mouth_lip_upper_inner_4_y',
    'mouth_lip_upper_inner_5_x',
    'mouth_lip_upper_inner_5_y',
    'mouth_lip_upper_inner_6_x',
    'mouth_lip_upper_inner_6_y',
    'mouth_lip_upper_inner_7_x',
    'mouth_lip_upper_inner_7_y',
    'mouth_lip_upper_inner_8_x',
    'mouth_lip_upper_inner_8_y',
    'mouth_lip_upper_inner_9_x',
    'mouth_lip_upper_inner_9_y',
    'mouth_lip_upper_outer_1_x',
    'mouth_lip_upper_outer_1_y',
    'mouth_lip_upper_outer_10_x',
    'mouth_lip_upper_outer_10_y',
    'mouth_lip_upper_outer_11_x',
    'mouth_lip_upper_outer_11_y',
    'mouth_lip_upper_outer_2_x',
    'mouth_lip_upper_outer_2_y',
    'mouth_lip_upper_outer_3_x',
    'mouth_lip_upper_outer_3_y',
    'mouth_lip_upper_outer_4_x',
    'mouth_lip_upper_outer_4_y',
    'mouth_lip_upper_outer_5_x',
    'mouth_lip_upper_outer_5_y',
    'mouth_lip_upper_outer_6_x',
    'mouth_lip_upper_outer_6_y',
    'mouth_lip_upper_outer_7_x',
    'mouth_lip_upper_outer_7_y',
    'mouth_lip_upper_outer_8_x',
    'mouth_lip_upper_outer_8_y',
    'mouth_lip_upper_outer_9_x',
    'mouth_lip_upper_outer_9_y',
    'nose_bridge_1_x',
    'nose_bridge_1_y',
    'nose_bridge_2_x',
    'nose_bridge_2_y',
    'nose_bridge_3_x',
    'nose_bridge_3_y',
    'nose_left_contour_1_x',
    'nose_left_contour_1_y',
    'nose_left_contour_2_x',
    'nose_left_contour_2_y',
    'nose_left_contour_3_x',
    'nose_left_contour_3_y',
    'nose_left_contour_4_x',
    'nose_left_contour_4_y',
    'nose_left_contour_5_x',
    'nose_left_contour_5_y',
    'nose_left_contour_6_x',
    'nose_left_contour_6_y',
    'nose_left_contour_7_x',
    'nose_left_contour_7_y',
    'nose_middle_contour_x',
    'nose_middle_contour_y',
    'nose_right_contour_1_x',
    'nose_right_contour_1_y',
    'nose_right_contour_2_x',
    'nose_right_contour_2_y',
    'nose_right_contour_3_x',
    'nose_right_contour_3_y',
    'nose_right_contour_4_x',
    'nose_right_contour_4_y',
    'nose_right_contour_5_x',
    'nose_right_contour_5_y',
    'nose_right_contour_6_x',
    'nose_right_contour_6_y',
    'nose_right_contour_7_x',
    'nose_right_contour_7_y',
    'nose_tip_x',
    'nose_tip_y'
]

FACE_SIZE = (256, 256)
calib = np.array([[90.81143, 116.88686],
                  [165.18857, 116.88686],
                  [128., 156.55542],
                  [97.0583, 198.88002],
                  [158.94173, 198.88002]], dtype="float32")

if not os.path.exists(root_dir):
    os.mkdir(root_dir)

if not os.path.exists(aligned_dir):
    os.mkdir(aligned_dir)

easy_data_loc = os.path.join(root_dir, dataset_name + '_easy_data_%d.csv')
hard_data_loc = os.path.join(root_dir, dataset_name + '_hard_data_%d.csv')
########################################################################################
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
log_dir = os.path.join(root_dir, '%s.log' % dataset_name)
if os.path.exists(log_dir):
    os.remove(log_dir)

# DEBUG
# INFO
# NOTICE
# WARNING
# ERROR
# CRITICAL
logging.basicConfig(filename=log_dir, level=logging.DEBUG, format=LOG_FORMAT,
                    datefmt=DATE_FORMAT)

start = time.time()
token_index = 0
token_info_list = [
    # Qiuchengyang
    ('zbaepQRIStmUoxnGjeM7RpVy', '2DTvk09y6M30b3gkfqEpNLFogbHycEDh'),
    # Jiangsong
    ('vwfsmcdxfqretbp3d1nz7U9Z', 'iWSGctFFMx9EdV9RZBB80f7zZqHR7b6k'),
    # Mine
    ('GhdpeLBBaIfTcQf7F36yh0Iq', 'ljKXlpxK2rHIAjZ30DrFq7pUK77bbPmY'),
    # quancx
    ('NZQTmuBBPqslMU0W6R8knlzQ', 'Tpx0x33sB6PHB5EoLBqCeBaeag2gKOK7'),
    # zhuda
    ('9yUmxAfcV4LY8Z0ri2NrBf0o', 'ikGrVXjpKwMB5kY29Gs1pNYH80MwXUHx'),
    # chenweida
    ('ohuZjHOM7IHW7EnojyIki6R8', 'gfcMLFP70A1suZ3HS7ZaWggmXnu7NEBg'),
    # caifei
    ('eeGv5pGY2ub922gAVk1rd8rP', 'W5nk7wkv9oGwziL822uyRxtOGh18wWSw'),
    # sunchao
    ('QFxSZtDHvEbqVr726g4vhcCE', 'OLwqMhASiM71dwRQsqWXXGq4fWMpRInw'),
    # wukefan
    ('WOVaba2ODiIFZL3Z53UEPSix', 'vlyBOYStf5UVM3v6nGvhY0hZw9F9YZkE'),

    # lijiangming
    ('z3KyEi1ghpy7HYBPuMCxRakj', 'Uau87nk2ObkGsVd8FrpE6k5U3y4hmVdR'),
    # liuxiaolong
    ('ze6NYmvDbyjHyNe36CVXcDuC', 'gUsGbjj2pOPGBFAknO1T0zG8kRw1j6ay'),
    # wangjun
    ('B8PAT0pwzI112G7YubNENGds', 'A8kSrbvcWtSwViyG68bSrzu0A0impVRG'),
]
access_token_list = []


def get_access_token(client_id, client_secret):
    params = (
        ('grant_type', 'client_credentials'),
        ('client_id', client_id),
        ('client_secret', client_secret),
    )
    try:
        response = requests.get('https://aip.baidubce.com/oauth/2.0/token', params=params, verify=False)
        result = eval(str(response.content, 'utf-8'))
    except Exception as e:
        logging.critical(e)
        return None

    return result['access_token']


def cv2_base64(image):
    base64_str = cv2.imencode('.jpg', image)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return base64_str


def get_baidu_info(img_loc, base64_img):
    error_code = 18
    global token_index
    while error_code == 18:
        url = 'https://aip.baidubce.com/rest/2.0/face/v3/detect?access_token=' + access_token_list[token_index]
        params = {
            'image': '' + str(base64_img, 'utf-8') + '',
            'image_type': 'BASE64',
            'face_field': 'age,gender,eye_status,emotion,face_type,landmark150,quality'}
        params = urlencode(params)
        request = http.request('POST',
                               url,
                               body=params,
                               headers={'Content-Type': 'application/json'})
        result = eval(str(request.data, 'utf-8').replace('null', 'None'))
        error_code = result['error_code']
        if error_code == 18:
            logging.debug("Now is using the %d token!" % token_index)
            logging.debug(result['error_msg'])
            token_index = (token_index + 1) % len(access_token_list)
        elif error_code == 111:
            logging.info(result['error_msg'])
            return None

    if result['error_code'] != 0:
        token_index = (token_index + 1) % len(access_token_list)
        logging.info(
            "This request have error! error code: %d,  error: %s" % (result['error_code'], result['error_msg']))
        logging.info("This image: %s is bad!" % img_loc)
        return None

    face_info = result['result']['face_list'][0]
    if not filter_img(face_info):
        logging.info('This image: %s is bad!' % img_loc)
        return None
    return face_info


def get_special_info(img_loc, face_info):
    # collect the speical info
    res = ['%s_%s' % (dataset_name, os.path.basename(img_loc)),
           face_info['angle']['yaw'],
           face_info['angle']['pitch'],
           face_info['age'],
           0 if face_info['gender']['type'] == 'male' else 1,
           face_info['eye_status']['left_eye'],
           face_info['eye_status']['right_eye'],
           face_info['emotion']['type'],
           face_info['emotion']['probability']]

    for k, v in face_info['landmark150'].items():
        res.append(v['x'])
        res.append(v['y'])
    return res


def filter_img(face_info):
    if face_info['emotion']['type'] not in emotion_dict.keys():
        logging.info('This emotion type is wrong : %s' % (
            face_info['emotion']['type']))
        return False

    if emotion_dict_num[face_info['emotion']['type']] != -1 and emotion_dict[face_info['emotion']['type']] >= \
            emotion_dict_num[face_info['emotion']['type']]:
        logging.info('This emotion type is enough : %s' % (
            face_info['emotion']['type']))
        return False

    if face_info['location']['height'] < 100 and face_info['location']['width'] < 100:
        logging.info('This face is too small, its height:%.2f, width:%.2f' % (
            face_info['location']['height'], face_info['location']['width']))
        return False

    if face_info['face_probability'] < 0.8:
        logging.info('This face probaility is too low: %f' % face_info['face_probability'])
        return False

    if abs(face_info['angle']['yaw']) > 50 or abs(face_info['angle']['pitch']) > 50:
        logging.info('This angle of face is too large, its yaw:%f, pitch:%f' % (
            face_info['angle']['yaw'], face_info['angle']['pitch']))
        return False

    if face_info['face_type']['type'] == 'cartoon' or face_info['face_type']['probability'] < 0.7:
        logging.info('This face is cartoon! ')
        return False

    if face_info['quality']['illumination'] < 75:
        logging.info('This illumination is too low: %d!' % face_info['quality']['illumination'])
        return False

    # if face_info['quality']['completeness'] != 1:
    #     logging.info('This completeness is not 1: %d' % face_info['quality']['completeness'])
    #     return False

    if face_info['quality']['blur'] != 0:
        logging.info('This blur is not 0: %d' % face_info['quality']['blur'])
        return False

    return True


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
tform = trans.SimilarityTransform()


def face_align_img(img_loc, coords):
    try:
        img = cv2.imread(img_loc)
        M = cv2.estimateAffine2D(coords, calib)
        face_aligned = cv2.warpAffine(img, M[0], FACE_SIZE)
    except Exception as e:
        logging.warning(e)
        logging.warning(img_loc)
        return None
    return face_aligned


def time_format(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return '%02d:%02d:%02d' % (h, m, s)


def face_align_img_fa(img_loc):
    try:
        img = cv2.imread(img_loc)
        lmk = fa.get_landmarks(img)
    except Exception as e:
        logging.warning(e)
        logging.warning(img_loc)
        return None
    eye_left = np.mean(lmk[0][36:42], 0).tolist()
    eye_right = np.mean(lmk[0][42:48], 0).tolist()
    nose = lmk[0][30].tolist()
    mouse_left = lmk[0][48].tolist()
    mouse_right = lmk[0][54].tolist()
    coords = np.array([eye_left, eye_right, nose, mouse_left, mouse_right], dtype=np.float32)

    tform.estimate(coords, calib)
    M = tform.params[0:2, :]
    face_aligned = cv2.warpAffine(img, M, FACE_SIZE, borderValue=0.0)
    return face_aligned


def data_process(len_files_list, file_index, thread_index, files):
    # 307 columns
    easy_data_list = []
    hard_data_list = []
    last_time = time.time()
    len_files = len(files)

    for index, img_loc in enumerate(files):
        if index % 100 == 0:
            time_100 = time.time() - last_time

            logging.critical('=' * 100)
            logging.critical('%s: %d/%d(%.2f%%)' % (
                dataset_name, file_index, len_files_list, file_index / len_files_list * 100))
            logging.critical('thread: %d' % thread_index)
            logging.critical('%-13s: %9d imgs' % ('files', len_files))
            logging.critical('Now index is : %9d imgs（%.2f%%)' % (index, float(index) / len_files * 100))
            logging.critical("Good images  : %9d imgs (%.2f%%)" % (
                len(hard_data_list), len(hard_data_list) / float(index + 1) * 100))
            logging.critical('Speed : %.2fs / 100 imgs' % time_100)
            logging.critical('Predict time   : %s' % time_format((len_files - index) / 100.0 * time_100))
            logging.critical("Durations time : %s" % time_format(time.time() - start))

            last_time = time.time()

        try:
            face_aligned = face_align_img_fa(img_loc)
            if face_aligned is None:
                continue

            base64_img = cv2_base64(face_aligned)
            face_info = get_baidu_info(img_loc, base64_img)
            new_loc = os.path.join(aligned_dir, '%s_%s' % (dataset_name, os.path.basename(img_loc)))
            if face_info is None:
                logging.info("**************This error image is an aligned image!**************")
                continue
            else:
                cv2.imwrite(new_loc, face_aligned)
        except Exception as e:
            logging.info(e)
            continue
        info = get_special_info(img_loc, face_info)
        # calculate the number of emotions
        emotion_dict[info[7]] = emotion_dict[info[7]] + 1
        if abs(info[1]) < 20:
            easy_data_list.append(info)
        hard_data_list.append(info)

    return [easy_data_list, hard_data_list]


def find_all_imgs(dir, files_list):
    for root, dirs, files in os.walk(dir):
        files_list.extend(
            [os.path.join(root, i) for i in files if i.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'tif', 'bmp']])
        for d in dirs:
            find_all_imgs(os.path.join(root, d), files_list)
        break


file_split_number = 10000


def data_process_thread(num_processes, Async):
    logging.critical('Start: %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    files = []
    find_all_imgs(dataset_img_loc, files)
    # test
    # files = files[:101]

    # n = int(len(files) / num_processes)
    # img_list = [files[i:i + n] for i in range(0, len(files), n)]

    files_list = [files[i:i + file_split_number] for i in range(0, len(files), file_split_number)]
    len_files_list = len(files_list)
    logging.critical("A: There are %d split lists" % len_files_list)

    for file_index, img_list in enumerate(files_list):
        # if file_index < 36:
        #     print("The %d split is runned!" % file_index)
        #     continue
        pool = ThreadPool(processes=num_processes)
        thread_list = []

        logging.critical('A: ' + '*' * 100)
        logging.critical('A: Now is %d split, the files is %d imgs' % (file_index, len(img_list)))
        ss = time.time()
        n = int(len(files) / num_processes)
        the_img_list = [img_list[i:i + n] for i in range(0, len(img_list), n)]
        for thread_index, l in enumerate(the_img_list):
            logging.critical("thread %d start!" % thread_index)
            if Async:
                out = pool.apply_async(func=data_process,
                                       args=(len_files_list, file_index, thread_index, l))
            else:
                out = pool.apply(func=data_process, args=(len_files_list, file_index, thread_index, l))
            thread_list.append(out)

        pool.close()
        pool.join()

        thread_easy_list = []
        thread_hard_list = []

        for p in thread_list:
            if Async:
                res = p.get()
            else:
                res = p
            thread_easy_list.extend(res[0])
            thread_hard_list.extend(res[1])

        easy_data = pd.DataFrame(thread_easy_list, columns=COLUMNS)
        hard_data = pd.DataFrame(thread_hard_list, columns=COLUMNS)
        easy_data.to_csv(easy_data_loc % file_index)
        hard_data.to_csv(hard_data_loc % file_index)

        ee = time.time()

        logging.critical('A:' + '*' * 100)
        logging.critical('A: There are %d split lists' % len_files_list)
        logging.critical('A: The %d files is ended' % ((file_index + 1) * file_split_number))
        logging.critical('A: Previous time: %s' % time_format(ee - ss))
        logging.critical('A: Duration time: %s' % time_format(ee - start))
        logging.critical('A: Predict time: %s' % time_format((len_files_list - file_index - 1) * (ee - ss)))
        logging.critical('A:' + str(emotion_dict))

        sum_number = 0
        for v in emotion_dict.values():
            sum_number += v
        logging.critical("A: Sum: %d" % sum_number)
        if list(filter(lambda x: x < 0, emotion_dict_num.values())) == [] and sum_number >= sum(
                emotion_dict_num.values()):
            logging.critical("It's enough!")
            break

    logging.critical('End: %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


for t in token_info_list:
    access_token_list.append(get_access_token(t[0], t[1]))
access_token_list = list(filter(None, access_token_list))
logging.critical('the access token list is already!')
data_process_thread(num_processes=1, Async=False)

import shutil

shutil.rmtree(dataset_img_loc)
# os.rename(dataset_img_loc, dataset_img_loc+"_past")
os.rename(aligned_dir, aligned_dir.rstrip("_new"))
