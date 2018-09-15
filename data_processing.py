import cv2
import numpy as np
from glob import glob
import os
import tifffile as tiff


def show(img_cv2, res=(700, 700)):
    cv2.imshow('image', cv2.resize(img_cv2, res, cv2.INTER_CUBIC))
    cv2.waitKey()


def show_(img_cv2):
    cv2.imshow('image', img_cv2)
    cv2.waitKey()


def preprocess_images(img_path, hstride=3337, wstride=5309, normalize=True):
    print('Splitting file at', img_path)
    im = tiff.imread(img_path)
    max_pix = im.max()

    im_name = img_path.split(os.sep)[-1].replace('.tiff', '')
    targ_root_dir = os.sep.join(img_path.split(os.sep)[:-1]) + os.sep + im_name + '_processed'
    targ_split_dir = targ_root_dir + os.sep + 'splitted'
    if not os.path.isdir(targ_split_dir):
        os.makedirs(targ_split_dir)

    clahe = cv2.createCLAHE()

    im_8bit = ((im / max_pix) * 255.).astype(np.uint8)
    # im_norm = clahe.apply(im_8bit)
    im_norm = cv2.equalizeHist(im_8bit)

    im_outpath = targ_root_dir + os.sep + 'histogram_equalized.png'
    cv2.imwrite(im_outpath, im_norm)

    for i in range(0, im.shape[0], hstride):
        for j in range(0, im.shape[1], wstride):
            cropped_im = im[i:i+hstride, j:j+wstride]
            if normalize:
                cropped_im_abs = ((cropped_im / max_pix) * 255.).astype(np.uint8)
                # cropped_im = clahe.apply(cropped_im_abs)
                cropped_im = cv2.equalizeHist(cropped_im_abs)
            targ_name = str(i) + 'r_' + str(j) + 'c' + '.png'
            targ_path = targ_split_dir + os.sep + targ_name
            print('Writing split files to', targ_path)
            print(cv2.imwrite(targ_path, cropped_im))


def viz_im(im_16bit, normalize=False):
    im = ((im_16bit / im_16bit.max()) * 255.).astype(np.uint8)
    clahe = cv2.createCLAHE()
    im = clahe.apply(im)
    return im


def force_makedir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def find_roadish_surface_mask(im_vh, im_vv):
    vh_vv_mul = cv2.multiply(im_vh, im_vv)
    m = vh_vv_mul.max()
    vh_vv_mul[vh_vv_mul > (vh_vv_mul.mean() - vh_vv_mul.std())] = 0
    vh_vv_mul[vh_vv_mul > 0] = m
    return vh_vv_mul


def sar_image_analyze_explore(sar_folder_path,
                              create_split_viz=False,
                              reorient=True,
                              bypass_ops=False):
    img_list_suffix = 'measurement' + os.sep + '*.tiff'

    img_paths = glob(sar_folder_path + os.sep + img_list_suffix)

    if create_split_viz: #creates chunked viewable 8-bit histogram normalized images for human viewing;
        for img_path in img_paths:
            preprocess_images(img_path)

    vh_path = [p for p in img_paths if '-vh-' in p][0]
    vv_path = [p for p in img_paths if '-vv-' in p][0]

    im_vh = tiff.imread(vh_path)
    im_vv = tiff.imread(vv_path)

    out_dir = sar_folder_path.split(os.sep)[-1] + '_analysis'
    force_makedir(out_dir)

    if reorient: #original images are north-south flipped, this corrects it
        im_vh = np.flipud(im_vh)
        im_vv = np.flipud(im_vv)
        tiff.imsave(open(vh_path.replace('.tiff', '_reoriented.tiff'), 'wb'), im_vh)
        tiff.imsave(open(vv_path.replace('.tiff', '_reoriented.tiff'), 'wb'), im_vv)

    if not bypass_ops:

        vh_vv_diff = cv2.subtract(im_vh, im_vv)
        vv_vh_diff = cv2.subtract(im_vv, im_vh)

        vhvv_diff_path = out_dir + os.sep + 'vh_vv_diff.tiff'
        vvvh_diff_path = out_dir + os.sep + 'vv_vh_diff.tiff'

        tiff.imsave(open(vhvv_diff_path, 'wb'), vh_vv_diff)
        tiff.imsave(open(vvvh_diff_path, 'wb'), vv_vh_diff)

        vh_vv_add = cv2.add(im_vh, im_vv)
        vh_vv_mul = cv2.multiply(im_vh, im_vv)

        vhvv_add_path = out_dir + os.sep + 'vh_vv_add.tiff'
        vhvv_mul_path = out_dir + os.sep + 'vh_vv_mul.tiff'

        tiff.imsave(open(vhvv_add_path, 'wb'), vh_vv_add)
        tiff.imsave(open(vhvv_mul_path, 'wb'), vh_vv_mul)

    return im_vh, im_vv, out_dir


def sar_road_mask_postprocess(roadish_mask_raw, out_dir):
    roadish_mask_raw_morphclosed = cv2.morphologyEx(roadish_mask_raw, cv2.MORPH_CLOSE,
                                                    cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    tiff.imsave(open(out_dir + os.sep + 'roadish_mask_morphclosed.tiff', 'wb'), roadish_mask_raw_morphclosed)

    roadish_mask_raw_morphclosed_morphopened = cv2.morphologyEx(roadish_mask_raw_morphclosed, cv2.MORPH_OPEN,
                                                                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                                                                iterations=1)
    tiff.imsave(open(out_dir + os.sep + 'roadish_mask_morphclosed_morphopened.tiff', 'wb'),
                roadish_mask_raw_morphclosed_morphopened)

    _, polys, _ = cv2.findContours(roadish_mask_raw_morphclosed_morphopened.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    poly_areas = np.array([cv2.contourArea(poly) for poly in polys])
    polys_mask = poly_areas > (poly_areas.mean() + poly_areas.std() / 4.)
    chosen_polys = np.array(polys)[polys_mask]
    roadish_mask_raw_morphclosed_morphopened_polycleaned = np.zeros_like(roadish_mask_raw_morphclosed_morphopened)
    cv2.drawContours(roadish_mask_raw_morphclosed_morphopened_polycleaned, chosen_polys, -1,
                     65535, -1)
    tiff.imsave(open(out_dir + os.sep + 'roadish_mask_morphclosed_morphopened_polycleaned.tiff', 'wb'),
                roadish_mask_raw_morphclosed_morphopened_polycleaned)

    return roadish_mask_raw_morphclosed_morphopened_polycleaned


def sar_image_process(sar_folder_path, create_split_viz=False, reorient=True, bypass_exploration=False):
    im_vh, im_vv, out_dir = sar_image_analyze_explore(sar_folder_path,
                                                      create_split_viz=create_split_viz,
                                                      reorient=reorient,
                                                      bypass_ops=bypass_exploration)
    roadish_mask_raw = find_roadish_surface_mask(im_vh, im_vv)
    tiff.imsave(open(out_dir + os.sep + 'roadish_mask_raw.tiff', 'wb'), roadish_mask_raw)

    roadish_mask_postprocessed = sar_road_mask_postprocess(roadish_mask_raw, out_dir)

    k = 0



if __name__ == '__main__':
    data_folders = glob('data' + os.sep + '*')
    sentinel_1_SAR_folders = [d for d in data_folders if 'S1' in d]

    s1a_201808_folder = [f for f in sentinel_1_SAR_folders if '201808' in f][0]
    s1a_201809_folder = [f for f in sentinel_1_SAR_folders if '201809' in f][0]

    # sar_image_process(s1a_201808_folder, create_split_viz=False, reorient=True, bypass_exploration=True)
    sar_image_process(s1a_201809_folder, create_split_viz=False, reorient=True, bypass_exploration=True)

    k = 0