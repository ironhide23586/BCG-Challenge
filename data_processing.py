import cv2
import numpy as np
from glob import glob
import os
import tifffile as tiff
import glymur as glm

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
    im_norm = clahe.apply(im_8bit)
    # im_norm = cv2.equalizeHist(im_8bit)

    im_outpath = targ_root_dir + os.sep + 'histogram_equalized.png'
    cv2.imwrite(im_outpath, im_norm)

    for i in range(0, im.shape[0], hstride):
        for j in range(0, im.shape[1], wstride):
            cropped_im = im[i:i+hstride, j:j+wstride]
            if normalize:
                cropped_im_abs = ((cropped_im / max_pix) * 255.).astype(np.uint8)
                cropped_im = clahe.apply(cropped_im_abs)
                # cropped_im = cv2.equalizeHist(cropped_im_abs)
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


def find_waterish_surface_mask(im_vh, im_vv):
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


def sar_water_mask_postprocess(waterish_mask_raw, out_dir):
    waterish_mask_raw_morphclosed = cv2.morphologyEx(waterish_mask_raw, cv2.MORPH_CLOSE,
                                                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    tiff.imsave(open(out_dir + os.sep + 'waterish_mask_morphclosed.tiff', 'wb'), waterish_mask_raw_morphclosed)

    waterish_mask_raw_morphclosed_morphopened = cv2.morphologyEx(waterish_mask_raw_morphclosed, cv2.MORPH_OPEN,
                                                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                                                iterations=1)
    tiff.imsave(open(out_dir + os.sep + 'waterish_mask_morphclosed_morphopened.tiff', 'wb'),
                waterish_mask_raw_morphclosed_morphopened)

    _, polys, _ = cv2.findContours(waterish_mask_raw_morphclosed_morphopened.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    poly_areas = np.array([cv2.contourArea(poly) for poly in polys])
    polys_mask = poly_areas > poly_areas.mean()
    chosen_polys = np.array(polys)[polys_mask]
    waterish_mask_raw_morphclosed_morphopened_polycleaned = np.zeros_like(waterish_mask_raw_morphclosed_morphopened)
    cv2.drawContours(waterish_mask_raw_morphclosed_morphopened_polycleaned, chosen_polys, -1,
                     65535, -1)
    tiff.imsave(open(out_dir + os.sep + 'waterish_mask_morphclosed_morphopened_polycleaned.tiff', 'wb'),
                waterish_mask_raw_morphclosed_morphopened_polycleaned)

    return waterish_mask_raw_morphclosed_morphopened_polycleaned


def sar_image_process(sar_folder_path, create_split_viz=False, reorient=True, bypass_exploration=False):
    im_vh, im_vv, out_dir = sar_image_analyze_explore(sar_folder_path,
                                                      create_split_viz=create_split_viz,
                                                      reorient=reorient,
                                                      bypass_ops=bypass_exploration)
    waterish_mask_raw = find_waterish_surface_mask(im_vh, im_vv)
    tiff.imsave(open(out_dir + os.sep + 'waterish_mask_raw.tiff', 'wb'), waterish_mask_raw)

    waterish_mask_postprocessed = sar_water_mask_postprocess(waterish_mask_raw, out_dir)

    return waterish_mask_postprocessed


def create_bgr_img(b, g, r, rgb=False):
    im = np.vstack([[b], [g], [r]])
    im = np.rollaxis(im, 0, 3)
    if rgb:
        im[:, :, [0, 2]] = im[:, :, [2, 0]]
    return im


def reconstruct_org_tcl_img(b_path, g_path, r_path, rgb=True):
    b = glm.Jp2k(b_path)[:]
    g = glm.Jp2k(g_path)[:]
    r = glm.Jp2k(r_path)[:]
    im, _, _, _ = reconstruct_color_img(b, g, r, rgb=rgb)
    return im, b, g, r


def reconstruct_color_img(b, g, r, rgb=True):
    clahe = cv2.createCLAHE()
    b_pp = clahe.apply(b)
    g_pp = clahe.apply(g)
    r_pp = clahe.apply(r)
    im = create_bgr_img(b_pp, g_pp, r_pp, rgb=rgb)
    return im, b_pp, g_pp, r_pp


def extract_cloudmasks(cloudmasks_raw):
    mask_tags_raw = np.array([t for t in cloudmasks_raw if 'gml:posList' in str(t)])

    # mask_type_tags_raw = np.array([str(t) for t in cloudmasks_raw if 'maskType' in str(t)])
    # opaque_indices = np.array([i for i in range(len(mask_type_tags_raw)) if 'OPAQUE' in mask_type_tags_raw[i]])
    # mask_tags_raw = mask_tags_raw[opaque_indices]

    mask_tags_str = [str(mask_tag.strip())[2:-1]
                     [len('<gml:posList srsDimension="2">')
                      :-len('</gml:posList srsDimension="2">')]
                     for mask_tag in mask_tags_raw]
    cloud_polypoints_raw = [np.array(list(map(int, mask_tag_str.split(' ')))).reshape([-1, 2])[:-1]
                            for mask_tag_str in mask_tags_str]
    bl_xy = [300004, 5690215]
    tr_xy = [409805, 5800015]
    local_height = tr_xy[1] - bl_xy[1]
    local_width = tr_xy[0] - bl_xy[0]
    offsetting_fn = lambda cld_pp_raw: np.abs([0, tr_xy[1]] - (cld_pp_raw - [bl_xy[0], 0])) / \
                                       [local_width, local_height]
    cloud_polypoints_local = list(map(offsetting_fn, cloud_polypoints_raw))
    return cloud_polypoints_local


def msi_l1c_image_process(folder_path):
    print('Processing', folder_path)
    root_dir_path_prefix = folder_path + os.sep + 'GRANULE'
    root_dir_path = glob(root_dir_path_prefix + os.sep + '*')[0]

    img_dir_path = root_dir_path + os.sep + 'IMG_DATA'
    gml_dir_path = root_dir_path + os.sep + 'QI_DATA'

    out_dir = folder_path.split(os.sep)[-1]
    force_makedir(out_dir)

    spectral_img_paths = glob(img_dir_path + os.sep + '*_B*.jp2')
    if len(spectral_img_paths) == 0:
        tmp = img_dir_path + os.sep + 'R60m'
        suffix = '_' + tmp.split(os.sep)[-1][1:]
        spectral_img_paths = np.array(glob(tmp + os.sep + '*_B*.jp2'))
        spectral_indices = np.array([int(p.replace(suffix, '').replace('8A', '08')
                                         .split('_')[-1][1:].replace('.jp2', ''))
                                     for p in spectral_img_paths])
        spectral_indices[spectral_indices > 9] = spectral_indices[spectral_indices > 9] - 1
        cloud_removal_bgr_extraction_channel_indices = [5, 9, 10]
    else:
        spectral_img_paths = np.array([p for p in spectral_img_paths if '_B8A' not in p])
        spectral_indices = np.array([int(p.split('_')[-1][1:].replace('.jp2', '')) for p in spectral_img_paths])
        cloud_removal_bgr_extraction_channel_indices = [5, 10, 11]

    spectral_indices = spectral_indices - 1
    spectral_indices_argsorted = np.argsort(spectral_indices)
    spectral_img_paths_sorted = spectral_img_paths[spectral_indices_argsorted]

    spectral_imgs = np.array(list(map(glm.Jp2k, spectral_img_paths_sorted)))
    tcl_height, tcl_width = spectral_imgs[1].shape

    # ---------------------True Color Image reconstruction from spectral images--------------------- #

    print('Reconstructing true color image...')
    reconstructed_tcl_img, b_true, g_true, r_true = reconstruct_org_tcl_img(spectral_img_paths_sorted[1],
                                                                            spectral_img_paths_sorted[2],
                                                                            spectral_img_paths_sorted[3],
                                                                            rgb=True)
    reconstructed_tcl_img_8bit = ((reconstructed_tcl_img / 65535) * 255).astype(np.uint8)
    true_imgs = np.array([b_true, g_true, r_true])
    recons_path = out_dir + os.sep + 'true_color_reconstructed_bands234.tiff'
    tiff.imsave(open(recons_path, 'wb'), reconstructed_tcl_img)

    # ---------------------True Color Image reconstruction from spectral images--------------------- #

    # ---------------------Cloud coverage polygon reading--------------------- #

    print('Parsing cloud coverage files and generating masks')
    cloudmask_filepath = gml_dir_path + os.sep + 'MSK_CLOUDS_B00.gml'
    with open(cloudmask_filepath, 'rb') as f:
        cloudmasks_raw = f.readlines()
    cloudmasks_normalized_xy = extract_cloudmasks(cloudmasks_raw)

    scale_fn = lambda cmsk_norm: (cmsk_norm * [tcl_width, tcl_height]).astype(np.int32)
    cloudmasks_scaled = list(map(scale_fn, cloudmasks_normalized_xy))

    cloudmask_reconstructed = np.zeros([tcl_height, tcl_width], dtype=np.uint8)
    for i in range(len(cloudmasks_scaled)):
        cv2.drawContours(cloudmask_reconstructed, cloudmasks_scaled, i, 255, -1)
    cloudmask_img_path = out_dir + os.sep + 'gml_cloudmasks_reconstructed.png'
    cv2.imwrite(cloudmask_img_path, cloudmask_reconstructed)

    # ---------------------Cloud coverage polygon reading--------------------- #

    # ---------------------Cloud Removal--------------------- #

    print('Cleaning clouds')
    cloudmask = cloudmask_reconstructed.astype(np.uint16)
    cloudmask[cloudmask > 0] = 1

    extraction_masks_raw = spectral_imgs[cloud_removal_bgr_extraction_channel_indices]
    extraction_masks = []
    for em in extraction_masks_raw:
        im = cv2.resize(em[:], (tcl_width, tcl_height), interpolation=cv2.INTER_LANCZOS4)
        extraction_masks.append(im)

    extraction_masks[0] = extraction_masks[1] # DOING AWAY WITH BLUE CHANNEL AS IT IS TOO CLOUDY
    b_ext, g_ext, r_ext = extraction_masks
    ext_reconstructed, _, _, _ = reconstruct_color_img(b_ext, g_ext, r_ext, rgb=True)

    ext_bands = np.array(cloud_removal_bgr_extraction_channel_indices) + 1
    ext_recons_path = out_dir + os.sep + 'true_color_reconstructed_bands_' +\
                      '-'.join(list(map(str, ext_bands))) + '.tiff'
    tiff.imsave(open(ext_recons_path, 'wb'), ext_reconstructed)

    extraction_masks = np.array(extraction_masks)
    cloudmask_repeated = np.tile(cloudmask, [3, 1, 1])
    extraction_masks_multipliers = (extraction_masks * cloudmask_repeated) / 65535
    extraction_masks_multipliers[cloudmask_repeated == 0] = 1.

    cloud_cleaned_img_raw = (true_imgs * (1 - cloudmask_repeated)) + (extraction_masks * cloudmask_repeated)
    cloud_cleaned_img_raw = cloud_cleaned_img_raw.astype(np.uint16)

    cloud_cleaned_img, _, _, _ = reconstruct_color_img(cloud_cleaned_img_raw[0],
                                                       cloud_cleaned_img_raw[1],
                                                       cloud_cleaned_img_raw[2],
                                                       rgb=True)
    cloud_cleaned_img_8bit = ((cloud_cleaned_img / 65535) * 255).astype(np.uint8)
    cloudcleaned_img_path = out_dir + os.sep + 'cloud_cleaned.tiff'
    tiff.imsave(open(cloudcleaned_img_path, 'wb'), cloud_cleaned_img)

    return cloud_cleaned_img, cloudmask_reconstructed, cloud_cleaned_img_8bit, reconstructed_tcl_img_8bit


if __name__ == '__main__':
    data_folders = glob('data' + os.sep + '*')

    # ---------------------SAR (Synthetic Aperture Radar) PROCESSING--------------------- #

    sentinel_1_SAR_folders = [d for d in data_folders if 'S1' in d]

    s1a_201808_folder = [f for f in sentinel_1_SAR_folders if '201808' in f][0]
    s1a_201809_folder = [f for f in sentinel_1_SAR_folders if '201809' in f][0]

    water_mask_201808 = sar_image_process(s1a_201808_folder, create_split_viz=False,
                                          reorient=True, bypass_exploration=True)
    water_mask_201809 = sar_image_process(s1a_201809_folder, create_split_viz=False,
                                          reorient=True, bypass_exploration=True)

    # ---------------------SAR (Synthetic Aperture Radar) PROCESSING--------------------- #

    # ---------------------MSI (Multi Spectral Imager) PROCESSING--------------------- #

    sentinel_2_MSI_folders = [d for d in data_folders if 'S2' in d]

    # south
    s2a_201609_l1c_t33uut_folder = [f for f in sentinel_2_MSI_folders if '201609' in f and 'T33UUT' in f][0]

    # north (includes Berlin)
    s2a_201609_l1c_t33uuu_folder = [f for f in sentinel_2_MSI_folders if '201609' in f and 'T33UUU' in f][0]

    # south (covers same area as 201609 T3UUT)
    s2a_201809_l2a_t33uut_folder = [f for f in sentinel_2_MSI_folders if '201809' in f and 'T33UUT' in f][0]

    # cloud_cleaned_img_16bit_s2a_201609_l1c_t33uut, cloudmask_reconstructed_8bit_s2a_201609_l1c_t33uut, \
    # cloud_cleaned_img_8bit_s2a_201609_l1c_t33uut, reconstructed_tcl_img_8bit_s2a_201609_l1c_t33uut \
    #     = msi_l1c_image_process(s2a_201609_l1c_t33uut_folder)

    # cloud_cleaned_img_16bit_s2a_201609_l1c_t33uuu, cloudmask_reconstructed_8bit_s2a_201609_l1c_t33uuu, \
    # cloud_cleaned_img_8bit_s2a_201609_l1c_t33uuu, reconstructed_tcl_img_8bit_s2a_201609_l1c_t33uuu \
    #     = msi_l1c_image_process(s2a_201609_l1c_t33uuu_folder)

    cloud_cleaned_img_16bit_s2a_201809_l2a_t33uut, cloudmask_reconstructed_8bit_s2a_201809_l2a_t33uut, \
    cloud_cleaned_img_8bit_s2a_201809_l2a_t33uut, reconstructed_tcl_img_8bit_s2a_201809_l2a_t33uut \
        = msi_l1c_image_process(s2a_201809_l2a_t33uut_folder)

    # ---------------------MSI (Multi Spectral Imager) PROCESSING--------------------- #