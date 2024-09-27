import os
import glob
import random
import itertools

from tqdm import tqdm
import numpy as np
import skimage
from utils import read_json, remove_extension, write_json, listdir, colors
import functools
import math
from monai.transforms import LoadImage, Orientation, ScaleIntensityRange, ScaleIntensity
import copy

random.seed(0)

modality = "MRI"
input_image_dir = "/Users/hroth/Data/MSD/Task03_Liver"
input_label_dir = "/Users/hroth/Data/VLM/ct2D/liver"

output_dir="../../data/experts/brats/slices"

MIN_VOL = 1000  # 1 cm^3
OVERWRITE = False


def get_centroid_slice_ids(label, min_vol, vox_vol):
    label = np.squeeze(label)
    center_slices = []
    print(f"Find connected components in {label.shape} volume ...")
    connects, num_connects = skimage.measure.label(label, return_num=True)  # uses ndim connectivity by default
    for connect_id in range(1, num_connects + 1):
        volume = np.sum(connects == connect_id) * vox_vol
        if volume > min_vol:
            obj_center = skimage.measure.centroid(connects == connect_id)

            center_slices.append(math.floor(obj_center[-1]))

    # assert len(center_slices) > 0
    print(f"Found {len(center_slices)} centroid slices ...")
    return sorted(center_slices)


def main():
    out_meta = []
    image_files = list(sorted(glob.glob(os.path.join(input_image_dir, "imagesTr", "*.nii*"), recursive=True)))
    label_files = list(sorted(glob.glob(os.path.join(input_label_dir, "labelsTr", "*.nii*"), recursive=True)))
    assert len(image_files) > 0
    assert len(label_files) > 0
    assert len(image_files) == len(label_files), f"{len(image_files)} image vs. {len(label_files)} labels"
    print(f"Found {len(image_files)} image label pairs")

    os.makedirs(output_dir, exist_ok=True)
    if len(listdir(output_dir)) > 0 and (not OVERWRITE):
        raise ValueError(f"Output directory not empty! {output_dir}")

    loader = LoadImage(image_only=True, ensure_channel_first=True)
    orientation = Orientation(axcodes="RAS")

    if modality == "CT":
        # abdomen soft tissue https://radiopaedia.org/articles/windowing-ct
        window_center = 50
        window_width = 400
        scaler = ScaleIntensityRange(a_min=window_center - window_width / 2, a_max=window_center + window_width / 2,
                                     b_min=0, b_max=255, clip=True)
    elif modality == "MRI":
        scaler = ScaleIntensity(minv=0, maxv=255, channel_wise=True)
    else:
        raise ValueError("No such modality!")

    # iterate through images
    count = 0
    n_target_labels = []
    # max_vista_label = get_max_label(vista_labels)
    for image_file, label_file in zip(image_files, label_files):
        out_img_dir = os.path.join(output_dir, os.path.basename(remove_extension(image_file)))
        os.makedirs(out_img_dir, exist_ok=True)

        print("Loading", os.path.basename(image_file), os.path.basename(label_file))
        image_volume = orientation(loader(image_file))
        label_volume = orientation(loader(label_file))
        assert np.all(np.diag(image_volume.affine) == np.diag(label_volume.affine))
        vox_vol = np.prod(np.diag(image_volume.affine))

        orig_image_volume = scaler(image_volume).numpy()
        orig_label_volume = label_volume.numpy()

        assert orig_image_volume.shape == orig_label_volume.shape
        print(f"image: {orig_image_volume.shape}, label: {orig_label_volume.shape}")

        slice_ids = get_centroid_slice_ids(label_volume, min_vol=MIN_VOL,
                                           vox_vol=vox_vol)  # TODO: call only once per image
        for slice_id in tqdm(slice_ids, desc="extracting slices..."):
            # select a slice
            image_slice = np.squeeze(orig_image_volume[0, :, :, [slice_id]][0, ...])
            label_slice = np.squeeze(orig_label_volume[0, :, :, [slice_id]][0, ...])

            # get color label names
            out_color_label_names = []
            color_cycle = itertools.cycle(colors)  # resets color cycle
            unique_labels = np.unique(label_slice)[1::]
            for label_id, label_color in zip(unique_labels,
                                             color_cycle):
                out_color_label_names.append(f"{label_color}: {label_names[int(label_id) - 1]}")
            out_color_label_names = ", ".join(out_color_label_names)

            img_id = f"{modality.lower()}_{count:05d}"
            out_prefix = os.path.join(out_img_dir, img_id)

            # save label
            label_outname = out_prefix + "_label.png"
            color_label = skimage.color.label2rgb(label_slice, colors=colors,
                                                  image=image_slice / 255) * 255  # If the number of labels exceeds the number of colors, then the colors are cycled.
            color_label = np.rot90(np.swapaxes(color_label.astype(np.uint8), 0, 1), k=2)
            skimage.io.imsave(label_outname, color_label)

            # save image as RGB
            image_outname = out_prefix + "_img.png"
            image_slice = np.repeat(np.expand_dims(image_slice, axis=-1), axis=-1, repeats=3)
            image_slice = np.rot90(np.swapaxes(image_slice.astype(np.uint8), 0, 1), k=2)
            skimage.io.imsave(image_outname, image_slice)

            # check for tumors
            found_tumors, found_tumor_ids, num_tumors = [], [], []
            for label_id in unique_labels:
                for tumor_label_id, tumor_label in tumor_label_ids.items():
                    if label_id == tumor_label_id:
                        found_tumors.append(tumor_label)
                        found_tumor_ids.append(tumor_label_id)

            if len(found_tumors) > 0:
                assert len(found_tumors) == len(found_tumor_ids)
                for tumor_label_id in found_tumor_ids:
                    tumor_connect, num_tumor = skimage.measure.label(label_slice == tumor_label_id, return_num=True)
                    num_tumors.append(num_tumor)

            out_meta.append(
                {
                    "image": os.path.basename(image_outname),
                    "label": os.path.basename(label_outname),
                    "target_label_names": target_label_names,
                    "target_label_ids": target_label_ids,
                    "n_target_labels": len(target_label_ids),
                    "orig_image": image_file,
                    "orig_shape": [int(i) for i in orig_image_volume.shape],
                    "orig_label": label_file,
                    "label_colors": out_color_label_names,
                    "found_tumor": found_tumors,
                    "num_tumors": num_tumors
                }
            )
            n_target_labels.append(len(target_label_ids))

            count += 1

            # return

    print(f"Exported a total of {count} images, with {len(np.unique(n_target_labels))} types of labels.")
    print("Extracted label frequency")
    print(np.histogram(n_target_labels))

    write_json(out_meta, os.path.join(output_dir, "extracted_slices_meta.json"))


if __name__ == "__main__":
    main()
