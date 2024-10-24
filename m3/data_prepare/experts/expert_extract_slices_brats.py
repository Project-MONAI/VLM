# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import os
import random

import numpy as np
import skimage
from data_utils import colors, listdir, read_json, remove_extension, write_json
from monai.transforms import LoadImage, Orientation, ScaleIntensity, ScaleIntensityRange
from tqdm import tqdm

random.seed(0)

modality = "MRI"

MIN_VOL = 1000  # 1 cm^3
OVERWRITE = False

modalities = ["t1ce", "t1", "t2", "flair"]


def main(args):
    """Extract image slices from brain MRI."""
    datalist = read_json(args.datalist_filename)

    assert len(datalist) > 0, "Datalist is empty!"

    os.makedirs(args.output_dir, exist_ok=True)
    if len(listdir(args.output_dir)) > 0 and (not OVERWRITE):
        raise ValueError(f"Output directory not empty! {args.output_dir}")

    out_meta = []
    for dataset_name in ["training", "validation", "testing"]:
        dataset_list = datalist[dataset_name]
        print(f"Processing {len(dataset_list)} images in {dataset_name} set")
        # get images & seg labels
        image_files, label_files = [], []
        for entry in dataset_list:
            images = entry["image"]
            basename = os.path.basename(os.path.dirname(images[0]))  # get folder name
            label_filename = glob.glob(
                os.path.join(args.input_label_dir, "**", f"{basename}*_seg.nii.gz"), recursive=True
            )
            assert len(label_filename) == 1, f"Couldn't find matching *_seg.nii.gz file for {basename}"
            label_filename = label_filename[0]
            # get images
            mod_filenames = []
            for mod in modalities:
                image_filename = glob.glob(
                    os.path.join(args.input_image_dir, "**", f"{basename}_{mod}.nii.gz"), recursive=True
                )
                assert len(image_filename) == 1, f"Couldn't find matching *_{mod}.nii.gz file for {basename}"
                mod_filenames.append(image_filename[0])

            image_files.append(mod_filenames)
            label_files.append(label_filename)

        assert len(image_files) > 0
        assert len(label_files) > 0
        assert len(image_files) == len(label_files), f"{len(image_files)} image vs. {len(label_files)} labels"
        print(f"Found {len(image_files)} image label pairs")

        loader = LoadImage(image_only=True, ensure_channel_first=True)
        orientation = Orientation(axcodes="RAS")

        if modality == "CT":
            # abdomen soft tissue https://radiopaedia.org/articles/windowing-ct
            window_center = 50
            window_width = 400
            scaler = ScaleIntensityRange(
                a_min=window_center - window_width / 2,
                a_max=window_center + window_width / 2,
                b_min=0,
                b_max=255,
                clip=True,
            )
        elif modality == "MRI":
            scaler = ScaleIntensity(minv=0, maxv=255, channel_wise=True)
        else:
            raise ValueError("No such modality!")

        # iterate through images
        count = 0
        for image_files, label_file in zip(image_files, label_files):
            out_img_dir = os.path.join(args.output_dir, os.path.basename(remove_extension(image_files[0])))
            os.makedirs(out_img_dir, exist_ok=True)

            label_volume = orientation(loader(label_file))

            orig_image_volumes = []
            for image_file in image_files:
                print("Loading", os.path.basename(image_file), os.path.basename(label_file))
                image_volume = orientation(loader(image_file))
                orig_image_volume = scaler(image_volume).numpy()
                orig_image_volumes.append(orig_image_volume)
                assert np.all(np.diag(image_volume.affine) == np.diag(label_volume.affine))

            orig_label_volume = label_volume.numpy()

            assert orig_image_volume.shape == orig_label_volume.shape
            print(f"image: {orig_image_volume.shape}, label: {orig_label_volume.shape}")

            slice_ids = np.unique(np.where(orig_label_volume > 0)[-1])  # all slices with non-zero values
            assert len(slice_ids) > 0
            for slice_id in tqdm(slice_ids, desc="extracting slices..."):
                # select a slice
                image_slices = []
                for orig_image_volume in orig_image_volumes:
                    image_slices.append(np.squeeze(orig_image_volume[0, :, :, [slice_id]][0, ...]))
                label_slice = np.squeeze(orig_label_volume[0, :, :, [slice_id]][0, ...])

                # only extract slices with all tumor labels
                if 1 not in label_slice or 2 not in label_slice or 4 not in label_slice:
                    continue

                img_id = f"{modality.lower()}_{count:05d}"
                out_prefix = os.path.join(out_img_dir, img_id)

                # save label
                label_outname = out_prefix + "_label.png"
                color_label = (
                    skimage.color.label2rgb(label_slice, colors=colors, image=image_slices[0] / 255) * 255
                )  # If the number of labels exceeds the number of colors, then the colors are cycled.
                color_label = np.rot90(np.swapaxes(color_label.astype(np.uint8), 0, 1), k=2)
                skimage.io.imsave(label_outname, color_label)

                # save image as RGB
                image_outnames = []
                for mod in modalities:
                    image_outname = out_prefix + f"_img_{mod}.png"
                    image_slice = np.repeat(np.expand_dims(image_slices[0], axis=-1), axis=-1, repeats=3)
                    image_slice = np.rot90(np.swapaxes(image_slice.astype(np.uint8), 0, 1), k=2)
                    skimage.io.imsave(image_outname, image_slice)
                    image_outnames.append(image_outname.replace(args.output_dir + os.path.sep, ""))

                out_meta.append(
                    {
                        "image": image_outnames,
                        "label": label_outname.replace(args.output_dir + os.path.sep, ""),
                        "orig_images": image_files,
                        "orig_shape": [int(i) for i in orig_image_volume.shape],
                        "orig_label": label_file,
                    }
                )

                count += 1

                # return

    print(f"Exported a total of {count} images")

    write_json(out_meta, os.path.join(args.output_dir, "extracted_slices_meta.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datalist_filename", type=str, required=True)
    parser.add_argument("--input_image_dir", type=str, required=True)
    parser.add_argument("--input_label_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main()
