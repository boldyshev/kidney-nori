import os
from pathlib import Path
from typing import Optional
import argparse
from math import sqrt

import cv2
import numpy as np
import skimage as sci
from tqdm import tqdm
import matplotlib.pyplot as plt

from hydra import initialize, compose

import torch

from src.models.unet import UNet


def read_image(
        data_dir: str,
        image_name: str,
) -> tuple[np.array, np.array, np.array]:
    """
    Load image, it's corresponding tubules, masks and tubules outlines to numpy arrays.

    Args:
        image_name (str): Image file name
        channel (int): Default to single channel lipid.
    """

    image_fp = os.path.join(data_dir, image_name)
    image = sci.io.imread(image_fp)

    mask_name = image_name.replace(".tif", "_Mask.png")
    mask_fp = os.path.join(data_dir, mask_name)
    mask = sci.io.imread(mask_fp)
    
    mask_name = image_name.replace(".tif", "_Mask.tif")
    mask_fp = os.path.join(data_dir, mask_name)
    mask_tif = sci.io.imread(mask_fp)
    outlines = np.zeros_like(mask_tif, dtype=np.uint8)
    masks_tif = masks_split_value(mask_tif)

    for m in masks_tif:
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(outlines, contours, -1, (255), thickness=1)

    return image, mask, outlines


def masks_split_contours(mask: np.array) -> np.array:
    """Split single image with `N` tubules masks to `N` images with single tubule mask"""

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    masks = np.zeros((len(contours), *mask.shape))
    for i, contour in enumerate(contours):
        cv2.drawContours(masks[i], [contour], -1, 1, thickness=cv2.FILLED)

    return masks.astype(np.uint8)


def masks_split_value(mask: np.array) -> np.array:
    """Split single image with `N` tubules masks to `N` images with single tubule mask"""

    pixel_values = np.unique(mask)[1:]
    masks = np.zeros((pixel_values.size, *mask.shape))
    for i, pixel_val in enumerate(pixel_values):
        mask_copy = np.copy(mask)
        mask_copy[mask_copy != pixel_val] = 0
        masks[i] = mask_copy

    return masks.astype(np.uint8)


def masks2bboxes(masks: np.array) -> np.array:
    """Get tubules bounding boxes from tubules masks"""

    bboxes = np.zeros((masks.shape[0], 4), dtype=int)

    for index, mask in enumerate(masks):
        y, x = np.where(mask != 0)

        x_left, y_top = np.min(x), np.min(y)
        width = np.max(x) - np.min(x)
        height = np.max(y) - np.min(y)
        bboxes[index] = np.array([x_left, y_top, width, height])

    return bboxes


def crop_images_bboxes(
        images: np.array,
        bboxes: np.array,
) -> tuple[np.array, np.array]:
    """Crop single tubule images to a fixed square size. Move bbox coordinates accordingly."""

    crop_size = np.max(bboxes[:, 2:])
    img_cropped = np.zeros((images.shape[0], crop_size, crop_size))
    bboxes_cropped = np.copy(bboxes)

    for i, img in enumerate(images):
        xmin, ymin, width, height = bboxes[i]

        pad_top = (crop_size - height) // 2
        pad_bottom = crop_size - height - pad_top
        pad_left = (crop_size - width) // 2
        pad_right = crop_size - width - pad_left
        pad_width = ((max(0, pad_top), max(0, pad_bottom)), (max(0, pad_left), max(0, pad_right)))

        crop_bbox = img[ymin:ymin + height, xmin:xmin + width]
        img_cropped[i] = np.pad(crop_bbox, pad_width)
        bboxes_cropped[i] = np.array([pad_left, pad_top, width, height])

    return img_cropped, bboxes_cropped


def insert_array(
        large_array: np.array,
        small_array: np.array,
        x: int,
        y: int
) -> np.array:
    """Insert small_array into large_array at coordinates (x, y)."""

    # Get the dimensions of the smaller array
    small_height, small_width = small_array.shape

    # Get the dimensions of the larger array
    large_height, large_width = large_array.shape

    # Calculate the region of interest where the smaller array will be inserted
    start_x = max(0, x)
    end_x = min(large_width, x + small_width)
    start_y = max(0, y)
    end_y = min(large_height, y + small_height)

    # Calculate the region of the smaller array that will be inserted
    small_start_x = max(0, -x)
    small_end_x = small_width - max(0, x + small_width - large_width)
    small_start_y = max(0, -y)
    small_end_y = small_height - max(0, y + small_height - large_height)

    # Insert the smaller array into the larger array
    large_array[start_y:end_y, start_x:end_x] += small_array[small_start_y:small_end_y, small_start_x:small_end_x]

    return large_array


def predict(model: torch.nn.Module, img: np.array) -> np.array:
    """Convert array to torch tensor, predict nuclei masks, convert masks to numpy"""

    img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = img.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        mask_pred = model(img)
        mask_pred = mask_pred.cpu()
        mask_pred = torch.sigmoid(mask_pred[0]) > 0.5
        mask_pred = mask_pred[0].squeeze().numpy().astype(np.uint8)

    return mask_pred


def filter_nuclei(mask: np.array, area_threshold: float = 12, ratio_threshold: float = 6) -> np.array:
    """Filter out all nuclei masks with `area / perimeter` ratio less than threshold"""

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask, dtype=np.uint8)
    area_lst, perimeter_lst = list(), list()
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if area > area_threshold and perimeter / sqrt(area) <= ratio_threshold:
            cv2.drawContours(result, np.expand_dims(cnt, axis=0), -1, 1, thickness=cv2.FILLED)
            area_lst.append(area)
            perimeter_lst.append(perimeter)

    return result, area_lst, perimeter_lst


def predict_image(
        model: torch.nn.Module,
        data_dir: str, 
        image_name: str,
) -> np.array:
    # Load images to numpy arrays of shape (H, W)
    print('Loading images')
    image, mask, outlines = read_image(data_dir, image_name)

    # Transform arrays to shape (N, H, W), where N is the number of individual tubules
    print('Crop tubules')
    masks = masks_split_contours(mask)
    images = image * masks
    bboxes = masks2bboxes(masks)
    n_tubules = masks.shape[0]

    # Square crop individual tubules (N, H, W) -> (N, S, S), S - largest tubule size
    # Save bbox coordinates for cropped images
    img_cropped, bboxes_cropped = crop_images_bboxes(images, bboxes)

    # Predict cropped nuclei masks
    mask_pred_cropped = np.zeros_like(img_cropped, dtype=np.uint8)
    areas, perimeters = list(), list()
    for i, img in tqdm(enumerate(img_cropped), total=n_tubules, desc='Predict masks'):
        mask_pred = predict(model, img)
        mask_pred_cropped[i], areas_i, perimeters_i = filter_nuclei(mask_pred)
        areas += areas_i
        perimeters += perimeters_i

    # Merge individual masks to a single image
    mask_pred_full = np.zeros_like(image, dtype=np.uint8)
    for i, mask_pred in enumerate(mask_pred_cropped):
        x, y, w, h = bboxes_cropped[i]
        mask_pred_crop = mask_pred[y:y + h, x:x + w]
        x, y, w, h = bboxes[i]
        mask_pred_full = insert_array(mask_pred_full, mask_pred_crop, x, y)

    areas = np.array(areas)
    perimeters = np.array(perimeters)

    return mask_pred_full, np.around(areas, decimals=2), np.around(perimeters, decimals=2)


def draw_nuclei(image, outlines, mask_pred, out_file=None, scale_pixels=170):
    mask_pred[mask_pred > 0] = 1
    mask_contour = mask_pred + np.clip(outlines, 0, 1)

    label_image = sci.color.label2rgb(
        mask_contour,
        image=image / scale_pixels,
        alpha=0.5
    )

    if not out_file:
        fig = plt.figure(figsize=(20, 20))
    plt.axis('off')
    plt.imshow(label_image)
    if out_file:
        plt.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches = 0)
    else:
        plt.show()


def draw_distributions(areas, perimeters, out_file=None):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the histograms
    axs[0].hist(areas, bins=np.unique(areas).size, color='blue', alpha=0.7)
    axs[0].set_title('Areas')
    axs[0].set_xlabel('Area')
    axs[0].set_ylabel('Nuclie Number')

    axs[1].hist(perimeters, bins=np.unique(perimeters).size, color='green', alpha=0.7)
    axs[1].set_title('Perimeters')
    axs[1].set_xlabel('Perimeter')
    axs[1].set_ylabel('Nuclie Number')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    if out_file:
        plt.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


def main():
    with initialize(version_base=None, config_path='conf'):
        cfg = compose(config_name='config.yaml')

    data_dir = cfg.paths.train_data
    checkpoints_path = cfg.paths.checkpoints

    parser = argparse.ArgumentParser()
    parser.add_argument('image_name', type=str, help='Image file to predict nuclei masks')
    parser.add_argument('-c', '--checkpoint', type=str,
                        default='unet_nuclei_epoch8.pth', help='Checkpoint file name')
    parser.add_argument('-o', '--out_dir', type=str,
                        default='.', help='Output directory')
    args = parser.parse_args()

    model = UNet(1, 1)
    model.load_state_dict(torch.load(str(Path(checkpoints_path) / args.checkpoint)))

    mask_pred, areas, perimeters = predict_image(model, data_dir, args.image_name)
    image, mask, outlines = read_image(data_dir, args.image_name)
    masks_path = args.image_name.replace('.tif', '_mask_pred.png')
    masks_path = os.path.join(args.out_dir, masks_path)
    draw_nuclei(image, outlines, mask_pred, out_file=masks_path)
    print(f'Mask predictions saved to {masks_path}')

    areas_path = os.path.join(args.out_dir, args.image_name.replace('.tif', '_areas.csv'))
    np.savetxt(str(areas_path), areas, delimiter=",")
    print(f'Areas array saved to {areas_path}')

    perimeters_path = os.path.join(args.out_dir, args.image_name.replace('.tif', '_perimeters.csv'))
    np.savetxt(str(perimeters_path), perimeters, delimiter=",")
    print(f'Perimeters array saved to {perimeters_path}')

    hist_path = os.path.join(args.out_dir, args.image_name.replace('.tif', '_areas_hist.png'))
    draw_distributions(areas, perimeters, out_file=hist_path)
    print(f'Histogram saved to {hist_path}')


if __name__ == '__main__':
    main()
