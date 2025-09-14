import cv2
import random
import torch


def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img

def center_crop(img_gt, crop_pad_size):
    # Add ensure_min_size check
    img_gt = ensure_min_size(img_gt, crop_pad_size)
    
    h, w = img_gt.shape[0:2]
    # pad
    if h < crop_pad_size or w < crop_pad_size:
        return img_gt
    # crop
    if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
        h, w = img_gt.shape[0:2]
        # randomly choose top and left coordinates
        top = (h - crop_pad_size)// 2
        left = (w - crop_pad_size) // 2
        img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
    return img_gt

def random_crop(img_gt, crop_pad_size_h, crop_pad_size_w=None):
    if crop_pad_size_w is None:
        crop_pad_size_w = crop_pad_size_h
    
    # Add ensure_min_size check
    img_gt = ensure_min_size(img_gt, crop_pad_size_h)
    
    h, w = img_gt.shape[0:2]
    # pad
    if h < crop_pad_size_h or w < crop_pad_size_w:
        pad_h = max(0, crop_pad_size_h - h)
        pad_w = max(0, crop_pad_size_w - w)
        img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    # crop
    if img_gt.shape[0] > crop_pad_size_h or img_gt.shape[1] > crop_pad_size_w:
        h, w = img_gt.shape[0:2]
        # randomly choose top and left coordinates
        top = random.randint(0, h - crop_pad_size_h)
        left = random.randint(0, w - crop_pad_size_w)
        img_gt = img_gt[top:top + crop_pad_size_h, left:left + crop_pad_size_w, ...]
    return img_gt

def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    # Apply ensure_min_size based on input type
    if input_type == 'Tensor':
        img_gts = [paired_ensure_min_size_tensor(gt, lq, gt_patch_size)[0] 
                  for gt, lq in zip(img_gts, img_lqs)]
        img_lqs = [paired_ensure_min_size_tensor(gt, lq, gt_patch_size)[1] 
                  for gt, lq in zip(img_gts, img_lqs)]
    else:
        img_gts = [paired_ensure_min_size(gt, lq, gt_patch_size)[0] 
                  for gt, lq in zip(img_gts, img_lqs)]
        img_lqs = [paired_ensure_min_size(gt, lq, gt_patch_size)[1] 
                  for gt, lq in zip(img_gts, img_lqs)]

    # Get updated dimensions
    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        # raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
        #                  f'({lq_patch_size}, {lq_patch_size}). '
        #                  f'Please remove {gt_path}.')
        if len(img_gts) == 1:
            img_gts = img_gts[0]
        if len(img_lqs) == 1:
            img_lqs = img_lqs[0]
        return img_gts,img_lqs

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
        
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img


def paired_ensure_min_size(img_gt, img_lq, gt_size):
    """Resize images if any dimension is smaller than gt_size while maintaining aspect ratio"""
    h_gt, w_gt = img_gt.shape[0:2]
    
    # Check if resizing is needed
    if h_gt < gt_size or w_gt < gt_size:
        # Calculate scaling factor to make smallest side equal to gt_size
        ratio = gt_size / min(h_gt, w_gt)
        new_h, new_w = int(h_gt * ratio), int(w_gt * ratio)
        
        # Resize both images to the same size
        img_gt = cv2.resize(img_gt, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img_lq = cv2.resize(img_lq, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return img_gt, img_lq

def ensure_min_size(img, min_size):
    """Resize single image if any dimension is smaller than min_size while maintaining aspect ratio"""
    h, w = img.shape[0:2]
    
    # Check if resizing is needed
    if h < min_size or w < min_size:
        # Calculate scaling factor to make smallest side equal to min_size
        ratio = min_size / min(h, w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        
        # Resize image
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return img

def paired_ensure_min_size_tensor(img_gt, img_lq, gt_size):
    """Resize tensor images if any dimension is smaller than gt_size while maintaining aspect ratio
    
    Args:
        img_gt (Tensor): GT image tensor in [C, H, W] format
        img_lq (Tensor): LQ image tensor in [C, H, W] format
        gt_size (int): Target minimum size for GT image
        
    Returns:
        tuple(Tensor, Tensor): Resized GT and LQ images
    """
    h_gt, w_gt = img_gt.size()[-2:]
    
    # Check if resizing is needed
    if h_gt < gt_size or w_gt < gt_size:
        # Calculate scaling factor to make smallest side equal to gt_size
        ratio = gt_size / float(min(h_gt, w_gt))
        new_h, new_w = int(h_gt * ratio), int(w_gt * ratio)
        
        # Resize both images using torch interpolate
        img_gt = torch.nn.functional.interpolate(
            img_gt.unsqueeze(0), size=(new_h, new_w), 
            mode='bilinear', align_corners=False).squeeze(0)
        img_lq = torch.nn.functional.interpolate(
            img_lq.unsqueeze(0), size=(new_h, new_w), 
            mode='bilinear', align_corners=False).squeeze(0)
    
    return img_gt, img_lq
