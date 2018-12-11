import numpy as np
import cv2

from core.config import cfg
import utils.blob as blob_utils
import roi_data.rpn


def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data']
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster R-CNN
        blob_names += roi_data.rpn.get_rpn_blob_names(is_training=is_training)
    elif cfg.RETINANET.RETINANET_ON:
        raise NotImplementedError
    else:
        # Fast R-CNN like models trained on precomputed proposals
        blob_names += roi_data.fast_rcnn.get_fast_rcnn_blob_names(
            is_training=is_training
        )
    return blob_names


def get_minibatch(roidb, valid_keys):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    if cfg.TRAIN.IGNORE_MASK:
        im_blob, im_ig_blob, im_scales = _get_image_blob(roidb)
        blobs['im_ig_blob'] = im_ig_blob
    else:
        im_blob, im_scales = _get_image_blob(roidb)
    blobs['data'] = im_blob
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster/Mask R-CNN
        valid = roi_data.rpn.add_rpn_blobs(blobs, im_scales, roidb, valid_keys=valid_keys)
    elif cfg.RETINANET.RETINANET_ON:
        raise NotImplementedError
    else:
        # Fast R-CNN like models trained on precomputed proposals
        valid = roi_data.fast_rcnn.add_fast_rcnn_blobs(blobs, im_scales, roidb)
    return blobs, valid


def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    if cfg.TRAIN.SCALES[0] == 0:
        target_size = cfg.TRAIN.MAX_SIZE
    else:
        target_size = np.random.randint(low=cfg.TRAIN.SCALES[0], high=cfg.TRAIN.SCALES[1], size=num_images)

    processed_ims = []
    if cfg.TRAIN.IGNORE_MASK:
        processed_ims_ig = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        assert im is not None, 'Failed to read image \'{}\''.format(roidb[i]['image'])
        if cfg.TRAIN.IGNORE_MASK:
            cv2.imread(roidb[i]['image'])
            im_ig = cv2.imread(roidb[i]['image'].replace('images', 'ignore_mask'))
        # If NOT using opencv to read in images, uncomment following lines
        # if len(im.shape) == 2:
        #     im = im[:, :, np.newaxis]
        #     im = np.concatenate((im, im, im), axis=2)
        # # flip the channel, since the original one using cv2
        # # rgb -> bgr
        # im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            if cfg.TRAIN.IGNORE_MASK:
                im_ig = im_ig[:, ::-1]

        #target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
        if cfg.TRAIN.IGNORE_MASK:
            im_ig_resized = cv2.resize(im_ig, None, None, fx=im_scale[0], fy=im_scale[0], interpolation=cv2.INTER_LINEAR)
            processed_ims_ig.append(im_ig_resized)
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])

    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)
    if cfg.TRAIN.IGNORE_MASK:
        blob_ig = blob_utils.im_list_to_blob(processed_ims_ig)
        return blob, blob_ig, im_scales
    else:
        return blob, im_scales
