#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Python wrapper for the SmartDoc2015-Challenge 1 dataset and tools.

This dataset is composed of a collection of video frames (extracted from
an original recording) captured when simulating the acquisition of an A4 paper
document with a mobile device. It also contains the original images and high
resolution pictures (captured with the same mobile device) of each of the
documents.

More details are available on the website:
    http://smartdoc.univ-lr.fr/

Each video frame contains an instance of a single document image, and the
corners of this documents are within the frame. The camera is never rotated
more than 90 degrees around the axis which is orthogonal to the document's
plane. This means that the user did not "turn" the camera, and also that
for each video frame, the closest corner of the document page to any corner
of the video frame remains always the same.

The original task this dataset was used for was locating the coordinates of
the page object within each frame, to simulate the behavior of a page
detection module.

A second task this dataset can be used for is document detection and tracking.
Using the reference image ("models") it is possible to train a detector
and track document objects across frames.

A last task is document classification: given a set of know document models,
can we recognize which of them is represented in a given frame.

The first task requires the ground truth for the coordinates, and any
comparison (distance, surface, etc.) should be computed in the dewarped
referential (as it the document was seen flat-parallel from above) in
order to have a homogeneous metric. The two following tasks require
model images and document class or document id.

Warning: the dataset is not balanced for tasks 2 and 3.

Frame count for each video sample, per background:

    bg_name       background01  background02  background03  background04  background05    All
    model_name
    datasheet001           235           222           215           164            97    933
    datasheet002           199           195           230           168            84    876
    datasheet003           210           201           184           160            94    849
    datasheet004           201           206           185           169            92    853
    datasheet005           235           210           209           143            83    880
    letter001              194           214           192           149            99    848
    letter002              200           192           189           145            97    823
    letter003              206           248           201           149            88    892
    letter004              211           187           176           141            84    799
    letter005              201           217           201           149           100    868
    magazine001            212           206           197           154            90    859
    magazine002            227           200           221           142            82    872
    magazine003            205           187           195           168            80    835
    magazine004            223           226           183           137            81    850
    magazine005            192           168           179           153            87    779
    paper001               197           261           185           123            78    844
    paper002               211           212           192           135            85    835
    paper003               205           214           195           140            79    833
    paper004               204           180           203           119            73    779
    paper005               187           197           199           131            92    806
    patent001              188           164           211           128            91    782
    patent002              193           167           199           126            85    770
    patent003              201           182           196           129            88    796
    patent004              223           210           189           135            80    837
    patent005              200           207           224           135            85    851
    tax001                 206           119           238           112            78    753
    tax002                 209           217           164           111            77    778
    tax003                 200           175           188           102            83    748
    tax004                 163           207           208           125            87    790
    tax005                 242           220           204           127            78    871
    All                   6180          6011          5952          4169          2577  24889


Archive format for frames
-------------------------
Please see the internal README.md file for more details.

The file hierarchy of this archive is:
    
    frames.tar.gz
    ├── README.md
    ├── LICENCE
    ├── original_datasets_files.txt
    ├── background01
    │   ├── datasheet001
    │   │   ├── frame_0001.jpeg
    │   │   ├── [...]
    │   │   └── frame_0235.jpeg
    │   ├── datasheet002
    │   │   └── [...]
    │   ├── datasheet003
    │   ├── datasheet004
    │   ├── datasheet005
    │   ├── letter001
    │   ├── letter002
    │   ├── letter003
    │   ├── letter004
    │   ├── letter005
    │   ├── magazine001
    │   ├── magazine002
    │   ├── magazine003
    │   ├── magazine004
    │   ├── magazine005
    │   ├── paper001
    │   ├── paper002
    │   ├── paper003
    │   ├── paper004
    │   ├── paper005
    │   ├── patent001
    │   ├── patent002
    │   ├── patent003
    │   ├── patent004
    │   ├── patent005
    │   ├── tax001
    │   ├── tax002
    │   ├── tax003
    │   ├── tax004
    │   └── tax005
    ├── background02
    │   └── [...]
    ├── background03
    │   └── [...]
    ├── background04
    │   └── [...]
    └── background05
        └── [...]

Metadata format for frames
--------------------------
The metadata file is a CSV file (separator: `,`, string quoting: None).
It is safe to split on `,` tokens as they do not appear elsewhere in this file.
Each row describes a video frame.
Columns are:
 - `bg_name`: Background name (example: `background01`). There are 5 backgrounds and they are named
   `background00N` with `N` between `1` and `5`.
 - `bg_id`: Background id (example: `0`), 0-indexed.
 - `model_name`: Model name (example: `datasheet001`). There are 30 models. See models description
   for more details.
 - `model_id`: Model id (example: `0`), 0-indexed. Value is between 0 and 29.
 - `modeltype_name`: Model type (example: `datasheet`). There are 6 model types. See models description
   for more details.
 - `modeltype_id`: Model type id (example: `0`), 0-indexed. Value is between 0 and 5.
 - `model_subid`: Model sub-index (example: `0`), 0-indexed. Value is between 0 and 4.
 - `image_path`: Relative path to the frame image (example: `background01/datasheet001/frame_0001.jpeg`)
   under the dataset home directory.
 - `frame_index`: Frame index (example: `1`), **1-indexed** (for compliance with the video version).
 - `model_width`: Width of the model object (example: `2100.0`). The size of the document along with the
   width / height ratio, are used to normalize the segmentation score among different models and frames.
   Here, 1 pixel represents 0.1 mm.
 - `model_height`: Height of the model object (example: `2970.0`).
 - `tl_x`: X coordinate of the top left point of the object in the current frame (example: `698.087`).
 - `tl_y`: Y coordinate of the top left point of the object in the current frame (example: `200.476`).
 - `bl_x`: X coordinate of the bottom left point of the object in the current frame (example: `692.141`).
 - `bl_y`: Y coordinate of the bottom left point of the object in the current frame (example: `891.077`).
 - `br_x`: X coordinate of the bottom right point of the object in the current frame (example: `1253.18`).
 - `br_y`: Y coordinate of the bottom right point of the object in the current frame (example: `869.656`).
 - `tr_x`: X coordinate of the top right point of the object in the current frame (example: `1178.15`).
 - `tr_y`: Y coordinate of the top right point of the object in the current frame (example: `191.515`).

Example of header + a random line:
    bg_name,bg_id,model_name,model_id,modeltype_name,modeltype_id,model_subid,image_path,frame_index,model_width,model_height,tl_x,tl_y,bl_x,bl_y,br_x,br_y,tr_x,tr_y
    background01,0,datasheet001,0,datasheet,0,0,background01/datasheet001/frame_0001.jpeg,1,2100.0,2970.0,698.087,200.476,692.141,891.077,1253.18,869.656,1178.15,191.515


Archive format for models
-------------------------
Please see the internal README.md file for more details.

The file hierarchy of this archive is:

    models.tar.gz
    ├── README.md
    ├── LICENCE
    ├── correct_perspective.m
    ├── original_datasets_files.txt
    ├── 01-original
    │   ├── datasheet001.png
    │   ├── [...]
    │   └── tax005.png
    ├── 02-edited
    │   ├── datasheet001.png
    │   ├── [...]
    │   └── tax005.png
    ├── 03-captured-nexus
    │   ├── datasheet001.jpg # JPG images here
    │   ├── [...]
    │   └── tax005.jpg
    ├── 04-corrected-nexus
    │   ├── datasheet001.png
    │   ├── [...]
    │   └── tax005.png
    └── 05-corrected-nexus-scaled33
        ├── datasheet001.png
        ├── [...]
        └── tax005.png

Metadata format for models
--------------------------
The metadata file is a CSV file (separator: `,`, string quoting: None).
It is safe to split on `,` tokens as they do not appear elsewhere in this file.
Each row describes a model image.
Columns are:
 - `model_cat`: Model category (example: `05-corrected-nexus-scaled33`). There are
   5 categories:
   - `01-original`: Original images extracted from the datasets described in `original_datasets_files.txt`.
   - `02-edited`: Edited images so they fit an A4 page and all have the same shape.
   - `03-captured-nexus`: Images captured using a Google Nexus 7 tablet, trying the keep the document
     part as rectangular as possible.
   - `04-corrected-nexus`: Image with perspective roughly corrected by manually selecting the four corners
     and warping the image to the quadrilateral of the edited image using the Matlab script `correct_perspective.m`.
   - `05-corrected-nexus-scaled33`: Corrected images scaled to roughly fit the size under which documents will be
     viewed in a full HD (1080 x 1920) preview frame captured in a regular smartphone.
 - `model_name`: Name of the document (example: `datasheet001`). There are 30 documents, 5 instances of each document
   class (see below for the list of document classes). Documents are named from `001` to `005`.
 - `model_id`: Model id (example: `0`), 0-indexed. Value is between 0 and 29.
 - `modeltype_name`: Document class (example: `datasheet`). There are 6 document classes:
   - `datasheet`
   - `letter`
   - `magazine`
   - `paper`
   - `patent`
   - `tax`
 - `modeltype_id`: Model type id (example: `0`), 0-indexed. Value is between 0 and 5.
 - `model_subid`: Document sub-index (example: `1`).
 - `image_path`: Relative path to the model image (example: `05-corrected-nexus-scaled33/datasheet001.png`)
   under the dataset home directory.

Example of header + a random line:
    model_cat,model_name,model_id,modeltype_name,modeltype_id,model_subid,image_path
    02-edited,paper005,19,paper,3,4,02-edited/paper005.png

"""
# Copyright (c) 2018 Joseph Chazalon
# License: MIT

# IMPORT
################################################################################
################################################################################
from __future__ import division, absolute_import, print_function
import six
from six.moves import range

import os
import tarfile

import numpy as np
import pandas as pd
from sklearn.datasets.base import get_data_home, _fetch_remote, RemoteFileMetadata
from sklearn.utils import Bunch, check_random_state
from skimage.io import imread
from skimage.transform import resize as imresize, estimate_transform
import Polygon

from .poly_utils import isSelfIntersecting


# CONSTANTS
################################################################################
################################################################################
ARCHIVE_BASE_URL = 'https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/download/v2.0.0'

ARCHIVE_MODELS_FILENAME = 'models.tar.gz'
ARCHIVE_MODELS = RemoteFileMetadata(
    filename=ARCHIVE_MODELS_FILENAME,
    url=ARCHIVE_BASE_URL + '/' + ARCHIVE_MODELS_FILENAME,
    checksum=('6f9068624073f76b20f88352b2bac60b9e5de5a59819fc9db37fba1ee07cce8a'))

ARCHIVE_FRAMES_FILENAME = 'frames.tar.gz'
ARCHIVE_FRAMES = RemoteFileMetadata(
    filename=ARCHIVE_FRAMES_FILENAME,
    url=ARCHIVE_BASE_URL + '/' + ARCHIVE_FRAMES_FILENAME,
    checksum=('3acb8be143fc86c507d90d298097cba762e91a3abf7e2d35ccd5303e13a79eae'))

DATASET_CONTENT = {
    "models": (ARCHIVE_MODELS, "390MB", "Model images"),
    "frames": (ARCHIVE_FRAMES, "972MB", "Dataset content and metadata")
}

SD15CH1_DIRNAME = "smartdoc15-ch1_home"

MODEL_VARIANT_01_ORIGINAL = "01-original"
MODEL_VARIANT_02_EDITED = "02-edited"
MODEL_VARIANT_03_CAPTURED = "03-captured-nexus"
MODEL_VARIANT_04_CORRECTED = "04-corrected-nexus"
MODEL_VARIANT_05_SCALED33 = "05-corrected-nexus-scaled33"
MODEL_VARIANTS = [
    MODEL_VARIANT_01_ORIGINAL,
    MODEL_VARIANT_02_EDITED,
    MODEL_VARIANT_03_CAPTURED,
    MODEL_VARIANT_04_CORRECTED,
    MODEL_VARIANT_05_SCALED33
]


# Naive logging helpers
# ##############################################################################
__silent_log = False
def __logmsg(lvl, msg):
    if not __silent_log:
        print("%s: %s" % (lvl, msg))
def __info(msg):
    __logmsg("INFO", msg)
def __warn(msg):
    __logmsg("WARNING", msg)
def __err(msg, exception=Exception):
    __logmsg("ERROR", msg)
    raise exception(msg)


# Core functions
################################################################################
################################################################################

def load_sd15ch1_frames(data_home=None,
                        sample=1.0,
                        shuffle=False,
                        random_state=0,
                        download_if_missing=True,
                        load_images=False,
                        resize=None,
                        color=False,
                        with_model_classif_targets=True,
                        with_modeltype_classif_targets=True,
                        with_segmentation_targets=True,
                        with_model_shapes=True,
                        return_X_y=False,
                        ):
    """Loader for the SmartDoc2015 Challenge 1 dataset from CVC & L3i.

    Read more at:
        http://l3i.univ-larochelle.fr/ICDAR2015SmartDoc

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all data is stored according to how scikit-learn stores datasets:
        in '~/scikit_learn_data' subfolders.
    shuffle : boolean, optional, default: False
        If True the order of the dataset is shuffled to avoid having
        images of the same person grouped.
    sample: float or integer, optional, default: 1.0
        If float, sample must be between 0.0 (exclusive) and 1.0 (inclusive),
        and is interpreted as a fraction of the dataset to use
        (with a least one image);
        If int, sample must be between 1 and the size of the dataset, and
        is interpreted as the maximum number of images to load.
    random_state : int, RandomState instance or None, optional, default: 0
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    download_if_missing : boolean, optional, default: True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.
    load_images: boolean, optional, default: False
        If True then the list of image file paths in the output will be replaced
        by a numpy array of images. See the "Returns" and "Data types" sections 
        below for more details.
    resize: int or float, optional default: None, must be > 0 and < 20
        if `load_images` is `True` then images will be scaled: each dimension of the
        resulting image will be `original_dimension * resize`
        if `with_segmentation_targets` is `True` then coordinates will be scaled
    with_model_classif_targets: boolean, optional, default: True
        If True, the output will contain a numpy array indicating the model_id
        of each frame (an int corresponding to "datasheet001" and so on). See the 
        "Returns" and "Data types" sections below for more details.
    with_modeltype_classif_targets: boolean, optional, default: True
        If True, the output will contain a numpy array indicating the modeltype_id
        of each frame (an int corresponding to "datasheet" and so on). See the 
        "Returns" and "Data types" sections below for more details.
    with_segmentation_targets: boolean, optional, default: True
        If True, the output will contain a numpy array indicating the coordinates
        of the four corners of the model representation within each frame. See the 
        "Returns" and "Data types" sections below for more details.
    with_model_shapes: boolean, optional, default: True
        If True, the output will contain a numpy array indicating the shape 
        (width, height) of the model. See the "Returns" and "Data types" sections below 
        for more details.
    return_X_y : boolean, default=False.
        If True, returns a tuple instead of a Bunch object.
        See the "Returns" and "Data types" sections below for more details.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'images', only present if `load_images` is True;
        'image_paths', only present if `load_images` is False;
        'target_model_ids', only present if `with_model_classif_targets` is True;
        'target_modeltype_ids', only present if `with_modeltype_classif_targets` is True;
        'target_segmentations', only present if `with_segmentation_targets` is True;
        'model_shapes', nly present if `with_model_shapes` is True;
        and 'DESCR', the full description of the dataset.
    (images_or_paths, target_model_ids, target_modeltype_ids, target_segmentations, model_shapes) : 
        tuple if ``return_X_y`` is True, element presence follow the same rules as the Bunch object;
        for example `target_model_ids` will only be present if 

    Data types
    ----------
    images : numpy array, shape `(frame_count, frame_height, frame_width)` if frames are greyscale
        or `(frame_count, frame_height, frame_width, color_depth)` otherwise
    image_paths : numpy array of strings, shape `(frame_count, )`
    target_model_ids : numpy array of ints, shape `(frame_count, )`
    target_modeltype_ids : numpy array of ints, shape `(frame_count, )`
    target_segmentations : numpy array of floats, shape `(frame_count, 8)` where the second axis values 
        are: 'tl_x', 'tl_y', 'bl_x', 'bl_y', 'br_x', 'br_y', 'tr_x', 'tr_y'
    model_shapes : numpy array of ints, shape `(frame_count, 2)` where the second axis values are:
        'model_width', 'model_height'

    Raises
    ------
    IOError : if the dataset cannot be loaded (for example when `download_if_missing` is
        false and local data cannot be found.)
    ValueError : if `sample` or `resize` parameters are not in the appropriate type/range
    RuntimeError : upon image codec error


    Examples
    --------
    TODO examples
    """
    # Check frames are locally available
    # if not, try to download them
    data_dirs = __download_open_dataset(data_home, download_if_missing)

    # Read metadata file
    frames_metadata_path = os.path.join(data_dirs["frames"], "metadata.csv.gz")
    __info('Loading frames metadata from %s' % (frames_metadata_path, ))
    df = pd.read_csv(frames_metadata_path)

    # Sampling
    if not (0.0 < sample <= 1.0):
        __err("sample parameter must be a > 0.0 and <= 1.0.", ValueError)

    # Shuffling
    df_filtered = None
    if shuffle:
        df_filtered = df.sample(frac=sample, random_state=check_random_state(random_state))
    else:
        df_filtered = df.head(max(1, min(int(sample * df.shape[0]), df.shape[0])))

    # Collect elements for output structure
    output_elements = []  # (keyname, value)

    # Load images -- this changes the type of returned content for X
    if load_images:
        __info("Loading %d frames." % (df_filtered.shape[0], ))
        images = read_sd15ch1_images(data_dirs["frames"], np.array(df_filtered["image_path"]), resize, color)
        __info("Done loading frames.")
        output_elements.append(("images", images))
    else:
        # If we do not load images, then we return the path to all frames
        output_elements.append(("image_paths", np.array(df_filtered["image_path"])))

    # Add extra elements and target to output structure, if requested
    if with_model_classif_targets:
        output_elements.append(("target_model_ids", np.array(df_filtered["model_id"])))
    if with_modeltype_classif_targets:
        output_elements.append(("target_modeltype_ids", np.array(df_filtered["modeltype_id"])))
    if with_segmentation_targets:
        coords = np.array(df_filtered[['tl_x', 'tl_y', 'bl_x', 'bl_y', 'br_x', 'br_y', 'tr_x', 'tr_y']])
        if resize is not None:
            coords = coords * resize
        output_elements.append(("target_segmentations", coords))
    if with_model_shapes:
        output_elements.append(("model_shapes", np.array(df_filtered[['model_width', 'model_height']])))

    # Build returned object
    result = None
    if return_X_y:
        result = tuple(value for key, value in output_elements)
    else:
        # pack the results as a Bunch instance
        result = Bunch(
            DESCR="SmartDoc2015-Challenge1 dataset",
            **dict(output_elements)
        )

    return result
# // load_sd15ch1_frames
###############################################################################


def load_sd15ch1_models(data_home=None,
                        download_if_missing=True,
                        load_images=False,
                        variant=MODEL_VARIANT_05_SCALED33,
                        color=False,
                        with_model_ids=True,
                        with_modeltype_ids=True,
                        return_X_y=False,
                        ):
    # Check frames are locally available
    # if not, try to download them
    data_dirs = __download_open_dataset(data_home, download_if_missing)

    # Read metadata file
    models_metadata_path = os.path.join(data_dirs["models"], "metadata.csv.gz")
    __info('Loading frames metadata from %s' % (models_metadata_path, ))
    df = pd.read_csv(models_metadata_path)

    # Filter the variant we want to load
    if variant not in MODEL_VARIANTS:
        __err("Unknown model variant: '%s'." % variant, ValueError)
    df = df[df["model_cat"] == variant]

    # Collect elements for output structure
    output_elements = []  # (keyname, value)

    # Load images -- this changes the type of returned content for X
    # If we need to load the images, there is a caveat:
    # for the variant "01-original", the images do not have the same shape, so we
    # return a list of numpy arrays, instead of a single array.
    if load_images:
        __info("Loading model images.")
        images = None
        if variant == MODEL_VARIANT_01_ORIGINAL:
            images = [read_sd15ch1_image(data_dirs["models"], path, None, color)
                      for path in df["image_path"]]
        else:
            images = read_sd15ch1_images(data_dirs["models"], np.array(df["image_path"]), None, color)
        __info("Done loading images.")
        output_elements.append(("images", images))
    else:
        # If we do not load images, then we return the path to all frames
        output_elements.append(("image_paths", np.array(df["image_path"])))

    if with_model_ids:
        output_elements.append(("model_ids", np.array(df["model_id"])))

    if with_modeltype_ids:
        output_elements.append(("modeltype_ids", np.array(df["modeltype_id"])))

    # Build returned object
    result = None
    if return_X_y:
        result = tuple(value for key, value in output_elements)
    else:
        # pack the results as a Bunch instance
        result = Bunch(
            DESCR="SmartDoc2015-Challenge1 models",
            **dict(output_elements)
        )

    return result
# // load_sd15ch1_models


def read_sd15ch1_image(root_dir,
                       image_relative_path,
                       resize=None,
                       color=False):
    real_path = os.path.join(root_dir, image_relative_path)
    __info("Loading image '%s'." % (real_path, ))
    img = imread(real_path, as_grey=(not color))
    # Checks if jpeg reading worked. Refer to skimage issue #3594 for more details.
    if img.ndim is 0:
        __err("Failed to read the image file %s, "
                           "Please make sure that libjpeg is installed"
                           % real_path, RuntimeError)
    if resize is not None:
        if not (0 < resize <= 20):
            __err("resize parameter but be > 0 and < 20.", ValueError)
        resize_f = float(resize)
        h, w = img.shape[0], img.shape[1]
        h = int(resize_f * h)
        w = int(resize_f * w)
        img = imresize(img, (h, w))

    return img
# // read_sd15ch1_image


def read_sd15ch1_images(root_dir,
                        image_relative_path_seq,
                        resize=None,
                        color=False):
    """
    WARNING
    -------
     - All images must have the same shape (this is the case for the frames, and all models but the
       ones of the "01-original" category).
     - Loading many images at one can quickly fill up your RAM.

    Returns
    -------
     - np.array((number_of_images, images_height, images_width)) if `color` is `False`
     - np.array((number_of_images, images_height, images_width, image_channels)) otherwise.
    """

    # Read first image, if any, to get image shape
    # Note: all images must have the same shape
    if len(image_relative_path_seq) == 0:
        return np.array([])

    # We have a least 1 element
    img0 = read_sd15ch1_image(root_dir, image_relative_path_seq[0], resize, color)

    # allocate some contiguous memory to host the decoded images
    dim_axis0 = (len(image_relative_path_seq), )  # make it a tuple
    dim_axis_others = img0.shape
    imgs_shape = dim_axis0 + dim_axis_others
    __info("About to allocate %d bytes for an array of shape %s." % (np.prod(imgs_shape) * 4, imgs_shape))
    imgs = np.zeros(imgs_shape, dtype=np.float32)

    # Handle first image
    imgs[0, ...] = img0

    # Loop over other images
    for ii, rel_path in enumerate(image_relative_path_seq[1:], start=1):
        imgi = read_sd15ch1_image(root_dir, rel_path, resize, color)
        if imgi.shape != dim_axis_others:
            __err("All images must have the same shape. Inconsistent dataset. Aborting loading.", RuntimeError)
        imgs[ii, ...] = imgi

    return imgs
# // read_sd15ch1_images


# Helpers needed when loading images manually with read_sd15ch_image{,s} functions
# ------------------------------------------------------------------------------
def get_sd15ch1_basedir_frames(data_home=None):
    data_home = get_data_home(data_home=data_home)
    sd15ch1_home = os.path.join(data_home, SD15CH1_DIRNAME)
    basedir = os.path.join(sd15ch1_home, "frames")
    return basedir

def get_sd15ch1_basedir_models(data_home=None):
    data_home = get_data_home(data_home=data_home)
    sd15ch1_home = os.path.join(data_home, SD15CH1_DIRNAME)
    basedir = os.path.join(sd15ch1_home, "models")
    return basedir


# Download management
# ------------------------------------------------------------------------------
def __download_open_dataset(data_home=None, download_if_missing=True):
    """Helper function to download any missing SD15-CH1 data.

    The dataset will be stored like this:
        ${data_home}/smartdoc15-ch1_home/frames:
        ├── background01
        │   ├── datasheet001
        │   │   ├── frame_0001.jpeg
        │   │   ├── [...]
        │   │   └── frame_0235.jpeg
        │   ├── [...]
        │   └── tax005
        │       └── [...]
        ├── background02
        |   └── [...]
        ├── background03
        |   └── [...]
        ├── background04
        |   └── [...]
        ├── background05
        |   └── [...]
        └── metadata.csv.gz

        ${data_home}/smartdoc15-ch1_home/models:
        ├── 01-original
        │   ├── datasheet001.png
        │   ├── [...]
        │   └── tax005.png
        ├── 02-edited
        │   ├── datasheet001.png
        │   ├── [...]
        │   └── tax005.png
        ├── 03-captured-nexus
        │   ├── datasheet001.jpg # JPG images here
        │   ├── [...]
        │   └── tax005.jpg
        ├── 04-corrected-nexus
        │   ├── datasheet001.png
        │   ├── [...]
        │   └── tax005.png
        ├── 05-corrected-nexus-scaled33
        │   ├── datasheet001.png
        │   ├── [...]
        │   └── tax005.png
        ├── correct_perspective.m
        └── original_datasets_files.txt
    """
    data_home = get_data_home(data_home=data_home)
    sd15ch1_home = os.path.join(data_home, SD15CH1_DIRNAME)

    if not os.path.exists(sd15ch1_home):
        os.makedirs(sd15ch1_home)

    data_dirs = {}
    for subdir, (archive, size, description) in six.iteritems(DATASET_CONTENT):
        data_folder_path = os.path.join(sd15ch1_home, subdir)
        data_dirs[subdir] = data_folder_path

        if not os.path.exists(data_folder_path):
            archive_path = os.path.join(sd15ch1_home, archive.filename)
            # (later) FIXME this is a naive test for existing files
            if not os.path.exists(archive_path):
                if download_if_missing:
                    __info("Downloading file %s (%s): %s" % (archive.filename, size, archive.url))
                    _fetch_remote(archive, dirname=sd15ch1_home)
                else:
                    __err("%s is missing" % archive_path, IOError)

            __info("Decompressing the data archive to %s" % (data_folder_path, ))
            tarfile.open(archive_path, "r:gz").extractall(path=data_folder_path)
            os.remove(archive_path)

    return data_dirs
# // __download_open_dataset




# Evaluation
# ------------------------------------------------------------------------------

def eval_sd15ch1_segmentations(segmentations, target_segmentations, model_shapes, frame_resize_factor=1.0, print_summary=False):
    # frame_resize_factor it the value of the resize factor applied to the frames. 
    # It will be inverted to recover the correct coordinates.

    # First check everything has the right type and shape
    # TODO check types
    seg_shape = segmentations.shape
    if len(seg_shape) != 2 or seg_shape[1] != 8:
        __err("eval_sd15ch1_segmentations: segmentations parameter "
              "must be a numpy array of shape (NUM_FRAMES, 8).",
              ValueError)
    tarseg_shape = target_segmentations.shape
    if len(seg_shape) != 2 or seg_shape[1] != 8:
        __err("eval_sd15ch1_segmentations: target_segmentations parameter "
              "must be a numpy array of shape (NUM_FRAMES, 8).",
              ValueError)
    mdlshapes_shape = model_shapes.shape
    if len(mdlshapes_shape) != 2 or mdlshapes_shape[1] != 2:
        __err("eval_sd15ch1_segmentations: model_shapes parameter "
              "must be a numpy array of shape (NUM_FRAMES, 2).",
              ValueError)
    num_frames = seg_shape[0]
    if tarseg_shape[0] != num_frames or mdlshapes_shape[0] != num_frames:
        __err("eval_sd15ch1_segmentations: 'segmentations', 'target_segmentations' and 'model_shapes' parameters "
              "must all have the same dimension on axis 0 (number of frames).", 
              ValueError)

    # Scale coordinates back to original frame size
    segmentations_scaled = segmentations / frame_resize_factor

    # Evaluate the segmentation for each frame result
    eval_result = np.zeros((num_frames))
    for ii in range(num_frames):
        # Warp coordinates so each pixel represents the same physical surface
        # in the real plane of the document object
        # point order: top-left, bottom-left, bottom-right, top-right
        # referential: x+ toward right, y+ toward down
        # witdh = object original size left to right
        # height = object original size top to bottom
        found_obj_coordinates_frame = segmentations_scaled[ii].reshape((-1, 2))
        true_obj_coordinates_frame = target_segmentations[ii].reshape((-1, 2))
        true_obj_width_real, true_obj_height_real = model_shapes[ii]
        true_obj_coordinates_real = np.array([[0, 0],
                                              [0, true_obj_height_real],
                                              [true_obj_width_real, true_obj_height_real],
                                              [true_obj_width_real, 0]])
        tform = estimate_transform('projective', true_obj_coordinates_frame, true_obj_coordinates_real)
        found_obj_coordinates_real = tform(found_obj_coordinates_frame)

        # Compute IoU
        poly_target = Polygon.Polygon(true_obj_coordinates_real)  #.reshape(-1,2))
        poly_test = Polygon.Polygon(found_obj_coordinates_real)  #.reshape(-1,2))
        poly_inter = None

        area_target = area_test = area_inter = area_union = 0.0
        # (sadly, we must check for self-intersecting polygons which mess the interection computation)
        if isSelfIntersecting(poly_target):
            __err("eval_sd15ch1_segmentations: target_segmentations[%d]: ground truth coordinates are not in right order "
                  "(the resulting polygon is self intersecting). Cannot compute overlap. Aborting." % (ii,),
                  ValueError)

        area_target = poly_target.area()
        
        if area_target < 0.0000000001:
            __err("eval_sd15ch1_segmentations: target_segmentations[%d]: ground truth coordinates form a "
                  "polygon degenerated to one point. Cannot compute overlap. Aborting." % (ii,),
                  ValueError)

        if isSelfIntersecting(poly_test):
            __warn("eval_sd15ch1_segmentations: segmentation[%d]: segmentation coordinates are not in right order "
                   "(the resulting polygon is self intersecting). Overlap assumed to be 0." % (ii,))
        else :
            poly_inter = poly_target & poly_test
            # Polygon.IO.writeSVG('_tmp/polys-%03d.svg'%fidx, [poly_target, poly_test, poly_inter]) # dbg
            # poly_inter should not self-intersect, but may have more than 1 contour
            area_test = poly_test.area()
            area_inter = poly_inter.area()

            # Little hack to cope with float precision issues when dealing with polygons:
            #   If intersection area is close enough to target area or GT area, but slighlty >,
            #   then fix it, assuming it is due to rounding issues.
            area_min = min(area_target, area_test)
            if area_min < area_inter and area_min * 1.0000000001 > area_inter :
                area_inter = area_min
                # __warn("Capping area_inter.")
            
        area_union = area_test + area_target - area_inter
        jaccard_index = area_inter / area_union
        eval_result[ii] = jaccard_index

    # Print summary if asked
    if print_summary:
        try:
            import scipy.stats
        except ImportError:
            __warn("eval_sd15ch1_segmentations: cannot import scipy.stats. Print summary deactivated. "
                   "Please install scipy to enable it.")
        else:
            values = eval_result
            # nobs, minmax, mean, variance, skewness, kurtosis = stats.describe(values)
            desc_res = scipy.stats.describe(values)
            std = np.sqrt(desc_res.variance)
            cil, cih = scipy.stats.norm.interval(0.95,loc=desc_res.mean,scale=std/np.sqrt(len(values)))

            __info("eval_sd15ch1_segmentations: ----------------------------------------------")
            __info("eval_sd15ch1_segmentations: Evaluation report")
            __info("eval_sd15ch1_segmentations: ----------------------------------------------")
            __info("eval_sd15ch1_segmentations: observations: %5d" % (desc_res.nobs, ))
            __info("eval_sd15ch1_segmentations: mean:         %8.2f (CI@95%%: %.3f, %.3f)" % (desc_res.mean, cil, cih))
            __info("eval_sd15ch1_segmentations: min-max:          %.3f - %.3f" % desc_res.minmax)
            __info("eval_sd15ch1_segmentations: variance:         %.3f (std: %.3f)" % (desc_res.variance, std))
            __info("eval_sd15ch1_segmentations: ----------------------------------------------")

    # return IoU output
    return eval_result


def eval_sd15ch1_classifications(labels, target_labels):
    # TODO doc
    # TODO forward to sklearn classifier accuracy evaluation?
    raise NotImplementedError("eval_sd15ch1_classifications: Not implemented yet.")


