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
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets.base import get_data_home, _fetch_remote, RemoteFileMetadata
from skimage.io import imread
from skimage.transform import resize as imresize


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


# Naive logging helpers
# ##############################################################################
class SimpleLoggerTrait(object):
    def __log_ensure_init(self):
        if not hasattr(self, "__silent_log"):
            self.__silent_log = True
    def _log_activate(self):
        self.__log_check_init()
        self.__silent_log = False
    def __logmsg(self, lvl, msg):
        self.__log_ensure_init()
        if not self.__silent_log:
            print("%s: %s" % (lvl, msg))
    def _info(self, msg):
        self.__logmsg("INFO", msg)
    def _warn(self, msg):
        self.__logmsg("WARNING", msg)
        warnings.warn(msg, stacklevel=2)
    def _err(self, msg, exception=Exception):
        self.__logmsg("ERROR", msg)
        raise exception(msg)



# Helper functions for downloading and loading
# ##############################################################################
def _ensure_dataset_is_downloaded(sd15ch1_home, download_if_missing):
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

    if not os.path.exists(sd15ch1_home):
        os.makedirs(sd15ch1_home)

    data_dirs = {}
    for subdir, (archive, size, description) in six.iteritems(DATASET_CONTENT):
        data_folder_path = os.path.join(sd15ch1_home, subdir)
        data_dirs[subdir] = data_folder_path

        # The existence of the target directory indicates a complete installation
        install_successful_filename = os.path.join(data_folder_path, "_INSTALL_SUCCESSFUL_")
        if not os.path.exists(install_successful_filename):
            archive_path = os.path.join(sd15ch1_home, archive.filename)
            # FIXME we should check the sum of the archive file
            if not os.path.exists(archive_path):
                if download_if_missing:
                    print("Downloading file %s (%s): %s" % (archive.filename, size, archive.url))
                    _fetch_remote(archive, dirname=sd15ch1_home)
                else:
                    msg = "%s is missing" % archive_path
                    print(msg)
                    raise IOError(msg)

            print("Decompressing the data archive to %s" % (data_folder_path, ))
            tarfile.open(archive_path, "r:gz").extractall(path=data_folder_path)
            os.remove(archive_path)
            # Touch indicator file
            with open(install_successful_filename, 'a') as f:
                f.write("OK\n")


def _read_image(full_path, color=False, scale_factor=None):
    final_scale_factor = None
    if scale_factor is not None:
        if not (0 < scale_factor <= 20):
            msg = "'scale_factor' parameter must be > 0 and < 20."
            print(msg)
            raise ValueError(msg)
        final_scale_factor = float(scale_factor)
    # print("Loading image '%s'." % full_path)
    img = imread(full_path, as_gray=(not color))
    # Checks if jpeg reading worked. Refer to skimage issue #3594 for more details.
    if img.ndim is 0:
        msg = ("Failed to read the image file %s, "
               "please make sure that libjpeg is installed."
               % full_path)
        print(msg)
        raise RuntimeError(msg)
    
    if final_scale_factor is not None and not np.isclose(final_scale_factor, 1.):
        h, w = img.shape[0], img.shape[1]
        h = int(final_scale_factor * h)
        w = int(final_scale_factor * w)
        img = imresize(img, (h, w))
    if not color:  # skimage.io.imread returns [0-1] float64 images when as_gray is True
        img = np.uint8(img * 255)
    return img



# Core functions
################################################################################
################################################################################
class Frame(dict, SimpleLoggerTrait):
#             dfi_dict["bg_name"] = dfi["bg_name"]  # string
#             dfi_dict["bg_id"] = dfi["bg_id"]  # int 0-indexed
#             dfi_dict["model_name"] = dfi["model_name"]  # string
#             dfi_dict["model_id"] = dfi["model_id"]  # int 0-indexed
#             dfi_dict["modeltype_name"] = dfi["modeltype_name"]  # string
#             dfi_dict["modeltype_id"] = dfi["modeltype_id"]  # int-0-indexed
#             dfi_dict["model_subid"] = dfi["model_subid"]  # int 0-indexed
#             dfi_dict["image_path_relative"] = dfi["image_path"]  # string (path)
#             dfi_dict["image_path_absolute"] = os.path.join(
#                     self.framesdir, dfi["image_path"])  # string (path)
#             dfi_dict["frame_index"] = dfi["frame_index"]  # int 1-indexed (WARNING)
#             dfi_dict["model_width"] = dfi["model_width"]  # float, 10^(-4) meter
#             dfi_dict["model_height"] = dfi["model_height"]  # float, 10^(-4) meter
#                 "tl_x" : dfi["tl_x"],  # float (pixel coordinates, frame image referential)
#                 "tl_y" : dfi["tl_y"],  # (same)
#                 "bl_x" : dfi["bl_x"],  # (same)
#                 "bl_y" : dfi["bl_y"],  # (same)
#                 "br_x" : dfi["br_x"],  # (same)
#                 "br_y" : dfi["br_y"],  # (same)
#                 "tr_x" : dfi["tr_x"],  # (same)
#                 "tr_y" : dfi["tr_y"]}  # (same)
#             dfi_dict["_scale_factor"] = self._scale_factor
    # Note: we do not store any scaled segmentation
    def __init__(self, value_dict):
        dict.__init__(self, value_dict)
        # This object should be unmutable, we could overwrite __setitem__ to raise an error
   
    def read_image(self, color=False, force_scale_factor=None):
        final_scale_factor = None
        if force_scale_factor is not None:
            if not (0 < force_scale_factor <= 20):
                self.__err("force_scale_factor parameter but be > 0 and < 20.", ValueError)
            final_scale_factor = float(force_scale_factor)

        if final_scale_factor is None:
            final_scale_factor = self["_scale_factor"]  # recover default scale factor

        real_path = self["image_path_absolute"]

        return _read_image(real_path, color, final_scale_factor)

    @property
    def segmentation_dict(self):
        segmentation = {
                k: self[k]
                for k in 
                ["tl_x", "tl_y", "bl_x", "bl_y", "br_x", "br_y", "tr_x", "tr_y"]
            }
        return segmentation
    
    @property
    def segmentation_dict_scaled(self):
        segmentation = self.segmentation_dict
        if self["_scale_factor"] is not None:
            segmentation = {k: v*self["_scale_factor"] for k, v in six.iteritems(segmentation)}
        return segmentation
    
    @property
    def segmentation_list(self):
        return [self[k]
                for k in 
                ["tl_x", "tl_y", "bl_x", "bl_y", "br_x", "br_y", "tr_x", "tr_y"]]
    
    @property
    def segmentation_list_scaled(self):
        segmentation = self.segmentation_list
        if self["_scale_factor"] is not None:
            segmentation = list(np.array(segmentation_list) * self["_scale_factor"])
        return segmentation
    

class Dataset(list, SimpleLoggerTrait):
    def __init__(self, 
                 data_home=None,
                 download_if_missing=True,
                 frame_scale_factor=None,
                 shuffle=False,
                 random_state=None):
        self.data_home = get_data_home(data_home=data_home)
        self.download_if_missing = download_if_missing

        self.sd15ch1_home = os.path.join(self.data_home, SD15CH1_DIRNAME)
        self.framesdir = os.path.join(self.sd15ch1_home, "frames")

        self._scale_factor = None
        if frame_scale_factor is not None:
            if not (0 < frame_scale_factor <= 20):
                self.__err("frame_scale_factor parameter but be > 0 and < 20.", ValueError)
            self._scale_factor = float(frame_scale_factor)

        # Open or try download or raise an exception if dataset is unavailable
        _ensure_dataset_is_downloaded(self.sd15ch1_home, self.download_if_missing)

        # Read metadata file
        frames_metadata_path = os.path.join(self.framesdir, "metadata.csv.gz")
        self._info('Loading frames metadata from %s' % (frames_metadata_path, ))
        self._rawdata = pd.read_csv(frames_metadata_path)

        if shuffle:
            self._rawdata = self._rawdata.sample(frac=1, axis=0, random_state=random_state).reset_index(drop=True)

        for _rid, rseries in self._rawdata.iterrows():
            dfi_dict = {}
            # Copy content
            for colname in self._rawdata:  # .keys()
                dfi_dict[colname] = rseries[colname]
            # Add extra entries
            dfi_dict["image_path_absolute"] = os.path.join(
                    self.framesdir, rseries["image_path"])  # string (path)
            dfi_dict["_scale_factor"] = self._scale_factor  # float
            # dfi_dict["frame_uid"] = rid  # int  # does not survive shuffling
            # hint: use df.reindex(np.random.permutation(df.index)) and keep original uids
            # Store
            self.append(Frame(dfi_dict))
        
        self._unique_background_ids = None
        self._unique_background_names = None
        self._unique_model_ids = None
        self._unique_model_names = None
        self._unique_modeltype_ids = None
        self._unique_modeltype_names = None

    @property
    def scale_factor(self):
        return self._scale_factor

    @property
    def model_classif_targets(self):
        return np.array(self._rawdata["model_id"])
    @property
    def modeltype_classif_targets(self):
        return np.array(self._rawdata["modeltype_id"])
    @property
    def segmentation_targets(self):
        coords = np.array(self._rawdata[['tl_x', 'tl_y', 'bl_x', 'bl_y', 'br_x', 'br_y', 'tr_x', 'tr_y']])
        if self._scale_factor is not None:
            coords = coords * self._scale_factor
        return coords
    @property
    def model_shapes(self):
        return np.array(self._rawdata[['model_width', 'model_height']])
    @property
    def background_labels(self):
        return np.array(self._rawdata['bg_id'])

    @property
    def raw_dataframe(self):
        return self._rawdata

    @property
    def unique_background_names(self):
        if self._unique_background_names is None:
            self._unique_background_names = self._rawdata['bg_name'].unique()
            self._unique_background_names.sort()
        return self._unique_background_names
    @property
    def unique_background_ids(self):
        if self._unique_background_ids is None:
            self._unique_background_ids = self._rawdata['bg_id'].unique()
            self._unique_background_ids.sort()
        return self._unique_background_ids
    @property
    def unique_model_names(self):
        if self._unique_model_names is None:
            self._unique_model_names = self._rawdata['model_name'].unique()
            self._unique_model_names.sort()
        return self._unique_model_names
    @property
    def unique_model_ids(self):
        if self._unique_model_ids is None:
            self._unique_model_ids = self._rawdata['model_id'].unique()
            self._unique_model_ids.sort()
        return self._unique_model_ids
    @property
    def unique_modeltype_names(self):
        if self._unique_modeltype_names is None:
            self._unique_modeltype_names = self._rawdata['modeltype_name'].unique()
            self._unique_modeltype_names.sort()
        return self._unique_modeltype_names
    @property
    def unique_modeltype_ids(self):
        if self._unique_modeltype_ids is None:
            self._unique_modeltype_ids = self._rawdata['modeltype_id'].unique()
            self._unique_modeltype_ids.sort()
        return self._unique_modeltype_ids

    def iter_frame_images(self, color=False, force_scale_factor=None):
        for frame in self:
            yield frame.read_image(color, force_scale_factor)


class Model(dict, SimpleLoggerTrait):
#                 mdli_dict["model_cat"] = mdli["model_cat"]  # string
#                 mdli_dict["model_name"] = mdli["model_name"]  # string
#                 mdli_dict["model_id"] = mdli["model_id"]  # int 0-indexed
#                 mdli_dict["modeltype_name"] = mdli["modeltype_name"]  # string
#                 mdli_dict["modeltype_id"] = mdli["modeltype_id"]  # int-0-indexed
#                 mdli_dict["model_subid"] = mdli["model_subid"]  # int 0-indexed
#                 mdli_dict["image_path_relative"] = mdli["image_path"]  # string (path)
#                 mdli_dict["image_path_absolute"] = os.path.join(
#                         self.modelsdir, mdli["image_path"])  # string (path)
    def __init__(self, value_dict):
        dict.__init__(self, value_dict)
    def read_image(self, color=False, scale_factor=None):
        real_path = self["image_path_absolute"]
        return _read_image(real_path, color, scale_factor)


class Models(list, SimpleLoggerTrait):
    VARIANT_01_ORIGINAL = "01-original"
    VARIANT_02_EDITED = "02-edited"
    VARIANT_03_CAPTURED = "03-captured-nexus"
    VARIANT_04_CORRECTED = "04-corrected-nexus"
    VARIANT_05_SCALED33 = "05-corrected-nexus-scaled33"
    VARIANTS = [
        VARIANT_01_ORIGINAL,
        VARIANT_02_EDITED,
        VARIANT_03_CAPTURED,
        VARIANT_04_CORRECTED,
        VARIANT_05_SCALED33]

    def __init__(self, 
            data_home=None,
            download_if_missing=True,
            variant="05-corrected-nexus-scaled33"):

        self.data_home = get_data_home(data_home=data_home)
        self.download_if_missing = download_if_missing

        self.sd15ch1_home = os.path.join(self.data_home, SD15CH1_DIRNAME)
        self.modelsdir = os.path.join(self.sd15ch1_home, "models")

        # Open or try download or raise an exception if dataset is unavailable
        _ensure_dataset_is_downloaded(self.sd15ch1_home, self.download_if_missing)

        # Read metadata file
        models_metadata_path = os.path.join(self.modelsdir, "metadata.csv.gz")
        self._info('Loading frames metadata from %s' % (models_metadata_path, ))
        df = pd.read_csv(models_metadata_path)

        # Filter the variant we want to load
        if variant not in Models.VARIANTS:
            self._err("Unknown model variant: '%s'." % variant, ValueError)
        self._rawdata = df[df["model_cat"] == variant]

        
        for _rid, rseries in self._rawdata.iterrows():
            mdli_dict = {}
            # Copy content
            for colname in self._rawdata:  # .keys()
                mdli_dict[colname] = rseries[colname]
            # Add extra entries
            mdli_dict["image_path_absolute"] = os.path.join(
                    self.modelsdir, rseries["image_path"])  # string (path)
            # Store
            self.append(Model(mdli_dict))
            
        self._unique_model_ids = None
        self._unique_model_names = None
        self._unique_modeltype_ids = None
        self._unique_modeltype_names = None

            
    @property
    def raw_dataframe(self):
        return self._rawdata

    @property
    def model_ids(self):
        return np.array(self._rawdata["model_id"])
    @property
    def modeltype_ids(self):
        return np.array(self._rawdata["modeltype_id"])

    @property
    def unique_model_names(self):
        if self._unique_model_names is None:
            self._unique_model_names = self._rawdata['model_name'].unique()
            self._unique_model_names.sort()
        return self._unique_model_names
    @property
    def unique_model_ids(self):
        if self._unique_model_ids is None:
            self._unique_model_ids = self._rawdata['model_id'].unique()
            self._unique_model_ids.sort()
        return self._unique_model_ids
    @property
    def unique_modeltype_names(self):
        if self._unique_modeltype_names is None:
            self._unique_modeltype_names = self._rawdata['modeltype_name'].unique()
            self._unique_modeltype_names.sort()
        return self._unique_modeltype_names
    @property
    def unique_modeltype_ids(self):
        if self._unique_modeltype_ids is None:
            self._unique_modeltype_ids = self._rawdata['modeltype_id'].unique()
            self._unique_modeltype_ids.sort()
        return self._unique_modeltype_ids

    def iter_model_images(self, color=False, scale_factor=None):
        for mdl in self:
            yield mdl.read_image(color, scale_factor)
