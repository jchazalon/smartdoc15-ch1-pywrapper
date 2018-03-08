Python wrapper for the SmartDoc 2015 - Challenge 1 Dataset
==========================================================

`The source for this project is available here <https://github.com/jchazalon/smartdoc15-ch1-pywrapper>`_.


`The SmartDoc 2015 Challenge 1 dataset <http://smartdoc.univ-lr.fr/>`_ was originally created for the SmartDoc 2015 competition focusing on the evaluation of document image acquisition method using smartphones. The challenge 1, in particular, consisted in detecting and segmenting document regions in video frames extracted from the preview stream of a smartphone.

This dataset was packaged in a new format and a python wrapper (this current package) was created to facilitate its usage.


Dataset version
---------------
The version of the dataset used by this wrapper is: **2.0.0**.

The source for the dataset is here: https://github.com/jchazalon/smartdoc15-ch1-dataset


Sample usage
------------
There are 3 tasks this Python package helps you to test your methods against, but first of all you have to properly install this package:

.. code:: console

    pip install smartdoc_ch1

A good practice is to install such package in a virtual environment.
We recommend to use `Virtualenv Wrapper <http://virtualenvwrapper.readthedocs.org/>`_ to use virtual environments.


Task 1: Segmentation
''''''''''''''''''''
**Segmentation**: this is the original task.
    Inputs are video frames, and expected output is a composed of the coordinated of the four corners of the document image in each frame (top left, bottom left, bottom right and top right).
    The evaluation is performed by computing the intersection over union ("IoU" or also "Jaccard index") of the expected document region and the found region. The tricky thing is that the coordinates are projected to the document referential in order to allow comparisons between different frames and different document models.
    The original evaluation code is available at https://github.com/jchazalon/smartdoc15-ch1-eval, and the Python wrapper also contains an implementation using the new data format.

    - read dataset
    - [opt. read models]
    - [opt. train/test split + train]
    - test
    - eval


Task 2: Model classification
''''''''''''''''''''''''''''
**Model classification**: this is a new task.
    Inputs are video frames, and expected output is the identifier of the document model represented in each frame.
    There are 30 models named "datasheet001", "datasheet002", ..., "tax005".
    The evaluation is performed as any multi-class classification task.

    - read dataset
    - [opt. read models]
    - [opt. train/test split + train]
    - test
    - eval


Task 3: Model type classification
'''''''''''''''''''''''''''''''''
**Model type classification**: this is a new task.
    Inputs are video frames, and expected output is the identifier of the document model **type** represented in each frame.
    There are 6 models types, each having 5 members, named "datasheet", "letter", "magazine", "paper", "patent" and "tax".
    The evaluation is performed as any multi-class classification task.

    - read dataset
    - [opt. read models]
    - [opt. train/test split + train]
    - test
    - eval

Optional: Using model images
''''''''''''''''''''''''''''


Manual download option
----------------------

If you are behind a proxy, have a slow connexion or for any other reason, you may want to download the dataset manually instead of letting the Python wrapper do it for you.
This is simple: 

1. download the ``frames.tar.gz`` and ``models.tar.gz`` files from https://github.com/jchazalon/smartdoc15-ch1-dataset/releases to some local directory;

2. choose where you want to store the files and manually create the file hierarchy (the ``smartdoc_ch1_home`` intermediate directory is important here):

.. code:: console

    mkdir -p PATH_TO_STORAGE_DIR/smartdoc_ch1_home/frames
    mkdir -p PATH_TO_STORAGE_DIR/smartdoc_ch1_home/models

3. extract the archives to their target directories:

.. code:: console

    tar -xzf PATH_TO_FRAMES.TAR.GZ -C PATH_TO_STORAGE_DIR/smartdoc_ch1_home/frames
    tar -xzf PATH_TO_MODELS.TAR.GZ -C PATH_TO_STORAGE_DIR/smartdoc_ch1_home/models

Then, make sure you specify ``data_home=PATH_TO_STORAGE_DIR`` and ``download_if_missing=False`` when you call the ``load_sd15ch1_frames`` and ``load_sd15ch1_models`` functions. The functions ``get_sd15ch1_basedir_frames`` and
``get_sd15ch1_basedir_models`` also require that you specify ``data_home=PATH_TO_STORAGE_DIR``.

By default, the path to local dataset storage complies with Scikit-learn standard location: ``PATH_TO_STORAGE_DIR=~/scikit_learn_data``


API
---
TODO DOC

.. code:: python

    MODEL_VARIANT_01_ORIGINAL = "01-original"
    MODEL_VARIANT_02_EDITED = "02-edited"
    MODEL_VARIANT_03_CAPTURED = "03-captured-nexus"
    MODEL_VARIANT_04_CORRECTED = "04-corrected-nexus"
    MODEL_VARIANT_05_SCALED33 = "05-corrected-nexus-scaled33"

    load_sd15ch1_frames(data_home=None,
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
                            )

    load_sd15ch1_models(data_home=None,
                            download_if_missing=True,
                            load_images=False,
                            variant=MODEL_VARIANT_05_SCALED33,
                            color=False,
                            with_model_ids=True,
                            with_modeltype_ids=True,
                            return_X_y=False,
                            )

    read_sd15ch1_image(root_dir,
                           image_relative_path,
                           resize=None,
                           color=False)

    read_sd15ch1_images(root_dir,
                            image_relative_path_seq,
                            resize=None,
                            color=False)
                            
    get_sd15ch1_basedir_frames(data_home=None)

    get_sd15ch1_basedir_models(data_home=None)

    eval_sd15ch1_segmentations(segmentations, 
                               target_segmentations, 
                               model_shapes, 
                               frame_resize_factor=1.0, 
                               print_summary=False)

    eval_sd15ch1_classifications(labels, 
                                 target_labels)


