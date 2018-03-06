Python wrapper for the SmartDoc 2015 - Challenge 1 Dataset
==========================================================

Overview + Smartdoc URL (github dataset + original dataset)
2 words about the dataset


Important resources
-------------------
Version, important URLS, etc.

`The source for this project is available here
<https://github.com/jchazalon/smartdoc15-ch1-pywrapper>`_.

The version of the dataset used by this wrapper is:
2.0.0 [TODO URL release]


Sample usage
------------

- installation
- Task 1: segmentation (workflow overview + commands)
	- read dataset
	- [opt. read models]
	- [opt. train/test split + train]
	- test
	- eval
- Task 2: model classification (workflow overview + commands)
	- read dataset
	- [opt. read models]
	- [opt. train/test split + train]
	- test
	- eval
- Task 3: model type classification (workflow overview + commands)
	- read dataset
	- [opt. read models]
	- [opt. train/test split + train]
	- test
	- eval



API
---
TODO

``load_sd15ch1_frames(data_home=None,
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
                        )``


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


TODO eval task1 / seg

TODO eval task2 / mdl clf

TODO eval task3 / mdl type clf
