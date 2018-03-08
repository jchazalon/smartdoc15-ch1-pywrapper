#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the python wrapper for the SmartDoc2015-Challenge 1 dataset and tools.
"""
from __future__ import division, absolute_import, print_function
import six

import unittest

import os
import shutil

import numpy as np


from smartdoc15_ch1  import (load_sd15ch1_frames, load_sd15ch1_models,
    MODEL_VARIANT_01_ORIGINAL, MODEL_VARIANT_02_EDITED, MODEL_VARIANT_03_CAPTURED,
    MODEL_VARIANT_04_CORRECTED, MODEL_VARIANT_05_SCALED33, eval_sd15ch1_segmentations,
    eval_sd15ch1_classifications, get_sd15ch1_basedir_frames, get_sd15ch1_basedir_models)


class Sd15LoaderTestCases(unittest.TestCase):
    tmpdir = "/tmp/testsd15/"
    def __clean_tmpdir(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.makedirs(self.tmpdir)

    def setUp(self):
        self.__clean_tmpdir()

    def tearDown(self):
        self.__clean_tmpdir()


    def test_lf_download_if_missing_false(self):
        with self.assertRaises(IOError):
            load_sd15ch1_frames(data_home=self.tmpdir,
                            download_if_missing=False)


    def test_lf_default(self):
        load_sd15ch1_frames(data_home=self.tmpdir)


    def test_lf_sample_no_shuffle(self):
        load_sd15ch1_frames(data_home=self.tmpdir,
                            sample=0.1,
                            shuffle=False)

    def test_lf_no_sample_shuffle(self):
        load_sd15ch1_frames(data_home=self.tmpdir,
                            sample=1.0,
                            shuffle=True)
    
    def test_lf_sample_shuffle(self):
        load_sd15ch1_frames(data_home=self.tmpdir,
                            sample=0.01,
                            shuffle=True)

    def test_lf_return_X_y(self):
        load_sd15ch1_frames(data_home=self.tmpdir,
                            return_X_y=True)

    def test_lf_sample_min(self):
        res = load_sd15ch1_frames(data_home=self.tmpdir,
                            sample=0.000001,
                            return_X_y=True)
        self.assertEqual(len(res[0]), 1)
        

    def test_lf_load_images(self):
        res = load_sd15ch1_frames(data_home=self.tmpdir,
                    sample=0.0001,
                    load_images=True,
                    resize=None,
                    color=False)

        for k, v in six.iteritems(res):
            print(k, v)
        
    def test_lf_load_images_resize_color(self):
        load_sd15ch1_frames(data_home=self.tmpdir,
                    sample=0.0001,
                    load_images=True,
                    resize=0.25,
                    color=True)

    def test_lf_with_all(self):
        res = load_sd15ch1_frames(data_home=self.tmpdir,
                        sample=0.1,
                        with_model_classif_targets=True,
                        with_modeltype_classif_targets=True,
                        with_segmentation_targets=True,
                        with_model_shapes=True,
                        return_X_y=True)
        self.assertEqual(len(res), 5)



    def test_lm_no_load(self):
        model_data = load_sd15ch1_models(
                data_home=self.tmpdir, 
                download_if_missing=True,
                load_images=False,
                variant=MODEL_VARIANT_05_SCALED33,
                color=False,
                with_model_ids=True,
                with_modeltype_ids=True,
                return_X_y=False)
        for k, v in six.iteritems(model_data):
            print(k, v)
    
    def test_lm_return_X_y_withall(self):
        model_data = load_sd15ch1_models(data_home=self.tmpdir,
            load_images=False,
            with_model_ids=True,
            with_modeltype_ids=True,
            return_X_y=True)
        self.assertEqual(len(model_data), 3)


    def test_lm_return_X_y_withnone(self):
        model_data = load_sd15ch1_models(data_home=self.tmpdir,
            load_images=False,
            with_model_ids=False,
            with_modeltype_ids=False,
            return_X_y=True)
        self.assertEqual(len(model_data), 1)

    
    def test_lm_all_variants_load(self):
        for variant in [MODEL_VARIANT_01_ORIGINAL, MODEL_VARIANT_02_EDITED, 
                MODEL_VARIANT_03_CAPTURED, MODEL_VARIANT_04_CORRECTED, 
                MODEL_VARIANT_05_SCALED33]:
            load_sd15ch1_models(data_home=self.tmpdir,
                    load_images=True,
                    variant=variant,
                    color=False)


    def test_lm_all_variants_no_load(self):
        for variant in [MODEL_VARIANT_01_ORIGINAL, MODEL_VARIANT_02_EDITED, 
                MODEL_VARIANT_03_CAPTURED, MODEL_VARIANT_04_CORRECTED, 
                MODEL_VARIANT_05_SCALED33]:
            load_sd15ch1_models(data_home=self.tmpdir,
                    load_images=False,
                    variant=variant,
                    color=False)
    

    def test_eval_seg_gt_1(self):
        resize_factor = 0.5

        data = load_sd15ch1_frames(data_home=self.tmpdir,
                        sample=0.01,
                        shuffle=True,
                        resize=resize_factor,
                        with_model_classif_targets=False,
                        with_modeltype_classif_targets=False,
                        with_segmentation_targets=True,
                        with_model_shapes=True,
                        return_X_y=False)

        evalres = eval_sd15ch1_segmentations(
            data.target_segmentations, 
            data.target_segmentations,
            data.model_shapes,
            frame_resize_factor=resize_factor)

        self.assertTrue(np.allclose(
            evalres,
            np.ones_like(evalres)))





if __name__ == '__main__':
    unittest.main()
