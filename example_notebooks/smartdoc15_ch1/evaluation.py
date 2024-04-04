#!/usr/bin/env python
# -*- coding: utf-8 -*-


# IMPORT
################################################################################
################################################################################
from __future__ import division, absolute_import, print_function
from six.moves import range

import numpy as np
from skimage.transform import estimate_transform
from sklearn import metrics
import Polygon

from .poly_utils import isSelfIntersecting


def _err(msg, exception=Exception):
    print("ERROR", msg)
    raise exception(msg)

def _warn(msg):
    print("WARNING", msg)
    warnings.warn(msg, stacklevel=2)


# Evaluation
# ------------------------------------------------------------------------------

def evaluate_segmentation(segmentation_results, target_segmentations, model_shapes, frame_scale_factor=1.0, print_summary=False):
    # frame_scale_factor is the value of the resize factor applied to the frames. 
    # It will be inverted to recover the correct coordinates.

    # First check everything has the right type and shape
    # TODO check types
    # TODO accept :
    # - list of list of floats or int
    # - dict of str: float or int as input as well for segmentation_results
    # - array 2D (dim = n*8)

    if type(segmentation_results) == list:
        segmentation_results = np.array(segmentation_results)

    seg_shape = segmentation_results.shape
    if len(seg_shape) != 2 or seg_shape[1] != 8:
        _err("evaluate_segmentation: segmentation_results parameter "
              "must be a numpy array of shape (NUM_FRAMES, 8).",
              ValueError)
    tarseg_shape = target_segmentations.shape
    if len(seg_shape) != 2 or seg_shape[1] != 8:
        _err("evaluate_segmentation: target_segmentations parameter "
              "must be a numpy array of shape (NUM_FRAMES, 8).",
              ValueError)
    mdlshapes_shape = model_shapes.shape
    if len(mdlshapes_shape) != 2 or mdlshapes_shape[1] != 2:
        _err("evaluate_segmentation: model_shapes parameter "
              "must be a numpy array of shape (NUM_FRAMES, 2).",
              ValueError)
    num_frames = seg_shape[0]
    if tarseg_shape[0] != num_frames or mdlshapes_shape[0] != num_frames:
        _err("evaluate_segmentation: 'segmentation_results', 'target_segmentations' and 'model_shapes' parameters "
              "must all have the same dimension on axis 0 (number of frames).", 
              ValueError)

    # Scale coordinates back to original frame size
    segmentations_scaled = segmentation_results / frame_scale_factor

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
            _err("evaluate_segmentation: target_segmentations[%d]: ground truth coordinates are not in right order "
                  "(the resulting polygon is self intersecting). Cannot compute overlap. Aborting." % (ii,),
                  ValueError)

        area_target = poly_target.area()
        
        if area_target < 0.0000000001:
            _err("evaluate_segmentation: target_segmentations[%d]: ground truth coordinates form a "
                  "polygon degenerated to one point. Cannot compute overlap. Aborting." % (ii,),
                  ValueError)

        if isSelfIntersecting(poly_test):
            _warn("evaluate_segmentation: segmentation[%d]: segmentation coordinates are not in right order "
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
                # _warn("Capping area_inter.")
            
        area_union = area_test + area_target - area_inter
        jaccard_index = area_inter / area_union
        eval_result[ii] = jaccard_index

    # Print summary if asked
    if print_summary:
        try:
            import scipy.stats
        except ImportError:
            _warn("evaluate_segmentation: cannot import scipy.stats. Print summary deactivated. "
                   "Please install scipy to enable it.")
        else:
            values = eval_result
            # nobs, minmax, mean, variance, skewness, kurtosis = stats.describe(values)
            desc_res = scipy.stats.describe(values)
            std = np.sqrt(desc_res.variance)
            cil, cih = scipy.stats.norm.interval(0.95,loc=desc_res.mean,scale=std/np.sqrt(len(values)))

            print("----------------------------------------------")
            print("   evaluate_segmentation: Evaluation report")
            print("----------------------------------------------")
            print("metric: IoU (aka Jaccard index)")
            print("----------------------------------------------")
            print("observations: %5d" % (desc_res.nobs, ))
            print("mean:         %8.2f (CI@95%%: %.3f, %.3f)" % (desc_res.mean, cil, cih))
            print("min-max:          %.3f - %.3f" % desc_res.minmax)
            print("variance:         %.3f (std: %.3f)" % (desc_res.variance, std))
            print("----------------------------------------------")

    # return IoU output
    return eval_result


def evaluate_classification(predicted_labels, target_labels, label_names=None, print_summary=False):
    '''
    Evaluates the performance of a classification task (applicable for tasks 2 and 3).

    This is a generic function and users have to provide the appropriate target labels (opt. names)
    as a parameter. This enables them to use this function even if they sub-sampled the dataset.

    This evaluation function is provided for the sake of completeness, as it provides a limited view of 
    the results.

    Parameters:
    -----------
    predicted_labels: numpy.array, 1 dimension, contains integers
        Labels returned by the method under evaluation, encoded as integers.

    Returns:
    --------
    mean_accuracy, confusion_matrix: tuple of (float, numpy.array)
        - mean_accuracy is the sum of correct results (label == target_label) divided by the number
          of values.
        - confusion_matrix is the confusion matrix of the prediction, as return by 
          `sklearn.metrics.confusion_matrix`. Rows are target values, columns indicate predicted 
          values.

    Notes:
    ------
    If `print_summary` is True, then detailed statistics are printed.
    '''
    # TODO doc
    # TODO forward to sklearn classifier accuracy evaluation?

    if print_summary:
        print(metrics.classification_report(target_labels, predicted_labels,
                target_names=label_names))

    mean_accuracy = np.mean(predicted_labels == target_labels)
    confusion_matrix = metrics.confusion_matrix(target_labels, predicted_labels)

    return mean_accuracy, confusion_matrix


