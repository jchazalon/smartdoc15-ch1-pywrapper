#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
# Imports
from collections import namedtuple

import Polygon
import Polygon.Utils
# import Polygon.IO # dbg


# ==============================================================================


# Test samples
# poly_regular = Polygon.Polygon([(0,0), (0, 10), (10, 10), (10, 0)])
# poly_crossed = Polygon.Polygon([(0,0), (0, 10), (10, 0), (10, 10)])

""" Polygon point. """
PPoint = namedtuple("PPoint", ["x", "y"])
""" Polygon edge. """
# start = start PPoint
# end   = end PPoint
## cidx  = contour index
## eidx  = edge index in contour
PEdge = namedtuple("PEdge", ["start", "end"]) #, "cidx", "eidx"])

def _checkPoly1Contour(poly):
    """
    Check that poly complies with current limitations.
    """
    # if len(poly) == 0:
    #     logger.warning("Polygon is empty. Accepting with warning.")

    if len(poly) > 1:
        msg = "Error: Current version of eval_seg cannot handle polygons with multiple contours."
        logger.error(msg)
        logger.error("Poly: %s" % poly)
        raise ValueError(msg)


def _polyEdges(poly):
    """
    Return a list of `PEdge` for the first contour of poly.
    It includes the segment between last point and first point.
    """
    _checkPoly1Contour(poly) # /!\ if accept more than 1 contour, you may have to store contour index
    edges = []
    for c in poly:
        for i in range(len(c)):
            (x0, y0) = c[i-1]
            (x1, y1) = c[i]
            edges.append(PEdge(PPoint(float(x0), float(y0)), PPoint(float(x1), float(y1)))) #, 1, i))
    return edges

def _isLeft(P0, P1, P2):
    """
    Test if point P2 is Left|On|Right of the line P0 to P1.
    returns: >0 for left, 0 for on, and <0 for right of the line.
    """
    return (P1.x - P0.x)*(P2.y - P0.y) - (P2.x - P0.x)*(P1.y -  P0.y)

def _intersect(edge1, edge2):
    """
    Test if two `PEdge`s are intersecting.
    Warning: calling this function with the same edge twice returns True.
    """
    # consecutive edges connexions are not intersections
    if edge1.end == edge2.start or edge2.end == edge1.start:
        return False

    # test for existence of an intersect point
    lsign = rsign = 0.0
    lsign = _isLeft(edge1.start, edge1.end, edge2.start)  #  edge2 start point sign
    rsign = _isLeft(edge1.start, edge1.end, edge2.end)    #  edge2 end point sign
    if (lsign * rsign > 0): # edge2 endpoints have same sign  relative to edge1
        return False       # => on same side => no intersect is possible
    lsign = _isLeft(edge2.start, edge2.end, edge1.start)  #  edge1 start point sign
    rsign = _isLeft(edge2.start, edge2.end, edge1.end)    #  edge1 end point sign
    if (lsign * rsign > 0): # edge1 endpoints have same sign  relative to edge2
        return False       # => on same side => no intersect is possible
    # the segments edge1 and edge2 straddle each other
    return True            # => an intersect exists


def isSelfIntersecting(poly):
    """
    Simple detection of self-intersection.
    Naive O(n^2) implementation for small, 1 contour polygons.
    """
    # For possible improvements, see:
    # http://en.wikipedia.org/wiki/Bentley%E2%80%93Ottmann_algorithm
    # http://geomalgorithms.com/a09-_intersect-3.html

    # We start be removing duplicate points and points on straight lines
    polyPruned = Polygon.Utils.prunePoints(poly)

    # Polygon must not have any self-intersection for each contour, but also between contours
    # This version only manages single contour polygons, without holes, because we don't need them yet
    # We assume at least, and at most, one contour.
    _checkPoly1Contour(polyPruned)

    # Get edges
    edges = _polyEdges(polyPruned)

    # Look for intersections
    for i in range(len(edges)):
        e1 = edges[i]
        for j in range(i+1, len(edges)):
            e2 = edges[j]
            if _intersect(e1, e2):
                # logger.error("Intersection: e1= %s ; e2= %s", e1, e2)
                return True

    return False


