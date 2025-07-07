import pytest
import pymupdf
from src.identifiers.title_page import find_aligned_clusters
from src.text_objects import TextLine, TextWord

def x0(line): return line.rect.x0

def test_x0_cluster():
    # Three horizontally x0 aligned lines, vertically close
    lines = [
        TextLine([TextWord(pymupdf.Rect(100, 100, 200, 110), "Line 1",0)]),
        TextLine([TextWord(pymupdf.Rect(100, 145, 200, 155), "Line 2",0)]),
        TextLine([TextWord(pymupdf.Rect(100, 190, 200, 200), "Line 3",0)])
    ]
    clusters = find_aligned_clusters(lines, key_func= lambda l: l.rect.x0, threshold=5)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3

def test_two_x0_clusters():
    # Two x0 aligned clusters far apart in x0
    lines = [
        TextLine([TextWord(pymupdf.Rect(100, 100, 200, 110), "Line 1 in cluster 1",0)]),
        TextLine([TextWord(pymupdf.Rect(102, 130, 200, 140), "Line 2 in cluster 1",0)]),
        TextLine([TextWord(pymupdf.Rect(300, 100, 400, 110), "Line 1 in cluster 2",0)]),
        TextLine([TextWord(pymupdf.Rect(298, 130, 400, 140), "Line 2 in cluster 2",0)]),
    ]
    clusters = find_aligned_clusters(lines, key_func=lambda l: l.rect.x0, threshold=5)
    assert len(clusters) == 2
    for cluster in clusters:
        assert len(cluster) == 2

def test_transitive_inclusion():
    # A close to B, B close to C, A not close to C
    lines = [
        TextLine([TextWord(pymupdf.Rect(100, 100, 200, 110), "Line A",0)]),  # A
        TextLine([TextWord(pymupdf.Rect(104, 130, 204, 140), "Line B",0)]),  # B
        TextLine([TextWord(pymupdf.Rect(108, 160, 208, 170), "Line C",0)]),  # C
    ]
    clusters = find_aligned_clusters(lines, key_func=lambda l: l.rect.x0, threshold=5)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3

def test_no_clusters_x0():
    # All lines too far apart on x0
    lines = [
        TextLine([TextWord(pymupdf.Rect(100, 100, 200, 110), "Line A",0)]),
        TextLine([TextWord(pymupdf.Rect(300, 300, 400, 310), "Line B",0)]),
        TextLine([TextWord(pymupdf.Rect(500, 500, 600, 510), "Line C",0)]),
    ]
    clusters = find_aligned_clusters(lines, key_func=lambda l: l.rect.x0, threshold=5)
    assert len(clusters) == 0

def test_no_cluster_y0():
    # All lines too far apart on y0
    lines = [
        TextLine([TextWord(pymupdf.Rect(100, 100, 200, 110), "Line A", 0)]),
        TextLine([TextWord(pymupdf.Rect(100, 160, 200, 170), "Line B", 0)]), # > 5 * height away
        TextLine([TextWord(pymupdf.Rect(100, 210, 200, 220), "Line C", 0)]), # > 5 * height away
    ]
    clusters = find_aligned_clusters(lines, key_func=lambda l: l.rect.x0, threshold=5)
    assert len(clusters) == 0