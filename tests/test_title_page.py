import pymupdf
from swissgeol_doc_processing.text.textline import TextLine, TextWord

from src.identifiers.title_page import VERTICAL_SPACING_FACTOR, cluster_aligned_text_lines


def x0(line):
    return line.rect.x0


def test_x0_cluster():
    # Three horizontally x0 aligned lines, vertically close
    lines = [
        TextLine([TextWord(rect=pymupdf.Rect(100, 100, 200, 110), text="Line 1", page=0)]),
        TextLine([TextWord(rect=pymupdf.Rect(100, 145, 200, 155), text="Line 2", page=0)]),
        TextLine([TextWord(rect=pymupdf.Rect(100, 190, 200, 200), text="Line 3", page=0)]),
    ]
    clusters = cluster_aligned_text_lines(lines, key_func=lambda line: line.rect.x0, threshold=VERTICAL_SPACING_FACTOR)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_two_x0_clusters():
    # Two x0 aligned clusters far apart in x0
    lines = [
        TextLine([TextWord(rect=pymupdf.Rect(100, 100, 200, 110), text="Line 1 in cluster 1", page=0)]),
        TextLine([TextWord(rect=pymupdf.Rect(102, 130, 200, 140), text="Line 2 in cluster 1", page=0)]),
        TextLine([TextWord(rect=pymupdf.Rect(300, 100, 400, 110), text="Line 1 in cluster 2", page=0)]),
        TextLine([TextWord(rect=pymupdf.Rect(298, 130, 400, 140), text="Line 2 in cluster 2", page=0)]),
    ]
    clusters = cluster_aligned_text_lines(lines, key_func=lambda line: line.rect.x0, threshold=VERTICAL_SPACING_FACTOR)
    assert len(clusters) == 2
    for cluster in clusters:
        assert len(cluster) == 2


def test_transitive_inclusion():
    # A close to B, B close to C, A not close to C
    lines = [
        TextLine([TextWord(rect=pymupdf.Rect(100, 100, 200, 110), text="Line A", page=0)]),  # A
        TextLine([TextWord(rect=pymupdf.Rect(104, 130, 204, 140), text="Line B", page=0)]),  # B
        TextLine([TextWord(rect=pymupdf.Rect(108, 160, 208, 170), text="Line C", page=0)]),  # C
    ]
    clusters = cluster_aligned_text_lines(lines, key_func=lambda line: line.rect.x0, threshold=VERTICAL_SPACING_FACTOR)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_no_clusters_x0():
    # All lines too far apart on x0
    lines = [
        TextLine([TextWord(rect=pymupdf.Rect(100, 100, 200, 110), text="Line A", page=0)]),
        TextLine([TextWord(rect=pymupdf.Rect(300, 300, 400, 310), text="Line B", page=0)]),
        TextLine([TextWord(rect=pymupdf.Rect(500, 500, 600, 510), text="Line C", page=0)]),
    ]
    clusters = cluster_aligned_text_lines(lines, key_func=lambda line: line.rect.x0, threshold=VERTICAL_SPACING_FACTOR)
    assert len(clusters) == 0


def test_no_cluster_y0():
    # All lines too far apart on y0
    lines = [
        TextLine([TextWord(rect=pymupdf.Rect(100, 100, 200, 110), text="Line A", page=0)]),
        TextLine([TextWord(rect=pymupdf.Rect(100, 160, 200, 170), text="Line B", page=0)]),  # > 5 * height away
        TextLine([TextWord(rect=pymupdf.Rect(100, 210, 200, 220), text="Line C", page=0)]),  # > 5 * height away
    ]
    clusters = cluster_aligned_text_lines(lines, key_func=lambda line: line.rect.x0, threshold=VERTICAL_SPACING_FACTOR)
    assert len(clusters) == 0
