import math

import numpy as np
import pymupdf

from src.classifiers.classifier_types import ClassifierTypes, Classifier
from src.identifiers.boreprofile import identify_boreprofile, keywords_in_figure_description
from src.identifiers.map import identify_map
from src.identifiers.text import identify_text
from src.identifiers.title_page import identify_title_page
from src.line_detection import extract_geometric_lines
from src.page_classes import PageClasses
from src.page_structure import PageContext


class RuleBasedClassifier(Classifier):
    """
    Baseline classifier for single document pages based on layout, content, and geometric features.

    Subclasses can override `_detect_text`, `_detect_boreprofile`, or `_detect_map`
    to customize classification behavior.
    """
    def __init__(self, matching_params: dict):
        self.matching_params = matching_params
        self.type = ClassifierTypes.BASELINE

    def determine_class(self, page: pymupdf.Page, context: PageContext, **kwargs) -> PageClasses:
        """Determines the page class (e.g., BOREPROFILE, MAP) based on page content."""
        if self._detect_text(page, context):
            return PageClasses.TEXT

        if self._detect_boreprofile(page, context):
            return PageClasses.BOREPROFILE

        if self._detect_map(page, context):
            return PageClasses.MAP

        if identify_title_page(context, self.matching_params):
            return PageClasses.TITLE_PAGE

        return PageClasses.UNKNOWN

    def _detect_text(self, page: pymupdf.Page, context: PageContext) -> bool:
        return identify_text(context)

    def _detect_boreprofile(self, page: pymupdf.Page, context: PageContext) -> bool:
        return identify_boreprofile(context, self.matching_params)

    def _detect_map(self, page: pymupdf.Page, context: PageContext) -> bool:
        """
        Determines whether a page should be classified as a map page.

        Map detection relies on Line detection, which gets delayed until here.
        Short lines (often from text artifacts) are filtered out when text is present.
        """
        if not context.geometric_lines:
            geometric_lines = extract_geometric_lines(page)

            if len(context.words) > 7:
                mean_font_size = np.mean([line.font_size for line in context.lines])
                min_line_length = mean_font_size * math.sqrt(2)

                geometric_lines = [line for line in geometric_lines if line.length > min_line_length]

            context.geometric_lines = geometric_lines

        return identify_map(context, self.matching_params)


class ScannedRuleBasedClassifier(RuleBasedClassifier):
    pass


class DigitalRuleBasedClassifier(RuleBasedClassifier):
    """
    Baseline classifier for digitally born documents.

    Uses image coverage and figure metadata to adjust classification logic.
    """

    def _detect_text(self, page: pymupdf.Page, context: PageContext) -> bool:
        """Determines whether a page should be classified as a text page.

        For digitally born pages, we suppress text classification if images
        covers more than 70% of the total text page area."""
        total_image_coverage = sum(img.page_coverage(context.page_rect) for img in context.image_rects)
        return total_image_coverage < 0.70 and identify_text(context)

    def _detect_boreprofile(self, page: pymupdf.Page, context: PageContext) -> bool:
        if context.image_rects and keywords_in_figure_description(context, self.matching_params):
            return True
        return context.drawings and super()._detect_boreprofile(page, context)

    def _detect_map(self, page: pymupdf.Page, context: PageContext) -> bool:
        if not (context.image_rects or context.drawings):
            return False
        return super()._detect_map(page, context)


class BaselineClassifier(Classifier):
    """
        Rule-based page classifier that delegates to digital or scanned classifiers
        based on the page type.

        Attributes:
            type (ClassifierTypes): Identifier for the classifier type (BASELINE).
            scanned (ScannedPageClassifier): Classifier for scanned pages.
            digital (DigitalPageClassifier): Classifier for digital pages.
    """
    def __init__(self, matching_params: dict):
        self.type = ClassifierTypes.BASELINE
        self.scanned = ScannedRuleBasedClassifier(matching_params)
        self.digital = DigitalRuleBasedClassifier(matching_params)

    def determine_class(self, page:pymupdf.Page, context: PageContext, **kwargs) -> PageClasses:
        if context.is_digital:
            return self.digital.determine_class(page, context)
        return self.scanned.determine_class(page, context)