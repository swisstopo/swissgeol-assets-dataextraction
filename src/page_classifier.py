import math
from abc import ABC
import numpy as np

from .identifiers.boreprofile import identify_boreprofile, keywords_in_figure_description
from .identifiers.map import identify_map
from .identifiers.text import identify_text
from .identifiers.title_page import identify_title_page
from .line_detection import extract_geometric_lines
from .page_classes import PageClasses

class PageClassifier(ABC):
    def determine_class(self, page, context, matching_params) -> PageClasses:
        """Determines the page class (e.g., BOREPROFILE, MAP) based on page content."""
        if self._detect_text(page, context, matching_params):
            return PageClasses.TEXT

        if self._detect_boreprofile(page, context, matching_params):
            return PageClasses.BOREPROFILE

        if self._detect_map(page, context, matching_params):
            return PageClasses.MAP

        if identify_title_page(context, matching_params):
            return PageClasses.TITLE_PAGE

        return PageClasses.UNKNOWN

    def _detect_text(self, page, context, matching_params) -> bool:
        return identify_text(context)

    def _detect_boreprofile(self, page, context, matching_params) -> bool:
        return identify_boreprofile(context, matching_params)

    def _detect_map(self, page, context, matching_params) -> bool:
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

                geometric_lines = [
                    line for line in geometric_lines
                    if line.length > min_line_length
                ]

            context.geometric_lines = geometric_lines

        return identify_map(context, matching_params)

class ScannedPageClassifier(PageClassifier):
    pass

class DigitalPageClassifier(PageClassifier):
    def _detect_text(self, page, context, matching_params) -> bool:
        """Determines whether a page should be classified as a text page.

            For digitally born pages, we suppress text classification if images
            covers more than 70% of the total text page area."""
        total_image_coverage = sum(
            img.page_coverage(context.text_rect) for img in context.image_rects
        )
        return total_image_coverage< 0.70 and identify_text(context)

    def _detect_boreprofile(self, page, context, matching_params) -> bool:
        if context.image_rects and keywords_in_figure_description(context, matching_params):
            return True
        return context.drawings and super()._detect_boreprofile(page,context, matching_params)

    def _detect_map(self, page, context, matching_params) -> bool:
        if not (context.image_rects or context.drawings):
            return False
        return super()._detect_map(page, context, matching_params)
