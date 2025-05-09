import math
from abc import ABC, abstractmethod

import numpy as np

from .identifiers.boreprofile import identify_boreprofile, keywords_in_figure_description
from .identifiers.map import identify_map
from .identifiers.text import identify_text
from .identifiers.title_page import sparse_title_page
from .line_detection import extract_geometric_lines
from .page_classes import PageClasses


class PageClassifier(ABC):
    @abstractmethod
    def determine_class(self, page, context, matching_params, features) -> PageClasses:
        """Determines the page class (e.g., BOREPROFILE, MAP) based on page content."""
        pass

    def _shared_classification(self, page, context, matching_params) -> PageClasses:
        # Geometric lines extraction only when needed
        if not context.geometric_lines:
            _, geometric_lines = extract_geometric_lines(page)
            if len(context.words) > 7:
                mean_font_size = np.mean([line.font_size for line in context.lines])
                geometric_lines = [line for line in geometric_lines if line.length > mean_font_size * math.sqrt(2)]
            context.geometric_lines = geometric_lines

        if identify_map(context, matching_params):
            return PageClasses.MAP

        if sparse_title_page(context.lines):
            return PageClasses.TITLE_PAGE

        return PageClasses.UNKNOWN

class DigitalPageClassifier(PageClassifier):
    def determine_class(self, page, context, matching_params, features) -> PageClasses:
        if not context.images and identify_text(context, features):
            return PageClasses.TEXT

        if context.images:
            if keywords_in_figure_description(context, matching_params):
                return PageClasses.BOREPROFILE

        if context.drawings and identify_boreprofile(context, matching_params):
            return PageClasses.BOREPROFILE

        if (context.drawings or context.images):
            return self._shared_classification(page, context, matching_params)

        return PageClasses.UNKNOWN


class ScannedPageClassifier(PageClassifier):
    def determine_class(self, page, context, matching_params, features) -> PageClasses:
        if identify_text(context, features):
            return PageClasses.TEXT

        if identify_boreprofile(context, matching_params):
            return PageClasses.BOREPROFILE

        return self._shared_classification(page, context, matching_params)
