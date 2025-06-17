import fitz


def text_type(page: fitz.Page) -> str:
    bboxes = page.get_bboxlog()
    has_ignore_text = False

    for boxType, rectangle in bboxes:
        # Empty rectangle that should be ignored occurs sometimes, e.g. SwissGeol 44191 page 37.
        if (boxType == "fill-text" or boxType == "stroke-text") and not fitz.Rect(rectangle).is_empty:
            return "digital"
        if boxType == "ignore-text":
            has_ignore_text = True

    if has_ignore_text:
        return "ocr"

    else:
        return "none"

