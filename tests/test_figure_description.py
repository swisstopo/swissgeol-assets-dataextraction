import re
import pytest

# The pattern to test
figure_pattern = re.compile(
    r"^(?:"                        
    r"(?:fig(?:ure)?|abb(?:ildung)?|tab(?:le)?)\.?\s*[:.]?\s*"
    r")?"                               
    r"\d{1,2}(?:[.:]\d{1,2}){0,3}"       
    r"\b",
    flags=re.IGNORECASE
)

@pytest.mark.parametrize("line", [
    "Fig. 1",
    "Figure 2.1",
    "Tab 3.1.2",
    "1.2.3",
    "Abbildung 4.1",
    "Fig 1: something here",
])
def test_figure_pattern_matches(line):
    assert figure_pattern.match(line.strip()), f"Expected match: {line}"

@pytest.mark.parametrize("line", [
    "This is Fig. 1",
    "We refer to Table 2",
    "Abbildungen are useful",
    "Some text 3.1 appears here",
])
def test_figure_pattern_non_matches(line):
    assert not figure_pattern.match(line.strip()), f"Expected no match: {line}"
