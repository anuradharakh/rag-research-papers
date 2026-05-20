import re
from typing import List


def extract_figure_captions(
    text: str,
    patterns: List[str],
    max_caption_lines: int = 3,
) -> List[str]:
    """EXTRACT FIGURE AND TABLE CAPTIONS FROM PAGE TEXT. **"""
    captions = []

    lines = text.splitlines()

    for index, line in enumerate(lines):
        stripped_line = line.strip()

        for pattern in patterns:
            if re.match(pattern, stripped_line, flags=re.IGNORECASE):
                caption_lines = [stripped_line]

                for offset in range(1, max_caption_lines):
                    next_index = index + offset

                    if next_index >= len(lines):
                        break

                    next_line = lines[next_index].strip()

                    if not next_line:
                        break

                    caption_lines.append(next_line)

                captions.append(" ".join(caption_lines))
                break

    return captions