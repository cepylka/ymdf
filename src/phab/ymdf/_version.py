from typing import Tuple
from datetime import datetime

__version_info__: Tuple[int, int, int] = (0, 1, 0)
__version__: str = ".".join(map(str, __version_info__))

__copyright__: str = " ".join((
    f"Copyright (C) 2024-{datetime.now().year}",
    "Declaration of VAR"
))
