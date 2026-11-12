"""Pages and shared helpers for the ShopSmart Streamlit dashboard (app/streamlit_app.py).

`streamlit run app/streamlit_app.py` puts `app/` on sys.path, so this package imports
as `dashboard`.
"""

import os
import sys

# Make the `src/` layout importable even when Streamlit is launched from a Python
# that does not have the project installed (e.g. a global/conda `streamlit`).
_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
