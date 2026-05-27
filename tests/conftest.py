"""Pytest configuration shared by every test module.

Force matplotlib's non-interactive "Agg" backend before any test imports
``utils.visualize``. CI runners (Windows in particular) ship without a
working Tk/Tcl, so the default Tk backend errors out with
``_tkinter.TclError: Can't find a usable init.tcl`` when ``plt.subplots()``
is called. ``Agg`` is a pure-Python raster backend that works headlessly on
every platform and is the standard choice for tests that save figures to
disk rather than displaying them interactively.

This file is auto-discovered by pytest at collection time and runs before
any test module is imported, which means it executes before
``matplotlib.pyplot`` is first imported transitively through our
visualization module.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
