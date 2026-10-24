"""Georeferencing against OpenStreetMap, borrowing the system QGIS install.

QGIS is a Qt5/PyQt5 library and this app is Qt6/PySide6. The two load into one
process because Qt versions its symbols, but only on one condition: QGIS must
be initialised with its GUI disabled, so it never loads a Qt5 platform plugin
and never contests the event loop. PySide6 owns all the windowing. That is also
why nothing here hands back a Qt object — pixels leave as a numpy array and
points as plain tuples, so a Qt5 wrapper can never reach a PySide6 call.

What is borrowed is the tile stack (fetching, caching, zoom-level choice,
reprojection, label rendering) and the GCP fitting maths. The export warp goes
through the GDAL command line instead; see `export.warp_to_geotiff`.

QGIS is a system package rather than a wheel, so it may simply be absent.
`available()` is the gate every caller checks first.
"""

import os
import sys

import numpy as np

# Where Debian keeps the QGIS bindings. Appended rather than inserted, so the
# app's own (uv-managed) numpy still wins over the distribution's copy.
_DIST_PACKAGES = "/usr/lib/python3/dist-packages"

# Map coordinates are web mercator throughout: it is what the tiles render in
# and what the user's clicks are measured in, so the warp reprojects nothing.
MAP_CRS = "EPSG:3857"

# Half-width of the web mercator world square, in metres.
WORLD_EXTENT = 20037508.342789244

# Where the map pane opens when nothing has been georeferenced yet: Hamburg,
# which is where this project's maps come from. EPSG:3857 metres, taken from
# the city's rough bounding box (9.73..10.33 E, 53.39..53.74 N).
DEFAULT_MAP_RECT = (1083139.0, 7055466.0, 1149930.0, 7121068.0)

# How far beyond the control points the map pane opens, in metres. The points
# alone make a box of no size at all when there is only one of them, so this is
# what decides the zoom for the first point of a page.
POINT_MARGIN = 500.0

OSM_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_ATTRIBUTION = "© OpenStreetMap contributors"

# Transform methods offered to the user, in menu order. Both are linear, which
# is what lets the export reduce either one to three corner GCPs.
METHODS = ("Helmert", "Linear")
DEFAULT_METHOD = "Helmert"

_modules = None
_app = None
_layer = None


def _qgis():
    """The (core, analysis) QGIS modules, imported once; raises if unavailable."""
    global _modules
    if _modules is None:
        if _DIST_PACKAGES not in sys.path:
            sys.path.append(_DIST_PACKAGES)
        from qgis import core, analysis
        _modules = (core, analysis)
    return _modules


def available() -> bool:
    """Whether QGIS can be loaded at all. Everything else assumes it can."""
    try:
        _qgis()
    except Exception:
        return False
    return True


def ready() -> bool:
    """Whether QGIS is up. Rendering before `initialize` would crash, not fail."""
    return _app is not None


def _clear_opencv_qt_env():
    """Point QGIS' Qt5 away from the Qt plugins opencv bundles.

    Importing cv2 repoints QT_QPA_PLATFORM_PLUGIN_PATH at its own copy of the
    Qt5 plugins. PySide6's Qt6 has no use for them, but QGIS' Qt5 finds a
    libqxcb.so sitting there, loads it against a Qt5 it was not built for, and
    takes the whole process down with it.

    QGIS is headless here — it only ever renders into a QImage — so it is sent
    to the offscreen platform and away from that directory. The settings are
    left changed rather than restored: QGIS loads Qt5 plugins lazily for the
    rest of the session, so it needs to keep looking in the right place. Qt6
    read its own settings when the app started and will not read them again,
    and nothing else in this process is a Qt program.
    """
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    os.environ.pop("QT_QPA_FONTDIR", None)


def initialize():
    """Start QGIS headless. Call from the GUI thread before any rendering."""
    global _app
    if _app is not None:
        return
    core, _ = _qgis()
    _clear_opencv_qt_env()
    # GUI disabled, so QGIS brings up no window of its own.
    _app = core.QgsApplication([], False)
    _app.initQgis()


def _make_osm_layer():
    core, _ = _qgis()
    url = OSM_URL.replace("{", "%7B").replace("}", "%7D")
    return core.QgsRasterLayer(f"type=xyz&url={url}&zmax=19&zmin=0", "osm", "wms")


def _osm_layer():
    """The OSM tile layer, built once and shared by every render."""
    global _layer
    if _layer is None:
        _layer = _make_osm_layer()
    return _layer


def osm_layer_for_project():
    """A fresh OSM layer to hand to a QGIS project.

    Deliberately not the shared one: a project takes ownership of the layers
    added to it and deletes them with itself, which would leave the renderer
    holding a dangling layer.
    """
    return _make_osm_layer()


def render_osm(extent, width: int, height: int) -> np.ndarray:
    """Render OSM over `extent` (xmin, ymin, xmax, ymax in map units) as RGBA.

    Safe to call off the GUI thread — the render job runs its own threads and
    `waitForFinished` needs no event loop of ours. Returns a (height, width, 4)
    uint8 array; an all-transparent one if the layer failed to load.
    """
    core, _ = _qgis()
    layer = _osm_layer()
    width = max(1, int(width))
    height = max(1, int(height))
    if not layer.isValid():
        return np.zeros((height, width, 4), dtype=np.uint8)

    from qgis.PyQt.QtCore import QSize
    from qgis.PyQt.QtGui import QImage

    settings = core.QgsMapSettings()
    settings.setLayers([layer])
    settings.setDestinationCrs(core.QgsCoordinateReferenceSystem(MAP_CRS))
    settings.setExtent(core.QgsRectangle(*extent))
    settings.setOutputSize(QSize(width, height))
    job = core.QgsMapRendererParallelJob(settings)
    job.start()
    job.waitForFinished()

    img = job.renderedImage().convertToFormat(QImage.Format_RGBA8888)
    buf = bytes(img.constBits().asstring(img.sizeInBytes()))
    # Rows may be padded, so index by the stride rather than trusting width.
    arr = np.frombuffer(buf, np.uint8).reshape(img.height(), -1)
    return arr[:, :width * 4].reshape(img.height(), width, 4).copy()


def minimum_gcp_count(method: str = DEFAULT_METHOD) -> int:
    """How many control points `method` needs before it can be fitted."""
    _, analysis = _qgis()
    return analysis.QgsGcpTransformerInterface.create(
        _method_enum(method)).minimumGcpCount()


def _method_enum(method: str):
    _, analysis = _qgis()
    methods = analysis.QgsGcpTransformerInterface.TransformMethod
    return getattr(methods, method if method in METHODS else DEFAULT_METHOD)


def fit(gcps, method: str = DEFAULT_METHOD):
    """Fit `method` to `gcps` — [(img_x, img_y, map_x, map_y), ...].

    Returns a callable from image coords to map coords, or None when the points
    are too few or degenerate (two coincident points fix no scale).

    Image rows run downwards while map northings run upwards, so the mapping
    between them includes a reflection, and a similarity transform cannot
    express one — given raw rows, Helmert approximates the flip by rotating the
    map about 45 degrees instead. QGIS wants the reflection taken out of the
    points it is fitted on, but puts it back itself when transforming, so the
    row is negated here and not there. The asymmetry is odd enough that it is
    worth stating it was measured, not deduced: of the four sign combinations
    only this one reproduces a known north-up mapping exactly.

    Linear survives either convention, having no rotation to protect — it fits
    the axes independently and simply absorbs the flip as a negative y scale.
    Helmert is the one that needs this.
    """
    core, analysis = _qgis()
    transformer = analysis.QgsGcpTransformerInterface.create(_method_enum(method))
    minimum = transformer.minimumGcpCount()
    if len(gcps) < minimum:
        return None
    # Distinct source points, not merely enough of them. Two control points on
    # the same page pixel make QGIS' Helmert solver hand GSL a singular matrix,
    # and GSL's default error handler calls abort(): the process dies on the
    # spot, with no exception for anyone to catch. Marking one landmark twice
    # instead of deleting a row is an ordinary slip, so it has to be refused
    # here, before QGIS ever sees the points.
    if len({(g[0], g[1]) for g in gcps}) < minimum:
        return None
    src = [core.QgsPointXY(g[0], -g[1]) for g in gcps]
    dst = [core.QgsPointXY(g[2], g[3]) for g in gcps]
    if not transformer.updateParametersFromGcps(src, dst, True):
        return None

    def transform(x, y):
        ok, tx, ty = transformer.transform(x, y)
        if not ok:
            raise ValueError("point could not be transformed")
        return tx, ty

    # Distinct but still degenerate — every point on one vertical line, say —
    # is solved rather than refused, and comes back as NaN. Probe it once, so
    # a hopeless fit reads as no fit instead of reaching an export, or the
    # dialog, as a georeference that merely looks real.
    try:
        probe = transform(gcps[0][0], gcps[0][1])
    except ValueError:
        return None
    if not all(np.isfinite(value) for value in probe):
        return None

    return transform


def transform_points(gcps, method, points):
    """Push `points` through the fit of `gcps`; None when it cannot be fitted."""
    transform = fit(gcps, method)
    if transform is None:
        return None
    return [transform(x, y) for x, y in points]


def residuals(gcps, method: str = DEFAULT_METHOD):
    """Per-point fit error in map units, or None when there is no fit yet.

    Measured against the same points the fit was made from, so this is how far
    each control point missed, not an independent check.
    """
    transform = fit(gcps, method)
    if transform is None:
        return None
    out = []
    for src_x, src_y, dst_x, dst_y in gcps:
        tx, ty = transform(src_x, src_y)
        out.append(float(np.hypot(tx - dst_x, ty - dst_y)))
    return out
