"""Real satellite orbits for the realistic trace generator.

What this module does
---------------------
It answers one question for the simulator in ``simulator.py``: *where is every
satellite at second t, as seen from a point on the ground?*

1. :func:`load_orbits` reads a file of orbital elements (one record per
   satellite).  Two public formats are accepted: the classic **TLE** text
   format (two-line element set) and **OMM CSV**, the newer format that
   CelesTrak recommends because TLE cannot hold catalogue numbers above 99999.
   :func:`download_orbits` fetches the current Starlink file from CelesTrak.
2. :func:`build_ephemeris` propagates the orbits with **SGP4**, the standard
   analytical propagator for these element sets [vallado2006spacetrack], and keeps
   the satellites that come near the ground observer.  SGP4 is not optional:
   the elements are *mean* elements fitted with SGP4, so they give the right
   position only when propagated with the same model.
3. The result, an :class:`Ephemeris`, gives positions in an Earth-fixed frame
   at any second of the run, and from them elevation angles and
   inter-satellite distances.

This is the same chain as the simulator of Magliarisi et al.
[magliarisi2025orbit]: Starlink elements, SGP4, a ground observer, and
visibility decided by the elevation angle.

Frames and formulas
-------------------
SGP4 returns positions in TEME (true equator, mean equinox), an inertial frame.
A point fixed on the rotating Earth is obtained by rotating TEME about the z
axis by the Greenwich mean sidereal time θ (IAU 1982 formula, the convention
of the reference SGP4 code [vallado2006spacetrack])::

    x_ecef =  cos θ · x + sin θ · y
    y_ecef = −sin θ · x + cos θ · y
    z_ecef =  z

Polar motion (about 10 m) is ignored.  The observer's Earth-fixed position
comes from its latitude, longitude and height on the WGS-84 ellipsoid.  With
``up`` the unit vector normal to the ellipsoid at the observer and ρ the
vector from observer to satellite, the elevation angle is::

    elevation = asin( ρ · up / |ρ| )

An idealised constellation (:func:`walker_constellation`) is provided for the
unit tests and for trying the generator without a network connection.  It is
NOT Starlink: do not use it for results.

Requires the ``sgp4`` package (``pip install sgp4``) and numpy.
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Current Starlink element sets, OMM CSV format.  CelesTrak refreshes the data
# every 2 hours and answers 403 to a second download inside that period, so
# download once and reuse the file (see https://celestrak.org/usage-policy.php).
CELESTRAK_STARLINK_URL = (
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=csv"
)

# WGS-84 ellipsoid (observer position) and physical constants.
_WGS84_A_KM = 6378.137
_WGS84_F = 1.0 / 298.257223563
_MU_KM3_S2 = 398600.4418          # Earth's gravitational parameter
_SECONDS_PER_DAY = 86400.0


def _require_sgp4():
    try:
        import sgp4.api as api  # noqa: F401
        return api
    except ImportError as exc:  # pragma: no cover - message only
        raise SystemExit(
            "The realistic trace generator needs the 'sgp4' package "
            "(orbit propagation).  Install it with:  pip install sgp4"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Constellation: names + SGP4 records
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Constellation:
    """A set of satellites, each with a name and an SGP4 record."""

    names: List[str]
    satrecs: list                     # sgp4.api.Satrec
    source: str = ""                  # file or generator the set came from

    def __len__(self) -> int:
        return len(self.names)

    def epochs_jd(self) -> np.ndarray:
        """Epoch of each element set as a Julian date."""
        return np.array([s.jdsatepoch + s.jdsatepochF for s in self.satrecs])

    def median_epoch_jd(self) -> float:
        """Median epoch.  SGP4 is most accurate close to the epoch of the
        elements and its error grows with the distance in time from it, so
        simulations start here unless told otherwise."""
        return float(np.median(self.epochs_jd()))


def load_orbits(path: str) -> Constellation:
    """Read orbital elements from an OMM CSV file or a TLE file.

    The format is detected from the first line: an OMM CSV file starts with a
    header that contains ``OBJECT_NAME``.  A TLE file has two lines per
    satellite starting with ``1 `` and ``2 ``, optionally preceded by a name
    line (the "3LE" layout CelesTrak uses).  Records that SGP4 rejects are
    skipped.
    """
    api = _require_sgp4()
    file = Path(path)
    text = file.read_text(encoding="utf-8-sig")
    lines = [line.rstrip("\r\n") for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path}: empty orbit file")

    names: List[str] = []
    satrecs = []
    if "OBJECT_NAME" in lines[0]:
        import io
        from sgp4 import omm
        for fields in omm.parse_csv(io.StringIO("\n".join(lines) + "\n")):
            try:
                sat = api.Satrec()
                omm.initialize(sat, fields)
            except (KeyError, ValueError, TypeError):
                continue            # truncated or malformed record
            names.append(fields.get("OBJECT_NAME", f"SAT-{sat.satnum}").strip())
            satrecs.append(sat)
    else:
        pending_name: Optional[str] = None
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
                try:
                    sat = api.Satrec.twoline2rv(line, lines[i + 1])
                except ValueError:
                    sat = None
                if sat is not None:
                    names.append(pending_name or f"SAT-{sat.satnum}")
                    satrecs.append(sat)
                pending_name = None
                i += 2
            else:
                pending_name = line[2:].strip() if line.startswith("0 ") else line.strip()
                i += 1

    if not satrecs:
        raise ValueError(f"{path}: no usable orbital elements found")
    return Constellation(names=names, satrecs=satrecs, source=str(file))


def download_orbits(dest: str, url: str = CELESTRAK_STARLINK_URL,
                    timeout: float = 600.0) -> Path:
    """Download the current Starlink elements from CelesTrak to ``dest``.

    The request carries no personal data.  ``dest`` is written only when the
    whole file arrived, so neither a refusal nor a broken transfer ever
    replaces a good file.

    CelesTrak allows ONE download per data update (every 2 hours), counted per
    public IP address, and it counts a transfer that was cut off.  A refusal
    is HTTP 403 with a text that names the time of the last download; that
    text is shown.  If the transfer breaks, the complete records received so
    far are saved next to ``dest`` as ``<name>_incomplete.csv``: usable with
    ``--orbits``, but only part of the constellation.
    """
    import urllib.error

    path = Path(dest)
    others = sorted(p.name for p in path.parent.glob("*.csv")) if path.parent.is_dir() else []
    fallback = ("\nOrbit files already in " + str(path.parent) + ": " + ", ".join(others)
                + "\nUse one with:  --orbits " + str(path.parent / others[0])) if others else ""

    request = urllib.request.Request(url, headers={"User-Agent": "sec-trace-generator/1.0"})
    chunks, broken = [], None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                chunks.append(chunk)
    except urllib.error.HTTPError as exc:
        reason = exc.read(600).decode("utf-8", "replace").strip()
        raise SystemExit(
            f"CelesTrak refused the download (HTTP {exc.code}). Its answer:\n"
            + "\n".join("    " + line for line in reason.splitlines())
            + "\nCelesTrak allows one download per data update (every 2 hours) for each "
              "public IP address,\nand a download that was cut off counts. Try again after "
              "the next update: two hours\nafter the time above at the latest." + fallback
        ) from exc
    except Exception as exc:                 # connection lost, timeout, DNS, ...
        broken = exc
    body = b"".join(chunks)

    if broken is None and b"OBJECT_NAME" not in body[:200]:
        raise SystemExit(
            "CelesTrak did not return orbital elements. Its answer was:\n"
            + body[:300].decode("utf-8", "replace") + fallback)

    # Keep only complete records: every line must have as many fields as the header.
    lines = body.split(b"\n")
    fields = lines[0].count(b",") if lines else 0
    complete = broken is None and body.endswith(b"\n") and lines[-2].count(b",") == fields
    if not complete:
        good = [line for line in lines[:-1] if line.count(b",") == fields]
        detail = f": {broken}" if broken is not None else " (the file ends in the middle of a record)"
        if b"OBJECT_NAME" in body[:200] and len(good) > 1:
            partial = path.with_name(path.stem + "_incomplete.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(b"\n".join(good) + b"\n")
            raise SystemExit(
                f"The download from CelesTrak broke off{detail}.\n"
                f"Saved the {len(good) - 1} complete records that arrived to {partial}.\n"
                "That is only PART of the constellation. You can use it with\n"
                f"    --orbits {partial}\n"
                "and say so wherever you report results, or try --download again after "
                "CelesTrak's next data update (within 2 hours).")
        raise SystemExit(f"Could not download orbital elements from {url}{detail}." + fallback)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(body)
    tmp.replace(path)
    return path


def walker_constellation(planes: int = 120, per_plane: int = 40,
                         altitude_km: float = 550.0, inclination_deg: float = 53.0,
                         phasing: int = 1,
                         epoch_jd: float = 2460676.5) -> Constellation:
    """An idealised Walker-delta shell: circular orbits, evenly spaced planes.

    For the unit tests and for trying the generator offline.  The default
    (4800 satellites at 550 km and 53 degrees) is simply dense enough for a
    ground user at mid latitude to see about 15 satellites above 40 degrees.
    It is NOT a real constellation (one shell, no gaps, no orbit raising):
    never use it for results.  ``phasing`` is the Walker phasing factor: satellites in
    adjacent planes are offset by ``phasing * 360 / (planes * per_plane)``
    degrees.  The default epoch is 2025-01-01 00:00 UTC.
    """
    api = _require_sgp4()
    a_km = _WGS84_A_KM + altitude_km
    mean_motion_rad_min = math.sqrt(_MU_KM3_S2 / a_km ** 3) * 60.0
    epoch_days = epoch_jd - 2433281.5          # days since 1949-12-31 00:00 UT
    total = planes * per_plane

    names, satrecs = [], []
    for p in range(planes):
        raan = 2.0 * math.pi * p / planes
        for k in range(per_plane):
            anomaly = (2.0 * math.pi * k / per_plane
                       + 2.0 * math.pi * phasing * p / total) % (2.0 * math.pi)
            sat = api.Satrec()
            sat.sgp4init(api.WGS72, "i", 90000 + len(satrecs), epoch_days,
                         0.0, 0.0, 0.0,            # bstar, ndot, nddot: no drag
                         0.0001,                   # eccentricity (near-circular)
                         0.0,                      # argument of perigee
                         math.radians(inclination_deg),
                         anomaly, mean_motion_rad_min, raan)
            names.append(f"WALKER-P{p + 1:02d}-S{k + 1:02d}")
            satrecs.append(sat)
    return Constellation(names=names, satrecs=satrecs,
                         source=f"walker({planes}x{per_plane}, {altitude_km:g} km, "
                                f"{inclination_deg:g} deg)")


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Observer:
    """A point on the ground (the ground user)."""

    name: str
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0

    def ecef_km(self) -> np.ndarray:
        """Earth-fixed position on the WGS-84 ellipsoid."""
        lat, lon = math.radians(self.lat_deg), math.radians(self.lon_deg)
        e2 = _WGS84_F * (2.0 - _WGS84_F)
        n = _WGS84_A_KM / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
        return np.array([
            (n + self.alt_km) * math.cos(lat) * math.cos(lon),
            (n + self.alt_km) * math.cos(lat) * math.sin(lon),
            (n * (1.0 - e2) + self.alt_km) * math.sin(lat),
        ])

    def up(self) -> np.ndarray:
        """Unit vector normal to the ellipsoid at the observer."""
        lat, lon = math.radians(self.lat_deg), math.radians(self.lon_deg)
        return np.array([math.cos(lat) * math.cos(lon),
                         math.cos(lat) * math.sin(lon),
                         math.sin(lat)])


def gmst_rad(jd_ut1: np.ndarray) -> np.ndarray:
    """Greenwich mean sidereal time (IAU 1982), in radians."""
    t = (np.asarray(jd_ut1, dtype=float) - 2451545.0) / 36525.0
    seconds = (67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * t
               + 0.093104 * t ** 2 - 6.2e-6 * t ** 3)
    return np.deg2rad((seconds % _SECONDS_PER_DAY) / 240.0)


def teme_to_ecef(r_teme: np.ndarray, jd_ut1: np.ndarray) -> np.ndarray:
    """Rotate TEME positions ``(..., T, 3)`` to the Earth-fixed frame.

    ``jd_ut1`` has one Julian date per time step (shape ``(T,)``).
    """
    theta = gmst_rad(jd_ut1)
    c, s = np.cos(theta), np.sin(theta)
    out = np.empty_like(r_teme)
    out[..., 0] = c * r_teme[..., 0] + s * r_teme[..., 1]
    out[..., 1] = -s * r_teme[..., 0] + c * r_teme[..., 1]
    out[..., 2] = r_teme[..., 2]
    return out


def elevation_deg(sat_ecef: np.ndarray, observer: Observer) -> np.ndarray:
    """Elevation angle of each satellite above the observer's horizon."""
    rho = sat_ecef - observer.ecef_km()
    dist = np.linalg.norm(rho, axis=-1)
    return np.degrees(np.arcsin(np.clip(rho @ observer.up() / dist, -1.0, 1.0)))


# ─────────────────────────────────────────────────────────────────────────────
# Ephemeris: positions on a time grid
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Ephemeris:
    """Earth-fixed positions of the satellites near one observer.

    ``pos[i, k]`` is the position (km) of satellite ``i`` at ``k * step_s``
    seconds after ``start_jd``.  Between grid points positions are
    interpolated linearly; with a 1 s grid the error is a few metres.
    """

    names: List[str]
    pos: np.ndarray                  # (n_sat, n_steps, 3), km, Earth-fixed
    step_s: float
    start_jd: float
    observer: Observer
    source: str = ""

    @property
    def duration_s(self) -> float:
        return (self.pos.shape[1] - 1) * self.step_s

    def at(self, t: float) -> np.ndarray:
        """Positions ``(n_sat, 3)`` at second ``t`` (clamped to the grid)."""
        x = min(max(t / self.step_s, 0.0), self.pos.shape[1] - 1.0)
        k = min(int(x), self.pos.shape[1] - 2) if self.pos.shape[1] > 1 else 0
        if self.pos.shape[1] == 1:
            return self.pos[:, 0]
        w = x - k
        return (1.0 - w) * self.pos[:, k] + w * self.pos[:, k + 1]

    def elevations(self, t: float) -> np.ndarray:
        """Elevation (degrees) of every satellite at second ``t``."""
        return elevation_deg(self.at(t), self.observer)

    def elevation_series(self) -> np.ndarray:
        """Elevation of every satellite at every grid step: ``(n_sat, n_steps)``."""
        return elevation_deg(self.pos, self.observer)


def _propagate_ecef(satrecs: Sequence, start_jd: float,
                    seconds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """SGP4 positions in the Earth-fixed frame.

    Returns ``(error, pos)`` with shapes ``(n_sat, T)`` and ``(n_sat, T, 3)``.
    ``error`` is SGP4's code: 0 means the position is valid.
    """
    api = _require_sgp4()
    jd0 = math.floor(start_jd)
    fr = (start_jd - jd0) + seconds / _SECONDS_PER_DAY
    jd = np.full_like(fr, jd0)
    err, r, _ = api.SatrecArray(list(satrecs)).sgp4(jd, fr)
    return err, teme_to_ecef(r, jd + fr)


def build_ephemeris(constellation: Constellation, observer: Observer, *,
                    start_jd: Optional[float] = None, duration_s: float = 300.0,
                    step_s: float = 1.0, keep_radius_km: float = 4000.0) -> Ephemeris:
    """Propagate the constellation and keep the satellites near the observer.

    Only satellites that come within ``keep_radius_km`` (straight-line
    distance) of the observer during the run are kept: the others can neither
    be seen nor reached over a few inter-satellite hops, and dropping them
    makes every later step cheap.  Selection uses a coarse 30 s grid, the
    result a ``step_s`` grid.
    """
    if start_jd is None:
        start_jd = constellation.median_epoch_jd()
    if duration_s <= 0 or step_s <= 0:
        raise ValueError("duration_s and step_s must be > 0")

    coarse = np.arange(0.0, duration_s + 30.0, 30.0)
    err, pos = _propagate_ecef(constellation.satrecs, start_jd, coarse)
    dist = np.linalg.norm(pos - observer.ecef_km(), axis=-1)
    # A satellite moves about 7.6 km/s, so up to 115 km between coarse samples.
    near = (np.nanmin(np.where(err == 0, dist, np.inf), axis=1) <= keep_radius_km + 120.0)
    valid = (err == 0).all(axis=1)
    keep = np.flatnonzero(near & valid)
    if keep.size == 0:
        raise ValueError(
            f"No satellite of {constellation.source or 'the constellation'} comes within "
            f"{keep_radius_km:g} km of {observer.name} during the run.")

    fine = np.arange(0.0, duration_s + step_s, step_s)
    err, pos = _propagate_ecef([constellation.satrecs[i] for i in keep], start_jd, fine)
    ok = (err == 0).all(axis=1)
    keep, pos = keep[ok], pos[ok]
    return Ephemeris(names=[constellation.names[i] for i in keep], pos=pos,
                     step_s=step_s, start_jd=float(start_jd), observer=observer,
                     source=constellation.source)


def jd_to_iso(jd: float) -> str:
    """Julian date to an ISO-8601 UTC string (for logs and manifests)."""
    from datetime import datetime, timedelta, timezone
    moment = datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(days=jd - 2451545.0)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")
