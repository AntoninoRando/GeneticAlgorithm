"""Realistic trace generator: a task log like data/example.csv, from real orbits.

``python3 -m data_processing.realistic_trace --download`` writes
``data/realistic/`` (see :mod:`data_processing.realistic_trace.generate`).

Modules, in the order to read them:

* :mod:`.orbits`     real Starlink orbital elements, SGP4, elevation angles.
* :mod:`.scenario`   every parameter of the model, with its source.
* :mod:`.simulator`  the discrete-event simulation that produces the log rows.
* :mod:`.generate`   several runs, the CSV files, the worlds, the report.

Needs the ``sgp4`` package and numpy; nothing here is imported by ``main.py``
unless you run the generator.
"""
