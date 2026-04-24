import os, sys
sys.path.insert(0, os.path.abspath('../../'))   # so Sphinx can find mylib/
print("SPHINX SYS.PATH:", sys.path)
try:
    import scheduler
    print("SUCCESSFULLY IMPORTED SCHEDULER", scheduler)
    print("SCHEDULER DIR:", dir(scheduler))
except Exception as e:
    print("FAILED TO IMPORT SCHEDULER", e)

project = 'schedulerlib'
copyright = '2026, La Sapienza, Università di Roma, Emiliano Casalicchio, Antonino Rando'
author = 'La Sapienza, Università di Roma, Emiliano Casalicchio, Antonino Rando'
release = '0'

extensions = [
    'sphinx.ext.autodoc',          # core: reads docstrings
    'sphinx.ext.napoleon',         # supports NumPy/Google style docstrings
    'sphinx.ext.viewcode',         # adds [source] links
    'sphinx_autodoc_typehints',    # pulls type hints automatically
]

html_theme = 'furo'                # clean modern look

# Napoleon settings (for NumPy-style docstrings)
napoleon_numpy_docstring = True
napoleon_google_docstring = False

autodoc_mock_imports = ["schedulerlib"]