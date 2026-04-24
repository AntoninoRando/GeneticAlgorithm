# schedulerlib/__init__.py
import scheduler        # import the compiled C++ extension

def create_satellite(id: int, name: str) -> float:
    """
    Create a satellite using the C++ engine.

    Parameters
    ----------
    data : list of float
        Input values.
    mode : str
        Either ``"fast"`` or ``"precise"``.

    Returns
    -------
    float
        The computed result.

    Examples
    --------
    >>> from schedulerlib import compute
    >>> compute([1.0, 2.0, 3.0])
    2.0
    """
    return scheduler.Satellite(id, name)