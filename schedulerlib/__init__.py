# schedulerlib/__init__.py
import scheduler        # import the compiled C++ extension

def create_satellite(id: int, name: str) -> float:
    """
    TEST DOC
    """
    return scheduler.Satellite(id, name)