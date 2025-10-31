"""

"""

import importlib

def import_package(package_name: str, pip_name: str = None):
    """
    Ensure that a Python package is installed and importable. Raise a clear,
    user-friendly error if the package is missing.

    Parameters
    ----------
    package_name : str
        Name of the module or submodule to import. Example: "shapely.geometry".
    pip_name : str, optional
        Name of the package to install via pip if different from `package_name`.
        Example: package_name="shapely.geometry", pip_name="shapely"

    Returns
    -------
    module
        The imported module or submodule object.

    Raises
    ------
    ModuleNotFoundError
        If the package cannot be imported. Includes a pip install suggestion.
    """
    try:
        return importlib.import_module(package_name)
    except ModuleNotFoundError as e:
        pkg = pip_name or package_name.split('.')[0]  # top-level package
        raise ModuleNotFoundError(
            f"Failed to import `{package_name}`.\n"
            f"Install it with:\n    pip install {pkg}"
        ) from e
