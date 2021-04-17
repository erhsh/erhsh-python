import os
import types
from importlib import import_module

from setuptools import setup, find_packages


def get_version():
    machinery = import_module("importlib.machinery")
    version_path = os.path.join(os.path.dirname(__file__), "erhsh", "version.py")
    module_name = "__erhshversion__"
    loader = machinery.SourceFileLoader(module_name, version_path)
    version_module = types.ModuleType(module_name)
    loader.exec_module(version_module)
    return version_module.__version__


if __name__ == '__main__':
    version = get_version()
    print("version:", version)

    setup(
        name="erhsh-python",
        version=version,
        description="This is a util python package",
        author="along",
        author_email="erhsh_165@126.com",
        packages=find_packages(),
        license="Apache 2.0",
        url="https://www.erhsh.com",
        entry_points={
            "console_scripts": [
                "ems=erhsh.ms.cli:ms_cli",
                "emsd=erhsh.ms.demo.cli:msd_cli",
                "ept=erhsh.pt.cli:pt_cli",
                "eptd=erhsh.pt.demo.cli:ptd_cli",
                "etools=erhsh.tools.cli:tools_cli",
            ]
        },
        include_package_data=True,
        install_requires=[
            "click",
        ],
        zip_safe=False,
    )
