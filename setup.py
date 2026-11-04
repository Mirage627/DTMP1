"""Editable source install retaining the EfficientFlow import namespace."""
from setuptools import find_namespace_packages, setup

setup(
    name="DTMP1",
    version="2.1.0",
    description="Calibrated directional JVP correction for MP1",
    packages=["EfficientFlow"] + [
        "EfficientFlow." + name
        for name in find_namespace_packages(
            where="EfficientFlow",
            include=["codecs*", "common*", "dataset*", "env*", "gym_util*",
                     "model*", "policy*", "shared_memory*", "workspace*"],
        )
    ],
    package_data={"EfficientFlow": ["config/*.yaml", "config/task/*.yaml"]},
    python_requires=">=3.9",
)
