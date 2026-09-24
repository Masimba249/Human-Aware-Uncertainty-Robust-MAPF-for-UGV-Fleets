# Copyright (c) 2026 Collins Masimba
# SPDX-License-Identifier: Apache-2.0
from setuptools import find_packages, setup

package_name = 'remroc_ha'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'scipy', 'pyyaml'],
    zip_safe=True,
    maintainer='Collins Masimba',
    maintainer_email='Masimba249@gmail.com',
    description='Human-aware, uncertainty-robust MAPF core library for REMROC.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ha_sim_experiments = remroc_ha.sim.experiments:main',
            'ha_analyze = remroc_ha.analysis:main',
        ],
    },
)
