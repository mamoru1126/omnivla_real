from glob import glob

from setuptools import setup

package_name = "omnivla_real_ros"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mamoru1126",
    maintainer_email="sunottty1126@gmail.com",
    description="OmniVLA navigation node for real robots",
    license="MIT",
    entry_points={"console_scripts": ["navigator = omnivla_real_ros.navigator_node:main"]},
)
