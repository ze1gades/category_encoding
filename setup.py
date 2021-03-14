from setuptools import setup, find_packages

packages = find_packages()
packages_folders = ['/'.join(p.split('.')) for p in packages]
packages_dict = dict(zip(packages, packages_folders))

setup(
    name='category_encoding',
    vesion='0.0.4',
    url='https://github.com/ze1gades/category_encoding.git',
    packages=packages,
    package_dir=packages_dict
)