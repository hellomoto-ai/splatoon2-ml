"""Package spml module"""
import setuptools


def _set_up():
    setuptools.setup(
        name='spml',
        version='0.1.0',
        packages=setuptools.find_packages(
            exclude=['tests*'],
        ),
        test_suite='tests',
        install_requires=[
            'torch >=1.1.0, <2.0',
            'opencv-python',
        ],
        extras_require={
            'dev': [
                'pytest',
                'pylint',
                'flake8',
                'flake8-print',
                'pytest-cov',
            ]
        },
    )


if __name__ == '__main__':
    _set_up()
