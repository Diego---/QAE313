from setuptools import setup

setup(
    name='qae',
    version='0.1',
    python_requires=">=3.10",
    author='Diego Alberto Olvera Millán',
    author_email='dolveram@uni-mainz.de',
    packages=[
        'qae',
        'qae.circuits',
        'qae.evaluation',
        'qae.optimization',
        'qae.plotting',
        'qae.utils'
    ],
    install_requires=[
        'numpy',
        'matplotlib',
        'umz_sequence_generator',
        'umz_backend_connector',
        'qiskit == 1.1',
        'qiskit_experiments == 0.6.1',
        'qiskit_algorithms == 0.3.0'
    ],
)
