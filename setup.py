from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='mpol_DCT',
      version='0.1',
      description='This program simulates the qubit-cavity dynamics when the cavity is driven.',
      long_description=readme(),
      author='Nicolas Gheeraert',
      author_email='n.gheeraert@physics.iitm.ac.in',
      license='',
      packages=['mpol_DCT'],
      install_requires=[],
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],)
