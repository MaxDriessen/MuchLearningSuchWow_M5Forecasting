from distutils.core import setup

setup(
    name='m5_forecasting',
    version='0.1',
    packages=['m5_forecasting'],
    url='',
    license='',
    author='Roel Hacking',
    author_email='',
    description='',
    entry_points={
          'console_scripts': [
              'm5_forecasting = m5_forecasting.__main__:main'
          ]
      }
)
