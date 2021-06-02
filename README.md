# InterGen

Python interferogram generator.

## Getting Started

This project was created to make data set for neural network.

### Prerequisites

To run this project you have to install following packages:

```
pip install numpy
pip install matplotlib
pip install PIL
pip install numba
pip install cupy
```

### Running
#### Interferogram genrator

To run you have to run `runInterGen.py`. With examplary command

```
python runIntergen.py [folder for results]
```

For help with parsing arguments
```
python runIntergen.py --help
```

If you want to edit generation parameters modify file `settings.json`.

#### Chambolle labels genrator

To run you have to run `runChambolle.py`. With examplary command

```
python runChambolle.py [dataset directory] [output] [mode] [number of images] [starting image id]
```

For help with parsing arguments
```
python runChambolle.py --help
```

## Authors

* **Konstanty Szumigaj** - *Initial work* - [Konstantysz](https://github.com/Konstantysz)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
