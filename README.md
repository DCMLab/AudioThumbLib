# AudioThumbLib

AudioThumbLib is a Python library and command-line tool for audio thumbnailing. Building on the dynamic-programming algorithm of Müller *et al.* (2013) and Jiang *et al.* (2014), it attempts to determine the positions and durations of the most characteristic material in a given musical stream—for example, a main theme and its repetitions.

## Documentation

See docstrings within the [source code](https://github.com/DCMLab/AudioThumbLib/blob/master/src/AudioThumbLib.py).

## Installation

```bash
$ git clone git@github.com:DCMLab/AudioThumbLib.git
$ cd AudioThumbLib
$ python -m pip install -r requirements.txt
```

AudioThumbLib has been developed and tested on Python 3.10.5 but may work with earlier versions, as well.

## Use

### Command-line tool

For an overview of available parameters:

`$ python AudioThumbLib.py --help`

To produce a thumbnail using default parameter values, simply provide a filename:

`$ python AudioThumbLib.py ./tests/Monk.mp3`

### Python module

```python
from AudioThumbLib import AudioThumbnailer
import json

t = AudioThumbnailer('./tests/Monk.mp3')
t.run()

print(json.dumps(t.thumbnail, indent=2))
```

## Tests

```bash
$ cd ./tests
$ pytest test_AudioThumbLib.py -vv
```

## Contact

Yannis Rammos (`yannis.rammos@epfl.ch`)
