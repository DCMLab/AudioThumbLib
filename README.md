# AudioThumbLib

AudioThumbLib is a Python library and command-line tool for audio thumbnailing, building on the dynamic-programming algorithm of Müller *et al.* (2013) and Jiang *et al.* (2014). It attempts to determine both the placement and duration of the optimal thumbnail, within a range of acceptable durations and other parameters.

## Documentation

See docstrings within the [source code](https://github.com/DCMLab/AudioThumbLib/blob/master/src/AudioThumbLib.py).

## Installation

```bash
$ git clone git@github.com:DCMLab/AudioThumbLib.git
$ cd AudioThumbLib
$ python -m pip install -r requirements.txt
```

## Use

### Command-line tool

For an overview of available parameters:

`$ python AudioThumbLib.py --help`

To thumbnail with default parameter values, simply provide a filename:

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
