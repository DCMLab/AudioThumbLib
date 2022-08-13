# AudioThumbLib

A simple Python library and command-line tool for audio thumbnailing using the dynamic-programming algorithm of Müller *et al.* (2013) and Jiang *et al.* (2014).

## Installation

```bash
$ git clone git@github.com:DCMLab/AudioThumbLib.git
$ cd AudioThumbLib
$ python -m pip install -r requirements.txt
```

## Use

### Command-line tool

`$ python AudioThumbLib.py --help`

To thumbnail with default parameters, simply supply a valid filename argument:

`$ python AudioThumbLib.py ./tests/Monk.mp3`

### Python module

```python
from AudioThumbLib import AudioThumbnailer
import json

t = AudioThumbnailer('./tests/Monk.mp3')
t.run()

print(json.dumps(t.thumbnail, indent=2))
```

## Documentation

See docstrings within the [source code](https://github.com/DCMLab/AudioThumbLib/blob/master/src/AudioThumbLib.py).

## Tests

```bash
$ cd ./tests
$ pytest test_AudioThumbLib.py -vv
```
