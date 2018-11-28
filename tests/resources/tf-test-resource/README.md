This project is based on the [scio-template](https://github.com/spotify/scio-template).

To regenerate the stats files:
```py
from os.path import join
import tensorflow_data_validation as tfdv

for x in ('tf-records', 'tf-records-iris'):
  for y in ('eval', 'train'):
    dir = join('./', x, y)
    tfdv.generate_statistics_from_tfrecord(data_location=join(dir, '*.tfrecords'), output_path=join(dir, '_stats.pb'))
```
