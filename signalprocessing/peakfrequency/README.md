Base branch for signal processing

#### Peak Frequency extraction

* **input** is a numpy matrix 
  * (channel, samples)
* **output** is a dictionary containing the median instantaneous peaks for each channel grouped by the frequency band
  * \<key:'band',val:{}\>

## instantfreq_box.py

The box script for open vibe

Here are some screenshots for my configuration.

![Screenshot from 2019-12-19 11-45-56](/home/biropo/Downloads/Screenshot from 2019-12-19 11-45-56.png)

![Screenshot from 2019-12-19 11-45-56](/home/biropo/Downloads/Screenshot from 2019-12-19 11-48-28.png)

## peak_frequency.py & peak_frequency_parallel.py

The script itself

## worspace.ipynb

Contains the current status of my alpha range optimizing function