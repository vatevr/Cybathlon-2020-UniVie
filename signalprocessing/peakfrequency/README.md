Base branch for signal processing

#### Peak Frequency extraction

* **input** is a numpy matrix 
  * (channel, samples)
* **output** is a dictionary containing the median instantaneous peaks for each channel grouped by the frequency band
  * \<key:'band',val:{}\>
