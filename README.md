# neuralCI
---
In these two notebooks I outline the Word2Vec neural net model, and run it against CI Log Data from Red Hat. 

### Usage
**If you have the `rchi-moby` parquet:**
Place the rchi parquet into the same directory as the notebooks. Then, run it through the [dataFormatter](https://github.com/RobGeada/neuralCI/blob/master/dataFormatter.ipynb) notebook followed the [neuralCI](https://github.com/RobGeada/neuralCI/blob/master/neuralCI.ipynb) notebook.

**If you do not have the `rchi-moby` parquet:**
You cannot run the dataFormatter as you have no raw data to format, but I've included preformatted data that is read by the [neuralCI](https://github.com/RobGeada/neuralCI/blob/master/neuralCI.ipynb) notebook. Therefore, you can simply run neuralCI.

### Disclaimer
For general corporate compliance practices, I haven't committed the dataset, but the notebooks are configured to run from `rhci-moby` parquet data. However, I have included preformatted data, as outputted by the dataFormatter notebook, which allows you to run the neuralCI notebook. The numpy data is numeric and uninterpretable, and therefore should not be a compliance issue. 
