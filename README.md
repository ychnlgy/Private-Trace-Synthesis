# Private-Trace-Synthesis
Algorithms for synthesizing differentially-private traces.

### Installation
```bash
pip3 install -r requirements.txt
```

### Brinkhoff dataset
We use pre-generated [Brinkhoff data](https://github.com/git-disl/AdaTrace/blob/master/brinkhoff.dat) from the AdaTrace repository. This dataset contains 20,000 synthetic traces.

Suppose we downloaded the data to `~/Downloads/brinkhoff.dat`. To visualize the dataset, change working directories to `scripts/data` and run

```bash
python3 plot_trajectories.py --file ~/Downloads/brinkhoff.dat
```

This will produce something similar to the following:

![alt text](images/brinkhoff.png)
