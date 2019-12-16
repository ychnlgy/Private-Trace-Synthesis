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

### AdaTrace
To compile AdaTrace, clone [their repository](https://github.com/git-disl/AdaTrace/blob/master). Then change directories to `AdaTrace/src/` and do:

```bash
javac -cp ../commons-math3-3.4.1.jar:../kd.jar expoimpl/*.java *.java
```

To create 5 synthetic datasets using AdaTrace, first move `brinkhoff.dat` to `AdaTrace/src/`. Then do:

```bash
java -cp .:../commons-math3-3.4.1.jar:../kd.jar Main
```

The 5 datasets should now be in the current directory. To evaluate these datasets, first create the directory `AdaTrace/src/SYNTHETIC-DATASETS/` and move synthesized datasets there. Then do:

```bash
java -cp .:../commons-math3-3.4.1.jar:../kd.jar Experiments
```

Example results:

```
Filename:                       brinkhoff.dat-eps1.0-iteration0.dat
Query AvRE:                     0.10578427372064991
Location coverage kendall-tau:  0.7259920634920635
Frequent pattern F1:            0.68
Frequent pattern support:       0.42459108223522285
Trip error:                     0.01919559574935812
Diameter error:                 0.026553808548257285
Length error:                   0.05082140129080564
```

A visualization of an example synthetic dataset produced by AdaTrace is shown below:

![alt text](images/adatrace.png)

