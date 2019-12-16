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

Below is an example of a dataset of Brinkhoff synthesized using AdaTrace with `ε = 1`):

![alt text](images/adatrace.png)

### GAN Experiments

Using a simple GAN with no privacy, we produce the following example at 2000 epochs:

![alt text](images/public-gan.png)

Scores are:

```
Filename:                       public-gan.dat
Query AvRE:                     0.07663956899182468
Location coverage kendall-tau:  0.6784523809523809
Frequent pattern F1:            0.53
Frequent pattern support:       0.7274587704979105
Trip error:                     0.09878842858777957
Diameter error:                 0.04040937756783465
Length error:                   0.1808399631440616
```