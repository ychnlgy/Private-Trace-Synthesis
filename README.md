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

Using a simple GAN with no privacy, we produce the following 20,000 traces at 2000 epochs:

![alt text](images/public-gan.png)

And at 13000 epochs we get:

![alt text](images/public-gan-E13000.png)

Scores are (* means we do better):

```
Filename:                       public-gan.dat          brinkhoff.dat-eps1.0-iteration0.dat
*Query AvRE:                    0.07663956899182468     0.10578427372064991
Location coverage kendall-tau:  0.6784523809523809      0.7259920634920635
Frequent pattern F1:            0.53                    0.68
Frequent pattern support:       0.7274587704979105      0.42459108223522285
Trip error:                     0.09878842858777957     0.01919559574935812
Diameter error:                 0.04040937756783465     0.026553808548257285
Length error:                   0.1808399631440616      0.05082140129080564
```

At 3000 epochs, scores are:

```
Filename:                       public-gan.dat          brinkhoff.dat-eps1.0-iteration0.dat
*Query AvRE:                    0.05595802327291127     0.10578427372064991
*Location coverage kendall-tau: 0.7358333333333333      0.7259920634920635
Frequent pattern F1:            0.63                    0.68
Frequent pattern support:       0.5864136727030113      0.42459108223522285
Trip error:                     0.07436397572967487     0.01919559574935812
Diameter error:                 0.03077201828143057     0.026553808548257285
Length error:                   0.15743568535397803     0.05082140129080564
```

At 5000 epochs:

```
Filename:                       public-gan.dat          brinkhoff.dat-eps1.0-iteration0.dat
*Query AvRE:                    0.028146547445979442    0.10578427372064991
*Location coverage kendall-tau: 0.7406746031746032      0.7259920634920635
Frequent pattern F1:            0.61                    0.68
Frequent pattern support:       0.5314850423844831      0.42459108223522285
Trip error:                     0.053613072058763596    0.01919559574935812
*Diameter error:                0.013600251015592455    0.026553808548257285
Length error:                   0.09558990550827025     0.05082140129080564
```

At 13000 epochs:

```
Filename:                       public-gan.dat          brinkhoff.dat-eps1.0-iteration0.dat
*Query AvRE:                    0.04230939531525398     0.10578427372064991
*Location coverage kendall-tau: 0.7559920634920635      0.7259920634920635
Frequent pattern F1:            0.66                    0.68
*Frequent pattern support:      0.3586430371981911      0.42459108223522285
Trip error:                     0.050683045080263334    0.01919559574935812
*Diameter error:                0.008101308895193405    0.026553808548257285
*Length error:                  0.04971716163705869     0.05082140129080564
```

At 15000 epochs:

```
Filename:                       public-gan.dat          brinkhoff.dat-eps1.0-iteration0.dat
*Query AvRE:                    0.029894383218779765    0.10578427372064991
*Location coverage kendall-tau: 0.7732936507936508      0.7259920634920635
*Frequent pattern F1:           0.71                    0.68
*Frequent pattern support:      0.2834304722670945      0.42459108223522285
Trip error:                     0.051443139331389866    0.01919559574935812
*Diameter error:                0.006277135780115614    0.026553808548257285
*Length error:                  0.046926810938924285    0.05082140129080564
```

We also try a tiny GAN, which obtains (820 epochs)

```
Filename:                       public-gan.dat          brinkhoff.dat-eps1.0-iteration0.dat
Query AvRE:                     0.11798949194063456     0.10578427372064991
Location coverage kendall-tau:  0.6638492063492063      0.7259920634920635
Frequent pattern F1:            0.48                    0.68
Frequent pattern support:       1.1275608115817066      0.42459108223522285
Trip error:                     0.22076443233704635     0.01919559574935812
Diameter error:                 0.06777863541331408     0.026553808548257285
Length error:                   0.24211632049734993     0.05082140129080564
```



### Private GAN

First we try to train a private GAN using differentially-private SGD for the discriminator only. At 20000 epochs:

![alt text](images/private-D-I20000.png)

```
Filename:                       public-gan.dat          brinkhoff.dat-eps1.0-iteration0.dat
Query AvRE:                     0.636400129979293       0.10578427372064991
Location coverage kendall-tau:  0.4899603174603175      0.7259920634920635
Frequent pattern F1:            0.18                    0.68
Frequent pattern support:       3.3810501418785424      0.42459108223522285
Trip error:                     0.620681087727295       0.01919559574935812
Diameter error:                 0.4988209244361573      0.026553808548257285
Length error:                   0.4482031507718248      0.05082140129080564
```



### Poly Represent

```python
for traj in iter_trajectories("brinkhoff.dat"):
  
		x = traj[:,0]
		y = traj[:,1]
    
    # fit the original data to poly with args.degree
		pfit = np.polyfit(x, y, args.degree)
		func_y = np.poly1d(pfit) # generate function of f
		traj[:,1] = func_y(traj[:,0]) # replace the original y by the new y

		lines.append("#%d:" % i )
		lines.append(traj_to_string(traj))
		i += 1
    
	with open(save_dat, "w") as f:
		f.write("\n".join(lines))
```



Evaluation of the poly dataset generated above, we got:

```
Filename: d3.dat
Query AvRE:											0.11778861570219686
Location coverage kendall-tau:	0.799404761904762
Frequent pattern F1:						0.67
Frequent pattern support:				0.43828738301347625
Trip error:											0.041560107542598784
Diameter error:									0.0028318682182573262
Length error:										0.010689317359303062
```

```
Filename: d5.dat
Query AvRE:											0.12208954118604075
Location coverage kendall-tau:	0.8060714285714285
Frequent pattern F1:						0.67
Frequent pattern support:				0.48829091787160356
Trip error:											0.035773687582232004
Diameter error:									0.0018972783694032964
Length error:										0.01798421354993425

```

```
Filename: d6.dat
Query AvRE:											0.12313647520736361
Location coverage kendall-tau:	0.8028174603174603
Frequent pattern F1:						0.66
Frequent pattern support:				0.5274448307562064
Trip error:											0.033580478963772115
Diameter error:									0.001832723043432205
Length error:										0.020976995716376727
```

this would be the error caused by the poly representation itself. if we take this as the ground truth, and find something close to it, theoretically the metric would be pretty close to this one. This partially supports the proof that w-dist remains specific property.

and future work could be to find a better representation than poly that performs well on these metric; then trying to prove it has similar property with poly represent regarding in terms of remaining property under w-dist. that proof is likely to be okay with most of the continuous representation that keeps distance order in specific mapping.