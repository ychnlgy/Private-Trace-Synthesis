import math

from matplotlib import pyplot

def parse_adatrace_eval(s):
    lines = s.strip().split("\n")
    out = {}
    for line in lines:
        key, val = line.split(":")
        val = val.strip().split()[0]
        out[key.strip().lstrip("*")] = float(val.strip())#math.log(float(val.strip()), 10)
    return out

PRIVATE_GAN = [
    (
        2000,
        """
        Query AvRE:		0.5867354893834693
        Location coverage kendall-tau:		0.37202380952380953
        Frequent pattern F1:		0.16
        Frequent pattern support:		5.135030731279546
        Trip error:		0.597405384722441
        Diameter error:		0.21851076806483272
        Length error:		0.2836412416333293
        """
    ),
    (
        3000,
        """
        Query AvRE:		0.6548456429592038
        Location coverage kendall-tau:		0.27884920634920635
        Frequent pattern F1:		0.13
        Frequent pattern support:		16.34251608401513
        Trip error:		0.593959215534686
        Diameter error:		0.2235978491581211
        Length error:		0.3250256270903237
        """
    ),
    (
        5000,
        """
        Query AvRE:		0.6552354135890082
        Location coverage kendall-tau:		0.17257936507936508
        Frequent pattern F1:		0.06
        Frequent pattern support:		1.49122399693511
        Trip error:		0.6288128188164335
        Diameter error:		0.3358998551607076
        Length error:		0.3851548599511671
        """
    ),
]

PUBLIC_GAN = [
    (
        2000,
        """
        *Query AvRE:                    0.07663956899182468     0.10578427372064991
        Location coverage kendall-tau:  0.6784523809523809      0.7259920634920635
        Frequent pattern F1:            0.53                    0.68
        Frequent pattern support:       0.7274587704979105      0.42459108223522285
        Trip error:                     0.09878842858777957     0.01919559574935812
        Diameter error:                 0.04040937756783465     0.026553808548257285
        Length error:                   0.1808399631440616      0.05082140129080564
        """
    ),
    (
        3000,
        """
        *Query AvRE:                    0.05595802327291127     0.10578427372064991
        *Location coverage kendall-tau: 0.7358333333333333      0.7259920634920635
        Frequent pattern F1:            0.63                    0.68
        Frequent pattern support:       0.5864136727030113      0.42459108223522285
        Trip error:                     0.07436397572967487     0.01919559574935812
        Diameter error:                 0.03077201828143057     0.026553808548257285
        Length error:                   0.15743568535397803     0.05082140129080564
        """
    ),
    (
        5000,
        """
        *Query AvRE:                    0.028146547445979442    0.10578427372064991
        *Location coverage kendall-tau: 0.7406746031746032      0.7259920634920635
        Frequent pattern F1:            0.61                    0.68
        Frequent pattern support:       0.5314850423844831      0.42459108223522285
        Trip error:                     0.053613072058763596    0.01919559574935812
        *Diameter error:                0.013600251015592455    0.026553808548257285
        Length error:                   0.09558990550827025     0.05082140129080564
        """
    ),
    (
        13000,
        """
        *Query AvRE:                    0.04230939531525398     0.10578427372064991
        *Location coverage kendall-tau: 0.7559920634920635      0.7259920634920635
        Frequent pattern F1:            0.66                    0.68
        *Frequent pattern support:      0.3586430371981911      0.42459108223522285
        Trip error:                     0.050683045080263334    0.01919559574935812
        *Diameter error:                0.008101308895193405    0.026553808548257285
        *Length error:                  0.04971716163705869     0.05082140129080564
        """
    ),
    (
        15000,
        """
        *Query AvRE:                    0.029894383218779765    0.10578427372064991
        *Location coverage kendall-tau: 0.7732936507936508      0.7259920634920635
        *Frequent pattern F1:           0.71                    0.68
        *Frequent pattern support:      0.2834304722670945      0.42459108223522285
        Trip error:                     0.051443139331389866    0.01919559574935812
        *Diameter error:                0.006277135780115614    0.026553808548257285
        *Length error:                  0.046926810938924285    0.05082140129080564
        """
    )
]

ADATRACE = [
    (
        i,
        """
        Query AvRE:                     0.142
        Location coverage kendall-tau:  0.74
        Frequent pattern F1:            0.62
        Frequent pattern support:       0.38
        Trip error:                     0.045
        Diameter error:                 0.023
        Length error:                   0.041
        """
    ) for i in [2000, 15000]
]


def main():

    plt, axes = pyplot.subplots(nrows=7, sharex=True, figsize=(6, 14))

    titles = [
        "Query AvRE",
        "Location coverage kendall-tau",
        "Frequent pattern F1",
        "Frequent pattern support",
        "Trip error",
        "Diameter error",
        "Length error"
    ]

    short = [
        "Q-AvRE",
        "Loc. KT",
        "FP F1",
        "FP-AvRE",
        "Trip E",
        "Diam. E",
        "Len. E"
    ]

    axes[-1].set_xlabel("Epochs")

    for i, title in enumerate(short):

        axes[i].set_ylabel(title)

    for dset, name, col in [
        (PRIVATE_GAN, "Private GAN", "bx:"),
        (PUBLIC_GAN, "Public GAN", "rx:"),
        (ADATRACE, "AdaTrace", "g:")
    ]:
        x = []
        ys = [[] for _ in titles]
        for epoch, s_val in dset:
            x.append(epoch)
            val = parse_adatrace_eval(s_val)
            for k, title in enumerate(titles):
                ys[k].append(val[title])

        for i, y in enumerate(ys):
            axes[i].plot(x, y, col, label=name)

    pyplot.legend()

    pyplot.savefig("results.png", bbox_inches="tight")


if __name__ == "__main__":
    main()