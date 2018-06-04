# Output

*parset-titanic-{1}-{2}.jpg* - {1} is the random walk ID, {2} is the step ID.

*data-distance.csv* - [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) and [Jensen-Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) relative to the baseline dataset, which has uniform values.
These measures are computed for each random walk independently. Note that whenever a category has 0 counts the KL-divergence
infinite. Jensen-Shannon divergence does not have this issue.

*data-distance.png* - Plot of the Jensen-Shannon divergence curve for each random walk.

