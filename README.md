<h1> Participation 7 </h1>

<section>
  <p>
    This repo contains the methods that produce aggregate features for the classifier. The algorithm used for this classifier is logistic regression. It will take in a file path to the data and predict if there is video streaming in specified intervals of time.
  </p>
</section>

<section>
  <h3>Input Data</h3>
  <p>Users can input their own desired data to predict from by changing the file path in <code>config/data-fp.json</code>.
</section>

<section>
  <h3> Targets </h3>
  <p>
    <code>python run.py predict</code> will predict if there is video streaming in a network-stats session. The classifier chunks the data and outputs if video is detected in each chunk.
  </p>
  <p>
    <code>python run.py scores</code> will print out the <code>classification_report</code> method from
    <code>sklearn.metrics</code>.
  </p>
</section>
