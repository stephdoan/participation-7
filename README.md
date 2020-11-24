<h1> Participation 7-8 </h1>

<section>
  <p>
    This repo contains an early implementation of a classifier with spectral features.
  </p>
</section>

<section>
  <h3>Input Data</h3>
  <p>
    Users can input their own desired data to predict from by changing the file
    path in <code>config/data_params.json</code>.
    The "folder" argument is preset to a "data" folder.
  </p>
</section>

<section>
  <h3> Targets </h3>
  <p>
    <code>python run.py predict</code> will predict if there is video
    streaming in a network-stats session. The classifier chunks the data
    into uniform time periods and makes predictions on each time period. The
    default time period is 100 seconds. It will then print 1 or 0 - 1 means
    there is video and 0 means there is no video. 
  </p>
</section>
