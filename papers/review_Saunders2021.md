# A Review Of Saunders2021 

Alta Saunders, David M. Drew, Willie Brink, Machine learning models perform better than traditional empirical models for stomatal conductance when applied to multiple tree species across different forest biomes. Trees, Forests and People 6 (2021) https://doi.org/10.1016/j.tfp.2021.100139

This paper is being held up as an example of how in ecology, machine learning models may do better than empirical models at representing the data. The main results states that "A random forest model performed the best with an R2 of 75%, compared to the empirical Ball-Berry stomatal conductance model (BWB) (R2 = 41%)." After reading over the paper and studying the [code](https://github.com/altazietsman/ML-stomatal-conductance-models) and data, I have a few reservations.

In section 2.1, Saunders explains that they have used a dataset provided by [Anderegg2018](https://figshare.com/articles/dataset/AllData_EcologyLetters_Figshare_v1_3-18_csv/6066449/1) and an unpublished study where the data is available in their github [repository](https://github.com/altazietsman/ML-stomatal-conductance-models). At the end of section 2.4, they note that where net photosynthesis was not available, it was computed using the Farquhar1980 model. While this is true and a reality with which we must struggle, it would be more correct to have said that while Anderegg2018 has photosynthesis, their own data does not and so it was approximated. I would have appreciated them presenting the analysis separately for each dataset. As it stands, you cannot tease out if it is the Ball-Berry model providing a poor approximation of the data, or the combination of Ball-Berry and Farquhar1980. It is all mixed together with the measured photosynthesis from Anderegg2018.

Under closer examination of their code, I have found several errors in the Ball-Berry portion of the analysis. In [line 166](https://github.com/altazietsman/ML-stomatal-conductance-models/blob/master/Model%20development/BWB.py#L116) of ``BWB.py``, they implement a crude grid search minimization to find optimal Ball-Berry parameters.
```python
ca = 40
g0 = [0,0.5,1,1.5,2]
g1 = [5,10,15,20,25]
df_par = pd.DataFrame(list(product(g0, g1)), columns=['g0', 'g1'])
MSE_par = []
for i in range(0,5):
    gsw_bb = (df_par['g0'][i] + df_par['g1'][i] * ((x_train['Photo'] * x_train['RH']) / ca)) / 1000
    MSE = mean_squared_error(gsw_bb, y_train)
    MSE_par.append(MSE)
print('MSE min:', min(MSE_par))
```
They are selecting two arrays of possible search parameters and then using the ``product`` function to compute all combinations. While not the best choice of optimization scheme, the real error is in line 121 where they only loop over the first 5 entries of this product (there should be 25). Effectively, this means that ``g0`` is only ever tested as 0. This hides another mistake in the computation of the Ball-Berry stomatal conductance. The model they write uses a constant ``ca=40`` and divides the whole expression through by 1000. The division by 1000 is presumably for some unit conversion that is not needed. Anderegg2018 provides units for their data and it requires no conversion. This factor of 1000 means that the actual optimal ``g0`` as they have implemented things is closer to 36 (I ran the code myself with expanded parameters) and leads to erroneous results. I think obtaining reasonable optimal parameters probably gave them confidence that everything was correct.

I am not clear why they have used a constant ``ca=40``. This variable is the CO2 concentration near the leaf (in ppm, or umol/mol), which is a given column in the Anderegg2018 data. To make things worse, a value of 40 is far too small, by maybe a factor of 10, judging from the order of magnitude of the Anderegg2018 data.

When I implemented my own fitting [code](https://github.com/nocollier/MLPhotoSynthesis/blob/main/stomatal_conductance.py) (with ``scipy.optimize.minimize``), using Anderegg2018 and Saunders but as two separate pieces, I observe very good correlation (0.88, better than the random forest model in the paper) in the Anderegg2018 data. Also note my optimal Ball-Berry parameters are reasonable. To further explore this, I tested the Medlyn model as well with similar results. However, when I run against the Saunders data using their implementation of Farquhar1980 for photosynthesis and a constant ``ca=40``, I get terrible correlation.

```
---------------- Anderegg2018 ------------------
Removed 43 rows (6.9%) marked as outliers.
Optimized BallBerry parameters: gs0 = 0.01680881468313495, gs1 = 10.237721578983216
Optimized Medlyn parameter: gs1 = 3.385846454611909
            Photo        Cond        Tair         VPD          RH        CO2S  Cond_bb_opt  Cond_m_opt
count  579.000000  579.000000  579.000000  579.000000  579.000000  579.000000   579.000000  579.000000
mean     8.299746    0.126478   24.298877    2.034061   45.493520  382.930760     0.126478    0.130074
std      5.941308    0.106416    4.424791    1.235932   18.247999   92.503663     0.094069    0.088784
min     -1.339959   -0.000553   15.370000    0.548583    8.242000  148.200000    -0.004025   -0.024934
25%      3.979000    0.046252   20.355000    1.083026   32.835000  368.550000     0.049416    0.056173
50%      7.620000    0.093385   23.660000    1.649000   51.890000  389.450000     0.107139    0.117396
75%     11.148944    0.172158   27.300000    2.675000   58.105000  396.535000     0.179260    0.192601
max     30.000000    0.466000   37.580000    5.903000   83.600000  701.200000     0.420799    0.395699
Optimized BallBerry correlation: 0.8839713525917283
Optimized Medlyn correlation: 0.8844930430470752

---------------- Saunders ------------------
Photosynthesis added by Farquhar model
Assumed leaf CO2 concentration of 40 ppm as per author script
Removed 4 rows (0.8%) marked as outliers.
Optimized BallBerry parameters: gs0 = 0.13406822089781864, gs1 = 0.6897209169440246
Optimized Medlyn parameter: gs1 = -0.06332759705937147
            Photo        Cond        Tair         VPD          RH     ca  Cond_bb_opt  Cond_m_opt
count  494.000000  494.000000  494.000000  494.000000  494.000000  494.0   494.000000  494.000000
mean    13.258916    0.255931   25.205243    0.015404   53.767915   40.0     0.255931    0.254559
std      2.503487    0.149473    4.265948    0.005456    6.535166    0.0     0.022451    0.087634
min      7.694900    0.010000   17.890000    0.007060   36.670000   40.0     0.213750    0.083966
25%     11.707856    0.140000   22.150000    0.011211   48.705000   40.0     0.236978    0.191644
50%     13.130430    0.250000   25.150000    0.014458   53.905000   40.0     0.258409    0.244526
75%     15.193813    0.350000   28.650000    0.018951   58.665000   40.0     0.272934    0.316990
max     17.938834    0.700000   33.140000    0.031819   68.400000   40.0     0.308362    0.451996
Optimized BallBerry correlation: 0.15019907632007853
Optimized Medlyn correlation: 0.03252437403401084
```

I think there are a few possible explanations for the poor performance in the latter dataset.

* Notice that ``CO2S`` in Anderegg2018 (equates to ``ca`` in Saunders) is much larger. Could there be some units problem? No units are given for the Saunders data.
* Also notice that ``VPD`` of Saunders is about 100x smaller. Since this is used in the Farquhar model, then it could make the photosynthesis also off and affect the results.
* The photosynthesis of Saunders is a good bit larger than Anderegg. While it could unit errors in the components, there could also be a bug in the Farquhar implementation.

As a sanity check, I also downloaded the stomatal conductance data from Lin2015 also. I observe very similar data magnitudes and performance to the Anderegg2018 data. 

```
----------------- Lin2015 ------------------
Removed 962 rows (6.9%) marked as outliers.
Relative humidity computed from a model
Optimized BallBerry parameters: gs0 = 0.0218728345051859, gs1 = 10.00888345271472
Optimized Medlyn parameter: gs1 = 3.37116048660136
              Photo          Cond         Tleaf           VPD          CO2S            RH   Cond_bb_opt    Cond_m_opt
count  12885.000000  12885.000000  12885.000000  12885.000000  12885.000000  12885.000000  12885.000000  12885.000000
mean       8.836610      0.143984     27.260792      2.046309    362.730817     46.756326      0.143984      0.142332
std        6.591768      0.135185      6.181603      0.991907     23.495630     15.914310      0.110493      0.109956
min        0.011025      0.001750      8.430000      0.001000    256.300000    -53.899235     -0.122198      0.000161
25%        4.172000      0.056000     22.880000      1.320000    350.720000     35.445219      0.068372      0.064144
50%        6.970000      0.097800     27.990000      1.880000    364.090000     47.708427      0.111926      0.109801
75%       11.700000      0.180000     31.766000      2.640000    378.800000     58.588127      0.181560      0.191158
max       33.094098      0.728000     44.650000      5.933333    474.000000     99.909919      0.708137      0.683382
Optimized BallBerry correlation: 0.8173466271230776
Optimized Medlyn correlation: 0.8063589209134864
```

I am left with the conclusion that the authors have made a few mistakes that have made the Ball-Berry model appear to underperform dramatically. I would not cite this paper in support of machine learning in ecology at this time. While it is possible that I have misunderstood something in their code, the fact that I can independently fit parameters to Ball-Berry and Medlyn with good correlation to the observed leads me to think that I am correct about their mistakes.

Furthermore, this leaves us with a uncomfortable reality: machine learning models can do a good job reproducing even erroneous data. Every single data point added to a collection needs to be carefully thought out and vetted.

