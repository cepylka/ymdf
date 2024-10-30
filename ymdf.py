import numpy
import pandas
from astropy.stats import sigma_clip
from scipy.signal import savgol_filter

import pandera
from typing import Tuple, List


def findGaps(
    lightCurve: pandas.DataFrame,
    maximumGap: float = 30,
    minimumObservationPeriod: int = 10
) -> List[Tuple[int, int]]:
    """
    Find gaps in light curve.

    Parameters:

    - `maximumGap`: should be larger than step in time series
    - `minimumObservationPeriod`:
    """

    dt = numpy.diff(lightCurve.index)

    gap = numpy.where(numpy.append(0, dt) >= maximumGap)[0]

    # add start/end of lightCurve to loop over easily
    gapOut = numpy.append(0, numpy.append(gap, len(lightCurve.index)))

    # start
    left = gapOut[:-1]
    # end of data
    right = gapOut[1:]-1

    # drop too short observation periods
    tooShort = numpy.where(numpy.diff(gapOut) < minimumObservationPeriod)
    left = numpy.delete(left, tooShort)
    right = numpy.delete(right, tooShort)

    return list(zip(left, right))


def detrendSavGolUltraViolet(
    lightCurve: pandas.DataFrame,
    gaps: List[Tuple[int, int]],
    windowLength: int
) -> pandas.DataFrame:
    """
    Construct a light curve model. Based on original Appaloosa (Davenport 2016)
    with Savitzky-Golay filtering from `scipy` and iterative `sigma_clipping`
    adopted from K2SC (Aigrain et al. 2016).

    Parameters:

    - `lightCurve`: light curve
    - `gaps`: found gaps in series
    - `windowLength`: number of datapoints for Savitzky-Golay filter, either
    one value for entire light curve of piecewise for gaps
    """

    maximumWindowLength = 5
    if windowLength > maximumWindowLength:
        raise ValueError(
            f"Windows length cannot be larger than {maximumWindowLength}"
        )

    if gaps is None:
        raise ValueError(
            " ".join((
                "Gaps cannot be None, so if your series has no gaps,",
                "then pass an empty list"
            ))
        )

    lightCurve["fluxDetrended"] = pandas.array(
        [numpy.nan]*len(lightCurve.index), dtype=float
    )
    lightCurve["fluxModel"] = pandas.array(
        [numpy.nan]*len(lightCurve.index), dtype=float
    )

    for (le, ri) in gaps:
        # iterative sigma clipping
        correctValues = numpy.where(
            sigma_clip(lightCurve.iloc[le:ri]["flux"])
        )[0] + le
        # incorrect values (inverse of correct ones)
        outliers = list(set(list(range(le, ri))) - set(correctValues))

        betweenGaps = pandas.DataFrame(columns=lightCurve.columns)
        for index, row in lightCurve.iterrows():
            if index in correctValues:
                prwt = pandas.DataFrame([row], index=[index])
                betweenGaps = pandas.concat([betweenGaps, prwt])
            elif index in outliers:
                lightCurve.at[index, "detrended_flux"] = lightCurve.at[
                    index,
                    "flux"
                ]
                lightCurve.at[index, "fluxModel"] = numpy.nanmean(
                    lightCurve["flux"]
                )

        if not betweenGaps.empty:
            betweenGaps["fluxDetrended"] = savgol_filter(
                betweenGaps["flux"],
                wl,
                3,
                mode="nearest"
            )
            betweenGaps["fluxModel"] = (
                betweenGaps["flux"]
                -
                betweenGaps["fluxDetrended"]
                +
                numpy.nanmean(betweenGaps["fluxDetrended"])
            )

        for index, row in lightCurve.iterrows():
            if index in betweenGaps.index:
                lightCurve.at[index, "fluxDetrended"] = betweenGaps.at[
                    index,
                    "fluxDetrended"
                ]
                lightCurve.at[index, "fluxModel"] = betweenGaps.at[
                    index,
                    "fluxModel"
                ]

    # with pandas.option_context("mode.use_inf_as_null", True):
    lightCurve = lightCurve.replace([numpy.inf, -numpy.inf], numpy.nan)
    # lightCurve = lightCurve.dropna()
    return lightCurve


def findFlares(
    timeSeries: list[float],
    fluxSeries: list[float],
    fluxErrorSeries: list[float],
    doPeriodicityRemoving: bool = False
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:

    seriesLength = len(timeSeries)
    if seriesLength != len(fluxSeries) or seriesLength != len(fluxErrorSeries):
        raise ValueError("The length of all the series must be the same")

    # ---

    lightCurveTableSchema = pandera.DataFrameSchema(
        index=pandera.Index(float),  # also check that it is increasing
        columns={
            "flux": pandera.Column(float),
            "fluxError": pandera.Column(float)
        }
    )
    lightCurve = pandas.DataFrame(
        {
            "flux": fluxSeries,
            "fluxError": fluxErrorSeries
        },
        index=timeSeries
    )
    lightCurveTableSchema(lightCurve)

    # ---

    gaps = findGaps(lightCurve, 30, 10)

    # ---

    detrendedLightCurve = detrendSavGolUltraViolet(lightCurve, gaps, 5)
    print(detrendedLightCurve)

    # ---

    flaresTableSchema = pandera.DataFrameSchema(
        {
            "istart": pandera.Column(int, nullable=True),
            "istop": pandera.Column(int, nullable=True)
        }
    )

    flares = pandas.DataFrame()

    flares["istart"] = pandas.array([numpy.nan], dtype="Int64")
    flares["istop"] = pandas.array([numpy.nan], dtype="Int64")

    flaresTableSchema(flares)

    # ---

    # if lc["detrended_flux"].isna().all():
    #     raise TypeError('Flare finding only works on de-trended light curves.')

    # #Now work on periods of continuous observation with no gaps
    # for (le,ri) in gaps:
    #     error = lc.iloc[le:ri]["detrended_flux_err"].values
    #     flux = lc.iloc[le:ri]["detrended_flux"].values

    #     median = lc.iloc[le:ri]["it_med"].values

    #     time = lc.iloc[le:ri]['time']
    #     # run final flare-find on DATA - MODEL

    #     if sigma is None:
    #         isflare = find_flares_in_cont_obs_period(flux, median, error, **kwargs)
    #     else:
    #         isflare = find_flares_in_cont_obs_period(flux, median, error, sigma=sigma[le:ri], **kwargs)


    #     # now pick out final flare candidate indices
    #     candidates = numpy.where( isflare > 0)[0]
    #     if (len(candidates) < 1):#no candidates = no indices
    #         LOG.debug(f'INFO: No candidates were found in the ({le},{ri}) gap.')
    #         istart_gap = numpy.array([])
    #         istop_gap = numpy.array([])
    #     else:
    #         # find start and stop index, combine neighboring candidates
    #         # in to same events
    #         separated_candidates = numpy.where( (numpy.diff(candidates)) > minsep )[0]
    #         istart_gap = candidates[ numpy.append([0], separated_candidates + 1) ]
    #         istop_gap = candidates[ numpy.append(separated_candidates,
    #                                 [len(candidates) - 1]) ]

    #     #stitch indices back into the original light curve
    #     istart = numpy.array(numpy.append(istart, istart_gap + le), dtype='int')
    #     istop = numpy.array(numpy.append(istop, istop_gap + le), dtype='int')
    #     LOG.info('Found {} candidate(s) in the ({},{}) gap.'
    #              .format(len(istart_gap), le, ri))

    # if len(istart) > 0:
    #     l = [equivalent_duration(lc, i, j, err=True) for (i,j) in zip(istart, istop)]
    #     ed_rec, ed_rec_err = zip(*l)
    #     fl = lc["detrended_flux"].values
    #     ampl_rec = [numpy.max(fl[i:j]) / lc["it_med"].values[i] - 1. for (i,j) in zip(istart,istop)]
    #     # cstart = lc.cadenceno[istart].value
    #     # cstop = lc.cadenceno[istop].value
    #     tstart = lc.iloc[istart]['time'].values
    #     tstop = lc.iloc[istop]['time'].values

    #     new = pd.DataFrame({'ed_rec': ed_rec,
    #                        'ed_rec_err': ed_rec_err,
    #                        'ampl_rec': ampl_rec,
    #                        'istart': istart,
    #                        'istop': istop,
    #                        # 'cstart': cstart,
    #                        # 'cstop': cstop,
    #                        'tstart': tstart,
    #                        'tstop': tstop,
    #                        'total_n_valid_data_points': lc["flux"].values.shape[0],
    #                        'dur': tstop - tstart
    #                       })

    #     flares = pd.concat([flares, new], ignore_index=True)

    # return flares


timeSeries = [1.0, 1.1, 1.2]
fluxSeries = [2.0, 2.1, 2.2]
fluxErrorSeries = [3.0, 3.1, 3.2]
findFlares(timeSeries, fluxSeries, fluxErrorSeries)
