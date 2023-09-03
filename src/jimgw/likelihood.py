from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from abc import ABC, abstractmethod
from jaxtyping import Array, Float
from jimgw.waveform import Waveform
from jimgw.detector import Detector
import jax.numpy as jnp
from astropy.time import Time


class LikelihoodBase(ABC):
    """
    Base class for likelihoods.
    Note that this likelihood class should work for a some what general class of problems.
    In light of that, this class would be some what abstract, but the idea behind it is this
    handles two main components of a likelihood: the data and the model.

    It should be able to take the data and model and evaluate the likelihood for a given set of parameters.

    """

    @property
    def model(self):
        """
        The model for the likelihood.
        """
        return self._model

    @property
    def data(self):
        """
        The data for the likelihood.
        """
        return self._data

    @abstractmethod
    def evaluate(self, params) -> float:
        """
        Evaluate the likelihood for a given set of parameters.
        """
        raise NotImplementedError



class TransientLikelihoodFD(LikelihoodBase):

    detectors: list[Detector]
    waveform: Waveform

    def __init__(
        self,
        detectors: list[Detector],
        waveform: Waveform,
        trigger_time: float = 0,
        duration: float = 4,
        post_trigger_duration: float = 2,
    ) -> None:
        self.detectors = detectors
        self.waveform = waveform
        self.trigger_time = trigger_time
        self.gmst = (
            Time(trigger_time, format="gps").sidereal_time("apparent", "greenwich").rad
        )

        self.trigger_time = trigger_time
        self.duration = duration
        self.post_trigger_duration = post_trigger_duration

    @property
    def epoch(self):
        """
        The epoch of the data.
        """
        return self.duration - self.post_trigger_duration

    @property
    def ifos(self):
        """
        The interferometers for the likelihood.
        """
        return [detector.name for detector in self.detectors]

    def evaluate(
        self, params: Array, data: dict
    ) -> float:  # TODO: Test whether we need to pass data in or with class changes is fine.
        """
        Evaluate the likelihood for a given set of parameters.
        """
        log_likelihood = 0
        frequencies = self.detectors[0].frequencies
        df = frequencies[1] - frequencies[0]
        params['gmst'] = self.gmst
        waveform_sky = self.waveform(frequencies, params)
        align_time = jnp.exp(-1j * 2 * jnp.pi * frequencies * (self.epoch + params['t_c']))
        for detector in self.detectors:
            waveform_dec = (
                detector.fd_response(frequencies, waveform_sky, params)
                * align_time
            )
            match_filter_SNR = (
                4
                * jnp.sum(
                    (jnp.conj(waveform_dec) * detector.data) / detector.psd * df
                ).real
            )
            optimal_SNR = (
                4
                * jnp.sum(
                    jnp.conj(waveform_dec) * waveform_dec / detector.psd * df
                ).real
            )
            log_likelihood += match_filter_SNR - optimal_SNR / 2
        return log_likelihood
