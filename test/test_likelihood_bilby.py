import bilby
from gwpy.timeseries import TimeSeries
from bilby.gw.utils import greenwich_mean_sidereal_time
import numpy as np

np.random.seed(1)

logger = bilby.core.utils.logger
outdir = 'outdir'
label = 'GW150914'

# Data set up
trigger_time = 1126259462

roll_off = 0.4  # Roll off duration of tukey window in seconds, default is 0.4s
duration = 4  # Analysis segment duration
post_trigger_duration = 2  # Time between trigger time and end of segment
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

psd_duration = 32 * duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time

# We now use gwpy to obtain analysis and psd data and create the ifo_list
ifo_list = bilby.gw.detector.InterferometerList([])
for det in ["H1", "L1"]:
    logger.info("Downloading analysis data for ifo {}".format(det))
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    data = TimeSeries.fetch_open_data(det, start_time, end_time)
    ifo.strain_data.set_from_gwpy_timeseries(data)

    logger.info("Downloading psd data for ifo {}".format(det))
    psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time)
    psd_alpha = 2 * roll_off / duration
    psd = psd_data.psd(
        fftlength=duration,
        overlap=0,
        window=("tukey", psd_alpha),
        method="median"
    )
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value)
    ifo_list.append(ifo)

logger.info("Saving data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)

# We now define the prior.
# We have defined our prior distribution in a local file, GW150914.prior
# The prior is printed to the terminal at run-time.
# You can overwrite this using the syntax below in the file,
# or choose a fixed value by just providing a float value as the prior.
priors = bilby.gw.prior.BBHPriorDict(filename='GW150914.prior')

# In this step we define a `waveform_generator`. This is the object which
# creates the frequency-domain strain. In this instance, we are using the
# `lal_binary_black_hole model` source model. We also pass other parameters:
# the waveform approximant and reference frequency and a parameter conversion
# which allows us to sample in chirp mass and ratio rather than component mass
waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments={'waveform_approximant': 'IMRPhenomC',
                        'reference_frequency': 50})

# In this step, we define the likelihood. Here we use the standard likelihood
# function, passing it the data and the waveform generator.
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifo_list, waveform_generator, priors=priors, time_marginalization=False,
    phase_marginalization=False, distance_marginalization=False)

priors['geocent_time'] = float(likelihood.interferometers.start_time)

likelihood.parameters = priors.sample()
likelihood.parameters['t_c'] = greenwich_mean_sidereal_time(likelihood.parameters['geocent_time'])
likelihood.parameters['a_1'] = 0
likelihood.parameters['a_2'] = 0
likelihood.parameters['tilt_1'] = 0
likelihood.parameters['tilt_2'] = 0
likelihood.parameters['phi_12'] = 0
likelihood.parameters['phi_jl'] = 0
params = {}
q = likelihood.parameters['mass_ratio']
params['mass_1'] = likelihood.parameters['chirp_mass']/((q/(1+q)**2)**(3./5)*(1+q))
params['mass_2'] = q*params['mass_1']
params['spin_1'] = likelihood.parameters['a_1']
params['spin_2'] = likelihood.parameters['a_2']
params['luminosity_distance'] = likelihood.parameters['luminosity_distance']
params['theta_jn'] = likelihood.parameters['theta_jn']
params['psi'] = likelihood.parameters['psi']
params['ra'] = likelihood.parameters['ra']
params['dec'] = likelihood.parameters['dec']
params['phase_c'] = likelihood.parameters['phase']
params['t_c'] = likelihood.parameters['t_c']

waveform = likelihood.waveform_generator.frequency_domain_strain(likelihood.parameters)
frequency_array  = ifo_list[0].frequency_array

print("Likelihood value from bilby: "+str(likelihood.log_likelihood_ratio()))

import numpy as np
import bilby
import jax
import jax.numpy as jnp

from jax.config import config
from jaxgw.sampler.NF_proposal import nf_metropolis_kernel, nf_metropolis_sampler
config.update("jax_enable_x64", True)
from jaxgw.gw.likelihood.detector_projection import construct_arm, detector_tensor, antenna_response, get_detector_response
from jaxgw.gw.likelihood.utils import inner_product
from jaxgw.gw.likelihood.detector_preset import get_H1, get_L1
from jaxgw.gw.waveform.TaylorF2 import TaylorF2
from jaxgw.gw.waveform.IMRPhenomC import IMRPhenomC
from jax import random, grad, jit, vmap, jacfwd, jacrev, value_and_grad, pmap

H1, H1_vertex = get_H1()
L1, L1_vertex = get_L1()
strain_H1 = get_detector_response(frequency_array, waveform, likelihood.parameters, H1, H1_vertex)
strain_L1 = get_detector_response(frequency_array, waveform, likelihood.parameters, L1, L1_vertex)

jaxgw_H1_SNR = inner_product(ifo_list[0].strain_data.frequency_domain_strain, strain_H1, frequency_array, ifo_list[0].power_spectral_density_array)
bilby_H1_SNR = ifo_list[0].inner_product(strain_H1)

print(jaxgw_H1_SNR, bilby_H1_SNR) 