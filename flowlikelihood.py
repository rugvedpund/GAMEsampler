import numpy as np
import os
import torch
import corner
import lusee
import NormalizingFlow as nf

##---------------------------------------------------------------------------##

model = "GIS_ulsa_nside128_sigma2.0_galcut20.0_chromaticBeamFalse_combineSigma4_SNRpp1e+09_seed0_subsampleSigma2.0_gainFluctuation0.0_SVD_freqs1,51"
fg = "ulsa.fits"

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_tensor_type("torch.cuda.FloatTensor")

##---------------------------------------------------------------------------##
# load model, fg and do all the PCA stuff for 2+4

assert os.path.exists(model)
flow = nf.FlowAnalyzerV2(loadPath=model)

args = nf.Args()
args.SNRpp = 1e9
args.combineSigma = "4"
args.prettyprint()

flow.set_fg(args)

# set truth t21 template
t21 = lusee.MonoSkyModels.T_DarkAges_Scaled(
    flow.freqs, A=0.04, nu_rms=14.0, nu_min=16.4
)
flow.set_t21(t21)

def likelihood(arr):
    try:
        assert arr.shape[1] == 3  # =50*2 nfreqs bins for 2+4 model
    except AssertionError:
        print("likelihood query must be of shape (npoints,nfreqs)")

    return flow.get_likelihoodFromSamplesGAME(arr)

if __name__ == "__main__":
    arr = np.array([[1.0, 14.0, 16.4]])

    print(likelihood(arr))
