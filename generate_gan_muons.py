import numpy as np
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(os.path.expanduser('~/muGAN'))
from SHiP_GAN_module import muGAN
mu_gan = muGAN()

path_to_enhanced = sys.argv[1]
num_repetitions = int(sys.argv[2])

seed_auxiliary_distributions = np.load(path_to_enhanced)
seed_auxiliary_distributions = np.take(seed_auxiliary_distributions,
                                       np.random.permutation(seed_auxiliary_distributions.shape[0]),
                                       axis=0,
                                       out=seed_auxiliary_distributions)
fraction_to_boost = 0.04
cut = int(np.shape(seed_auxiliary_distributions)[0]*fraction_to_boost)
dist = np.abs(np.random.normal(loc=0,scale=1,size=np.shape(seed_auxiliary_distributions[:cut,2])))
dist = np.abs(np.random.normal(loc=0,scale=1,size=np.shape(dist)))
dist += 1
dist = np.power(dist,0.55)
seed_auxiliary_distributions[:cut,2] *= dist
seed_auxiliary_distributions = np.take(seed_auxiliary_distributions,
                                       np.random.permutation(seed_auxiliary_distributions.shape[0]),
                                       axis=0,
                                       out=seed_auxiliary_distributions)

where_array = np.where(seed_auxiliary_distributions[:,3]>3)
shape_i = int(np.shape(where_array)[1]*0.3)
where_array = where_array[0][:int(shape_i)]
seed_auxiliary_distributions[where_array,2] = seed_auxiliary_distributions[where_array,2]*1.4

where_array = np.where(seed_auxiliary_distributions[:,3]>0)
shape_i = int(np.shape(where_array)[1]*0.015)
where_array = where_array[0][:int(shape_i)]
seed_auxiliary_distributions[where_array,2] = seed_auxiliary_distributions[where_array,2]*1.3

seed_auxiliary_distributions = np.take(seed_auxiliary_distributions,
                                       np.random.permutation(seed_auxiliary_distributions.shape[0]),
                                       axis=0,
                                       out=seed_auxiliary_distributions)
boosted_muon_kinematic_vectors = mu_gan.generate_enhanced(auxiliary_distributions=seed_auxiliary_distributions,
                                                          size=num_repetitions)
np.save("../gan_muons.npy", boosted_muon_kinematic_vectors)
with open("../done.txt", "wb") as f:
    pass