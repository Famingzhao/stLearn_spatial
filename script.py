# Dependencies:
# pip: scikit-learn, anndata, scanpy
#
# Python starter kit for the NeurIPS 2021 Single-Cell Competition.
# Parts with `TODO` are supposed to be changed by you.
#
# More documentation:
#
# https://viash.io/docs/creating_components/python/

import logging
import anndata as ad

from scipy.sparse import csc_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor


import pickle
logging.basicConfig(level=logging.INFO)

## VIASH START
# Anything within this block will be removed by `viash` and will be
# replaced with the parameters as specified in your config.vsh.yaml.
par = {
    'input_train_mod1': 'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod1.h5ad',
    'input_train_mod2': 'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod2.h5ad',
    'input_test_mod1': 'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.test_mod1.h5ad',
    'distance_method': 'minkowski',
    'output': 'output.h5ad',
    'n_pcs': 50,
}
## VIASH END

# TODO: change this to the name of your method


import matplotlib.pyplot as plt

# from utils.parameter_intialization import ParamInit
import numpy as np
import crocodile.toolbox as tb
from numpy.linalg import pinv, inv

class ParamInit:
    @staticmethod
    def svd_quick(config, x):  # sparse / Truncated
        """Does not scale well. It can do for 1k order."""
        pass

    @staticmethod
    def svd_dense(config, x):  # Full SVD.
        pass

    @staticmethod
    def random_normal(config, x):
        alpha = np.random.normal(size=(config.i_dim, x.shape[0]))
        beta = np.random.normal(size=(x.shape[1], config.j_dim))
        return alpha, beta  # pivot_sample & pivot_feature

    @staticmethod
    def eigen_quick(config, x):  # sparse / truncated.
        pass

    @staticmethod
    def eigen_dense(config, x):  # full.
        pass
    
class GCProc(tb.Base):
    def __init__(self, i_dim=30, j_dim=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ====================== Model Configuration ====================================
        self.i_dim = i_dim  # size of the latent space.
        self.j_dim = j_dim  # size of the latent space.
        self.param_init = ParamInit.random_normal  # initializes alpha and beta.

        # ===================== Numerical Specs =========================================
        self.norm = tb.Struct(log=False, center=False, scale=False, bias=1)
        self.seed = 1  # for reproducibility of random initializations.
        self.verbose = False

        # ============== Optimizer Specs ================================================
        self.max_iter = 10
        self.count = 1  # iteration counter.
        self.score = []  # history of performance of mae, used by stopping criterion.
        self.score_batch_size = 4
        self.converg_thresh = 1e-3

        # ================== Configuration of Tasks and Inputs (Recover) ====================================
        self.task = ["regression", "classification", "imputation"][0]
        self.method = ["matrix.projection"]  # per task (list of lists). knn another choice.
        self.design = None  # specifies what to predict (i.e. where are the missing values).
        # It should be list with same length as data list (passed to the gcproc)
        self.join = tb.Struct(alpha=[], beta=[])  # which alphas and betas are common among datasets.
        # alpha = [1, 1, None]  # use None to signify unique alpha or beta.

        # ================ Transfer pretrained latents (output to be) ===============================
        self.data = None  # list of datasets passed.
        self.encode = None  # common for all datasets. alpha_x Ex beta_x == alpha_y Y beta_y = ENCODE (1)
        self.code = None  # alpha_d (x) beta_d
        self.prev_encode = None  # for convergence test.

    def init_single_dataset(self, x, update_encode_code=False, idx=None):
        x = self.prepare(x)
        alpha, beta = self.param_init(self, x)
        if update_encode_code or self.encode is None:
            self.encode = alpha @ x @ beta  # pinv(alpha) @ self.encode @ pinv(beta)
        return tb.Struct(x=x, alpha=alpha, beta=beta, recovered=None, idx=idx)

    def init_code_encode(self):
        assert self.data is not None, f"Initialize the parameters first!"
        d = self.data[-1]  # any dataset is good enough.
        self.encode = d.alpha @ d.x @ d.beta
        self.code = inv(d.alpha @ d.alpha.T) @ self.encode @ inv(d.beta.T @ d.beta)

    def check_convergenence(self) -> bool:
        if self.count < self.score_batch_size:
            return False  # too few iterations.
        else:
            mae_avg = np.mean(self.score[-self.score_batch_size:])
            if self.count < self.max_iter:
                return mae_avg < self.converg_thresh
            else:
                print(f"Failed to converge before max iteration {self.max_iter}. Latest loss = {mae_avg}")
                return True

    def fit(self, data_list):
        np.random.seed(self.seed)
        self.data = [self.init_single_dataset(data, idx=idx) for idx, data in enumerate(data_list)]
        self.init_code_encode()
        data = self.data
        while True:
            self.count += 1
            for idx, d in enumerate(data):
                self.update_set(d)
                self.join_params(data, idx, d)

            mae = np.mean(abs(self.prev_encode - self.encode))
            self.score.append(mae)
            if self.check_convergenence():
                break
            if self.verbose:
                # d = data[0]
                # print(f"Iteration #{self.count:3}. Loss = {np.abs(d.x - self.recover(d)).sum():1.0f}")
                fig, ax = plt.subplots()
                ax.plot(self.score)
                ax.set_title(f"GCProc Convergence")
                ax.set_xlabel(f"Iteration")
                ax.set_ylabel(f"Encode Mean Absolute Error")

    def fit_transform(self, data_list):
        """Runs the solver `fit` and then transform each of the datasets provided."""
        self.fit(data_list)
        for dat in self.data:
            self.encode(dat)

    @staticmethod
    def encode(dat):
        return dat.alpha @ dat.x @ dat.beta

    def join_params(self, data, idx, d):
        # ====================== Joining ===============================
        if self.join.alpha:
            indices = np.argwhere(self.join.alpha == self.join.alpha[idx])
            for tmp in data[indices]:
                tmp.alpha = d.alpha
        if self.join.beta:
            indices = np.argwhere(self.join.beta == self.join.beta[idx])
            for tmp in data[indices]:
                tmp.beta = d.beta

    def update_set(self, d):
        tmp = (self.code @ d.beta.T)
        d.alpha = (d.x @ pinv(tmp)).T  # update alpha using backward model (cheap)
        tmp = (d.alpha.T @ self.code)
        d.beta = (pinv(tmp) @ d.x).T  # update beta using backward model (cheap)
        self.prev_encode, self.encode = self.encode, d.alpha @ d.x @ d.beta  # update encode using forward model.
        # d.recovered = self.recover(d)  # Optional.
        return d

    def recover(self, d):  # reconstruct from a, b & z.
        return d.alpha.T @ self.encode @ d.beta.T

    def prepare(self, x):
        """ Based on this function: https://github.com/AskExplain/gcproc/blob/main/R/prepare_data.R
        :return:
        """
        if self.norm.log:
            x = np.log(x + self.norm.bias)
        if self.norm.center:
            x -= np.mean(x, axis=0)  # assumes that the layout is [samples, features]
        if self.norm.scale:
            x = x / np.linalg.norm(x, ord="fro") * np.prod(x.shape)
        return x



method_id = "python_starter_kit"

logging.info('Reading `h5ad` files...')
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(input_train_mod1.obs[["batch"]])

input_train = ad.concat(
    {"train": input_train_mod1, "test": input_test_mod1},
    axis=0,
    join="outer",
    label="group",
    fill_value=0,
    index_unique="-"
)

import scanpy as sc
sc.pp.highly_variable_genes(input_train, min_mean=0.0125, max_mean=10, min_disp=0.01)
input_train = input_train[:, input_train.var.highly_variable]

sc.pp.scale(input_train)

# TODO: implement own method

input_train_batch = enc.fit_transform(input_train.obs[["batch"]])
# tmp_adata = input_train.copy()

# tmp_adata.X = csc_matrix(tmp_adata.layers["counts"])

# sc.pp.combat(input_train, key='batch')

# Do PCA on the input data
logging.info('Performing dimensionality reduction on modality 1 values...')


# i_dim = 200  # But only for learning purpose, eventually, alpha will NOT process the rows.
# j_dim = 100

# gc1 = GCProc(i_dim=i_dim, j_dim=j_dim)

# gc1.fit([input_train.X.toarray()])

# dat = gc1.data[0]

# mod1_pca = dat.x @ dat.beta

# norm = np.linalg.norm(mod1_pca)
# mod1_pca = mod1_pca/norm


embedder_mod1 = TruncatedSVD(n_components=50)



from sklearn.preprocessing import StandardScaler
from scipy import sparse

scaler = StandardScaler(with_mean=False)
scaler.fit(input_train.X)

scaled_counts = sparse.csr_matrix(scaler.fit_transform(input_train.X))
# sel = VarianceThreshold(threshold=(0.5))
# scaled_counts = sel.fit_transform(scaled_counts)
# from scipy.sparse import hstack
# mod1_pca = mydmap.fit_transform(hstack([scaled_counts,input_train_batch]))




mod1_pca = embedder_mod1.fit_transform(scaled_counts)
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(0.8))
mod1_pca = sel.fit_transform(mod1_pca)



# tmp = input_train.obs

# import harmonypy as hm
# ho = hm.run_harmony(mod1_pca, tmp, "batch")
# mod1_pca = ho.Z_corr.T

logging.info('Performing dimensionality reduction on modality 2 values...')

embedder_mod2 = TruncatedSVD(n_components=50)
mod2_pca = embedder_mod2.fit_transform(input_train_mod2.X)

# split dimred back up
X_train = mod1_pca[input_train.obs['group'] == 'train']
X_test = mod1_pca[input_train.obs['group'] == 'test']
y_train = mod2_pca

assert len(X_train) + len(X_test) == len(mod1_pca)

# Get all responses of the training data set to fit the
# KNN regressor later on.
#
# Make sure to use `toarray()` because the output might
# be sparse and `KNeighborsRegressor` cannot handle it.

logging.info('Running Linear regression...')



# reg = MultiOutputRegressor(LGBMRegressor(n_jobs=4, num_leaves=128, max_depth=8, learning_rate=0.05,
#     feature_fraction= 0.9,bagging_fraction=0.7,bagging_freq=10,max_bin= 512,num_iterations=1000))
# reg = MultiOutputRegressor(LGBMRegressor(n_jobs=1))
# reg = LinearRegression()
reg = MultiOutputRegressor(CatBoostRegressor(learning_rate=0.1))
# Train the model on the PCA reduced modality 1 and 2 data
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Project the predictions back to the modality 2 feature space
y_pred = y_pred @ embedder_mod2.components_

# Store as sparse matrix to be efficient. Note that this might require
# different classifiers/embedders before-hand. Not every class is able
# to support such data structures.
y_pred = csc_matrix(y_pred)

adata = ad.AnnData(
    X=y_pred,
    obs=input_test_mod1.obs,
    var=input_train_mod2.var,
    uns={
        'dataset_id': input_train_mod1.uns['dataset_id'],
        'method_id': method_id,
    },
)

logging.info('Storing annotated data...')
adata.write_h5ad(par['output'], compression = "gzip")
