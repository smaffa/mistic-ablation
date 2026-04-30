import os
import time
import psutil as pst
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scvi
import anndata as ad
import pandas as pd
import seaborn as sns
import argparse

def load_data(cell_metadata_filepath, tx_metadata_filepath):
    transcript_metadata = pd.read_csv(tx_metadata_filepath).rename({'Unnamed: 0': 'molecule_id'}, axis=1)
    cell_metadata = pd.read_csv(cell_metadata_filepath, index_col=0)
    counts_df = pd.pivot_table(transcript_metadata, index='cell_id', columns='gene', aggfunc='count', values='molecule_id')
    return counts_df, cell_metadata

def create_adata_object(counts_df, cell_metadata, spatial_coordinate_columns=['center_x', 'center_y']):
    adata = sc.AnnData(counts_df)
    cell_metadata = cell_metadata.reindex(adata.obs.index)
    adata.obs = cell_metadata

    adata.layers['counts'] = adata.X
    adata.layers['counts'] = np.nan_to_num(adata.layers['counts'], nan=0)
    adata.obsm['X_spatial'] = adata.obs[spatial_coordinate_columns].to_numpy()
    return adata

def load_h5ad(h5ad_filepath, spatial_coordinate_columns=['center_x', 'center_y']):
    adata = sc.read_h5ad(h5ad_filepath)
    adata.layers['counts'] = adata.X
    adata.layers['counts'] = np.nan_to_num(adata.layers['counts'], nan=0)
    adata.obsm['X_spatial'] = adata.obs[spatial_coordinate_columns].to_numpy()
    return adata

def memory_usage():
    """Get current memory usage in MB"""
    process = pst.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Return memory in MB

def setup_resolvi(adata, accelerator='cpu'):
    scvi.external.RESOLVI.setup_anndata(adata, labels_key="cell_type", layer="counts")
    resolvi_model = scvi.external.RESOLVI(adata, semisupervised=True)
    resolvi_model.train(max_epochs=100, accelerator=accelerator)
    return resolvi_model

def resolvi_predict(adata, resolvi_model):
    adata.obsm["resolvi_celltypes"] = resolvi_model.predict(adata, num_samples=3, soft=True)
    adata.obs["resolvi_predicted"] = adata.obsm["resolvi_celltypes"].idxmax(axis=1)

def get_latent_representation_and_neighbors(adata, resolvi_model):
    adata.obsm["X_resolVI"] = resolvi_model.get_latent_representation(adata)
    adata.obsm["X_resolVI"] = resolvi_model.get_latent_representation(adata)
    sc.pp.neighbors(adata, use_rep="X_resolVI")
    sc.tl.umap(adata)

def sample_posterior_and_add_to_adata(adata, resolvi_model):
    # Sample posterior for corrected rates
    samples_corr = resolvi_model.sample_posterior(
        model=resolvi_model.module.model_corrected,
        return_sites=["px_rate"],
        summary_fun={"post_sample_q50": np.median},
        num_samples=10,
       summary_frequency=100
    )
    samples_corr = pd.DataFrame(samples_corr).T

    # Sample posterior for mixture proportions
    samples = resolvi_model.sample_posterior(
        model=resolvi_model.module.model_residuals,
        return_sites=["mixture_proportions"],
        summary_fun={"post_sample_means": np.mean},
        num_samples=10,
        summary_frequency=100
    )
    samples = pd.DataFrame(samples).T

    # Assign proportions and generated expression
    adata.obs[["true_proportion", "diffusion_proportion", "background_proportion"]] = samples.loc[
        "post_sample_means", "mixture_proportions"
    ]
    adata.layers["generated_expression"] = samples_corr.loc["post_sample_q50", "px_rate"]

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, help="input transcript file")
    parser.add_argument("-m", type=str, help="input cell metadata file")
    parser.add_argument("-h", type=str, help="input h5ad file")
    parser.add_argument("-o", type=str, help="h5ad output file")

    args = parser.parse_args()

    output_file = str(args.o)

    if args.t is None and args.m is None:
        assert args.h is not None, "if transcripts and cell metadata are not provided, then an anndata h5ad file must be provided"
        adata = load_h5ad(str(args.h))

    elif args.h is None:
        assert args.t is not None and args.m is not None, "if an anndata h5ad file is not provided, then transcript and cell metadata must be provided"
        counts_df, metadata = load_data(str(args.m), str(args.t))
        adata = create_adata_object(counts_df, metadata)

    start_memory = memory_usage()
    start_time = time.time()

    resolvi_model = setup_resolvi(adata, accelerator='cpu')

    resolvi_predict(adata, resolvi_model)

    get_latent_representation_and_neighbors(adata, resolvi_model)

    sample_posterior_and_add_to_adata(adata, resolvi_model)

    end_time = time.time()
    end_memory = memory_usage()
    
    execution_time = end_time - start_time
    memory_diff = end_memory - start_memory
    adata.uns['execution_time'] = execution_time
    adata.uns['memory_used'] = memory_diff

    adata.write_h5ad(output_file)
