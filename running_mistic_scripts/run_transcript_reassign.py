from MisTIC.mistic_class import mistic

import numpy as np

import pandas as pd
import os 
import gc


if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, help="tx #" )
    args = parser.parse_args()

    tx = str(args.t)
    fix_cellbygene = True
    cell_by_gene_counts = f"/home/jholz/mistic_bayesian/HumanLiverCancerPatient1Synth/synthetic_datasets/cell_by_gene_subset_tx{tx}.csv"
    detected_transcripts = f'/home/jholz/mistic_bayesian/HumanLiverCancerPatient1Synth/synthetic_datasets/synthetic_tx{tx}_transcript_meta.csv'
    cell_metadata = '/home/jholz/mistic_bayesian/HumanLiverCancerPatient1Synth/cell_meta.csv'
    cell_boundary_polygons = "/home/jholz/mistic_bayesian/HumanLiverCancerPatient1Synth/cell_boundaries_fixed.parquet"
    mistic_model_loc = f"/home/jholz/mistic_bayesian/mistic_runs/original_1/tx{tx}"
    new_mistic_model_loc =  f"/home/jholz/mistic_bayesian/mistic_runs/original_1/tx{tx}_withcorr"

    os.makedirs(new_mistic_model_loc, exist_ok=True)

    if fix_cellbygene:
        print("Creating a cell by gene matrix that matches the synthetic transcripts")
        tx_meta = pd.read_csv(detected_transcripts)
        counts_df = (
        tx_meta
        .groupby(['cell_id', 'gene'])
        .size()
        .unstack(fill_value=0)
        )
        counts_df = counts_df.astype(int)
        counts_df.to_csv(cell_by_gene_counts)

    m = mistic(cell_centroid_x_col='center_x',
                        cell_centroid_y_col='center_y',
                        celltype_col="cell_type",
                        tx_x_col='x',
                        tx_y_col='y',
                        gene_col='gene',
                        cell_col='cell_id')

    gc.collect()

    print("loading model at", mistic_model_loc)
    m.load_model(dir_name=mistic_model_loc,
                        model_name="mistic",)

    print("importing data")
    m.import_data(cell_by_gene_counts=cell_by_gene_counts,
                        cell_metadata=cell_metadata,
                        cell_boundary_polygons=cell_boundary_polygons,
                        detected_transcripts=detected_transcripts)

    print("computing probs")
    m.compute_reassign_probs()

    print("correcting tx")
    m.correct_tx(reassign_threshold_grid=np.arange(start=0.1, stop=0.6, step=0.1),
                remove_threshold_grid=np.arange(start=0, stop=0.4, step=0.1),
                choice_type="best")

    print("saving")
    m.save_model(dir_name=new_mistic_model_loc,
                        model_name="mistic",
                        save_correction_result=True)






