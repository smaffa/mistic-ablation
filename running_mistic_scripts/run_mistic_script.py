

if __name__ == "__main__":
    import argparse
    import warnings
    from MisTIC.mistic_class import mistic
    import numpy as np
    import os
    import gc
    import pandas as pd
    
    #turn off progress bars to not clog up the logs
    os.environ["TQDM_DISABLE"] = "True"

    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, help="location of transcript csv" )
    parser.add_argument("-o", type=str, help="location of output folder" )
    parser.add_argument("--c", type=str, default=None, help="location of transcript cell by gene -- for transcript assignment to work, cell by gene must match simulated dataset" )

    args = parser.parse_args()
    cell_by_gene_counts = args.c

    detected_transcripts = str(args.t)
    output_folder = str(args.o)
    if(args.c is None):
        print("Creating a cell by gene matrix that matches the synthetic transcripts")
        tx_tag = str(args.t).split("/")[-1].split("_")[1]
        synth_foldr = "/".join(str(args.t).split("/")[0:-1])
        cell_by_gene_counts = f"{synth_foldr}/cell_by_gene_subset_{tx_tag}.csv"
        tx_meta = pd.read_csv(detected_transcripts)
        counts_df = (
        tx_meta
        .groupby(['cell_id', 'gene'])
        .size()
        .unstack(fill_value=0)
        )
        counts_df = counts_df.astype(int)
        print("saving csv at ", cell_by_gene_counts)
        counts_df.to_csv(cell_by_gene_counts)
    #detected_transcripts = '/home/jholz/mistic_bayesian/HumanLiverCancerPatient1Synth/synthetic_datasets/tx05/synthetic_tx05_transcript_meta.csv'
    cell_metadata = '/orcd/compute/edsun/001/jholz/mistic/mistic_bayesian/HumanLiverCancerPatient1Synth/cell_meta.csv'
    cell_boundary_polygons = "/orcd/compute/edsun/001/jholz/mistic/mistic_bayesian/HumanLiverCancerPatient1Synth/cell_boundaries_fixed.parquet"
    
    m = mistic(cell_centroid_x_col='center_x',
                    cell_centroid_y_col='center_y',
                    celltype_col="cell_type",
                    tx_x_col='x',
                    tx_y_col='y',
                    gene_col='gene',
                    cell_col='cell_id')

    m.import_data(cell_by_gene_counts=cell_by_gene_counts,
                        cell_metadata=cell_metadata,
                        cell_boundary_polygons=cell_boundary_polygons,
                        detected_transcripts=detected_transcripts)
    
    m.patchfy_data()

    m.initialize_parameters()
    gc.collect()
    m.training_loop(n_epochs=20)
    gc.collect()
    m.save_model(dir_name=output_folder,
                model_name="mistic",
                save_correction_result=False)
    m.compute_reassign_probs()
    gc.collect()
    m.correct_tx(reassign_threshold_grid=np.arange(start=0.1, stop=0.6, step=0.1),
                remove_threshold_grid=np.arange(start=0, stop=0.4, step=0.1),
                choice_type="best")
    gc.collect()

    m.save_model(dir_name=output_folder,
                    model_name="mistic",
                    save_correction_result=True)
