

if __name__ == "__main__":
    import argparse
    import warnings
    from MisTIC.mistic_class import mistic
    import numpy as np
    import os
    import gc
    
    #turn off progress bars to not clog up the logs
    os.environ["TQDM_DISABLE"] = "True"

    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, help="location of transcript csv" )
    parser.add_argument("-o", type=str, help="location of output folder" )

    args = parser.parse_args()
    detected_transcripts = str(args.t)
    output_folder = str(args.o)
    cell_by_gene_counts = "/home/jholz/mistic_bayesian/HumanLiverCancerPatient1Synth/cell_by_gene_subset_fixed.csv"
    #detected_transcripts = '/home/jholz/mistic_bayesian/HumanLiverCancerPatient1Synth/synthetic_datasets/tx05/synthetic_tx05_transcript_meta.csv'
    cell_metadata = '/home/jholz/mistic_bayesian/HumanLiverCancerPatient1Synth/cell_meta.csv'
    cell_boundary_polygons = "/home/jholz/mistic_bayesian/HumanLiverCancerPatient1Synth/cell_boundaries_fixed.parquet"

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