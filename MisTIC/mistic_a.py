# Data IO
import os
import json 
import pandas as pd
import scanpy as sc
import geopandas as gpd
import polars as pl 
# Data manipulation 
import numpy as np
from scipy.special import softmax
from scipy.stats import ks_2samp, entropy
import torch 
import torch.nn as nn 
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
# Utility function 
from MisTIC.utility import import_data, calculate_mask_distance, binary_gumbel_softmax_sample,\
        make_reassignment_adata, calibrate_threshold, compute_gene_threshold, generate_count_patches,\
            diagLinear, Positive, JSONEncoder, even_split
from MisTIC.generate_tx_feature import generate_feature
from MisTIC.data_loader import generate_patch_coords, load_patch
# User entertainment
from tqdm.auto import tqdm
# Typing 
from typing import Union, Optional, Tuple 


class mistic_a(nn.Module):
    """Alternate version of mistic that uses only two factors for transcript
    reassignment inference:

        1. Gene expression compatibility  (``exp_feature``)
        2. Neighborhood expression support  (``neighbor_exp_feature``)

    The first factor used by the original mistic —
    *spatial proximity* (``distance_feature``) — is intentionally excluded.
    See the inline ``# [MODIFIED]`` comments for every location where the
    original code was changed and the reasoning behind each change.
    """

    def __init__(self,
                cell_centroid_x_col: str='center_x',
                cell_centroid_y_col: str='center_y',
                celltype_col: Optional[str]=None,
                tx_x_col: str='global_x',
                tx_y_col: str='global_y',
                gene_col: str='gene',
                cell_col: str='cell_id',
                leiden_res: float=1,
                dr_method: str='umap',
                max_centroid_dist: float=50,
                mask_dist_cutoff: float=5,
                nearest: int=3,
                prior_50_reassign_prob: float=0.01,
                prior_5_reassign_prob: float=0.99,
                seed: int=42,
                max_de_cells: int=100000,
                model_device: Optional[Union[str, torch.device]] = None) -> None:
        """Instantiate a mistic_a object.

        Parameters
        ----------
        (Identical to mistic — see original docstring for details.)
        """
        super().__init__()
        
        # Record parameters 
        self.import_data_par = {
            'cell_centroid_x_col': cell_centroid_x_col,
            'cell_centroid_y_col': cell_centroid_y_col,
            'tx_x_col': tx_x_col,
            'tx_y_col': tx_y_col,
            'gene_col': gene_col,
            'cell_col': cell_col,
            'celltype_col': celltype_col,
            'leiden_res': leiden_res,
            'dr_method': dr_method
        }
        self.calculate_mask_distance_par = {
            'max_centroid_dist': max_centroid_dist,
        }
        self.generate_feature_par = {
            'mask_dist_cutoff': mask_dist_cutoff,
            'nearest': nearest,
            'seed': seed,
            'max_de_cells': max_de_cells
        }
        
        # Create data 
        self.adata = None
        self.cell_coords = None
        self.tx_metadata = None
        self.current_layer = "counts_0"
        self.mask_distance = None
        self.intf_tx = None
        self.tx_reassign_info = None
        self.coord_list = {"neighbor{}".format(i): [] for i in range(nearest)}
        self.tx_to_reassign = None
        self.tx_to_remove = None
        self.criterion_df = None
        
        # Parameters for training 
        self.init_temperature = 1.0
        self.min_temperature = 0.5
        self.current_temperature = self.init_temperature
        self.ANNEAL_RATE = 0.00003
        self.log_interval = 10
        
        # Set model device
        if model_device is None:
            self.model_device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(model_device, str):
            self.model_device = torch.device(model_device)
        else:
            self.model_device = model_device
        self.n_genes = None
        self.n_leiden = None
        self.prior_50_reassign_prob = prior_50_reassign_prob
        self.prior_5_reassign_prob = prior_5_reassign_prob
        self.calibrator = None
        
        self.reassign_coefficients = None
        self.prior_reassign_coefficients = None

        self.cell_type_coefficients = None
        self.optimizer = None
        self.CEl_list = []
        self.KLD_list = []

    def import_data(self,
                    cell_metadata: Union[str, pd.DataFrame],
                    cell_boundary_polygons: Union[str, gpd.GeoDataFrame],
                    detected_transcripts: Union[str, pd.DataFrame, gpd.GeoDataFrame],
                    cell_by_gene_counts: Optional[Union[str, pd.DataFrame]]=None) -> None:
        """Read in cell metadata, cell boundary information, transcript
        information, and optionally a cell-by-gene matrix.

        (Identical to mistic.import_data — see original docstring.)
        """
        self.adata, self.cell_coords, self.tx_metadata = import_data(
            cell_metadata=cell_metadata,
            cell_boundary_polygons=cell_boundary_polygons,
            detected_transcripts=detected_transcripts,
            cell_by_gene_counts=cell_by_gene_counts,
            **self.import_data_par)
        self.n_genes = self.adata.uns['n_genes']
        self.n_leiden = self.adata.uns['n_leiden']
        self.mask_distance = calculate_mask_distance(adata=self.adata,
                                                      cell_coords=self.cell_coords,
                                                      **self.calculate_mask_distance_par)
        self.intf_tx = generate_feature(adata=self.adata,
                                        layer=self.current_layer,
                                        tx_metadata=self.tx_metadata,
                                        cell_coords=self.cell_coords,
                                        mask_distance=self.mask_distance,
                                        **self.generate_feature_par)
        
    def patchfy_data(self,
                     percent_cell_per_patch: float=0.1,
                     num_overlap: int=7) -> None:
        """Generate patches represented by their coordinates.

        (Identical to mistic.patchfy_data — see original docstring.)
        """
        for neighbor_index in range(self.generate_feature_par['nearest']):
            key = "neighbor"+str(neighbor_index)
            self.coord_list[key] = generate_patch_coords(adata=self.adata, 
                                                        intf_tx=self.intf_tx,
                                                        percent_cell_per_patch=percent_cell_per_patch,
                                                        num_overlap=num_overlap,
                                                        neighbor_index=neighbor_index) 
            if len(self.coord_list[key]) == 0:
                self.generate_feature_par['nearest'] -= 1
                del self.coord_list[key]
    
    def initialize_parameters(self) -> None:
        """Initialize model parameters.

        [MODIFIED] The original mistic used ``features=3`` for both
        ``prior_reassign_coefficients`` and ``reassign_coefficients``,
        corresponding to the three factors:
            [0] distance_feature
            [1] exp_feature
            [2] neighbor_exp_feature

        Here we use ``features=2`` to drop the first factor
        (``distance_feature``).  The bias and weight tensors are also
        shortened to length 2 accordingly.  Everything else is identical.
        """
        alpha_0 = -np.log(1/self.prior_50_reassign_prob-1+1e-20)
        temp = -np.log(0.05/0.95) 
        alpha_1 = (-np.log(1/self.prior_5_reassign_prob-1+1e-20) - alpha_0)/temp
        self.calibrator = calibrate_threshold(alpha_0=alpha_0,
                                              alpha_1=alpha_1)
        # [MODIFIED] features=2 instead of 3: only exp + neighbor_exp features
        self.prior_reassign_coefficients = diagLinear(features=2, bias=True)
        self.prior_reassign_coefficients.bias = nn.Parameter(
            torch.tensor([alpha_0]*2, dtype=torch.float32).reshape_as(
                self.prior_reassign_coefficients.bias))
        self.prior_reassign_coefficients.weight = nn.Parameter(
            torch.tensor([alpha_1]*2, dtype=torch.float32).reshape_as(
                self.prior_reassign_coefficients.weight))
        for param in self.prior_reassign_coefficients.parameters():
            param.requires_grad = False
        # [MODIFIED] features=2 instead of 3
        self.reassign_coefficients = diagLinear(features=2, bias=True,
                                                initial_weights=[np.log(np.exp(alpha_1)-1)]*2,
                                                initial_bias=[alpha_0]*2)
        parametrize.register_parametrization(self.reassign_coefficients, "weight", Positive())
        self.cell_type_coefficients = nn.Linear(in_features=self.n_genes, 
                                            out_features=self.n_leiden,
                                            bias=True)
        self.optimizer = optim.Adam([{'params': self.reassign_coefficients.parameters()},
                                    {'params': self.prior_reassign_coefficients.parameters()},
                                    {'params': self.cell_type_coefficients.parameters()}], lr=1e-3)
        self.to(self.model_device)
    
    def encode(self, 
               tx_features: torch.tensor,
               temperature: float) -> Tuple[torch.tensor, torch.tensor]:
        """Compute posterior probability given the features.

        [MODIFIED] ``tx_features`` is now a 2-column tensor
        [exp_feature, neighbor_exp_feature] rather than the original 3-column
        tensor.  The rest of the logic (Gumbel-softmax sampling, product
        over factors) is unchanged.
        """
        reassign_logits = self.reassign_coefficients(tx_features)
        reassign_probs = torch.sigmoid(reassign_logits)
        individual_reassign_hard = binary_gumbel_softmax_sample(logits=reassign_logits,
                                                        temperature=temperature,
                                                        model_device=self.model_device,
                                                        hard=True)
        reassign_hard = torch.prod(individual_reassign_hard, dim=1, keepdim=True)
        
        return reassign_hard, reassign_probs
        
    def decode(self, 
               updated_cell_by_gene_counts: torch.tensor,
               tx_prior_features: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """Compute the likelihood and the prior.

        [MODIFIED] ``tx_prior_features`` is now a 2-column tensor
        [prior_exp_feature, prior_neighbor_exp_feature].  The
        ``prior_distance_feature`` column is no longer passed in or
        consumed here.  The likelihood computation via
        ``cell_type_coefficients`` is unchanged.
        """
        cell_type_logits = self.cell_type_coefficients(updated_cell_by_gene_counts)
        prior_reassign_logits = self.prior_reassign_coefficients(tx_prior_features)
        prior_reassign_probs = torch.sigmoid(prior_reassign_logits)
        
        return cell_type_logits, prior_reassign_probs
    
    def forward(self,
                tx_features: dict, 
                tx_prior_features: dict,
                cell_by_gene_counts: torch.tensor, 
                row_index_self: torch.tensor,
                row_index_neighbor: torch.tensor,
                col_index: torch.tensor,
                temperature: float) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """The forward pass.

        [MODIFIED] Callers must supply ``tx_features`` and
        ``tx_prior_features`` as 2-column tensors (without the
        ``distance_feature`` / ``prior_distance_feature`` columns).
        All scatter-add indexing and count-matrix update logic is unchanged.
        """
        reassign_hard, reassign_probs = self.encode(tx_features=tx_features,
                                                    temperature=temperature)
        index_self = row_index_self * self.adata.uns['n_genes'] + col_index
        index_neighbor = row_index_neighbor * self.adata.uns['n_genes'] + col_index
        update_patch_self = torch.zeros((cell_by_gene_counts.shape[0])*(self.adata.uns['n_genes']), 
                                        1, dtype=torch.float32, device=self.model_device).scatter_add(0, index_self, reassign_hard)
        update_patch_neighbor = torch.zeros((cell_by_gene_counts.shape[0])*(self.adata.uns['n_genes']), 
                                            1, dtype=torch.float32, device=self.model_device).scatter_add(0, index_neighbor, reassign_hard)
        ordered_row_index = np.repeat(np.arange(cell_by_gene_counts.shape[0]), self.adata.uns['n_genes'])
        ordered_col_index = np.tile(np.arange(self.adata.uns['n_genes']), cell_by_gene_counts.shape[0])
        cell_by_gene_counts[ordered_row_index, ordered_col_index] -= update_patch_self.squeeze(1)
        cell_by_gene_counts[ordered_row_index, ordered_col_index] += update_patch_neighbor.squeeze(1)
        cell_type_logits, prior_reassign_probs = self.decode(updated_cell_by_gene_counts=cell_by_gene_counts,
                                                             tx_prior_features=tx_prior_features)
        
        return reassign_probs, cell_type_logits, prior_reassign_probs
    
    def loss_function(self,
                      cell_type_logits: torch.tensor, 
                      cell_type_labels: torch.tensor,
                      reassign_probs: torch.tensor,
                      prior_reassign_probs: torch.tensor) -> Tuple[torch.tensor, np.array, np.array]:
        """The loss function.

        [MODIFIED] The KLD sum now runs over 2 terms instead of 3, matching
        the reduced feature dimensionality.  The formula itself is identical.
        """
        CEl = F.cross_entropy(cell_type_logits,
                                cell_type_labels, reduction='mean')
        log_ratio_1 = torch.log(reassign_probs/(prior_reassign_probs+1e-20)+1e-20)
        log_ratio_0 = torch.log((1-reassign_probs)/(1-prior_reassign_probs+1e-20)+1e-20)
        KLD_reassign = torch.mean(reassign_probs * log_ratio_1 + (1-reassign_probs) * log_ratio_0, dim=0)
        
        return CEl + KLD_reassign.sum(), CEl.detach().cpu().numpy().item(), KLD_reassign.detach().cpu().numpy()
    
    def training_loop(self,
                      n_epochs: int,
                      early_stop_pval: float=0.5) -> None:
        """The training loop.

        [MODIFIED] The call to ``load_patch`` (from the data loader) must
        return 2-column feature tensors.  If ``load_patch`` is not modified
        upstream, the caller/data-loader layer must be responsible for
        dropping the ``neighbor_exp_feature`` column before passing data into
        this model.  The training loop structure here is otherwise identical
        to the original.
        """
        adata_w_leiden_xy = self.adata.to_df(self.current_layer).merge(self.adata.obs[['leiden','x','y']],
                                                                        how='left',
                                                                        left_index=True,
                                                                        right_index=True)
        adata_w_leiden_xy = pl.from_pandas(adata_w_leiden_xy, include_index=True)
        adata_var = pl.from_pandas(self.adata.var, include_index=True)
        
        self.train()
        for epoch in range(n_epochs):
            self.current_temperature = self.init_temperature
            CEl_epoch_list = []
            KLD_epoch_list = []
            if epoch >= 2:
                cel_previous_2 = np.array(self.CEl_list[epoch-2])
                cel_previous_1 = np.array(self.CEl_list[epoch-1])
                p_val = ks_2samp(cel_previous_2, cel_previous_1).pvalue
                if p_val > early_stop_pval:
                    print("="*30)
                    print("No improvement in the classification task detected. Stop early at epoch {}".format(epoch-1))
                    print("="*30)
                    break
            neighbor_indices = list(range(self.generate_feature_par['nearest']))
            np.random.shuffle(neighbor_indices)
            for neighbor_index in neighbor_indices:
                key = "neighbor"+str(neighbor_index)
                np.random.shuffle(self.coord_list[key])
                sub_intf_tx = self.intf_tx.filter(pl.col("neighbor_index") == neighbor_index)
                train_loss = 0.0
                for minibatch_ind, coord in tqdm(enumerate(self.coord_list[key]),
                                                total=len(self.coord_list[key]),
                                                desc=key):
                    # Load data
                    cell_by_gene_counts, tx_features, tx_prior_features, cell_type_labels, row_index_self, row_index_neighbor, col_index = load_patch(
                        adata_w_leiden_xy=adata_w_leiden_xy,
                        adata_var=adata_var,
                        intf_tx=sub_intf_tx,
                        coord_tuple=coord,
                        model_device=self.model_device)

                    # [MODIFIED] Drop the first column (distance_feature) from
                    # both the posterior features and the prior features before
                    # passing them into the model.  This is the single point where
                    # the ablation is enforced at inference time.
                    tx_features = tx_features[:, 1:]          # keep only exp + neighbor_exp
                    tx_prior_features = tx_prior_features[:, 1:]

                    reassign_probs, cell_type_logits, prior_reassign_probs = self(
                        tx_features=tx_features, 
                        tx_prior_features=tx_prior_features,                
                        cell_by_gene_counts=cell_by_gene_counts, 
                        row_index_self=row_index_self,
                        row_index_neighbor=row_index_neighbor,
                        col_index=col_index,
                        temperature=self.current_temperature)
                    self.optimizer.zero_grad()
                    loss, CEl, KLD = self.loss_function(cell_type_logits=cell_type_logits,
                                                        cell_type_labels=cell_type_labels,
                                                        reassign_probs=reassign_probs,
                                                        prior_reassign_probs=prior_reassign_probs)
                    loss.backward()
                    CEl_epoch_list.append(CEl)
                    KLD_epoch_list.append(KLD)
                    train_loss += loss.item()
                    self.optimizer.step()
                    if minibatch_ind % 100 == 1:
                        self.current_temperature = np.maximum(
                            self.current_temperature * np.exp(-self.ANNEAL_RATE * minibatch_ind),
                            self.min_temperature) 
                    if minibatch_ind % self.log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch, minibatch_ind, len(self.coord_list[key]),
                                    100. * minibatch_ind / len(self.coord_list[key]),
                                    loss.item()))
                print('====> Epoch: {} Average loss: {:.4f}'.format(
                        epoch, train_loss / len(self.coord_list[key])))
            self.CEl_list.append(np.array(CEl_epoch_list))
            self.KLD_list.append(np.array(KLD_epoch_list))
                
    def compute_reassign_probs(self) -> None:
        """Compute reassignment probabilities.

        [MODIFIED] The original selected three columns from ``intf_tx``:
            ['distance_feature', 'exp_feature', 'neighbor_exp_feature']
        Here we select only two:
            ['exp_feature', 'neighbor_exp_feature']

        This means the model scores each transcript–neighbor pair using only
        gene expression compatibility and neighborhood expression support,
        completely ignoring spatial proximity (distance_feature).
        """
        self.eval()
        with torch.no_grad():
            # [MODIFIED] Only two feature columns; 'distance_feature' is omitted.
            tx_features_chunks = even_split(
                array=self.intf_tx[['exp_feature', 'neighbor_exp_feature']].to_numpy(),
                chunk_size=np.ceil(self.intf_tx.shape[0]/100))
            reassign_probs = np.array([], dtype=float).reshape(0,1)
            for tx_features_chunk in tqdm(tx_features_chunks):
                tx_features = torch.tensor(tx_features_chunk, dtype=torch.float32, device=self.model_device)
                _, reassign_probs_chunk = self.encode(tx_features=tx_features,
                                                        temperature=self.min_temperature)
                reassign_probs_chunk = torch.prod(reassign_probs_chunk, dim=1, keepdim=True)
                reassign_probs = np.vstack([reassign_probs, reassign_probs_chunk.cpu().numpy()])
            self.intf_tx = self.intf_tx.with_columns(pl.Series(name="reassign_probs_raw", values=reassign_probs.squeeze(-1)))
            self.intf_tx = self.intf_tx.with_columns(pl.Series(name="reassign_probs", values=self.calibrator(reassign_probs.squeeze(-1))))
            self.tx_reassign_info = self.intf_tx.group_by("molecule_id").agg(pl.all().sort_by("reassign_probs", descending=False).last())

    def _correct_tx(self,
                    adata_obs: pl.DataFrame,
                    reassign_threshold: float,
                    remove_threshold: float) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Correct transcripts given thresholds.

        [MODIFIED] The ``drop`` call on ``tx_reassign_info`` originally
        removed three prior-feature columns:
            ['prior_distance_feature', 'prior_exp_feature',
             'prior_neighbor_exp_feature']
        Here we drop only two, because the prior was never fitted on
        ``distance_feature``:
            ['prior_exp_feature', 'prior_neighbor_exp_feature']

        ``prior_distance_feature`` is left as a harmless passenger column —
        it is present in ``intf_tx`` (generated upstream) but never read by
        the scoring logic.  All threshold logic and downstream filtering is
        unchanged.
        """
        # [MODIFIED] Drop only the two prior-feature columns we used.
        tx_to_change = self.tx_reassign_info.drop(['prior_exp_feature',
                                                    'prior_neighbor_exp_feature'])

        # Add reassign threshold 
        tx_to_change = tx_to_change.with_columns(pl.lit(reassign_threshold).cast(pl.Float64).alias("reassign_threshold"))
        # Add removal threshold 
        tx_to_change = tx_to_change.with_columns((pl.col("reassign_threshold")*(1-remove_threshold)).alias("remove_threshold"))
        # Dichotomize results 
        tx_to_change = tx_to_change.with_columns(pl.when((pl.col('reassign_probs')>pl.col("reassign_threshold"))).then(1).otherwise(0).alias('reassign'))
        # For not reassigned tx, remove them per threshold 
        tx_to_change = tx_to_change.with_columns(pl.when((pl.col("reassign")==0) & (pl.col('reassign_probs')>pl.col("remove_threshold"))).then(1).otherwise(0).alias('remove'))
        # Filter reassigned 
        tx_to_reassign = tx_to_change.filter(pl.col("reassign")==1).drop(["reassign", "remove", "reassign_threshold", "remove_threshold"])
        tx_to_remove = tx_to_change.filter(pl.col("remove")==1).drop(["reassign", "remove", "reassign_threshold", "remove_threshold"])
        # Rename for readability
        tx_to_reassign = tx_to_reassign.join(adata_obs.rename({"cell_type": "from_cell_type"}),
                                                how='left', left_on='cell_id', right_on="cell_id")
        tx_to_reassign = tx_to_reassign.join(adata_obs.rename({"cell_type": "to_cell_type"}),
                                                how='left', left_on='neighbor_cell_id', right_on="cell_id")
        tx_to_reassign = tx_to_reassign.drop(["cell_type", "neighbor_celltype"])
        # Filter removed 
        tx_to_remove = tx_to_remove.join(adata_obs.rename({"cell_type": "from_cell_type"}),
                                                how='left', left_on='cell_id', right_on="cell_id")
        tx_to_remove = tx_to_remove.drop(["cell_type", "neighbor_celltype"])
        
        return tx_to_reassign, tx_to_remove
    
    
    def _find_criteria(self,
                    adata_obs: pl.DataFrame,
                    reassign_threshold_grid: Union[np.array, list],
                    remove_threshold_grid: Union[np.array, list]) -> None:
        """Grid search for criteria. (Unchanged from original mistic.)"""
        criterion_df = []
        zero_counts = self.adata.to_df().copy()
        zero_counts.loc[:,:] = 0
        zero_counts = pl.from_pandas(zero_counts, include_index=True)
        for reassign_threshold in tqdm(reassign_threshold_grid, desc="Reassign grid"):
            for remove_threshold in tqdm(remove_threshold_grid, desc="Remove grid"):
                tx_to_reassign, tx_to_remove = self._correct_tx(adata_obs=adata_obs,
                                                            reassign_threshold=reassign_threshold,
                                                            remove_threshold=remove_threshold)
                counts_to_subtract, counts_to_add, rm_counts_to_subtract = generate_count_patches(
                    adata=zero_counts,
                    tx_to_reassign=tx_to_reassign,
                    tx_to_remove=tx_to_remove)
                X_chunks = even_split(
                    array=(self.adata.to_df("counts_0")+counts_to_add-counts_to_subtract-rm_counts_to_subtract).values,
                    chunk_size=np.ceil(self.adata.X.shape[0]/100))
                leiden_chunks = even_split(
                    array=self.adata.obs["counts_0_leiden"].astype(int).values,
                    chunk_size=np.ceil(self.adata.X.shape[0]/100))
                loss = 0
                with torch.no_grad():
                    for X, leiden in zip(X_chunks, leiden_chunks):
                        X = torch.tensor(X, dtype=torch.float32, device=self.model_device)
                        leiden = torch.tensor(leiden, dtype=torch.int64, device=self.model_device)
                        cell_type_logits = self.cell_type_coefficients(X)
                        loss += F.cross_entropy(cell_type_logits, leiden, reduction='sum').cpu().numpy().item()
                loss /= (self.adata.X.shape[0])
                criterion_df.append((reassign_threshold, remove_threshold, loss, tx_to_remove.shape[0]))
        self.criterion_df=pd.DataFrame(criterion_df, columns=['reassign_threshold', 'remove_threshold_percent', 'loss', "n_removed"])
        self.criterion_df.sort_values("loss", ascending=True, ignore_index=True, inplace=True)
        self.criterion_df.loc[:, "remove_threshold"] = self.criterion_df.loc[:, 'reassign_threshold'] * (1-self.criterion_df.loc[:, 'remove_threshold_percent'])
        self.criterion_df.loc[:, "loss_increase"] = self.criterion_df.loc[:, 'loss']/self.criterion_df.at[0, 'loss']-1
        
    def correct_tx(self,
                    reassign_threshold_grid: Union[int, float, np.array, list]=np.arange(start=0.1, stop=0.6, step=0.1),
                    remove_threshold_grid: Union[int, float, np.array, list]=np.arange(start=0, stop=0.4, step=0.1),
                    choice_type: str="conservative") -> None:
        """Generate transcript reassignment based on various criteria.

        (Unchanged from original mistic — the ablation is already enforced
        upstream in ``compute_reassign_probs`` and the parameter count in
        ``initialize_parameters``.)
        """
        adata_obs = pl.from_pandas(self.adata.obs["cell_type"], include_index=True)  
        if isinstance(reassign_threshold_grid, int) or isinstance(reassign_threshold_grid, float):
            reassign_threshold_grid = [reassign_threshold_grid]
        if isinstance(remove_threshold_grid, int) or isinstance(remove_threshold_grid, float):
            remove_threshold_grid = [remove_threshold_grid]
        if (len(reassign_threshold_grid) > 1) or (len(remove_threshold_grid) > 1):
            self._find_criteria(adata_obs=adata_obs,
                                reassign_threshold_grid=reassign_threshold_grid,
                                remove_threshold_grid=remove_threshold_grid)
            if choice_type == "best":
                ind = 0
            elif choice_type == "aggressive":
                ind = self.criterion_df.loc[self.criterion_df['loss_increase'] < 0.01, 'n_removed'].idxmax(axis=0)
            elif choice_type == "conservative":
                ind = self.criterion_df.loc[self.criterion_df['n_removed']==0, 'loss'].idxmin(axis=0)
            else: 
                raise ValueError('Invalid choice type')
            reassign_threshold = self.criterion_df.at[ind, "reassign_threshold"]
            remove_threshold = self.criterion_df.at[ind, "remove_threshold_percent"]
        else: 
            reassign_threshold = reassign_threshold_grid[0]
            remove_threshold = remove_threshold_grid[0]
        tx_to_reassign, tx_to_remove = self._correct_tx(adata_obs=adata_obs,
                                                        reassign_threshold=reassign_threshold,
                                                        remove_threshold=remove_threshold)
        trial_layer = self.current_layer+"_corrected"
        self.adata = make_reassignment_adata(adata=self.adata,
                                            layer=self.current_layer,
                                            tx_to_reassign=tx_to_reassign,
                                            tx_to_remove=tx_to_remove,
                                            trial_layer=trial_layer,
                                            dr_method=self.import_data_par['dr_method'])
        tx_to_reassign = tx_to_reassign.rename({"cell_id": "from_cell_id",
                                                "neighbor_cell_id": "to_cell_id"})
        tx_to_remove = tx_to_remove.rename({"cell_id": "from_cell_id"})
        
        self.tx_to_reassign = tx_to_reassign.to_pandas().copy()
        self.tx_to_remove = tx_to_remove.to_pandas().copy()
    
    def recluster(self,
                temperature: float=0.0,
                top_k: Optional[int]=None,
                new_layer: Optional[str]=None,
                overwrite_previous_trials: bool=False,
                update_leiden: bool=False) -> None:
        """Regenerate the clusters. (Unchanged from original mistic.)"""
        if new_layer is None:
            new_layer = self.current_layer+"_corrected"
        self.eval()
        with torch.no_grad():
            cell_by_gene_counts_chunks = even_split(array=self.adata.layers[new_layer],
                                                    chunk_size=np.ceil(self.adata.X.shape[0]/100))
            logits = torch.empty((0, self.adata.uns["n_leiden"]), dtype=torch.float32)
            cell_type_predict = torch.empty((0, 1), dtype=torch.int64)
            for cell_by_gene_counts_chunk in tqdm(cell_by_gene_counts_chunks):
                logits_chunk = self.cell_type_coefficients(torch.tensor(cell_by_gene_counts_chunk,
                                                                dtype=torch.float32, 
                                                                device=self.model_device)).cpu()
                logits = torch.cat((logits, logits_chunk), dim=0)
                if top_k is not None:
                    top_logits, _ = torch.topk(logits_chunk, top_k)
                    min_val = top_logits[:, [-1]]
                    logits_chunk = torch.where(
                        logits_chunk < min_val,
                        torch.tensor(float('-inf')).to(logits.device),
                        logits_chunk
                    )
                if temperature > 0.0:
                    logits_chunk = logits_chunk/temperature
                    probs = torch.softmax(logits_chunk, dim=-1)
                    cell_type_predict_chunk = torch.multinomial(probs, num_samples=1)
                else:
                    cell_type_predict_chunk = torch.argmax(logits_chunk, dim=-1, keepdim=True)
                cell_type_predict = torch.cat((cell_type_predict, cell_type_predict_chunk), dim=0)
        cell_type_predict = cell_type_predict.numpy()
        logits = logits.numpy()
        probs = softmax(logits, axis=1)
        perplexity = np.exp(entropy(probs, axis=1, keepdims=True))
        if overwrite_previous_trials:
            previous_trials = []
            i=0
            while True:
                new_leiden_name = new_layer + "_leiden_" + str(i)
                if new_leiden_name in self.adata.obs.columns:
                    previous_trials.append(new_leiden_name)
                    previous_trials.append(new_layer + "_cell_type_" + str(i))
                    previous_trials.append(new_leiden_name+"_perplexity")
                else:
                    break
                i += 1
            self.adata.obs.drop(columns=previous_trials, inplace=True)
        i=0
        while True:
            new_leiden_name = new_layer + "_leiden_" + str(i)
            new_cell_type_name = new_layer + "_cell_type_" + str(i)
            if new_leiden_name not in self.adata.obs.columns:
                break 
            i += 1
        self.adata.obs[new_leiden_name] = cell_type_predict
        self.adata.obs[new_leiden_name] = self.adata.obs[new_leiden_name].astype(str)
        
        temp_df = self.adata.obs[[new_leiden_name]].merge(self.adata.uns['cell_type_leiden_map'],
                                            how='left', left_on = new_leiden_name,
                                            right_on = "cell_type_index")
        self.adata.obs[new_cell_type_name] = temp_df['cell_type_name'].values.copy()
        
        self.adata.obs[new_leiden_name+"_perplexity"] = perplexity
        if update_leiden:
            self.adata.obs['leiden'] = self.adata.obs[new_leiden_name].copy()
    
    def save_model(self,
                   dir_name: str,
                   model_name: str,
                   save_correction_result: bool=True) -> None:
        """Save the model. (Unchanged from original mistic.)"""
        torch.save({'model_state_dict': self.state_dict()} | \
                {'optimizer_state_dict': self.optimizer.state_dict()}, 
                os.path.join(dir_name, model_name+".pt"))
        
        model_meta = {'import_data_par': self.import_data_par,
                      'calculate_mask_distance_par': self.calculate_mask_distance_par,
                      'generate_feature_par': self.generate_feature_par,
                      'n_genes': self.n_genes,
                      'n_leiden': self.n_leiden,
                      'prior_50_reassign_prob': self.prior_50_reassign_prob,
                      'prior_5_reassign_prob': self.prior_5_reassign_prob,
                      'current_layer': self.current_layer,
                      "save_correction_result": save_correction_result}
        
        with open(os.path.join(dir_name, model_name+"_meta.json"), "w") as f:
            json.dump(model_meta, f, cls=JSONEncoder)
        
        if save_correction_result:
            self.criterion_df.to_csv(os.path.join(dir_name, model_name+"_criteria_df.csv"),
                                     index=False)
            self.tx_to_reassign.to_parquet(os.path.join(dir_name, model_name+"_tx_to_reassign.parquet"))
            self.tx_to_remove.to_parquet(os.path.join(dir_name, model_name+"_tx_to_remove.parquet"))
            
    def load_model(self,
                   dir_name: str,
                   model_name: str) -> None:
        """Load the model. (Unchanged from original mistic.)"""
        model_meta = json.load(open(os.path.join(dir_name, model_name+"_meta.json")))
        self.import_data_par = model_meta['import_data_par']
        self.calculate_mask_distance_par = model_meta['calculate_mask_distance_par']
        self.generate_feature_par = model_meta['generate_feature_par']
        self.n_genes = model_meta['n_genes']
        self.n_leiden = model_meta['n_leiden']
        self.current_layer = model_meta['current_layer']
        self.prior_50_reassign_prob = model_meta['prior_50_reassign_prob']
        self.prior_5_reassign_prob = model_meta['prior_5_reassign_prob']
        save_correction_result = model_meta['save_correction_result']
        
        if save_correction_result:
            self.criterion_df = pd.read_csv(os.path.join(dir_name, model_name+"_criteria_df.csv"))
            self.tx_to_reassign = pd.read_parquet(os.path.join(dir_name, model_name+"_tx_to_reassign.parquet"))
            self.tx_to_remove = pd.read_parquet(os.path.join(dir_name, model_name+"_tx_to_remove.parquet"))
        
        self.initialize_parameters()
        checkpoint = torch.load(os.path.join(dir_name, model_name+".pt")) 
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
