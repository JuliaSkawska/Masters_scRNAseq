import os
import scanpy as sc
import anndata as ad
import logging
import numpy as np
import pandas as pd

sc.settings.set_figure_params(dpi=100, facecolor="white")

"""
The reason that all those steps are done in separate functions is due to the fact that often certian parts of scRNA-seq 
pipelien need to be repeated, this allows the user to only repeat the part that is necessery for what they are trying 
to do

Note to self 1: Need to add the ability for user to choose which parts of the code need to be run via console instead of 
messing with the code 

Note to self 2: Need to fix the image saving options, atm they dont get saved and need to be downloaded manually
"""

def setup_logging(log_file: str):
    """
    Sets up logging configuration.

    log_file: Path to where the log file should be saved.
    """
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def validate_path(path: str) -> bool:
    """
    Checks if a path exists ( and is a directory )
    """
    return os.path.isdir(path)

def log_information(log_text: str):
    """
    Logs the information as well as prints it in console
    """
    logging.info(log_text)
    print(log_text)#due to long computing times user is being visual updated that the program is still running and at which step


def mtx_to_h5ad(input_path: str, output_path: str) -> str:
    """
    Converts a 10x Genomics data file to AnnData format and saves it as h5ad file.
    Annotates mitochondrial genes and performs a quality check

    input_path: Path to the folder containing the 10x Genomics data.
    output_path: Path to the folder where the h5ad file and QC plot will be saved.

    :return: Path to the saved h5ad file.
    """

    if not validate_path(input_path):
        logging.error(f"Input path is not a valid directory: {input_path}")
        raise

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    try:
        file_name = os.path.basename(input_path.rstrip('/'))
        log_information(f"Pre-processing file: {file_name}")

        # Annotate mitochondrial genes
        adata = sc.read_10x_mtx(input_path, var_names='gene_symbols', cache=True)
        adata.var["mt"] = adata.var_names.str.startswith("MT-")

        # Perform Quality Control
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, log1p=True)
        output_path = os.path.join(output_path, f"qc.h5ad")
        sc.write(output_path, adata)

        # Plot QC metrics
        sc.pl.violin(
            adata,
            ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
            jitter=0.4,
            multi_panel=True
        )
        sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt")

        log_information(f"Successfully processed file: {file_name}")

        return output_path

    except Exception as e:
        logging.error(f"An error occurred while pre-processing {input_path}: {e}")


def get_ann(input_path: str) -> ad.AnnData:
    """
    Fetches AnnData from a specified file

    input_path: path to the AnnData file
    """
    try:
        logging.info(f"Retrieving anndata from: {input_path}")
        adata = ad.read_h5ad(input_path)
        log_information(f"Successfully retrieved anndata from: {input_path}")
        return adata
    except OSError as e:
        logging.error(f"An error occurred while retrieving {input_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while retrieving {input_path}: {e}")
        raise


def check_ann(adata: ad.AnnData):
    """
    Gives basic information about AnnData file

    Used for debugging purposes, or to check if the file has the expected data
    """
    try:
        logging.info(f"Retrieving information about .h5ad file")

        n_obs = adata.shape[0]
        n_var = adata.shape[1]

        print("Number of observations (cells):", n_obs)
        print("Number of variables (genes):", n_var)

        if hasattr(adata, 'var') and hasattr(adata.var, 'keys'):
            print("Available variables (annotations):", adata.var.keys())
        else:
            logging.warning("No variable annotations available.")

        if hasattr(adata, 'obs'):
            print("Available observations (annotations):", adata.obs)
        else:
            logging.warning("No observation annotations available.")

    except Exception as e:
        logging.error(f"An error occurred when retrieving information from .h5ad file: {e}")


def filter(adata: ad.AnnData, output_path: str) -> str:
    """
    Filters the AnnData object based on given criteria and saves it post filtering.

    adata: AnnData object.
    output_path: Path to save the filtered AnnData object.

    :return: Path to the saved filtered AnnData object.
    """
    min_cell = int(input("Minimum number of cells in which gene is expressed: "))
    max_cell = int(input("Maximum number of cells in which gene is expressed: "))
    min_genes = int(input("Minimum amount of genes per cell: "))
    max_genes = int(input("Maximum amonut of genes per cell: "))
    mito = int(input("Count mito: "))

    try:
        log_information(f"Filtering AnnData")

        output_path = os.path.join(output_path, f"filtered.h5ad")

        if min_cell > 0:
            sc.pp.filter_genes(adata, min_cells=min_cell)
            logging.info(f"Filtered genes that appear in less then {min_cell} cells")

        if max_cell > 0:
            sc.pp.filter_genes(adata, max_cells=max_cell)
            logging.info(f"Filtered genes that appear in more then {max_cell} cells")

        if min_genes > 0:
            sc.pp.filter_cells(adata, min_genes=min_genes)
            logging.info(f"Filtered cells with less then {min_genes} genes")

        if max_genes > 0:
            sc.pp.filter_cells(adata, max_genes=max_genes)
            logging.info(f"Filtered cells with more then {max_genes} genes")

        if mito > 0:
            if 'pct_counts_mt' not in adata.obs.columns:
                raise ValueError("'pct_counts_mt' column is missing in adata.obs")

            adata = adata[adata.obs['pct_counts_mt'] <= mito].copy()
            logging.info(f"Filtered cells with less then {mito}% mitchondrial genes")

        if os.path.exists(output_path):
            base_name, ext = os.path.splitext(output_path)
            count = 1
            while os.path.exists(f"{base_name}_{count}{ext}"):
                count += 1
            output_path = f"{base_name}_{count}{ext}"
            sc.write(output_path, adata)
        else:
            sc.write(output_path, adata)

        log_information(f"AnnData filtered successfully")

        return output_path

    except Exception as e:
        logging.error(f"An error occurred while filtering AnnData: {e}")
        raise


def normalize(adata: ad.AnnData, output_path: str) -> str:
    """
    Normalizes and logarithmizes, the AnnData object
    Detects doublets and highly variable genes in AnnData object then saves it.

    adata: AnnData object.
    output_path: Path to save the normalized AnnData object.
    :return: Path to the saved normalized AnnData object.
    """
    try:
        log_information(f"Normalizing AnnData")

        # Detecting doublets
        sc.pp.scrublet(adata)
        # Saving count data
        adata.layers["counts"] = adata.X.copy()
        # Normalizing to median total counts
        sc.pp.normalize_total(adata)
        # Logarithmizing the data
        sc.pp.log1p(adata)
        # Marking highly variable genes
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        sc.pl.highly_variable_genes(adata)

        output_path = os.path.join(output_path, f"normalized.h5ad")
        sc.write(output_path, adata)

        log_information(f"AnnData normalized successfully")

        return output_path

    except Exception as e:
        logging.error(f"An error occurred while normalizing AnnData: {e}")
        raise


def reduce_dimensions(adata: ad.AnnData, output_path: str) -> str:
    """
    Reduces dimensions of the AnnData object and saves it.

    adata: AnnData object.
    output_path: Path to save the reduced dimension AnnData object.
    :return: Path to the saved reduced dimension AnnData object.
    """
    try:
        log_information(f"Reducing dimensions of AnnData")

        sc.tl.pca(adata)
        sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.pl.umap(adata, size=2,)

        output_path = os.path.join(output_path, f"pca_umap.h5ad")
        sc.write(output_path, adata)

        log_information(f"Dimensions reduced successfully")

        return output_path

    except Exception as e:
        logging.error(f"Reducing dimensions failed: {e}")
        raise


def clustering(adata: ad.AnnData, output_path: str, n_iterations: int = 2, random_seed: int = 42, resolution: int = 1) -> str:
    """
    Clusters the AnnData object and saves it.

    adata: AnnData object.
    output_path: Path to save the clustered AnnData object.
    n_iterations: Number of iterations for the clustering algorithm.
    random_seed: Random seed for reproducibility.
    resolution: Resolution parameter for the clustering algorithm.
    :return: Path to the saved clustered AnnData object.
    """
    try:
        log_information("Attempting to cluster adata")

        # Setting the random seed for numpy and scanpy
        np.random.seed(random_seed)
        sc.settings.seed = random_seed

        # Performing clustering
        logging.info("Starting Louvain clustering.")

        sc.tl.louvain(adata, flavor="vtraag", random_state=random_seed, resolution=resolution)

        logging.info(f"Louvain clustering completed with resolution {resolution}")

        # Storing the clustering parameters in anndata object
        adata.uns['clustering_parameters'] = {
            'n_iterations': n_iterations,
            'random_seed': random_seed,
            'resolution': resolution
        }

        # Ploting UMAPs
        logging.info("Generating UMAP plots.")
        sc.pl.umap(adata, color=["louvain"], legend_loc="on data")
        sc.pl.umap(adata, color=["louvain", "predicted_doublet", "doublet_score"], wspace=0.5, size=3,)
        sc.pl.umap(adata, color=["louvain", "log1p_total_counts", "pct_counts_mt", "log1p_n_genes_by_counts"],
                   wspace=0.5, ncols=2,)

        output_path = os.path.join(output_path, f"clustered_louvain.h5ad")

        # Avoid overriding the previews clusterings
        if os.path.exists(output_path):
            base_name, ext = os.path.splitext(output_path)
            count = 1
            while os.path.exists(f"{base_name}_{count}{ext}"):
                count += 1
            output_path = f"{base_name}_{count}{ext}"

        sc.write(output_path, adata)
        log_information(f"Clustering succeeded")

        return output_path

    except Exception as e:
        logging.error(f"Clustering failed with error: {e}")
        raise


def mark_genes(adata: ad.AnnData, output_path: str) -> str:
    """
    Marks specific genes in the AnnData object.

    adata: AnnData object.
    output_path: Path to save outputs.
    """
    try:
        log_information("Marking genes in AnnData")

        # Perform ranking of genes based on groups
        sc.tl.rank_genes_groups(adata, groupby="louvain", method="wilcoxon")

        # Display a dot plot of ranked genes
        sc.pl.rank_genes_groups_dotplot(adata, groupby="louvain", standard_scale="var", n_genes=5)

        # Display UMAP plot colored by specific genes and louvain clusters
        sc.pl.umap(
            adata,
            color=["louvain", "SLPI", "ELANE", "COL3A1", "FBN1", "LUM"],
            frameon=False,
            ncols=3,
        )

        # Prompt for group number for further analysis
        group = input("Enter group number for further analysis (type END to end the program): ")

        while group != "END":
            # Get top 5 ranked genes for the specified group
            ranked_genes_df = sc.get.rank_genes_groups_df(adata, group=group)
            dc_cluster_genes = ranked_genes_df.head(5)["names"].tolist()

            # Update UMAP plot with top ranked genes for the specified group
            sc.pl.umap(
                adata,
                color=["louvain"] + dc_cluster_genes,  # Include louvain and top genes for color
                frameon=False,
                ncols=3,
            )

            # Prompt again for group number
            group = input("Enter group number for further analysis (type END to end the program): ")

        log_information("Genes marked successfully")

        try:
            log_information("Saving ranked genes to file")

            result = adata.uns['rank_genes_groups']
            groups = result['names'].dtype.names
            df = pd.DataFrame({group + '_' + key[:1]: result[key][group]
                               for group in groups for key in ['names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj']})

            output_path = os.path.join(output_path, f"ranked_genes.csv")

            if os.path.exists(output_path):
                base_name, ext = os.path.splitext(output_path)
                count = 1
                while os.path.exists(f"{base_name}_{count}{ext}"):
                    count += 1
                output_path = f"{base_name}_{count}{ext}"

            df.to_csv(output_path)

            log_information(f"Ranked genes exported to {output_path}")

            return output_path

        except Exception as e:
            logging.error(f"Saving ranked genes to excel file failed: {e}")

    except Exception as e:
        logging.error(f"Marking failed: {e}")


def compare_top_genes(adata: ad.AnnData, input_path: str, output_path: str):
    """
    Analyzes top expressed genes in a specific cluster and shows their expression in all clusters.

    adata: AnnData object with clustering results
    output_path: Directory to save output plots
    """

    cluster_id = input("Which clusters should be checked for top genes ( seperate numbers using ,): ")
    top_n = 200 # Number of top genes to select
    cluster_id_list = [id.strip() for id in cluster_id.split(',')]

    for elem in cluster_id_list:
        try:
            logging.info(f"Attempting to compare genes in {elem} cluster")

            # Reading the CSV file and filtering for top genes
            ranked_genes = pd.read_csv(input_path)
            gene_col = f"{elem}_n"  # Adjust according to how the column is named

            if gene_col in ranked_genes.columns:
                top_genes = ranked_genes[gene_col].head(top_n).tolist()
            else:
                print(f"Column '{gene_col}' not found in the CSV file.")
                return

            # Creating a heatmap
            try:
                logging.info("Creating a heatmap")

                sc.pl.heatmap(adata, var_names=top_genes, groupby="louvain", cmap="viridis",
                            figsize=(36, 24),
                            show_gene_labels=True)

                logging.info("Heatmap created successfully")

            except Exception as e:
                logging.error(f'Heatmap creation failed: {e}')

            # Filter the AnnData for top genes
            adata_filtered = adata[:, adata.var_names.isin(top_genes)]

            # Calculate the mean expression of top genes across clusters
            mean_expression = adata_filtered.to_df().groupby(adata.obs['louvain']).mean()
            mean_expression.reset_index(inplace=True)
            mean_expression_transposed = mean_expression.T

        except Exception as e:
            logging.error(f"Comparing top genes in cluster {elem} failed: {e}")

        try:
            mean_expression_transposed.to_csv(os.path.join(output_path, f"mean_expression_{elem}.csv"), header=False)

        except Exception as e:
            logging.error(f"Error saving file: {e}")


def run_analysis(input_path, output_path):
    current_file = mtx_to_h5ad(input_path, output_path)
    adata = get_ann(current_file)
    current_file = filter(adata, output_path)
    adata = get_ann(current_file)
    current_file = normalize(adata, output_path)
    adata = get_ann(current_file)
    current_file = reduce_dimensions(adata, output_path)
    adata = get_ann(current_file)
    current_file = clustering(adata, output_path)
    adata = get_ann(current_file)
    cvs_file_path = mark_genes(adata, output_path)
    compare_top_genes(adata, cvs_file_path, output_path)

if __name__ == "__main__":
    setup_logging("your_path")
    run_analysis("your_path_input_data", "your_path_results")
