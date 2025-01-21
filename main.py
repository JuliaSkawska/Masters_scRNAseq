import os
import scanpy as sc
from scipy.stats import median_abs_deviation
import anndata as ad
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sc.settings.set_figure_params(dpi=100, facecolor="white")

def setup_logging(log_file: str):
    """
    Sets up logging configuration, as well as a folder to save images in

    log_file: Path to where the log file should be saved.
    """
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def log_information(log_text: str):
    """
    Logs the information as well as prints it in console
    """
    logging.info(log_text)
    print(log_text)


def is_outlier(adata, metric, nmads):
    met = adata.obs[metric]
    outlier = (met < np.median(met) - nmads * median_abs_deviation(met)) | (
        np.median(met) + nmads * median_abs_deviation(met) < met
    )
    return outlier


def merge_sets_1(input_path: str, output_path: str) -> str:
    try:
        logging.info(f"Attempting to merge together files in {input_path} using merge_sets_1")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"The directory '{input_path}' does not exist.")
        if not os.path.isdir(input_path):
            raise NotADirectoryError(f"The path '{input_path}' is not a directory.")

        files = []

        for file in os.listdir(input_path):
            file_path = os.path.join(input_path, file)
            print(file_path)

            if file.endswith(".csv.gz"):
                df = pd.read_csv(file_path, compression='gzip', index_col=0)  
                X = df.iloc[:, :-1].values
                gene_names = df['gene_symbol'].values
                gene_ids = df.index.values
                adata = sc.AnnData(X)
                if adata.var.shape[0] != len(gene_ids):
                    adata.var = pd.DataFrame(index=gene_ids)#moim zdaniem po paru próbach tutaj coś wysiada ale nie wiem dla czego 
                adata.var['gene_ids'] = gene_ids
                adata.var['gene_symbol'] = gene_names
                adata.obs['cell_ids'] = df.columns[:-1].values
                files.append(adata)

            else:
                logging.warning(f"Skipping non-CSV file: {file_path}")

        merged_adata = files[0].concatenate(*files[1:], batch_key="batch")
        output_path = os.path.join(output_path, "merged_dataset_test_1.h5ad")
        sc.write(output_path, merged_adata)

        logging.info(f"Merged dataset saved at: {output_path}")

        return output_path

    except Exception as e:
        logging.error(f"An error while trying to merge files in {input_path}: {e}")


def merge_sets_2(input_path: str, output_path: str) -> str:
    try:
        logging.info(f"Attempting to merge together files in {input_path} using merge_sets_2")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"The directory '{input_path}' does not exist.")
        if not os.path.isdir(input_path):
            raise NotADirectoryError(f"The path '{input_path}' is not a directory.")

        files = []

        for file in os.listdir(input_path):
            file_path = os.path.join(input_path, file)
            print(file_path)

            if file.endswith(".csv.gz"):
                adata_df = pd.read_csv(file_path, compression='gzip', index_col=0)  
                adata_df = adata_df.apply(pd.to_numeric, errors='coerce') 
                gene_symbols = adata_df.pop('gene_symbol')  
                adata = sc.AnnData(adata_df) 
                adata.var['gene_symbol'] = gene_symbols  

                files.append(adata)
            else:
                logging.warning(f"Skipping non-CSV file: {file_path}")

        merged_adata = files[0].concatenate(*files[1:], batch_key="batch")
        output_path = os.path.join(output_path, "merged_dataset_test_2.h5ad")
        sc.write(output_path, merged_adata)

        logging.info(f"Merged dataset saved at: {output_path}")

        return output_path

    except Exception as e:
        logging.error(f"An error while trying to merge files in {input_path}: {e}")



def integrate_sets(adata: ad.AnnData, output_path: str) -> str:
    try:
        log_information(f"Integrating AnnData")
        adata_integrated = sc.tl.mnn_correct(adata, batch_key='batch') 
        output_path = os.path.join(output_path, f"integrated_dataset.h5ad")
        sc.write(output_path, adata_integrated)

        return output_path

    except Exception as e:
        logging.error(f"An error occured while trying to integrate AnnData: {e}")


def quality_check(input_path: str, output_path: str) -> str:
    """
    Annotates mitochondrial genes and performs a quality check

    input_path: Path to the folder containing the 10x Genomics data.
    output_path: Path to the folder where the h5ad file and QC plot will be saved.

    :return: Path to the saved h5ad file.
    """

    if not os.path.exists(input_path):
        logging.error(f"Input path does not exist: {input_path}")
        raise

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    try:
        file_name = os.path.basename(input_path.rstrip('/'))
        log_information(f"Pre-processing file: {file_name}")

        # Annotate mitochondrial genes
        adata = sc.read_h5ad(input_path)
        mito_genes = [gene for gene in adata.var_names if gene.lower().startswith("mt-")]
        adata.var["mt"] = adata.var_names.isin(mito_genes)

        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, log1p=True)

        adata.obs["outlier"] = (
                is_outlier(adata, "log1p_total_counts", 5)
                | is_outlier(adata, "log1p_n_genes_by_counts", 5)
        )
        adata.obs.outlier.value_counts()

        log_information(f"Outliers found: {adata.obs['outlier'].sum()} cells")

        # Setting a path for images to be saved in pytanie - nie wiem w sumie czy nie lepiej by to ustawiać poza funkcją
        image_path = os.path.join(output_path, "Images")
        os.makedirs(image_path, exist_ok=True)
        sc.settings.figdir = image_path

        output_path = os.path.join(output_path, f"qc.h5ad")
        sc.write(output_path, adata)

        # Plot QC metrics - pytanie czy ten histogram jest potrzebny
        plt.figure(figsize=(8, 6))
        sns.histplot(adata.obs["total_counts"], bins=100, kde=False)
        plt.title("Histogram of Total Counts")
        plt.xlabel("Total Counts")
        plt.ylabel("Frequency")
        hist_path = os.path.join(image_path, "total_counts_histogram.png")
        plt.savefig(hist_path)
        plt.show()
        plt.close()

        sc.pl.violin(adata, "pct_counts_mt", save="_plot.png")
        sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts", color="pct_counts_mt", save="_plot.png")

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
    """
    min_cell = int(input("Minimum number of cells in which gene is expressed: "))
    max_cell = int(input("Maximum number of cells in which gene is expressed: "))
    min_genes = int(input("Minimum amount of genes per cell: "))
    max_genes = int(input("Maximum amount of genes per cell: "))
    mito = int(input("Count mito: "))
    """
    #pytanie - czy każdy batch powinnien być filtrowany osobno czy można to zrobić globalnie ( tak jak jest teraz )
    min_cell = 3
    max_cell = 0
    min_genes = 200
    max_genes = 6000
    mito = 15

    logging.info(f"Filtering cells with parameters: \n min_cell = {min_cell} \n max_cell = {max_cell} \n min_genes = {min_genes} \n max_genes = {max_genes}")

    try:
        log_information(f"Filtering AnnData")
        log_information(f"Number of cells: {adata.n_obs} before filtering")
        log_information(f"Number of genes: {adata.n_vars} before filtering")

        output_path = os.path.join(output_path, f"filtered.h5ad")

        if min_cell > 0:
            sc.pp.filter_genes(adata, min_cells=min_cell)
            print(f"Number of genes after filtering out ones that appear in less then {min_cell} cells: {adata.n_vars}")

        if max_cell > 0:
            sc.pp.filter_genes(adata, max_cells=max_cell)
            print(f"Number of genes after filtering out ones that appear in more then {max_cell} cells: {adata.n_vars}")

        if min_genes > 0:
            sc.pp.filter_cells(adata, min_genes=min_genes)
            print(f"Number of cells after filtering out ones with less then {min_genes} genes: {adata.n_obs}")

        if max_genes > 0:
            sc.pp.filter_cells(adata, max_genes=max_genes)
            print(f"Number of cells after filtering out ones with more then {max_genes} genes: {adata.n_obs}")

        if mito > 0:
            if 'pct_counts_mt' not in adata.obs.columns:
                raise ValueError("'pct_counts_mt' column is missing in adata.obs")

            adata.obs["mt_outlier"] = (
                    is_outlier(adata, "pct_counts_mt", 3)
                    | (adata.obs["pct_counts_mt"] > mito)
            )

            adata = adata[(~adata.obs.get("outlier", False)) & (~adata.obs["mt_outlier"])].copy()
            print(f"Number of cells after filtering of low quality cells ( including outliers ): {adata.n_obs}")

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

        sc.pp.scrublet(adata)
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        #sc.pp.highly_variable_genes(adata)
        sc.pp.highly_variable_genes(adata, batch_key="batch")

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
        sc.pl.umap(adata, size=2, save="_dimension_reduction.png")

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
        if "batch" in adata.obs:
            color = ["louvain","batch"]
            color_dub = ["louvain", "batch", "predicted_doublet", "doublet_score"]
            color_pct = ["louvain", "batch", "pct_counts_mt", "log1p_n_genes_by_counts"]
        else:
            color = ["louvain"]
            color_dub = ["louvain", "predicted_doublet", "doublet_score"]
            color_pct = ["louvain", "pct_counts_mt", "log1p_n_genes_by_counts"]

        log_information("Generating UMAP plots")
        sc.pl.umap(adata, color=color, legend_loc="on data", save="_on_data.png")
        sc.pl.umap(adata, color=color_dub, wspace=0.5, size=3, save="_doublets.png")
        sc.pl.umap(adata, color=color_pct, wspace=0.5, ncols=2, save="_quality.png")

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

        # Display UMAP plot colored by specific genes and louvain clusters
        sc.pl.umap(
            adata,
            color=["louvain", "SLPI", "ELANE", "COL3A1", "FBN1", "LUM"],
            frameon=False,
            ncols=3,
            save="_fibroblast_marker.png",
        )

        # Prompt for group number for further analysis
        group_id = input("Which groups should be checked for ranked genes ( separate numbers using ,): ")
        group_list = [id.strip() for id in group_id.split(',')]

        for elem in group_list:
            # Get top 5 ranked genes for the specified group
            ranked_genes_df = sc.get.rank_genes_groups_df(adata, group=elem)
            dc_cluster_genes = ranked_genes_df.head(5)["names"].tolist()

            # Update UMAP plot with top ranked genes for the specified group
            sc.pl.umap(
                adata,
                color=["louvain"] + dc_cluster_genes,  # Include louvain and top genes for color
                frameon=False,
                ncols=3,
                save=f"_top_genes_{elem}.png",
            )

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
                            show_gene_labels=True,
                            save=f"_{elem}.png")

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
    save_directory = os.path.join(output_path, "Images")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print(f"Folder 'Images' created at {save_directory}")
    else:
        print(f"Folder 'Images' already exists at {save_directory}")

    current_file = quality_check(input_path, output_path)
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
    setup_logging("C:\\Users\\Julia\\Desktop\\scrna\\dane\\mainlog.log")
    min_test = ["C:\\Users\\Julia\\Desktop\\scrna\\dane\\testres\\merged_dataset_test_unspecified.h5ad"]
    list_test = ["C:\\Users\\Julia\\Desktop\\scrna\\dane\\testres\\merged_dataset_test_outer.h5ad","C:\\Users\\Julia\\Desktop\\scrna\\dane\\testres\\merged_dataset_test_inner.h5ad", "C:\\Users\\Julia\\Desktop\\scrna\\dane\\testres\\merged_dataset_test_unspecified.h5ad"]
    a = 0
    if a == 1:
        for elem in list_test:
            print(elem)
            adata=get_ann(elem)
            check_ann(adata)
            print(adata.obs[['batch']].head())
            print(adata.var[['gene_symbol']].head())

    else:
        merge_sets("C:\\Users\\Julia\\Desktop\\scrna\\dane\\test","C:\\Users\\Julia\\Desktop\\scrna\\dane\\testres")

    #print(adata.var.index[:10])
    
    #quality_check("C:\\Users\\Julia\\Desktop\\scrna\\dane\\testres\\merged_dataset.h5ad","C:\\Users\\Julia\\Desktop\\scrna\\dane\\testres")

    #run_analysis("C:\\Users\\Julia\\Desktop\\scrna\\dane\\testres\\merged_dataset.h5ad","C:\\Users\\Julia\\Desktop\\scrna\\dane\\testres")

    """
    task = input("Which task should be run: ( chose number ) \n 1. Merge sets \n 2. Run analysis \n 3. END \n")
    input_path = input()
    output_path = input()
    task = task.strip()

    if task == "1":
        merge_sets(input_path, output_path)
    elif task == "2":
        #run_analysis(input_path, output_path)
        run_analysis("C:\\Users\\User\\Desktop\\pythonProject1\\testcase\\test2","C:\\Users\\User\\Desktop\\pythonProject1\\rescase\\test2")
    """

