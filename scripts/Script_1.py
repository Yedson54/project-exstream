#project-exstream
# -*- coding: utf-8 -*-
# YANN CHOHO & AGNIMO YEDIDIA

# Import librairies.
import glob
import warnings
from itertools import groupby
from typing import List, Dict, Tuple, Optional, Literal, Union, Iterable
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings(action="ignore")


# Get files
DATAPATH =  "../data/custom_no_streaming_8/folder_1"
files = glob.glob(rf"{DATAPATH}/*")


# Read anomaly files and concatenate dataframes
dfs = []
for file in files[:-1]:
    df = pd.read_csv(file)
    # Extract the filename without the parent folder path and extension
    filename = file.split('\\')[-1].split('.')[0]
    # Add a new column "anomaly_type" with the extracted filename
    df.insert(1, column="trace_id", value=filename)
    df.rename({"Unnamed: 0": "time"}, axis=1, inplace=True)
    dfs.append(df)
# Concatenate all dataframes except labels.csv
anomaly_df = pd.concat(dfs)

# Create a separate dataframe for labels.csv
labels_df = pd.read_csv(files[-1], index_col=0)


def _filter_ano_time(x: pd.DataFrame):
    """
    Filter the DataFrame based on time intervals.

    Parameters:
    x (pd.DataFrame): The input DataFrame.

    Returns:
    np.ndarray: A boolean array indicating whether each row satisfies the time intervals.
    """
    return np.logical_or(x['time'].between(x['ref_start'], x['ref_end']),
                         x['time'].between(x['ano_start'], x['ano_end']))

# Merge anomaly_df and labels and filter based on time.
# Keep only point observed on annotated period. (#TODO: REMIND THIS FOR LATER)
anomaly_df = anomaly_df.merge(
    labels_df,
    how='inner',
    on='trace_id',
    suffixes=('_anomaly', '_labels'),
    validate="m:m" # remind what was expected (yes, really m:m)
).loc[lambda x: _filter_ano_time(x)]

# Add "period_type" column based on conditions
anomaly_df['period_type'] = np.where(
    anomaly_df['time'].between(anomaly_df['ref_start'], anomaly_df['ref_end']), 'I_R',
    np.where(anomaly_df['time'].between(anomaly_df['ano_start'], anomaly_df['ano_end']), 'I_A', "Not In period")
)



# CHANGE THE NAME OF THE COLUMNS for shorter names
continuous_features = [
        'driver_BlockManager_memory_memUsed_MB_value',
        'driver_jvm_heap_used_value',
        'avg_jvm_heap_used_value',
        'avg_executor_filesystem_hdfs_write_ops_value_1_diff',
        'avg_executor_cpuTime_count_1_diff',
        'avg_executor_runTime_count_1_diff',
        'avg_executor_shuffleRecordsRead_count_1_diff',
        'avg_executor_shuffleRecordsWritten_count_1_diff'
    ]
features_code_to_labels = {
    f"feature_{code}": feature_name 
    for code, feature_name in enumerate(continuous_features)
}
features_label_to_code = {
    value: key for key, value in features_code_to_labels.items()
}
features_code = features_code_to_labels.keys()
anomaly_df.rename(columns=features_label_to_code, inplace=True)

## Sufficient features space

def sufficient_features_space(data: pd.DataFrame, features: List[str]) -> List[str]:
    """
    Filter the relevant features based on the given data and return the list of features with sufficient variance.

    Args:
        data (pd.DataFrame): The input dataframe containing the data.
        features (List[str]): The list of features to consider.

    Returns:
        List[str]: The list of features with sufficient variance.

    Raises:
        AttributeError: If `data` is not a dataframe.
    """
    df = data.copy()
    if isinstance(data, pd.DataFrame):
        relevant_data = df[(df["ref_start"] <= df["time"]) & (df["time"] <= df["ref_end"]) &
                             (df["ano_start"] <= df["time"]) & (df["time"] <= df["ano_end"])]
        relevant_data = relevant_data[features]
        return [col for col in relevant_data.columns if relevant_data[col].var() > 1e-16]  # Adjust var threshold
    else:
        raise AttributeError("`data` is not a dataframe")


def tsa_tsr_series_split(
        df: pd.DataFrame,
        trace_id: str,
        ano_id: int
        ) -> Tuple[pd.Series, pd.Series]:
    """Get annotated time series: TSA (Time Series Anomaly) and 
    TSR (Time Series Reference) of a given event, characterized by its 
    `trace_id` and its `ano_id`. 

    Parameters:
    - df (pd.DataFrame): The DataFrame containing time series data.
    - trace_id (str): The identifier for the time series.
    - ano_id (int): The identifier for the anomaly.

    Returns:
    tuple: A tuple containing two DataFrames - TSA and TSR.
    """
    tsa = df[(df['trace_id'] == trace_id)
             & (df['ano_id'] == ano_id)
             & (df['period_type'] == 'I_A')]
    tsr = df[(df['trace_id'] == trace_id)
             & (df['ano_id'] == ano_id)
             & (df['period_type'] == 'I_R')]
    
    return tsa, tsr

## Single feature reward (section 4)
def class_entropy(ts_anomaly: Iterable[float], ts_reference: Iterable[float]) -> float:
    """
    Calculate the class entropy based on the number of instances in
    TSA (True Samples for Anomaly) and TSR (True Samples for Regular).

    Parameters:
    - TSA (Iterable[float]): List or array containing instances
        classified as True Anomalies.
    - TSR (Iterable[float]): List or array containing instances
        classified as True Regular.

    Returns:
    float: The calculated class entropy.
    """
    p_A = len(ts_anomaly) / (len(ts_anomaly) + len(ts_reference))
    p_R = len(ts_reference) / (len(ts_anomaly) + len(ts_reference))

    H_Class_f = -(p_A * np.log2(p_A) + p_R * np.log2(p_R))
    return H_Class_f


### Segmentation entropy 

def generate_TS(df: pd.DataFrame, variable: str) -> pd.Series:
    """
    Generate a time series (TS) for a given variable based on unique combinations
    of 'variable' and 'period_type' in the input DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the data.
    - variable (str): The variable for which the time series is generated.

    Returns:
    pd.Series: The generated time series.
    """
    # Use set to get unique combinations of 'variable' and 'period_type'
    unique_values = df[[variable, 'period_type']].drop_duplicates()
    
    # Determine the period type for each unique value
    TS = unique_values.groupby(variable)['period_type'].apply(
        lambda x: 'I_M' if set(x) == {'I_R', 'I_A'} else x.iloc[0]
    )
    return TS

def penalized_segmentation_entropy(TS: pd.Series) -> float:
    """
    Calculate the segmentation entropy with a penalty for mixed segments.

    Parameters:
    - TS (pd.Series): Time series data.

    Returns:
    float: The calculated segmentation entropy with mixed segment penalty.
    """
    # Group the series by consecutive identical values and get segment lengths
    segments = [(k, sum(1 for _ in g)) for k, g in groupby(TS)]
    
    # Filter segments for 'I_R' and 'I_A' only
    relevant_segments = [(label, length) for label, length in segments
                         if label in ['I_R', 'I_A']]
    
    # Calculate segment probabilities
    total_length = sum(length for _, length in relevant_segments)
    probabilities = [length / total_length for _, length in relevant_segments
                     if total_length > 0]
    
    # Calculate segmentation entropy
    H_Segmentation_f = -np.sum(np.fromiter(
        (p * np.log2(p) for p in probabilities if p > 0), dtype=float)
    )  # avoid log(0)

    # Penalty for mixed segments
    penalty = np.sum(
        np.fromiter(
            (
                1/length * np.log2(1/length) for label, length in segments 
                if label == 'I_M' for _ in range(length)
            ),
            dtype=float)
    )

    return H_Segmentation_f - penalty  # Subtract the penalty

def entropy_based_reward(
        class_entropy_value: float, 
        penalized_segmentation_entropy_value: float) -> float:
    """
    Calculate the entropy-based feature distance.

    Args:
    - class_entropy_value (float): The class entropy of the feature.
    - penalized_segmentation_entropy_value (float): The regularized segmentation 
      entropy of the feature.

    Returns:
    float: The normalized entropy-based feature distance.
    """
    return (class_entropy_value / penalized_segmentation_entropy_value
            if penalized_segmentation_entropy_value != 0 else 0)



def single_feature_reward(df, feature, trace_id, ano_id):
    """
    Calculate the reward for a single feature based on entropy.

    Parameters:
    df (DataFrame): The input dataframe.
    feature (str): The name of the feature.
    trace_id (str): The trace ID column name.
    ano_id (str): The anomaly ID column name.

    Returns:
    float: The entropy-based reward for the feature.
    """
    
    ts_ano, ts_ref = tsa_tsr_series_split(df, trace_id=trace_id, ano_id=ano_id)
    ts = generate_TS(pd.concat([ts_ano, ts_ref]), feature)
    ts_class_entropy = class_entropy(ts_ano, ts_ref)
    ts_segment_entropy = penalized_segmentation_entropy(ts)
    entropy_based_reward(ts_class_entropy, ts_segment_entropy)

    return entropy_based_reward(ts_class_entropy, ts_segment_entropy)

    
## Constructing explanations (section 5)

# Step 1

def compute_feature_rewards(df: pd.DataFrame, trace_id: str, ano_id: int, features_code: List[str]) -> Dict[str, float]:
    """
    Computes feature rewards for continuous features.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the features.
    - trace_id (str): The trace ID.
    - ano_id (int): The anomaly ID.
    - features_code (List[str]): List of feature names.

    Returns:
    - dict: Dictionary containing feature names and their corresponding rewards.
    """
    feature_rewards = {
        feature: single_feature_reward(
            df, feature=feature, trace_id=trace_id, ano_id=ano_id
        )
        for feature in features_code
    }
    return feature_rewards

import numpy as np
import matplotlib.pyplot as plt

def select_feature_on_rewards(df, trace_id, ano_id, features_code, ignore_first=False, plot=True, verbose=True):
    """
    Plots the feature rewards for continuous features and identifies the elbow point.
    Optionally ignores the first sharpest drop if it is the first feature.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the features.
    - trace_id (str): The trace ID.
    - ano_id (int): The anomaly ID.
    - features_code (List[str]): List of feature names.
    - ignore_first (bool, optional): Whether to ignore the first sharpest drop. Defaults to False.
    - plot (bool, optional): Whether to plot the feature rewards. Defaults to True.
    - verbose (bool, optional): Whether to print detailed information. Defaults to True.

    Returns:
    - List[str]: The significant features identified up to the elbow point.
    """
    feature_rewards = compute_feature_rewards(
        df, trace_id=trace_id, ano_id=ano_id, features_code=features_code
        )
    sorted_features = sorted(feature_rewards.items(), key=lambda x: x[1], reverse=True)
    variables, rewards = zip(*sorted_features)

    # Calculate differences between successive feature rewards
    differences = np.abs(np.diff(rewards))
    elbow_point = np.argmax(differences)

    # Check if the elbow point is the first feature and whether to ignore it
    if ignore_first and elbow_point == 0:
        second_largest_drop = np.argmax(differences[1:]) + 1  # Offset by 1 due to slicing
        elbow_point = second_largest_drop

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(variables, rewards, marker='o', linestyle='--', color='b')
        plt.xlabel('Feature')
        plt.ylabel('Feature Reward')
        plt.title('Feature Rewards for Continuous Features')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.axvline(x=variables[elbow_point], color='r', linestyle='--')
        plt.show()

    significant_features = list(variables[:elbow_point + 1])  # Include elbow_point

    if verbose:
        print("Significant features:", significant_features)

    return significant_features

# Step 2: not achieved yet

# Step 3: filtering by correlation clustering
def process_trace_anomaly(
    df: pd.DataFrame,
    trace_id: str,
    ano_id: int,
    selected_features: List[str],
    plot_correlation: bool = False,
    threshold: float = 0.8,
    linkage: str = 'complete',
    verbose: bool = True,
) -> List[str]:
    """
    Process the trace anomaly by performing the following steps:
    1. Retrieve the TSA (Trace Start Anomaly) and TSR (Trace Stop Anomaly) data
    for the given trace_id and ano_id.
    2. Concatenate the TSA and TSR data and select the specified features.
    3. Calculate the correlation matrix of the selected features.
    4. Cluster the strongly correlated features using agglomerative clustering.
    5. Identify the representative features for each cluster.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the features.
    - trace_id (str): The ID of the trace.
    - ano_id (int): The ID of the anomaly.
    - selected_features (List[str]): The list of selected features.
    - threshold (float, optional): The threshold for identifying strong
      correlations. Defaults to 0.8.
    - linkage (str, optional): The linkage method for agglomerative clustering.
      Defaults to 'complete'.

    Returns:
    - representative_features (List[str]): The list of representative features
      for each cluster.

    #TODO: Idea for False Positive Filtering (why don't we simply add time in 
    #   agglomerative clustering)  Then, we remove all features in the same 
    #   cluster than him. 
    # Pay attention to exclude time when looking for best "single_feature_reward"
    #   in its cluster.
    # ==> This is equivalent to simply drop the cluster containing afterward.
    """
    # Check that only one feature is selected
    if len(selected_features) == 1:
        if verbose:
            print(f"Only one feature selected: {selected_features[0]}")
        return selected_features
    # Retrieve the TSA (Trace Start Anomaly) and TSR (Trace Stop Anomaly) data
    # for the given trace_id and ano_id
    tsa, tsr = tsa_tsr_series_split(df, trace_id=trace_id, ano_id=ano_id)

    # Concatenate the TSA and TSR data and select the specified features
    data: pd.DataFrame = pd.concat([tsa, tsr])[selected_features]

    # Calculate the correlation matrix of the selected features
    corr_matrix = data.corr().abs()
    # Create a mask to hide the upper triangle of the correlation matrix for aesthetics.
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    if plot_correlation:
        # Create a heatmap to visualize the correlation matrix without redundancy
        plt.figure(figsize=(6, 6))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                    cmap='coolwarm', square=True)
        plt.title('Correlation Matrix without Redundancy Heatmap')
        plt.show()

    # Cluster the strongly correlated features using agglomerative clustering.
    clustering = AgglomerativeClustering(
        n_clusters=None, affinity='precomputed', linkage=linkage,
        distance_threshold=threshold)
    clustering.fit(1 - corr_matrix)  # 1 - corr_matrix b/c agglomerative clustering uses distances

    # Identify the representative features for each cluster.
    cluster_labels = clustering.labels_
    unique_clusters = np.unique(cluster_labels)
    representative_features = []

    for cluster in unique_clusters:
        # Select the features in the cluster.
        indices = np.where(cluster_labels == cluster)[0] # feature indices
        features_in_cluster = corr_matrix.columns[indices] # feature names
        if "time" in features_in_cluster: # just in case someone include "time" by mistake.
            features_in_cluster.drop("time")

        # Calculate single feature reward for each feature in the cluster.
        feature_rewards = {
            feature: single_feature_reward(df, feature, trace_id, ano_id)
            for feature in features_in_cluster
        }
        # Select the feature with the highest reward.
        selected_feature = max(
            feature_rewards, key=feature_rewards.get
        )
        representative_features.append(selected_feature)

    # print the representative features and associated features for each cluster
    for cluster in unique_clusters:
        indices = np.where(cluster_labels == cluster)[0]
        if verbose:
            print(f"Cluster {cluster}:", ", ".join(data.columns[indices]))
            print(f"Representative features: {representative_features[cluster]}\n")

    if verbose:
        print("Selected representative characteristics: ",
              representative_features)

    return representative_features


def build_explanation(
        df,
        trace_id,
        ano_id,
        features,
        plot_rewards=False,
        plot_correlation=False,
        threshold=0.8,
        linkage='complete', 
        ignore_first : bool = False,
        verbose: bool = False,
        cluster: bool = True):
    # Step 1: Reward leap filtering (select features based on reward)
    selected_features = select_feature_on_rewards(
        df, 
        trace_id=trace_id,
        ano_id=ano_id,
        features_code=features,
        plot=plot_rewards,
        verbose=verbose,
        ignore_first=ignore_first,

    )

    # Step 2: False Positive Filtering (#TODO: Coming soon...)

    selected_features_without_fp = selected_features.copy()
    if cluster == False:
        return selected_features_without_fp
    # Step 3: Filtering by correlation clustering
    representative_features = process_trace_anomaly(df, 
                                                    trace_id=trace_id, 
                                                    ano_id=ano_id, 
                                                    selected_features=selected_features, 
                                                    plot_correlation=plot_correlation, 
                                                    threshold=threshold, 
                                                    linkage=linkage, 
                                                    verbose=verbose)
    
    return representative_features


## Evaluation

## Conciseness



def select_features_using_decision_tree(df, features_code, plot=True, verbose=True):
    """
    Selects significant features using a decision tree and identifies the elbow point
    for feature importances.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the features and target variable.
    - features_code (List[str]): List of feature names.
    - plot (bool, optional): Whether to plot the feature importances. Defaults to True.
    - verbose (bool, optional): Whether to print detailed information. Defaults to True.

    Returns:
    - List[str]: The significant features identified up to the elbow point.
    """
    # Normalisation des données avec StandardScaler
    scaler = StandardScaler()
    X = df[features_code]
    y = df['period_type']  # Assuming 'period_type' is the target variable
    X_scaled = scaler.fit_transform(X)

    # Créer et entraîner l'arbre de décision
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_scaled, y)

    # Obtenir l'importance des caractéristiques
    feature_importances = pd.Series(clf.feature_importances_, index=X.columns)

    # Calculer les différences entre les importances des caractéristiques successives
    feature_importances_sorted = feature_importances.sort_values(ascending=False)
    differences = np.abs(np.diff(feature_importances_sorted))
    elbow_point = np.argmax(differences)

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(feature_importances_sorted.index, feature_importances_sorted, marker='o', linestyle='--', color='b')
        plt.xlabel('Feature')
        plt.ylabel('Feature Importance')
        plt.title('Feature Importances for Decision Tree')
        plt.xticks(rotation=90)
        plt.axvline(x=feature_importances_sorted.index[elbow_point], color='r', linestyle='--')
        plt.tight_layout()
        plt.show()

    significant_features = list(feature_importances_sorted.index[:elbow_point + 1])

    if verbose:
        print("Significant features:", significant_features)

    return significant_features


def instability(
    df: pd.DataFrame,
    trace_id: str,
    ano_id: int,
    features: List[str],
    plot_rewards: bool = False,
    plot_correlation: bool = False,
    verbose: bool = False,
    threshold: float = 0.8,
    linkage: str = 'complete',
    repeat: int = 5,
    sample_frac: int = 0.8,
    replace: bool = False,
    random_state: int = 42,
    cluster: bool = True
):
    """
    Calculate the instability (consistency as entropy) of feature explanations for a specific anomaly in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        trace_id (str): The trace ID for the anomaly.
        ano_id (int): The anomaly ID.
        features (List[str]): The list of features to analyze.
        repeat (int): Number of times to repeat the process for stability estimation.
        sample_frac (float): Fraction of the DataFrame to use for each sample during instability calculation.
        replace (bool): Whether to sample with replacement.
        random_state (int): Random seed for reproducibility.

    Returns:
        float: Instability (consistency as entropy) of feature explanations.
    """
    np.random.seed(random_state) # for reproducibility

    # 1. Filter the DataFrame to the specified anomaly
    single_ano_df = df[(df["trace_id"] == trace_id) & (df["ano_id"] == ano_id)]
    
    # 2. Get all the set of explanation features.
    all_explanatory_features = [
        build_explanation(
            single_ano_df.sample(frac=sample_frac, replace=replace),
            trace_id=trace_id,
            ano_id=ano_id,
            features=features,
            plot_rewards=plot_rewards,
            plot_correlation=plot_correlation,
            threshold=threshold,
            linkage=linkage,
            verbose=verbose,
            cluster=cluster
        ) for _ in range(repeat)
    ]

    # 3. Compute the frequency for each feature in the list.
    feature_counts = pd.Series(
        np.concatenate(all_explanatory_features)).value_counts(normalize=True)

    # 4. Compute consistency as entropy using the previous frequency.
    consistency = -np.sum(feature_counts * np.log2(feature_counts))

    return consistency

### output folder_1_results.csv


# Get the list of anomalies
anomalies = labels_df[['trace_id', 'ano_id']].values.tolist()

# Liste pour stocker les dictionnaires de résultats
results_list = []

for trace_id, ano_id in anomalies:
    # Générer l'explication
    explanation_features = build_explanation(
        anomaly_df[anomaly_df["trace_id"] == trace_id], 
        trace_id=trace_id, 
        ano_id=ano_id, 
        features=features_code,
        plot_rewards=False, 
        plot_correlation=False,
        verbose=False,
        ignore_first=True,
        cluster=True
    )
    
    # Calculer l'instabilité
    exp_size = len(explanation_features)
    exp_instability = instability(
        anomaly_df[anomaly_df["trace_id"] == trace_id], 
        trace_id=trace_id, 
        ano_id=ano_id, 
        features=features_code
    )
    
    # Ajouter les résultats à la liste
    results_list.append({
        'trace_id': trace_id,
        'ano_id': ano_id,
        'exp_size': exp_size,
        'exp_instability': exp_instability,
        'explanation': ', '.join(explanation_features)
    })

# Créer DataFrame à partir de la liste de dictionnaires
results_df = pd.DataFrame(results_list)

# Exporter en CSV
results_df.to_csv('../data/custom_no_streaming_8/folder_1_results.csv', index=False)
