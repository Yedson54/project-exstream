# project-extsream

# Project README

## Data Streaming Processing Class Project

This project focuses on implementing an anomaly explainability method named "EXStream" for data streaming processing. The aim is to provide interpretable explanations for anomalies detected in time-series data.

### Archive Content

The project archive includes:

- **src/ :** Contains the source code for our implementation of the EXStream anomaly explainability method.
- **data/ :** Sample datasets used for testing.
- **notebooks/ :** Notebooks outputing the results.

### Main Functions

1. **tsa_tsr_series_split:**
   - Description: Splits the anomaly dataframe into a Time Series Anomaly (TSA) and a Time Series Reference (TSR) for a specific anomaly ID in a given trace ID.
   - Signature: `tsa_tsr_series_split(df: pd.DataFrame, trace_id: str, ano_id: int) -> Tuple[pd.Series, pd.Series]`

2. **single_feature_reward:**
   - Description: Calculates the reward for a single feature based on entropy.
   - Signature: `single_feature_reward(df, feature, trace_id, ano_id) -> float`

3. **select_feature_on_rewards:**
   - Description: Plots the feature rewards for continuous features and identifies the elbow point. Optionally ignores the first sharpest drop if specified.
   - Signature: `select_feature_on_rewards(df, trace_id, ano_id, features_code, ignore_first=False, plot=True, verbose=True) -> List[str]`

4. **process_trace_anomaly:**
   - Description: Processes the trace anomaly by retrieving TSA and TSR data, selecting specified features, calculating feature correlations, clustering correlated features, and identifying representative features.
   - Signature: `process_trace_anomaly(df, trace_id, ano_id, selected_features, plot_correlation=False, threshold=0.8, linkage='complete', verbose=True) -> List[str]`

5. **build_explanation:**
   - Description: Builds an explanation for an anomaly using bursty input data, trace ID, anomaly ID, features, and other optional parameters.
   - Signature: `build_explanation(bursty_input1_df, trace_id="1_1", ano_id=1, features=features_code, plot_rewards=False, plot_correlation=False, verbose=False, ignore_first=False, cluster=True) -> Set[str]`

6. Certainly, let's integrate the `instability` function into the README:

### Additional Function

6. **instability:**
   - Description: Calculates the instability (consistency as entropy) of feature explanations for a specific anomaly in a DataFrame. It repeats the explanation process multiple times on different samples and computes the frequency of each feature in the list to determine consistency.
   - Signature: `instability(df, trace_id, ano_id, features, plot_rewards=False, plot_correlation=False, verbose=False, threshold=0.8, linkage='complete', repeat=5, sample_frac=0.8, replace=False, random_state=42) -> float`


### Usage

1. Clone the repository.
2. Install required dependencies (`pip install -r requirements.txt`).
3. Use the main functions in your project as needed.

For detailed usage and examples, refer to the documentation in the "docs/" directory.

### Contributors

- [Yedidia AGNIMO]
- [Yann Eric CHOHO CHOHO]

Feel free to contribute, report issues, or suggest improvements.
