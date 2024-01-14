# Import librairies.
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Get files
DATAPATH =  "../data/custom_no_streaming_8/folder_2"
files = glob.glob(rf"{DATAPATH}/*")
files

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
anomaly_df

# Rename values in the "anomaly_type" column based on the specified mapping
mapping = {
    "1_3": "bursty_input",
    "1_4": "bursty_input",
    "2_3": "stalled_input",
    "2_4": "stalled_input",
    "3_3": "cpu_contention",
    "3_4": "cpu_contention"
}
anomaly_df.insert(2, column="anomaly_type", value=anomaly_df['trace_id'].replace(mapping))

cols = ['driver_jvm_heap_used_value', 'avg_jvm_heap_used_value',
       'avg_executor_filesystem_hdfs_write_ops_value_1_diff',
       'avg_executor_cpuTime_count_1_diff',
       'avg_executor_runTime_count_1_diff',
       'avg_executor_shuffleRecordsRead_count_1_diff',
       'avg_executor_shuffleRecordsWritten_count_1_diff']
features = len(cols)



import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# Load the data (example for trace_id = '1_3')
df = anomaly_df[anomaly_df['trace_id'] == '1_3'].copy()

# Select the relevant numerical columns (excluding 'time' and 'trace_id')
cols = df.columns[3:]
n_features = len(cols)

def anomalies_detection(df, cols, epochs=20, n_layers=10, verbose=False):
    """
    Detect anomalies in a time series using an LSTM autoencoder.
    :param df: The dataframe containing the time series data
    :param cols: The columns containing the time series data
    :param epochs: The number of epochs for training the LSTM autoencoder
    :param n_layers: The number of layers for the LSTM autoencoder
    :param verbose: Whether or not to display the training progress
    :return: The anomalies and the predictions
    """

    # Normalize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[cols])

    # Reshape data for LSTM autoencoder
    timesteps = 1  # Adjust this according to your time sequence
    n_features = len(cols)
    data = df_scaled.reshape(df_scaled.shape[0], timesteps, n_features)

    # Construct the LSTM autoencoder
    inputs = Input(shape=(timesteps, n_features))
    encoded = LSTM(n_layers, return_sequences=False)(inputs)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(n_features, return_sequences=True)(decoded)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Train the model
    autoencoder.fit(data, data, epochs=epochs, batch_size=128, validation_split=0.2, shuffle=False, verbose=verbose)

    # Predict the reconstructed data
    predictions = autoencoder.predict(data)

    # Calculate the reconstruction error
    mse = np.mean(np.power(data - predictions, 2), axis=1)
    error_df = pd.DataFrame({'Reconstruction_error': mse[:,0]})

    # Identify anomalies (You can set an error threshold)
    threshold = np.percentile(error_df.Reconstruction_error.values, 95)
    anomalies = error_df[error_df.Reconstruction_error > threshold]

    return anomalies, predictions


def plot_reconstruction(df, predictions, cols,n_features):
    """
    Plot the original and reconstructed time series data.
    
    
    Parameters
    ----------
    df : pd.DataFrame
        The original dataframe.
        predictions : np.array
        The reconstructed data.
    cols : list

    scaler : StandardScaler
        The scaler used to scale the data.

    Returns
    -------
    None.            
    """
    # normalize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[cols])
    
    # Rescale predictions back to original scale
    predictions_rescaled = scaler.inverse_transform(predictions.reshape(predictions.shape[0], n_features))

    # Set Seaborn style
    sns.set(style='whitegrid')

    # Number of variables (excluding 'time' and 'trace_id')
    num_vars = len(cols)

    # Create subplots
    # add 3 columns for the time, trace_id, and anomaly_type columns


    fig, axes = plt.subplots(num_vars, figsize=(15, 10), sharex=True)

    for i, col in enumerate(cols):
        # Original Data
        axes[i].plot(df['time'], df[col], label='Original', color='blue', linewidth=1)

        # Reconstructed Data
        axes[i].plot(df['time'], predictions_rescaled[:, i], label='Reconstructed', color='orange', linewidth=1)

        # Titles and Labels
        axes[i].set_title(f'Reconstruction Comparison for {col}')
        #axes[i].set_ylabel(col)
        axes[i].legend()

    # Set common X label
    axes[-1].set_xlabel('Time')

    # Adjust layout for better readability
    plt.tight_layout()
    plt.show()

def plot_anomalies(df, anomalies, cols, save=False, name='anomalies'):
    """
    Function to plot anomalies detected by the LSTM autoencoder.
    Parameters:
        df (DataFrame): DataFrame containing the original data.
        anomalies (DataFrame): DataFrame containing the detected anomalies.
        cols (list): List of columns to plot.
    """
    # Add the 'time' column for temporal reference
    anomalies['time'] = anomalies.index

    # Set the Seaborn style for better aesthetics
    sns.set(style='whitegrid')

    # Define color palette for the time series
    colors = sns.color_palette('tab10', len(cols))

    # Create a figure and axis for the plot
    plt.figure(figsize=(15, 6))
    ax = plt.gca()

    # Plot each time series with a thinner line
    for i, col in enumerate(cols):
        ax.plot(df['time'], df[col], color=colors[i], linewidth=1, label=col)

    # Highlight anomaly points with red color
    ax.scatter(anomalies['time'], df.loc[anomalies.index, cols[0]], color='red', s=50, label='Anomalies', zorder=5)

    # Set title and labels with appropriate font sizes
    ax.set_title('Time Series with Anomaly Points', fontsize=16)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Normalized Values', fontsize=14)

    # Place the legend in the upper right corner of the plot
    ax.legend(loc='upper right', frameon=True)

    # Save the plot if 'save' is True
    if save:
        plt.savefig('../results/{}.png'.format(name))

    # Display the plot
    plt.show()


results_list = []
traces = anomaly_df['trace_id'].unique().tolist()

# load results
results_to_fill = pd.read_csv('../data/custom_no_streaming_8/folder_2_results.csv', index_col=0)

for trace_id in traces:
    # Load the data (example for each trace_id)
    df = anomaly_df[anomaly_df['trace_id'] == trace_id].copy()
    cols = df.columns[3:]

    # Detect anomalies for each trace_id
    anomalies, predictions = anomalies_detection(df, cols, epochs=50, n_layers=20, verbose=False)

    # Retrieve the min and max of the time in anomalies for each trace_id
    if not anomalies.empty:
        min_time = anomalies.index.min()
        max_time = anomalies.index.max()
    else:
        min_time = np.nan
        max_time = np.nan
    
    results_list.append({
        'ano_start': min_time,
        'ano_end': max_time,
        'trace_id': trace_id
    })

results_df = pd.DataFrame(results_list)

# Merge anomalies and results
final_results = pd.merge(results_to_fill.drop(['ano_start', 'ano_end'], axis=1), results_df, on='trace_id', how='inner')


results_df.to_csv('../results/folder_2_results.csv')