import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, ReLU
from tensorflow.keras.callbacks import EarlyStopping

def train_stock_model(
    dataset,
    seq_length=60,
    train_split=0.7,
    epochs=20,
    batch_size=32,
    model_filename=None,
    output_dir="/content",
    features=None
):
    """
    Train a Bidirectional LSTM model on stock data for one-step-ahead close price prediction.

    Args:
        dataset (pd.DataFrame): Input DataFrame with columns ['time', 'open', 'high', 'low', 'close']
        seq_length (int): Length of input sequences for LSTM
        train_split (float): Fraction of data for training (default: 0.7)
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        model_filename (str): Name for the saved model file (default: derived from dataset name)
        output_dir (str): Base directory to save outputs; a stock-specific subdirectory is created
        features (list): List of feature names (default: ['open', 'high', 'low', 'close', 'ema_5', 'ema_10', 'ema_15'])

    Returns:
        dict: Contains model, feature_scaler, target_scaler, predictions_df, metrics, history, model_path, predictions_csv, features_csv, plot_path
    """
    # Validate input dataset
    required_columns = ['time', 'open', 'high', 'low', 'close']
    if not all(col in dataset.columns for col in required_columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")

    # Default features if not provided
    if features is None:
        features = ['open', 'high', 'low', 'close', 'ema_5', 'ema_10', 'ema_15']

    # Derive stock name and set stock-specific output directory
    if model_filename is None:
        indice_name = "stock_model"
    else:
        indice_name = os.path.splitext(model_filename)[0].replace(",", "")
    stock_output_dir = os.path.join(output_dir, indice_name)
    os.makedirs(stock_output_dir, exist_ok=True)

    # Copy dataset to avoid modifying the input
    df = dataset.copy()

    # Preprocess time
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df.dropna(subset=['time'], inplace=True)
    df.sort_values(by='time', inplace=True)
    df.set_index('time', inplace=True)

    # Compute technical indicators
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()

    # Target: next period's close
    df['target'] = df['close'].shift(-1)  # Shift close by -1 to predict next period
    df = df[:-1]  # Remove last row where target is NaN

    # Scale features
    feature_scaler = MinMaxScaler()
    df[features] = feature_scaler.fit_transform(df[features])

    # Scale target
    target_scaler = MinMaxScaler()
    df[['target']] = target_scaler.fit_transform(df[['target']])

    # Split train/test
    train_size = int(len(df) * train_split)
    train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

    X_train_raw, y_train_raw = train_df[features], train_df['target']
    X_test_raw, y_test_raw = test_df[features], test_df['target']

    # Create sequences for one-step-ahead prediction
    def create_sequences(X, y, seq_length):
        Xs, ys = [], []
        for i in range(seq_length, len(X)):
            Xs.append(X.iloc[i-seq_length:i].values)
            ys.append(y.iloc[i])
        return np.array(Xs), np.array(ys)

    X_train, y_train = create_sequences(X_train_raw, y_train_raw, seq_length)
    X_test, y_test = create_sequences(X_test_raw, y_test_raw, seq_length)

    # Build model
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.001))
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.001))
    model.add(ReLU())
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    # Save model
    model_filename = f"{indice_name}_prediction_model.keras"
    model_path = os.path.join(stock_output_dir, model_filename)
    model.save(model_path)
    print(f"💾 Model saved as: {model_path}")

    # Predict and inverse scale
    pred_scaled = model.predict(X_test)
    y_test_scaled = y_test.reshape(-1, 1)

    pred_actual = target_scaler.inverse_transform(pred_scaled)
    y_actual = target_scaler.inverse_transform(y_test_scaled)

    # Save predictions
    predictions_csv = os.path.join(stock_output_dir, f"{indice_name}_predictions.csv")
    predictions_df = pd.DataFrame({
        'datetime': test_df.index[seq_length:],
        'actual': y_actual.flatten(),
        'predicted': pred_actual.flatten()
    })
    predictions_df.to_csv(predictions_csv, index=False)
    print(f"✅ Predictions saved to: {predictions_csv}")

    # Save features
    features_csv = os.path.join(stock_output_dir, 'Features.csv')
    if os.path.exists(features_csv):
        os.remove(features_csv)
        print("Existing 'Features.csv' deleted.")
    features_data = {
        'Dataset': [indice_name],
        'Features': [features]
    }
    features_df = pd.DataFrame(features_data)
    features_df['Features'] = features_df['Features'].apply(lambda x: ", ".join([f"'{item}'" for item in x]))
    features_df.to_csv(features_csv, index=False)
    print(f"✅ Features saved to: {features_csv}")

    # Evaluate
    mae = mean_absolute_error(y_actual, pred_actual)
    rmse = np.sqrt(mean_squared_error(y_actual, pred_actual))
    mape = mean_absolute_percentage_error(y_actual, pred_actual)
    accuracy = 100 - (mape * 100)

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'accuracy': accuracy
    }

    # Plot and save
    plt.figure(figsize=(12, 6))
    plt.plot(y_actual, label='Actual')
    plt.plot(pred_actual, label='Predicted')
    plt.title('LSTM One-Step-Ahead Prediction vs Actual')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(stock_output_dir, f"{indice_name}_prediction_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Plot saved to: {plot_path}")

    return {
        'model': model,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'predictions_df': predictions_df,
        'metrics': metrics,
        'history': history,
        'model_path': model_path,
        'predictions_csv': predictions_csv,
        'features_csv': features_csv,
        'plot_path': plot_path
    }
