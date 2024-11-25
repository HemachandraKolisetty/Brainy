import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from encoder import rate_encoding, temporal_encoding
import torch

def load_and_merge_data(root, velocity, texture):
    velocity_folder = os.path.join(root, f"pickles_{velocity}")
    texture_folder = os.path.join(velocity_folder, texture)
    
    baro_file = os.path.join(texture_folder, "full_baro.csv")
    imu_file = os.path.join(texture_folder, "full_imu.csv")
    
    baro_df = pd.read_csv(baro_file)
    
    imu_df = pd.read_csv(imu_file)
    
    imu_df['baro'] = baro_df['baro']
        
    return imu_df

def load_data_label_encoded(root, file_name, spike_encoding='rate', num_steps=100, batch_size=32, device='cpu'):
    file_path = os.path.join(root, file_name)
    df = pd.read_csv(file_path)

    y = df['Texture']
    X = df.drop(['Texture'], axis=1)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    num_features = X.shape[1]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    if spike_encoding == 'rate':
        X_train_tensor = rate_encoding(X_train, num_steps=num_steps, device=device)
        X_val_tensor = rate_encoding(X_val, num_steps=num_steps, device=device)
        X_test_tensor = rate_encoding(X_test, num_steps=num_steps, device=device)

        y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long, device=device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)
    else:
        X_train_tensor = temporal_encoding(X_train, num_steps=num_steps, device=device)
        X_val_tensor = temporal_encoding(X_val, num_steps=num_steps, device=device)
        X_test_tensor = temporal_encoding(X_test, num_steps=num_steps, device=device)

        y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long, device=device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)


    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes, num_features

def load_data_one_hot_encoded(root, file_name, spike_encoding='rate', num_steps=50, batch_size=32, device='cpu'):
    file_path = os.path.join(root, file_name)
    df = pd.read_csv(file_path)

    y = df['Texture']
    X = df.drop(['Texture'], axis=1)

    y_encoded = pd.get_dummies(y)
    num_classes = len(y_encoded.columns)
    num_features = X.shape[1]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train.idxmax(axis=1)
    )

    if spike_encoding == 'rate':
        X_train_tensor = rate_encoding(X_train, num_steps=num_steps, device=device)
        X_val_tensor = rate_encoding(X_val, num_steps=num_steps, device=device)
        X_test_tensor = rate_encoding(X_test, num_steps=num_steps, device=device)

        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32, device=device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32, device=device)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32, device=device)
    else:
        X_train_tensor = temporal_encoding(X_train, num_steps=num_steps, device=device)
        X_val_tensor = temporal_encoding(X_val, num_steps=num_steps, device=device)
        X_test_tensor = temporal_encoding(X_test, num_steps=num_steps, device=device)

        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32, device=device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32, device=device)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32, device=device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes, num_features


def load_data_for_ann(root, file_name, batch_size=32, device='cpu'):
    file_path = os.path.join(root, file_name)
    df = pd.read_csv(file_path)

    y = df['Texture']
    X = df.drop(['Texture'], axis=1)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    num_features = X.shape[1]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes, num_features