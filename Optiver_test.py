import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error

file = pd.read_csv('train.csv')
file.drop(["date_id","seconds_in_bucket","row_id"], axis=1, inplace=True)
stock_id_list = []
for stock_id, df in file.groupby(['stock_id']):
    ## fill the missing far_price values with the mean of the far_price
    df['far_price'].fillna(df['far_price'].mean(), inplace=True)
    df['near_price'].fillna(df['near_price'].mean(), inplace=True)
    df = df[['stock_id', 'imbalance_size', 'imbalance_buy_sell_flag','reference_price', 'matched_size', 'far_price', 'near_price','bid_price', 'bid_size', 'ask_price', 'ask_size', 'wap', 'time_id','target']]
    
    if len(df) < 26455:
        continue
    else:
        stock_id_list.append(df)

## change the stock_id_list dimension to (time,stock_id, 13)

stock_id_list = np.array(stock_id_list).astype(np.float32) ## (stock_id,time,13) 
features = stock_id_list[:,:,:-1]
target = stock_id_list[:,:,-1]

#do the same to testfile
testfile = pd.read_csv('./example_test_files/test.csv')
testfile.drop(["date_id","seconds_in_bucket","row_id"], axis=1, inplace=True)
test_stock_id_list = []
for stock_id, df in testfile.groupby(['stock_id']):
    ## fill the missing far_price values with the mean of the far_price
    df['far_price'].fillna(df['far_price'].mean(), inplace=True)
    df['near_price'].fillna(df['near_price'].mean(), inplace=True)
    df = df[['stock_id', 'imbalance_size', 'imbalance_buy_sell_flag','reference_price', 'matched_size', 'far_price', 'near_price','bid_price', 'bid_size', 'ask_price', 'ask_size', 'wap', 'time_id']]
    
    test_stock_id_list.append(df)
test_stock_id_list = np.array(test_stock_id_list).astype(np.float32) ## (stock_id,time,13)
test_features = test_stock_id_list[:,:,:]


def lstmmodel(features=features,target=target):    
    # Remove sequences with NaN in target
    mask = ~np.isnan(target).any(axis=1)
    features = features[mask]
    target = target[mask]

    def break_into_subsequences_with_padding(data, target, length):
        subsequences = []
        subtargets = []
        total_sequences = data.shape[1] // length
        for i in range(total_sequences):
            subsequences.append(data[:, i*length:(i+1)*length])
            subtargets.append(target[:, i*length:(i+1)*length])
        
        # Handle the last subsequence which might need padding
        if data.shape[1] % length != 0:
            padding_size = length - data.shape[1] % length
            start_idx = total_sequences * length  # start index for the last subsequence
            last_subseq = np.pad(data[:, start_idx:], ((0,0), (0, padding_size), (0,0)), mode='constant')
            last_target = np.pad(target[:, start_idx:], ((0,0), (0, padding_size)), mode='constant')
            subsequences.append(last_subseq)
            subtargets.append(last_target)
        
        return np.concatenate(subsequences, axis=0), np.concatenate(subtargets, axis=0)

    # Define the subsequence length
    subseq_length = 256  # for example

    # Break sequences into smaller subsequences and pad if necessary
    features_subseq, target_subseq = break_into_subsequences_with_padding(features, target, subseq_length)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_subseq, target_subseq, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers,output_size):
            super(LSTMModel, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
            self.fc2 = nn.Linear(hidden_size*output_size, output_size)

        def forward(self, x):
            # Set initial hidden and cell states
            h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
            c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
            x = F.relu(self.fc1(x))
            
            # Forward propagate LSTM
            out, _ = self.lstm(x, (h0, c0))
            out = F.relu(out)
            # Decode the hidden state of the last time step
            out = out.reshape(out.shape[0], -1)
            out = self.fc2(out)
            return out

    # Hyperparameters
    input_size = features_subseq.shape[2] # 13
    hidden_size = 50
    num_layers = 2
    output_size = features_subseq.shape[1] #256
    learning_rate = 1e-5

    # Create the model
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size, hidden_size, num_layers,output_size).to(device)

    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for batch_features, batch_labels in train_loader:
            # Forward pass
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)
            
            loss = criterion(outputs, batch_labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)
        test_predictions = model(X_test_tensor)
        print(y_test_tensor.shape,test_predictions.shape)
        test_loss = criterion(test_predictions, y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}')

        # Calculate MAE
        y_pred = test_predictions.cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()
        mae = mean_absolute_error(y_true, y_pred)
        print(f'MAE: {mae:.4f}')




def xgmodel(features=features,target=target):
    mask = ~np.isnan(target).any(axis=1)
    features = features[mask]
    target = target[mask]

    def break_into_subsequences_with_padding(data, target, length):
        subsequences = []
        subtargets = []
        total_sequences = data.shape[1] // length
        for i in range(total_sequences):
            subsequences.append(data[:, i*length:(i+1)*length])
            subtargets.append(target[:, i*length:(i+1)*length])
        
        # Handle the last subsequence which might need padding
        if data.shape[1] % length != 0:
            padding_size = length - data.shape[1] % length
            start_idx = total_sequences * length  # start index for the last subsequence
            last_subseq = np.pad(data[:, start_idx:], ((0,0), (0, padding_size), (0,0)), mode='constant')
            last_target = np.pad(target[:, start_idx:], ((0,0), (0, padding_size)), mode='constant')
            subsequences.append(last_subseq)
            subtargets.append(last_target)
        
        return np.concatenate(subsequences, axis=0), np.concatenate(subtargets, axis=0)

    # Define the subsequence length
    subseq_length = 256  # for example

    # Break sequences into smaller subsequences and pad if necessary
    features_subseq, target_subseq = break_into_subsequences_with_padding(features, target, subseq_length)
    features = features_subseq
    target = target_subseq
    
    features_2d = features.reshape(-1, features.shape[-1])
    target_2d = target.reshape(-1)

    # Combine into a single DataFrame
    data = pd.DataFrame(features_2d)
    data['target'] = target_2d

    # Drop rows where 'target' is NaN
    data = data.dropna(subset=['target'])

    # Split data into training and testing sets
    X = data.drop(columns=['target'])
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model, using MAE as the objective function
    model = xgb.XGBRegressor(objective='reg:squarederror', reg_alpha=0.5)

    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    print(f'MAE: {mae:.4f}')
    

    test_features_2d = test_features.reshape(-1, test_features.shape[-1])
    test_data = pd.DataFrame(test_features_2d)
    # Predict using the trained model
    y_pred = model.predict(test_data)
    return y_pred
    

if __name__ == "__main__":
    # lstmmodel()
    xg_pred = xgmodel()
    
    #read the sample_sumission.csv and change the target
    import pandas as pd
    submit = pd.read_csv("./example_test_files/sample_submission.csv")
    
    # Convert the predictions to a DataFrame
    submission_df = pd.DataFrame({
        'time_id': submit['time_id'],
        'row_id': submit['row_id'],
        'target': xg_pred  # assuming this is the name of your numpy array with predictions
    })

    # Save the submission DataFrame to a CSV file
    submission_df.to_csv('./results/xgsubmission.csv', index=False)

    

    

