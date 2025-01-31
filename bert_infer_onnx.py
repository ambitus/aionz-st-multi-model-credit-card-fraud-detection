import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from data.vocab import Vocabulary 
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sn
import warnings
import matplotlib.pyplot as plt
import onnxruntime as ort
warnings.filterwarnings("ignore")

database=pd.read_csv('CCF_data/train_set_all_fraud.csv',index_col=False)
database=database.drop(columns=['Unnamed: 0'])

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

class SinglePointDataset(Dataset):
    def __init__(self, single_point_data, vocab, seq_len=10, flatten=False):
        self.single_point_data = single_point_data
        self.vocab = vocab
        self.encoder_fit = self.load_encoders('CCF_data/preprocessed_final/train_set_all_fraud.encoder_fit.pkl')
        self.num_bins = 10
        # Ensure vocab is loaded
        if not self.vocab:
            raise ValueError("Vocabulary is not loaded.")

        self.seq_len = seq_len
        self.flatten = flatten
        self.data = self.prepare_data()

    def load_encoders(self, file_path):
        # Load encoders from pickle file
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def prepare_data(self):
        # Encode the single data point
        encoded_data = self.encode_single_data_point(self.single_point_data)
        # If the length of data is less than seq_len, pad it
        final_encoded_data = [item for sublist in encoded_data for item in sublist]
        return [final_encoded_data]

    def encode_single_data_point(self, single_data_point):
        # Initialize an empty DataFrame for a single data point
        data=single_data_point
        # Apply preprocessing

        # Apply the same preprocessing steps as during training
        data['Errors?'] = self.nanNone(data['Errors?'])
        data['Is Fraud?'] = self.fraudEncoder(data['Is Fraud?'])
        data['Zip'] = self.nanZero(data['Zip'])
        data['Merchant State'] = self.nanNone(data['Merchant State'])
        data['Use Chip'] = self.nanNone(data['Use Chip'])
        data['Amount'] = self.amountEncoder(data['Amount'])

        # Label encoding for columns
        sub_columns = ['Errors?', 'MCC', 'Zip', 'Merchant State', 'Merchant City', 'Merchant Name', 'Use Chip']
        for col_name in sub_columns:
            # if col_name=='Merchant Name':
            #     import pdb;pdb.set_trace()
            col_data = data[col_name]
            col_fit = self.encoder_fit.get(col_name)
            if col_fit:
                data[col_name] = col_fit.transform(col_data)

        # Timestamp encoding
        timestamp = self.timeEncoder(data[['Year', 'Month', 'Day', 'Time']])
        timestamp_fit = self.encoder_fit.get('Timestamp')
        if timestamp_fit:
            data['Timestamp'] = timestamp_fit.transform(timestamp)
            bin_edges, _, _ = self.encoder_fit.get('Timestamp-Quant', ([], [], []))
            data['Timestamp'] = self._quantize(data['Timestamp'].values, bin_edges)

        # Amount quantization
        coldata = data['Amount'].values
        bin_edges, _, _ = self.encoder_fit.get('Amount-Quant', ([], [], []))
        data['Amount'] = self._quantize(coldata, bin_edges)

        # Select required columns
        columns_to_select = ['User', 'Card', 'Timestamp', 'Amount', 'Use Chip', 'Merchant Name', 'Merchant City', 'Merchant State', 'Zip', 'MCC', 'Errors?']
        encoded_data = data[columns_to_select].values.tolist()  # Convert to list format
        #import pdb;pdb.set_trace()
        final_encoded_data = [item for sublist in encoded_data for item in sublist]
        return self.format_trans(final_encoded_data,['User', 'Card', 'Timestamp', 'Amount', 'Use Chip', 'Merchant Name', 'Merchant City', 'Merchant State', 'Zip', 'MCC', 'Errors?', 'Is Fraud?'])

    def format_trans(self, trans_lst, column_names):
        #import pdb;pdb.set_trace()
        trans_lst = list(divide_chunks(trans_lst, 11))  # 2 to ignore isFraud and SPECIAL
        user_vocab_ids = []

        sep_id = self.vocab.get_id(self.vocab.sep_token, special_token=True)

        for trans in trans_lst:
            vocab_ids = []
            for jdx, field in enumerate(trans):
                try:
                    vocab_id = self.vocab.get_id(str(int(field)), column_names[jdx])
                except:
                    vocab_id = self.vocab.get_id(str(field), column_names[jdx])
                vocab_ids.append(vocab_id)

            # TODO : need to handle ncols when sep is not added
            #if self.mlm:  # and self.flatten:  # only add [SEP] for BERT + flatten scenario
            vocab_ids.append(sep_id)
            user_vocab_ids.append(vocab_ids)

        return user_vocab_ids

    def __getitem__(self, index):
        data_point = self.data[index]
        if self.flatten:
            return torch.tensor(data_point, dtype=torch.long)
        else:
            return torch.tensor(data_point, dtype=torch.long).reshape(self.seq_len, -1)

    def __len__(self):
        return len(self.data)

    def nanNone(self, series):
        return series.fillna("None")
    def fraudEncoder(self, series):
        return series.map({'No': 0, 'Yes': 1})

    def nanZero(self, series):
        return series.fillna(0)

    def amountEncoder(self, series):
        return series

    @staticmethod
    def label_fit_transform(column, enc_type="label"):
        if enc_type == "label":
            mfit = LabelEncoder()
        else:
            mfit = MinMaxScaler()
        mfit.fit(column)

        return mfit, mfit.transform(column)
    @staticmethod
    def timeEncoder(X):
        X_hm = X['Time'].str.split(':', expand=True)
        d = pd.to_datetime(dict(year=X['Year'], month=X['Month'], day=X['Day'], hour=X_hm[0], minute=X_hm[1])).astype(
            int)
        return pd.DataFrame(d)

    def _quantization_binning(self, data):
        qtls = np.arange(0.0, 1.0 + 1 / self.num_bins, 1 / self.num_bins)
        bin_edges = np.quantile(data, qtls, axis=0)  # (num_bins + 1, num_features)
        bin_widths = np.diff(bin_edges, axis=0)
        bin_centers = bin_edges[:-1] + bin_widths / 2  # ()
        return bin_edges, bin_centers, bin_widths

    def _quantize(self, inputs, bin_edges):
        quant_inputs = np.zeros(inputs.shape[0])
        for i, x in enumerate(inputs):
            quant_inputs[i] = np.digitize(x, bin_edges)
        quant_inputs = quant_inputs.clip(1, self.num_bins) - 1  # Clip edges
        return quant_inputs

def classify_single_input(session, vocab, input_data, device):
    # Initialize the dataset for a single data point
    try:
        dataset = SinglePointDataset(input_data, vocab, seq_len=10, flatten=True)
    
    except:
        return 
    # Create DataLoader with batch size of 1
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for data in dataloader:
        data = data[0].unsqueeze(0).to(device)
        # Forward pass through the model
        data = np.array(data, dtype=np.int64) 

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_dict = {input_name: data}

        output = session.run([output_name], input_dict)
        prediction=np.argmax(output[0])
        return prediction.item()

def data_retreive(data):
    ip_user = data['User'].values[0]
    user_df = database[database['User'] == ip_user]
    user_df = user_df.sample(n=9, random_state=42)
    final_input=pd.concat([data,user_df],ignore_index=True)
    return final_input

def time(data):
    X=data[['Year', 'Month', 'Day', 'Time']]
    X_hm = X['Time'].str.split(':', expand=True)
    data['DateTime']= pd.to_datetime(dict(year=X['Year'], month=X['Month'], day=X['Day'], hour=X_hm[0], minute=X_hm[1]))
    return data

def data_retreive1(database,input_data):
    database_time=time(database)
    input_data_time=time(input_data)
    input_user_data=database_time[database_time['User']==input_data['User'].values[0]]
    past_user_data=input_user_data[input_user_data['DateTime']<input_data_time['DateTime'].values[0]]
    past_user_data=past_user_data.sort_values(by=['DateTime'],ascending=False)
    final_data=pd.concat([input_data_time,past_user_data.head(9)])
    return final_data

def results(actual_labels, predicted_labels):
    tn, fp, fn, tp = confusion_matrix(actual_labels, predicted_labels).ravel()
    cf_matrix=confusion_matrix(actual_labels, predicted_labels)
    sn.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
    plt.show()
    acc=(tp+tn)/(tn+fp+fn+tp)
    recall=(tp)/(tp+fn)
    precision=(tp)/(tp+fp)
    f1=(2*(recall)*(precision))/(recall+precision)
    print("Accuracy : {:.2f}%".format(acc*100))
    print("Recall : {:.2f}%".format(recall*100))
    print("Precision :{:.2f}%".format(precision*100))
    print("F1 score : {:.2f}%".format(f1*100))

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load vocab
    vocab = Vocabulary(10 ** 8)
    vocab.load_vocab(fname='op_1/vocab.nb')

    test_dataset=pd.read_csv('CCF_data/balanced_test_data_full.csv')
    test_dataset.drop(columns=["Unnamed: 0","Unnamed: 0.1"],inplace=True)
    predicted_labels=[]
    actual_labels=[]

    # Initialize model
    session = ort.InferenceSession("quantized_model.onnx")
    print("Model loaded successfully!")

    for i in range(test_dataset.shape[0]):
        data=test_dataset.iloc[i]
        input_data=pd.DataFrame([data])
        input_data=data_retreive1(database,input_data).sort_values(by=['DateTime'])
        if input_data.shape[0]!=10:
            try:
                input_data=data_retreive(input_data)
            except:
                continue           
        #break
        if input_data.shape[0]!=10:
            print("Not enough data to predict")
            continue
        
        #Run classification
        prediction = classify_single_input(session, vocab, input_data, device)
        if prediction is not None:
            print("{} Predicted class: {}".format(i,prediction))
            predicted_labels.append(prediction)
            actual_labels.append(1 if data["Is Fraud?"]=="Yes" else 0)
        if i%500==0 and i!=0:
            results(actual_labels,predicted_labels)
    results(actual_labels,predicted_labels)