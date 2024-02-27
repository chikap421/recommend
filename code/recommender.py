
import torch  # download & install PyTorch here:  https://pytorch.org/get-started/locally/
import torch.nn as nn
import pandas as pd
import numpy as np

torch.manual_seed(0)

# You also need to install the libraries listed above
# This can be done with: pip install pandas numpy


##########################################
##  Helper Functions (Provided for you) ##
##########################################

def train_model(X_train, y_train, k, max_iter=50, batch_size=32, print_n=10, verbose=True):
    '''
    Trains neural network model on X_train, y_train data.

    Parameters
    ----------
    X_train: np.array
        matrix of training data features
    y_train: np.array
        vector of training data labels
    k: int
        size of hidden layer to use in neural network
    max_iter: int
        maximum number of iterations to train for
    batch_size: int
        batch size to use when training w/ SGD
    print_n: int
        print training progress every print_n steps

    Returns
    ----------
    nn_model: torch.nn.Module
        trained neural network model
    '''
    # convert to tensors (for Pytorch)
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    # intialize neural network
    n_samples, n_features = X_train_tensor.shape
    nn_model = NN(n_features, k)
    nn_model.train()  # put model in train mode
    # initialize mse loss function
    mse_loss = torch.nn.MSELoss()
    # train with (mini-batch) SGD; initialize optimizer
    opt = torch.optim.SGD(nn_model.parameters(), lr=0.001)
    for it in range(max_iter):
        # save losses across all batches
        losses = []
        # loop through data in batches
        for batch_start in range(0, n_samples, batch_size):
            # reset gradients to zero
            opt.zero_grad()
            # form batch
            X_batch = X_train_tensor[batch_start:batch_start+batch_size]
            y_batch = y_train_tensor[batch_start:batch_start+batch_size]
            # pass batch through neural net to get prediction
            y_pred = nn_model(X_batch.float())
            # compute MSE loss
            loss = mse_loss(y_pred, y_batch[:, None].float())
            # back-propagate loss
            loss.backward()
            # update model parameters based on backpropogated gradients
            opt.step()
            losses.append(loss.item())
        if verbose and it % print_n == 0:
            print(f"Mean Train MSE at step {it}: {np.mean(losses)}")
    return nn_model


def evaluate_model(nn_model, X_eval, y_eval, batch_size=32):
    '''
    Evaluates trained neural network model on X_eval, y_eval data.

    Parameters
    ----------
    nn_model: torch.nn.Module
        trained neural network model
    X_eval: np.array
        matrix of training data features
    y_eval: np.array
        vector of training data labels
    batch_size: int
        batch size to looping over dataset to generate predictions

    Returns
    ----------
    mse: float
        MSE of trained model on X_eval, y_eval data
    '''
    # initialize mse loss function
    mse_loss = torch.nn.MSELoss()
    # convert to tensors (for Pytorch)
    X_eval_tensor = torch.tensor(X_eval)
    y_eval_tensor = torch.tensor(y_eval)
    n_samples = X_eval_tensor.shape[0]
    nn_model.eval() # put in eval mode
    # loop over data and generate predictions
    preds = []
    for batch_start in range(0, n_samples, batch_size):
        # form batch
        X_batch = X_eval_tensor[batch_start:batch_start+batch_size]
        y_batch = y_eval_tensor[batch_start:batch_start+batch_size]
        with torch.no_grad():  # no need to compute gradients during evaluation
            # pass batch through neural net to get prediction
            y_pred = nn_model(X_batch.float())
            preds.append(y_pred)
    # compute MSE across all samples
    all_preds = torch.cat(preds)
    loss = mse_loss(all_preds, y_eval_tensor[:, None].float()).item()
    return loss


def read_data(ratings_fp):
    '''
    Reads book ratings from book_ratings.csv file, splits into train, validation, and test groups,
    and produces a Pandas dataframe with the ratings for each. Each dataframe contains the
    columns: UserID, bookID, rating. There is a row for each rating in the dataset.

    Parameters
    ----------
    ratings_fp: str
        Path to book_ratings.csv file

    Returns
    ----------
    train_df: pd.DataFrame
        DataFrame for each book rating in the train dataset
    val_df: pd.DataFrame
        DataFrame for each book rating in the train dataset
    test_df: pd.DataFrame
        DataFrame for each book rating in the train dataset
    n_users: int
        total number of users in dataset
    n_books: int
        total number of books in dataset
    '''
    # read data
    ratings_df = pd.read_csv(ratings_fp)
    n_users = len(ratings_df['userId'].unique())
    n_books = len(ratings_df['bookId'].unique())
    # split into train, val, test
    train_df = ratings_df[ratings_df["split"] == "train"]
    val_df = ratings_df[ratings_df["split"] == "val"]
    test_df = ratings_df[ratings_df["split"] == "test"]
    # reset their indices
    train_df = train_df.reset_index()
    val_df = val_df.reset_index()
    test_df = test_df.reset_index()
    return train_df, val_df, test_df, n_users, n_books


def get_genre_id(genre_str):
    '''
    Returns unique integer ID (single digit in range 0-19) associated with the given genre.
    
    Parameters
    ----------
    genre_str: string specifying a single genre name: e.g., "Fantasy"

    Returns
    ----------
    genre_id: int
        ID in range [0, 19]
    '''
    
    GENRE_ID_DICT = {'Fantasy': 0,
                     'Science Fiction': 1,
                     'Mystery': 2,
                     'Romance': 3,
                     'Thriller': 4,
                     'Horror': 5,
                     'Non-fiction': 6,
                     'Historical Fiction': 7,
                     'Biography': 8,
                     'Self-help': 9,
                     'Young Adult': 10,
                     'Poetry': 11,
                     'Humor': 12,
                     'Cooking': 13,
                     'Travel': 14,
                     'Art': 15,
                     'Business': 16,
                     'Science': 17,
                     'Health': 18,
                     'Fitness': 19
                     }
    return GENRE_ID_DICT[genre_str]


#####################################
##  Neural Network Implementation  ##
#####################################


class NN(nn.Module):
    '''
    Class for fully connected neural net.
    '''
    def __init__(self, input_dim, hidden_dim):
        '''
        Parameters
        ----------
        input_dim: int
            input dimension (i.e., # of features in each example passed to the network)
        hidden_dim: int
            number of nodes in hidden layer
        '''
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


#######################
##  Data Processing  ##
#######################


def process_data_one_hot(ratings_df, N, M):
    X = np.zeros((len(ratings_df), N + M))
    y = np.zeros(len(ratings_df))
    for i, row in ratings_df.iterrows():
        user_id = int(row["userId"]) - 1
        book_id = int(row["bookId"]) - 1

        X[i, user_id] = 1
        X[i, N + book_id] = 1

        y[i] = row["rating"]

    return X, y


def process_data_with_genres(ratings_df, N, M):
    N_GENRES = 20  
    X = np.zeros((len(ratings_df), N + M + N_GENRES))
    y = np.zeros(len(ratings_df))
    
    for i, row in ratings_df.iterrows():
        user_id = int(row["userId"]) - 1
        book_id = int(row["bookId"]) - 1
        genres = row["genres"].split('|') if pd.notna(row["genres"]) else []
        
        X[i, user_id] = 1
        X[i, N + book_id] = 1

        genre_vector = np.zeros(N_GENRES)
        for genre in genres:
            genre_id = get_genre_id(genre)
            genre_vector[genre_id] = 1
        if genres:
            genre_vector /= genre_vector.sum()

        X[i, -N_GENRES:] = genre_vector
        
        y[i] = row["rating"]

    return X, y


#########################################
##  Code to Run Training & Evaluation  ##
#########################################


def main():
    # Load data used for training and evaluation
    path_to_ratings_file = "book_ratings.csv"
    train_df, val_df, test_df, n_users, n_books = read_data(path_to_ratings_file)

    METHOD = "genre_info"  # change this to "genre_info" to test the other approach!

    # get feature label pairs (x, y) from ratings dataframes
    if METHOD == "one_hot":
        X_train, y_train = process_data_one_hot(train_df, n_users, n_books)
        X_val, y_val = process_data_one_hot(val_df, n_users, n_books)
    else: # method is "genre_info"
        X_train, y_train = process_data_with_genres(train_df, n_users, n_books)
        X_val, y_val = process_data_with_genres(val_df, n_users, n_books)

    # train NN model to predict rating from user + book features using train data
    for k in [20, 60, 100]:
        print('_________Performing Training for %d Hidden Dimension_______'%k)
        nn_model = train_model(X_train, y_train, k)
        # evaluate performance of final model on train data
        train_mse = evaluate_model(nn_model, X_train, y_train)
        print(f"Train MSE for feature method {METHOD}, k={k} is: {train_mse}")
        # evaluate performance of model on validation data
        val_mse = evaluate_model(nn_model, X_val, y_val)
        print(f"Validation MSE for feature method {METHOD}, k={k} is: {val_mse}")


if __name__ == '__main__':
    main()