import random
import pandas as pd
import numpy as np
import os

# Function to normalize ratings
def normRating(original_csv_path):
    df = pd.read_csv(original_csv_path)
    df["norm"] = df["rating"] / 5
    return df

# Function to combine book IDs with their normalized ratings
def combineBooksWithRates(df):
    # Create a new column 'book_rating' which is a list of [book_id, norm]
    df["book_rating"] = df[["book_id", "norm"]].values.tolist()
    return df

# Function to transpose the DataFrame, grouping by 'user_id'
def transpose(df):
    # Group by 'user_id' and aggregate 'book_rating' into lists
    grouped_df = (
        df.groupby("user_id")["book_rating"]
        .apply(list)
        .reset_index(name="book_rating_list")
    )
    return grouped_df

# Function to split a list randomly into two sublists (80% and 20%)
def split_list_randomly(items):
    # Shuffle the items randomly
    random.shuffle(items)
    
    # Determine the split index
    if len(items) > 4:
        split_index = int(0.8 * len(items))  # 80% split if more than 4 items
    elif len(items) > 1:
        split_index = len(items) - 1  # Leave one item for test if more than 1 item
        print("User with low number of readings")
    
    # Split the list into two sublists
    list_80 = items[:split_index]
    list_20 = items[split_index:]
    
    return [list_80, list_20]

# Function to split the DataFrame into train and test sets
def split(df, train_txt_path, test_txt_path, ratings_txt_path, small=False):
    # Open train, test, and ratings files for writing
    with open(train_txt_path, "w") as train, open(test_txt_path, "w") as test, open(ratings_txt_path, "w") as ratings:
        # Determine the size of the data to process
        size = 5000 if small else len(df.index)
        
        for i in range(0, size):
            user = str(df.at[i, "user_id"])
            items = df.at[i, "book_rating_list"]
            
            if len(items) > 2:
                # Split the items into training and testing sets
                random_lists = split_list_randomly(items)
                
                # Extract book IDs and ratings for training set
                books_train = [sublist[0] for sublist in random_lists[0]]
                int_books_train = [int(s) for s in books_train]
                ratings_train = [sublist[1] for sublist in random_lists[0]]
                
                # Extract book IDs for testing set
                books_test = [sublist[0] for sublist in random_lists[1]]
                int_books_test = [int(s) for s in books_test]
                
                # Write the training data to the train file
                k = user + " " + " ".join(map(str, int_books_train))
                train.write(k)
                train.write("\n")
                
                # Write the testing data to the test file
                k = user + " " + " ".join(map(str, int_books_test))
                test.write(k)
                test.write("\n")
                
                # Write the ratings data to the ratings file
                ratings.write(", " + ", ".join(map(str, ratings_train)))
        
        # Ensure all data is written to the files
        train.flush()
        test.flush()
        ratings.flush()

def split_dual(df, train_txt_path, test_txt_path, ratings_txt_path, small=False):
    # Open train, test, and ratings files for writing
    with open(train_txt_path, "w") as train, open(test_txt_path, "w") as test, open(ratings_txt_path, "w") as ratings:
        # Determine the size of the data to process
        size = 5000 if small else len(df.index)
        
        for i in range(0, size):
            user = str(df.at[i, "user_id"])
            items = df.at[i, "book_rating_list"]
            
            if len(items) > 2:
                # Split the items into training and testing sets
                random_lists = split_list_randomly(items)
                
                # Extract book IDs and ratings for training set
                books_train = [sublist[0] for sublist in random_lists[0]]
                int_books_train = [int(s) for s in books_train]
                ratings_train = [sublist[1] for sublist in random_lists[0]]
                
                # Extract book IDs for testing set
                books_test = [sublist[0] for sublist in random_lists[1]]
                int_books_test = [int(s) for s in books_test]
                
                # Write the training data to the train file
                k = user + " " + " ".join(map(str, int_books_train))
                train.write(k)
                train.write("\n")
                
                # Write the testing data to the test file
                k = user + " " + " ".join(map(str, int_books_test))
                test.write(k)
                test.write("\n")
                
                # Write the ratings data to the ratings file
                ratings.write(", " + ", ".join(map(str, ratings_train)))
        
        # Ensure all data is written to the files
        train.flush()
        test.flush()
        ratings.flush()

def split_dual(df, threshold, train_good_txt, test_good_txt, ratings_good_txt, train_bad_txt, test_bad_txt, ratings_bad_txt, small=False):
    # Create empty lists to store good and bad interactions
    user_good_ratings = []
    user_bad_ratings = []
    
    # Iterate over each user and split their ratings into good and bad
    for user_id, user_df in df.groupby("user_id"):
        good_ratings = user_df[user_df['rating'] >= threshold][["book_id", "norm"]].values.tolist()
        bad_ratings = user_df[user_df['rating'] < threshold][["book_id", "norm"]].values.tolist()
        
        if good_ratings:
            user_good_ratings.append({"user_id": user_id, "book_rating_list": good_ratings})
        if bad_ratings:
            user_bad_ratings.append({"user_id": user_id, "book_rating_list": bad_ratings})
    
    # Convert lists to DataFrames
    df_good = pd.DataFrame(user_good_ratings)
    df_bad = pd.DataFrame(user_bad_ratings)
    
    # Process good ratings
    split(df_good, train_good_txt, test_good_txt, ratings_good_txt, small)
    
    # Process bad ratings
    split(df_bad, train_bad_txt, test_bad_txt, ratings_bad_txt, small)

# Main script execution
dual = False                # Set to False for non-dual case
small = True                # Flag to determine if we process a small subset of data
file_path = 'ratings.csv'   # Path to the ratings CSV file
threshold = 3               # Rating threshold (used in dual case)

if dual:
    # Dual case: Split data into good and bad ratings based on threshold
    df = pd.read_csv(file_path)

else:
    # Non-dual case: Normalize ratings, combine with books, and transpose data
    df = normRating(file_path)
    df = combineBooksWithRates(df)
    df = transpose(df)
    # Uncomment the following line to split the data into train and test sets
    split(df, "train_medium.txt", "test_medium.txt", "ratings_medium.txt", small=small)



