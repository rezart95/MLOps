import argparse
import pandas as pd

def parse_args():    
    parser = argparse.ArgumentParser()    
    parser.add_argument("--input-path", dest="input_path", required=True)    
    parser.add_argument("--output-dataset", dest="output_dataset", required=True)    
    
    return parser.parse_args()
    
    
if __name__ == "__main__":    
    args = parse_args()    
    
    # get data from run context    
    df = pd.read_csv(args.input_path)    
    
    ##### YOUR CODE HERE #####    
    # Hint: Wrap your pandas functions in print() statements so that the output will be captured in AML logs    
    # The head() function will provide a slice of the dataframe
    print(df.head())
                                    
    # The describe() function will provide a summary statistics of 
    # the columns in the dataframe
    print(df.describe())
    # ##### YOUR CODE HERE #####   
     
    df.to_csv(args.output_dataset, index=False)
