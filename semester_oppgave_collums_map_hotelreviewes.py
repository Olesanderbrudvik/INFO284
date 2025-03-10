import os
import re
import pandas as pd
import hashlib
import numpy as np
from num2words import num2words

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Use relative paths so that the file "Hotel_Reviews.csv" is expected to be in the same directory
input_file = os.path.join(current_dir, "Hotel_Reviews.csv")
output_file = os.path.join(current_dir, "hotel_reviews_processed.csv")
filtered_file = os.path.join(current_dir, "hotel_reviews_filtered.csv")
summary_csv = os.path.join(current_dir, "5_reviews_summary.csv")

def replace_numbers_with_words(text: str) -> str:
    """
    Replace numbers in a text with their English word representation.
    The process is:
      1. Replace ordinal numbers (e.g. "30th" -> "thirtieth").
      2. Insert spaces between numbers and letters (e.g. "30pm" becomes "30 pm").
      3. Replace standalone numbers (including decimals) with words.
    """
    # 1. Replace ordinals (e.g. "30th" -> "thirtieth")
    def ordinal_repl(match):
        try:
            number = int(match.group(1))
            return num2words(number, ordinal=True)
        except Exception:
            return match.group(0)
    text = re.sub(r'\b(\d+)(st|nd|rd|th)\b', ordinal_repl, text)
    
    # 2. Insert spaces between numbers and letters to separate them.
    #    This handles cases like "30pm" or "240v" regardless of order.
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
    
    # 3. Replace standalone numbers (including decimals)
    def cardinal_repl(match):
        num_str = match.group(0)
        num_str_fixed = num_str.replace(',', '.')
        try:
            if '.' in num_str_fixed:
                integer_part, fractional_part = num_str_fixed.split('.')
                words = num2words(int(integer_part))
                fractional_words = " ".join(num2words(int(digit)) for digit in fractional_part)
                return f"{words} point {fractional_words}"
            else:
                number = int(num_str_fixed)
                return num2words(number)
        except Exception:
            return num_str
    text = re.sub(r'\b\d+([,.]\d+)?\b', cardinal_repl, text)
    return text

class HotelReviewProcessor:
    """
    A class for reading, cleaning, and organizing a large CSV file containing hotel reviews.
    The data is processed in chunks to reduce memory usage and saved to a new CSV file.
    """
    
    def __init__(self, input_file: str, output_file: str, chunk_size: int = 10000):
        self.input_file = input_file
        self.output_file = output_file
        self.chunk_size = chunk_size

    def reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Move the review columns ('negative_review' and 'positive_review') to the end.
        """
        review_cols = ['negative_review', 'positive_review']
        other_cols = [col for col in df.columns if col not in review_cols]
        new_order = other_cols + review_cols
        return df[new_order]

    def process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Process a chunk of data by:
          - Renaming columns to a consistent structure.
          - Converting dates to datetime.
          - Cleaning numeric columns and recalculating word counts.
          - Replacing missing review texts with an empty string.
          - Reorganizing columns so that review texts are at the end.
        """
        rename_map = {
            'Hotel_Address': 'address',
            'Additional_Number_of_Scoring': 'scoring_extra',
            'Review_Date': 'review_date',
            'Average_Score': 'avg_score',
            'Hotel_Name': 'hotel_name',
            'Reviewer_Nationality': 'reviewer_nationality',
            'Negative_Review': 'negative_review',
            'Review_Total_Negative_Word_Counts': 'neg_word_count',
            'Total_Number_of_Reviews': 'total_reviews',
            'Positive_Review': 'positive_review',
            'Review_Total_Positive_Word_Counts': 'pos_word_count',
            'Total_Number_of_Reviews_Reviewer_Has_Given': 'total_reviews_by_reviewer',
            'Reviewer_Score': 'reviewer_score',
            'Tags': 'tags',
            'days_since_review': 'days_since_review',
            'lat': 'latitude',
            'lng': 'longitude'
        }
        chunk.rename(columns=rename_map, inplace=True)
        
        # Clean numeric columns by removing commas and converting to numeric type.
        if 'neg_word_count' in chunk.columns:
            chunk['neg_word_count'] = chunk['neg_word_count'].astype(str).str.replace(',', '')
        if 'pos_word_count' in chunk.columns:
            chunk['pos_word_count'] = chunk['pos_word_count'].astype(str).str.replace(',', '')
        
        chunk['neg_word_count'] = pd.to_numeric(chunk['neg_word_count'], errors='coerce')
        chunk['pos_word_count'] = pd.to_numeric(chunk['pos_word_count'], errors='coerce')
        
        if 'review_date' in chunk.columns:
            chunk['review_date'] = pd.to_datetime(chunk['review_date'], errors='coerce')
        
        # Replace missing or empty review texts with an empty string.
        chunk['negative_review'] = chunk['negative_review'].fillna('').astype(str)
        chunk['positive_review'] = chunk['positive_review'].fillna('').astype(str)
        
        # Recalculate word counts from the review text.
        chunk['neg_word_count'] = chunk['negative_review'].apply(lambda x: len(str(x).split()))
        chunk['pos_word_count'] = chunk['positive_review'].apply(lambda x: len(str(x).split()))
        
        # Reorganize the columns so that review texts appear at the end.
        chunk = self.reorder_columns(chunk)
        return chunk

    def process_file(self) -> None:
        """
        Read the CSV file in chunks, process each chunk, and write the result to a new CSV file.
        The header is written only for the first chunk.
        """
        first_chunk = True
        with open(self.output_file, 'w', encoding='utf-8', newline='') as f_out:
            for chunk in pd.read_csv(self.input_file, chunksize=self.chunk_size):
                processed_chunk = self.process_chunk(chunk)
                processed_chunk.to_csv(f_out, index=False, header=first_chunk)
                first_chunk = False

def create_summary_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new column 'full_review' that aggregates all information for each review
    into a formatted text paragraph.
    """
    review_cols = ['negative_review', 'positive_review']
    other_cols = [col for col in df.columns if col not in review_cols]
    summary_list = []
    
    for i, (_, row) in enumerate(df.iterrows()):
        lines = []
        lines.append(f"Review {i + 1}:")
        for col in other_cols:
            lines.append(f"{col}: {row[col]}")
        lines.append("\n--- StringValue Review ---")
        for col in review_cols:
            lines.append(f"{col}:")
            lines.append(f"{row[col]}")
            lines.append("-" * 20)
        summary_text = "\n".join(lines)
        summary_list.append(summary_text)
    
    summary_df = pd.DataFrame({'full_review': summary_list})
    return summary_df

def count_nulls_per_column(file_path: str, chunk_size: int = 10000):
    """
    Count the number of null values for each column in the CSV file using chunked reading.
    """
    null_counts = {}
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk_nulls = chunk.isnull().sum().to_dict()
        for col, count in chunk_nulls.items():
            null_counts[col] = null_counts.get(col, 0) + count
    sorted_null_counts = dict(sorted(null_counts.items(), key=lambda item: item[1], reverse=True))
    return sorted_null_counts

def export_summary_csv(processed_file: str, summary_csv: str, nrows: int = 5):
    """
    Read the processed CSV file, create a summary view with a specified number of reviews (default is 5),
    and export it to a CSV file.
    """
    df = pd.read_csv(processed_file, nrows=nrows)
    summary_df = create_summary_column(df)
    summary_df.to_csv(summary_csv, index=False)
    print(f"{nrows} reviews have been exported to the CSV file: {summary_csv}")

def count_duplicates(file_path: str, chunk_size: int = 10000) -> int:
    """
    Count duplicate rows in the CSV file using a hashing approach.
    This function attempts to count duplicates even across different chunks.
    """
    seen_hashes = set()
    duplicate_count = 0
    
    def hash_row(row: pd.Series) -> str:
        row_bytes = row.to_json().encode('utf-8')
        return hashlib.md5(row_bytes).hexdigest()
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            row_hash = hash_row(row)
            if row_hash in seen_hashes:
                duplicate_count += 1
            else:
                seen_hashes.add(row_hash)
    return duplicate_count

def get_row_hash(row: pd.Series) -> str:
    """
    Generate an MD5 hash for a given DataFrame row.
    """
    row_bytes = row.to_json().encode('utf-8')
    return hashlib.md5(row_bytes).hexdigest()

def filter_processed_csv(processed_file: str, filtered_file: str, chunk_size: int = 10000) -> None:
    """
    Read the processed CSV file in chunks, filter out all columns except
    the specified ones ('negative_review', 'positive_review', 'pos_word_count', 'neg_word_count',
    'avg_score', 'reviewer_score'), and remove duplicate rows globally,
    keeping only the first occurrence of each duplicate.
    """
    columns_to_keep = ['negative_review', 'positive_review', 'pos_word_count', 'neg_word_count', 'avg_score', 'reviewer_score']
    seen_hashes = set()
    first_chunk = True
    
    for chunk in pd.read_csv(processed_file, chunksize=chunk_size):
        chunk_filtered = chunk[columns_to_keep]
        indices_to_keep = []
        for index, row in chunk_filtered.iterrows():
            row_hash = get_row_hash(row)
            if row_hash not in seen_hashes:
                seen_hashes.add(row_hash)
                indices_to_keep.append(index)
        chunk_unique = chunk_filtered.loc[indices_to_keep]
        if first_chunk:
            chunk_unique.to_csv(filtered_file, index=False, mode='w', header=True)
            first_chunk = False
        else:
            chunk_unique.to_csv(filtered_file, index=False, mode='a', header=False)


def count_reviews(file_path: str, chunk_size: int = 10000) -> tuple:
    """
    Read the processed CSV file in chunks and count the number of non-empty
    negative_review and positive_review entries.
    """
    neg_count = 0
    pos_count = 0
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        neg_count += chunk['negative_review'].apply(lambda x: 1 if str(x).strip() != '' else 0).sum()
        pos_count += chunk['positive_review'].apply(lambda x: 1 if str(x).strip() != '' else 0).sum()
    return neg_count, pos_count

def convert_reviews_to_lowercase_and_words(file_path: str, chunk_size: int = 10000) -> None:
    """
    Read the filtered CSV file in chunks, convert the text in 'negative_review' and 'positive_review'
    to lowercase and replace numbers with their word representation, then write the result back to the file.
    """
    temp_file = file_path + ".tmp"
    first_chunk = True
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        if 'negative_review' in chunk.columns:
            chunk['negative_review'] = chunk['negative_review'].str.lower().apply(replace_numbers_with_words)
        if 'positive_review' in chunk.columns:
            chunk['positive_review'] = chunk['positive_review'].str.lower().apply(replace_numbers_with_words)
        chunk.to_csv(temp_file, index=False, mode='w' if first_chunk else 'a', header=first_chunk)
        first_chunk = False
    os.replace(temp_file, file_path)

def calculate_word_count_stats(file_path: str, chunk_size: int = 10000) -> dict:
    """
    Calculate maximum, minimum, median, and average (mean) values for the 'pos_word_count' and 'neg_word_count' columns.
    
    This function analyzes the distribution of word counts in both positive and negative reviews.
    Knowing these statistics is necessary to determine the most appropriate normalization method for further processing.
    """
    pos_counts = []
    neg_counts = []
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        pos_counts.extend(chunk['pos_word_count'].tolist())
        neg_counts.extend(chunk['neg_word_count'].tolist())
    
    stats = {
        'pos_word_count': {
            'max': max(pos_counts),
            'min': min(pos_counts),
            'median': np.median(pos_counts),
            'mean': np.mean(pos_counts)
        },
        'neg_word_count': {
            'max': max(neg_counts),
            'min': min(neg_counts),
            'median': np.median(neg_counts),
            'mean': np.mean(neg_counts)
        }
    }
    return stats

def main():
    chunk_size = 10000

    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: The input file '{input_file}' does not exist.")
        print("Please ensure that the file is available before running this script.")
        return

    # Process the raw file into a processed CSV file
    processor = HotelReviewProcessor(input_file, output_file, chunk_size)
    processor.process_file()
    print("Data has been processed and saved to:", output_file)

    # Export a summary (5 reviews) to a CSV file with formatted text
    export_summary_csv(output_file, summary_csv, nrows=5)
    
    # Filter the processed CSV file to keep selected columns and remove duplicates
    filter_processed_csv(output_file, filtered_file, chunk_size)
    print("Filtered data (selected columns and deduplicated) has been saved to:", filtered_file)

    # Convert review texts to lowercase and replace numbers with words in the filtered file
    convert_reviews_to_lowercase_and_words(filtered_file, chunk_size)
    print("Review texts have been converted to lowercase and numbers replaced with words in:", filtered_file)

    # Count and print the number of null values per column
    null_counts = count_nulls_per_column(output_file, chunk_size)
    print("Null values per column:")
    for col, count in null_counts.items():
        print(f"{col}: {count}")

    # Count duplicate rows
    total_duplicates = count_duplicates(output_file, chunk_size)
    print(f"\nTotal duplicate rows in processed file: {total_duplicates}")

    # Count the number of non-empty negative and positive reviews
    neg_count, pos_count = count_reviews(output_file, chunk_size)
    print(f"\nTotal number of negative reviews: {neg_count}")
    print(f"Total number of positive reviews: {pos_count}")
    
    # Calculate and print word count statistics for 'pos_word_count' and 'neg_word_count'
    # We do this calculation to decide what kind of normalization we should use. 
    # Note: We use the filtered file since it is the final version used for AI model training.
    stats = calculate_word_count_stats(filtered_file, chunk_size)
    print("\nWord count statistics for 'pos_word_count':")
    print(f"Max: {stats['pos_word_count']['max']}")
    print(f"Min: {stats['pos_word_count']['min']}")
    print(f"Median: {stats['pos_word_count']['median']}")
    print(f"Average: {stats['pos_word_count']['mean']}")
    
    print("\nWord count statistics for 'neg_word_count':")
    print(f"Max: {stats['neg_word_count']['max']}")
    print(f"Min: {stats['neg_word_count']['min']}")
    print(f"Median: {stats['neg_word_count']['median']}")
    print(f"Average: {stats['neg_word_count']['mean']}")

if __name__ == '__main__':
    main()
