import pandas as pd
import hashlib

class HotelReviewProcessor:
    """
    A class for reading, cleaning, and organizing a large CSV file containing hotel reviews.
    The data is processed in chunks to reduce memory usage and saved to a new CSV file.
    """
    
    def __init__(self, input_file: str, output_file: str, chunk_size: int = 10000):
        """
        Initializes the object with file paths and chunk size.
        
        :param input_file: The path to the original CSV file.
        :param output_file: The path to the processed CSV file.
        :param chunk_size: The number of rows to process per chunk.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.chunk_size = chunk_size

    def reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Moves the review columns ('negative_review' and 'positive_review') to the end.
        
        :param df: DataFrame with all columns.
        :return: DataFrame with the review columns moved to the end.
        """
        review_cols = ['negative_review', 'positive_review']
        other_cols = [col for col in df.columns if col not in review_cols]
        new_order = other_cols + review_cols
        return df[new_order]

    def process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a chunk of data by:
         - Renaming columns to a consistent structure.
         - Converting dates to datetime.
         - Reorganizing columns so that the review texts are at the end.
        
        Note: This version does not drop duplicates or fill null values.
        Future enhancements can add functionality to search for duplicates or null values.
        
        :param chunk: A Pandas DataFrame representing a chunk of data.
        :return: The processed chunk.
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
        
        # Future improvement: Search for duplicate rows without dropping them.
        # duplicates = chunk[chunk.duplicated()]
        # if not duplicates.empty:
        #     print("Found duplicates in this chunk:")
        #     print(duplicates)
        
        # Future improvement: Search for null values in review columns without modifying them.
        # null_reviews = chunk[chunk['negative_review'].isnull() | chunk['positive_review'].isnull()]
        # if not null_reviews.empty:
        #     print("Found null review values in this chunk:")
        #     print(null_reviews)
                
        if 'review_date' in chunk.columns:
            chunk['review_date'] = pd.to_datetime(chunk['review_date'], errors='coerce')
        
        # Reorganize the columns so that review texts appear at the end
        chunk = self.reorder_columns(chunk)
        return chunk

    def process_file(self) -> None:
        """
        Reads the CSV file in chunks, processes each chunk, and writes the result to a new CSV file.
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
    Creates a new column 'full_review' that aggregates all information for each review
    into a formatted text paragraph.
    
    :param df: DataFrame with the processed data.
    :return: A new DataFrame containing only the 'full_review' column.
    """
    review_cols = ['negative_review', 'positive_review']
    other_cols = [col for col in df.columns if col not in review_cols]
    summary_list = []
    
    for idx, row in df.iterrows():
        lines = []
        lines.append(f"Review {idx + 1}:")
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

def export_summary_csv(processed_file: str, summary_csv: str, nrows: int = 5):
    """
    Reads the processed CSV file, creates a summary view with a specified number of reviews (default is 5),
    and exports it to a CSV file.
    
    :param processed_file: File path to the processed CSV file.
    :param summary_csv: File path to the new, summarized CSV file.
    :param nrows: Number of reviews to include.
    """
    df = pd.read_csv(processed_file, nrows=nrows)
    summary_df = create_summary_column(df)
    summary_df.to_csv(summary_csv, index=False)
    print(f"{nrows} reviews have been exported to the CSV file: {summary_csv}")

def count_null_values(file_path: str, chunk_size: int = 10000) -> int:
    """
    Counts the total number of null values in the CSV file using chunked reading.
    
    :param file_path: Path to the CSV file.
    :param chunk_size: Number of rows to process per chunk.
    :return: Total count of null values in the file.
    """
    total_nulls = 0
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        total_nulls += chunk.isnull().sum().sum()
    return total_nulls

def count_duplicates(file_path: str, chunk_size: int = 10000) -> int:
    """
    Counts duplicate rows in the CSV file using a hashing approach.
    This function attempts to count duplicates even across different chunks.
    
    :param file_path: Path to the CSV file.
    :param chunk_size: Number of rows to process per chunk.
    :return: Total count of duplicate rows.
    """
    seen_hashes = set()
    duplicate_count = 0
    
    def hash_row(row: pd.Series) -> str:
        # Create a hash for the row using a stable encoding
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

def main():
    input_file = 'hotel_reviews.csv'               # Original CSV file
    output_file = 'hotel_reviews_processed.csv'      # Processed CSV file
    summary_csv = '5_reviews_summary.csv'            # New CSV file with 5 summarized reviews
    chunk_size = 10000

    # Process the large file
    processor = HotelReviewProcessor(input_file, output_file, chunk_size)
    processor.process_file()
    print("Data has been processed and saved to:", output_file)
    
    # Export a summary (5 reviews) to a CSV file with formatted text
    export_summary_csv(output_file, summary_csv, nrows=5)
    
    # Count null values and duplicate rows
    total_nulls = count_null_values(output_file, chunk_size)
    total_duplicates = count_duplicates(output_file, chunk_size)
    print("Total null values in processed file:", total_nulls)
    print("Total duplicate rows in processed file:", total_duplicates)

if __name__ == '__main__':
    main()


