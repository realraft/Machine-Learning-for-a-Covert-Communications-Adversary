import gzip
import shutil
import argparse
import os

def compress_file(input_path, output_path=None):
    if output_path is None:
        output_path = input_path + '.gz'
        
    print(f"Compressing {input_path} to {output_path}...")
    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    orig_size = os.path.getsize(input_path) / (1024*1024)
    comp_size = os.path.getsize(output_path) / (1024*1024)
    print(f"Done! Size reduced from {orig_size:.2f} MB to {comp_size:.2f} MB")
    print(f"Compression ratio: {orig_size/comp_size:.2f}x")

def decompress_file(input_path, output_path=None):
    if output_path is None:
        if input_path.endswith('.gz'):
            output_path = input_path[:-3]
        else:
            output_path = input_path + '.uncompressed'
            
    print(f"Decompressing {input_path} to {output_path}...")
    with gzip.open(input_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick CSV compression utility")
    parser.add_argument('-d', '--decompress', action='store_true', help="Decompress instead of compress")
    parser.add_argument('-i', '--input', type=str, default='../data/data.csv', help="Input file path")
    parser.add_argument('-o', '--output', type=str, default=None, help="Output file path (optional)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
    elif args.decompress:
        decompress_file(args.input, args.output)
    else:
        compress_file(args.input, args.output)
