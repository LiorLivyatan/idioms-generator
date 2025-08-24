#!/usr/bin/env python3
"""
Main script for Confusing Context Variant Generator

This script initializes/installs dependencies and runs the samples generator
with user-configurable options.
"""

import os
import sys
import subprocess
import random
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

def install_requirements():
    """Install required packages from requirements.txt"""
    print("🔧 Installing dependencies...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ["GITHUB_TOKEN_MODEL", "GITHUB_BASE_URL"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        print("Example .env file:")
        print("GITHUB_TOKEN_MODEL=your_github_token")
        print("GITHUB_BASE_URL=https://models.inference.ai.azure.com")
        return False
    
    return True

def check_input_file():
    """Check if the input TSV file exists"""
    tsv_file = "./test_english.tsv"
    if not os.path.exists(tsv_file):
        print(f"❌ Input file not found: {tsv_file}")
        print("Please ensure test_english.tsv is in the current directory.")
        return False
    
    print(f"✅ Input file found: {tsv_file}")
    return True

def get_user_input():
    """Get user configuration for the variant generation"""
    print("\n📋 Configuration Options:")
    print("=" * 40)
    
    # Number of variants per sentence
    while True:
        try:
            num_variants = input("Number of variants per sentence (default: 3): ").strip()
            if not num_variants:
                num_variants = 3
            else:
                num_variants = int(num_variants)
            if num_variants < 1:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Number of sentences to process
    while True:
        try:
            max_sentences_input = input("Number of sentences to process (default: all): ").strip()
            if not max_sentences_input or max_sentences_input.lower() == "all":
                max_sentences = None
            else:
                max_sentences = int(max_sentences_input)
                if max_sentences < 1:
                    print("Please enter a positive number or 'all'.")
                    continue
            break
        except ValueError:
            print("Please enter a valid number or 'all'.")
    
    # Random or first sentences
    if max_sentences:
        while True:
            random_choice = input("Take random sentences? (y/n, default: n): ").strip().lower()
            if random_choice in ['', 'n', 'no']:
                use_random = False
                break
            elif random_choice in ['y', 'yes']:
                use_random = True
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")
    else:
        use_random = False
    
    # Output format
    while True:
        output_format = input("Output format (json/csv/pickle, default: json): ").strip().lower()
        if not output_format:
            output_format = "json"
        if output_format in ["json", "csv", "pickle"]:
            break
        else:
            print("Please choose from: json, csv, pickle")
    
    return num_variants, max_sentences, use_random, output_format

def run_samples_generator(num_variants: int, max_sentences: Optional[int], 
                         use_random: bool, output_format: str):
    """Run the samples generator with the specified configuration"""
    
    # Import the samples generator module
    try:
        from samples_generator import generate_variants_dataframe, save_variants_to_file, display_sample_variants
        import pandas as pd
        from datetime import datetime
    except ImportError as e:
        print(f"❌ Error importing samples_generator: {e}")
        print("Make sure all dependencies are installed.")
        return False
    
    # Read the TSV file to get sentence count for random selection
    if use_random and max_sentences:
        try:
            from samples_generator import read_bio_tsv
            original_df = read_bio_tsv("./test_english.tsv")
            total_sentences = len(original_df)
            
            if max_sentences >= total_sentences:
                print(f"Requested {max_sentences} sentences, but only {total_sentences} available. Using all.")
                max_sentences = None
                use_random = False
            else:
                # Select random indices
                random_indices = sorted(random.sample(range(total_sentences), max_sentences))
                original_df = original_df.iloc[random_indices].reset_index(drop=True)
                
                # Save temporary file with selected sentences
                temp_file = "./temp_random_sentences.tsv"
                
                # Convert back to TSV format
                with open(temp_file, 'w', encoding='utf-8') as f:
                    for _, row in original_df.iterrows():
                        tokens = row['tokens']
                        tags = row['tags']
                        for token, tag in zip(tokens, tags):
                            f.write(f"{token}\t{tag}\n")
                        f.write("\n")  # Empty line between sentences
                
                tsv_file_path = temp_file
                print(f"✅ Selected {max_sentences} random sentences")
        except Exception as e:
            print(f"❌ Error selecting random sentences: {e}")
            return False
    else:
        tsv_file_path = "./test_english.tsv"
    
    # Generate variants
    try:
        print(f"\n🚀 Starting variant generation...")
        print(f"  - Variants per sentence: {num_variants}")
        print(f"  - Sentences to process: {max_sentences if max_sentences else 'All'}")
        print(f"  - Selection method: {'Random' if use_random else 'Sequential'}")
        print(f"  - Output format: {output_format}")
        print("-" * 50)
        
        variants_df = generate_variants_dataframe(
            tsv_file_path=tsv_file_path,
            num_variants=num_variants,
            max_sentences=max_sentences if not use_random else None
        )
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"variants_output_{timestamp}.{output_format}"
        
        # Save results
        print(f"\n💾 Saving results to {output_filename}...")
        save_variants_to_file(variants_df, output_filename, output_format)
        
        # Display sample results
        display_sample_variants(variants_df, num_samples=2)
        
        # Clean up temporary file if created
        if use_random and max_sentences and os.path.exists("./temp_random_sentences.tsv"):
            os.remove("./temp_random_sentences.tsv")
        
        print(f"\n✅ Process completed successfully!")
        print(f"Results saved to: {output_filename}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during variant generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🎯 Confusing Context Variant Generator - Main Script")
    print("=" * 60)
    
    # Step 1: Install dependencies
    print("\n📦 Step 1: Installing dependencies")
    install_requirements()
    
    # Step 2: Check environment
    print("\n🔍 Step 2: Checking environment")
    if not check_environment():
        sys.exit(1)
    
    # Step 3: Check input file
    print("\n📄 Step 3: Checking input file")
    if not check_input_file():
        sys.exit(1)
    
    # Step 4: Get user configuration
    print("\n⚙️  Step 4: User configuration")
    num_variants, max_sentences, use_random, output_format = get_user_input()
    
    # Step 5: Run the generator
    print("\n🔄 Step 5: Running variant generator")
    success = run_samples_generator(num_variants, max_sentences, use_random, output_format)
    
    if success:
        print("\n🎉 All done! Your variants have been generated successfully.")
    else:
        print("\n💥 Something went wrong. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)