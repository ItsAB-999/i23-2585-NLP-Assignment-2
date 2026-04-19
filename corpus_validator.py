"""
NLP CORPUS VALIDATION UTILITY - Verification of dataset integrity and basic statistics.
This script ensures that all necessary files are present before the natural language processing
pipeline begins its execution.
"""

import os
import re
import sys


def SetupStandardOutputEncodingToUtf8():
    """
    Adjusts the standard output encoding to UTF-8. 
    This is critical for Windows environments where the default code page might 
    not support Urdu characters, preventing potential runtime errors during printing.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def CalculateTotalLogicalDocumentCount(ContentOfCorpusText):
    """
    Identifies and counts unique document markers within the cleaned corpus.
    Each distinct article in the 'cleaned.txt' file is preceded by a numerical 
    identifier in brackets, such as [1], [2], etc.
    """
    # Regex pattern to match the document start markers at the beginning of lines
    MarkerIdentificationPattern = r"^\[?(\d+)\]?\s*$"
    
    # Extract all occurrences that match the pattern across the entire text
    ListOfFoundMarkers = re.findall(MarkerIdentificationPattern, ContentOfCorpusText, flags=re.MULTILINE)
    
    # The number of matches corresponds to the total document count
    return len(ListOfFoundMarkers)


def RunCorpusSanityCheckSequence():
    """
    Primary execution thread for validating the corpus.
    1. Sets up the environment.
    2. Verifies the existence of mandatory files.
    3. Computes and displays corpus-wide statistics.
    """
    SetupStandardOutputEncodingToUtf8()

    # Define the set of files that must exist in the root directory for the pipeline to function
    EssentialDatasetFiles = ["cleaned.txt", "raw.txt", "Metadata.json"]
    
    for CurrentFileName in EssentialDatasetFiles:
        if not os.path.isfile(CurrentFileName):
            print(f"CRITICAL ERROR: The required dataset file '{CurrentFileName}' was not found.")
            sys.exit(1)

    # Load the entire purified corpus into a string variable for analysis
    try:
        with open("cleaned.txt", mode="r", encoding="utf-8") as PurifiedCorpusFileObject:
            FullStringRepresentationOfCorpus = PurifiedCorpusFileObject.read()
    except Exception as FileReadException:
        print(f"IO ERROR: Unable to read 'cleaned.txt'. Details: {FileReadException}")
        sys.exit(1)

    # Generate a list of tokens based on whitespace separation
    ListOfAllCorpusTokens = FullStringRepresentationOfCorpus.split()
    
    # Determine unique types by converting the list to a set
    SetOfUniqueLexicalTypes = set(ListOfAllCorpusTokens)

    # Aggregate statistical measurements
    FinalDocumentArchitectureCount = CalculateTotalLogicalDocumentCount(FullStringRepresentationOfCorpus)
    AggregateTokenFrequencyCount = len(ListOfAllCorpusTokens)
    DistinctVocabularySizeMetric = len(SetOfUniqueLexicalTypes)

    # Output the calculated metrics to the console for verification
    print("Aggregate Logical Document Count:", FinalDocumentArchitectureCount)
    print("Grand Total of Tokens in Corpus:", AggregateTokenFrequencyCount)
    print("Unique Vocabulary Size (Type Count):", DistinctVocabularySizeMetric)


if __name__ == "__main__":
    # Initiate the sanity check process
    RunCorpusSanityCheckSequence()
