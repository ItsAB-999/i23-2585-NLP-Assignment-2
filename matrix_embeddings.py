"""
CORE MATRIX EMBEDDING PIPELINE - TF-IDF, PPMI, and Dimensionality Reduction (t-SNE).
This module processes the cleaned corpus to generate sparse and dense vector representations 
of documents and words. It also visualizes the lexical space using t-SNE.
"""

import os
import re
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict


def PrepareStandardUtf8OutputSystem():
    """
    Stabilizes the output stream by forcing UTF-8 encoding.
    Ensures that Urdu script renders correctly without crashing 
    the system console on Windows.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def SegmentCorpusIntoArticleDictionary(SourceFilePath):
    """
    Parses the 'cleaned.txt' file and separates it into individual articles.
    Returns a dictionary where keys are integer Article IDs and values are 
    lists of whitespace-separated tokens.
    """
    try:
        with open(SourceFilePath, mode="r", encoding="utf-8") as InputDataStream:
            AllRawLinesFromCorpus = InputDataStream.read().splitlines()
    except Exception as ErrorOpeningFile:
        print(f"FAILED TO OPEN CORPUS FILE: {ErrorOpeningFile}")
        return {}

    DictionaryMappingDocsToTokens = {}
    TrackedActiveDocumentIdentifier = None
    
    for CurrentRawLineContent in AllRawLinesFromCorpus:
        StrippedLineText = CurrentRawLineContent.strip()
        # Look for [N] bracketed markers indicating the start of a document
        ArticleDelimiterMatch = re.match(r"^\[?(\d+)\]?\s*$", StrippedLineText)
        
        if ArticleDelimiterMatch:
            TrackedActiveDocumentIdentifier = int(ArticleDelimiterMatch.group(1))
            DictionaryMappingDocsToTokens[TrackedActiveDocumentIdentifier] = []
        elif TrackedActiveDocumentIdentifier is not None:
            DictionaryMappingDocsToTokens[TrackedActiveDocumentIdentifier].append(CurrentRawLineContent)

    FinalTokenizedDocumentStorage = {}
    for DocumentKeyID, ListOfLineFragments in DictionaryMappingDocsToTokens.items():
        ConcatenatedBodyParagraph = " ".join(ListOfLineFragments)
        FinalTokenizedDocumentStorage[DocumentKeyID] = ConcatenatedBodyParagraph.split()
        
    return FinalTokenizedDocumentStorage


def ConstructCappedVocabularyWithUnknownSlot(TokenizedArticleStorage, VocabularySizeLimit):
    """
    Analyzes token frequencies and retains the top most frequent words.
    Adds a special '<UNK>' token for out-of-vocabulary terms.
    """
    GlobalWordFrequencyCounter = Counter()
    for ListOfTokensInOneDoc in TokenizedArticleStorage.values():
        GlobalWordFrequencyCounter.update(ListOfTokensInOneDoc)

    StringForUnknownTypes = "<UNK>"
    # Select the most common words up to the requested limit
    ListOfTopWordsInOrder = [LexicalItem for LexicalItem, _count in GlobalWordFrequencyCounter.most_common(VocabularySizeLimit)]
    SetOfTopWordsForFastLookup = set(ListOfTopWordsInOrder)

    MappingFromWordToIndex = {LexicalItem: IndexPosition for IndexPosition, LexicalItem in enumerate(ListOfTopWordsInOrder)}
    MappingFromWordToIndex[StringForUnknownTypes] = len(ListOfTopWordsInOrder)
    
    OrderedListOfVocabularyStrings = ListOfTopWordsInOrder + [StringForUnknownTypes]
    
    return MappingFromWordToIndex, OrderedListOfVocabularyStrings, SetOfTopWordsForFastLookup


def ConvertSurfaceTokensToIntegerIndices(RawTokenSequence, WordToIndexMap, ValidTopWordSet):
    """
    Transforms a list of strings into a list of integer IDs based on the vocabulary.
    Items not in the top word set are mapped to the index of '<UNK>'.
    """
    IntegerValueOfUnknownToken = WordToIndexMap["<UNK>"]
    return [WordToIndexMap[CurrentToken] if CurrentToken in ValidTopWordSet else IntegerValueOfUnknownToken for CurrentToken in RawTokenSequence]


def ExecuteMatrixGenerationPipeline():
    """
    Manages the end-to-end workflow for Part 1 of the assignment:
    - Load corpus and metadata.
    - Generate TF-IDF matrix.
    - Generate PPMI matrix.
    - Plot t-SNE clusters.
    - Find semantic neighbors.
    """
    PrepareStandardUtf8OutputSystem()
    
    # Ensure the directory for storing output embeddings exists
    os.makedirs("embeddings", exist_ok=True)

    # Basic file presence validation
    if not os.path.isfile("cleaned.txt") or not os.path.isfile("Metadata.json"):
        print("MISSING NECESSARY INPUT FILES (cleaned.txt or Metadata.json). ABORTING.")
        sys.exit(1)

    with open("Metadata.json", mode="r", encoding="utf-8") as MetadataFileHandle:
        DictionaryOfArticleMetadata = json.load(MetadataFileHandle)

    FullCollectionOfDocuments = SegmentCorpusIntoArticleDictionary("cleaned.txt")
    TotalNumberOfDocumentsFound = len(FullCollectionOfDocuments)
    MaxThresholdForKnownVocabulary = 10000

    MappingFromWordsToIDs, CompleteSortedVocabulary, SetOfRecognizedKeywords = ConstructCappedVocabularyWithUnknownSlot(
        FullCollectionOfDocuments, MaxThresholdForKnownVocabulary
    )
    TotalLengthOfVocabulary = len(CompleteSortedVocabulary)

    # Phase 1: TF-IDF Construction
    # Initialize a sparse-style matrix for raw counts
    TfidfComputationMatrix = np.zeros((TotalNumberOfDocumentsFound, TotalLengthOfVocabulary), dtype=np.float64)
    TrackingDocumentFrequencyOfEachType = np.zeros(TotalLengthOfVocabulary, dtype=np.float64)
    SortedListOfDocumentKeys = sorted(FullCollectionOfDocuments.keys())

    for RowPositionIndex, DocumentIDKey in enumerate(SortedListOfDocumentKeys):
        SequenceOfTokensInThisDoc = FullCollectionOfDocuments[DocumentIDKey]
        SequenceOfIntegerIndices = ConvertSurfaceTokensToIntegerIndices(
            SequenceOfTokensInThisDoc, MappingFromWordsToIDs, SetOfRecognizedKeywords
        )
        PerDocumentFrequencyDistribution = Counter(SequenceOfIntegerIndices)
        
        for LexicalTypeIndex, ObservationCount in PerDocumentFrequencyDistribution.items():
            TfidfComputationMatrix[RowPositionIndex, LexicalTypeIndex] = float(ObservationCount)
        
        for LexicalTypeIndex in PerDocumentFrequencyDistribution.keys():
            TrackingDocumentFrequencyOfEachType[LexicalTypeIndex] += 1.0

    # Calculate Inverse Document Frequency: log( N / (1 + df(w)) )
    InverseDocumentFrequencyVector = np.log(TotalNumberOfDocumentsFound / (1.0 + TrackingDocumentFrequencyOfEachType))
    TfidfComputationMatrix = TfidfComputationMatrix * InverseDocumentFrequencyVector
    
    # Persist the TF-IDF matrix to disk
    np.save("embeddings/tfidf_matrix.npy", TfidfComputationMatrix.astype(np.float32))

    # Save the word-to-index mapping for later use in other scripts
    with open("embeddings/word2idx.json", mode="w", encoding="utf-8") as JsonOutputPointer:
        json.dump(MappingFromWordsToIDs, JsonOutputPointer, ensure_ascii=False)

    # Analyze categories per document for reporting
    MappingCategoryToArticleIndices = defaultdict(list)
    for DocumentIDKey in SortedListOfDocumentKeys:
        StringVersionOfKey = str(DocumentIDKey)
        if StringVersionOfKey not in DictionaryOfArticleMetadata:
            continue
        AssignedCategoryLabel = DictionaryOfArticleMetadata[StringVersionOfKey].get("category", "general")
        MappingCategoryToArticleIndices[AssignedCategoryLabel].append(SortedListOfDocumentKeys.index(DocumentIDKey))

    print("\nPROMINENT DISCRIMINATIVE KEYWORDS PER CATEGORY (Top 10 by mean TF-IDF):")
    for CategoryTitle in sorted(MappingCategoryToArticleIndices.keys()):
        IndicesOfArticlesInCategory = MappingCategoryToArticleIndices[CategoryTitle]
        if not IndicesOfArticlesInCategory:
            continue
        AverageTfidfScoresForThisCategory = TfidfComputationMatrix[IndicesOfArticlesInCategory].mean(axis=0)
        IndicesOfTopTenWords = np.argsort(-AverageTfidfScoresForThisCategory)[:10]
        StringsOfTopTenWords = [CompleteSortedVocabulary[Idx] for Idx in IndicesOfTopTenWords]
        print(f"{CategoryTitle}: {' '.join(StringsOfTopTenWords)}")

    # Phase 2: Co-occurrence and PPMI
    # Construct a Symmetric Window Co-occurrence Matrix
    CoOccurrenceFrequencyMatrix = np.zeros((TotalLengthOfVocabulary, TotalLengthOfVocabulary), dtype=np.float32)
    NeighborSearchWindowRadius = 5
    
    for DocumentIDKey in SortedListOfDocumentKeys:
        SequenceOfTokensInThisDoc = FullCollectionOfDocuments[DocumentIDKey]
        SequenceOfIntegerIndices = ConvertSurfaceTokensToIntegerIndices(
            SequenceOfTokensInThisDoc, MappingFromWordsToIDs, SetOfRecognizedKeywords
        )
        DocumentSequenceLength = len(SequenceOfIntegerIndices)
        
        for CenterWordPosition in range(DocumentSequenceLength):
            LowerBoundOfWindow = max(0, CenterWordPosition - NeighborSearchWindowRadius)
            UpperBoundOfWindow = min(DocumentSequenceLength, CenterWordPosition + NeighborSearchWindowRadius + 1)
            
            for ContextWordPosition in range(LowerBoundOfWindow, UpperBoundOfWindow):
                if ContextWordPosition == CenterWordPosition:
                    continue
                LeftTypeID = SequenceOfIntegerIndices[CenterWordPosition]
                RightTypeID = SequenceOfIntegerIndices[ContextWordPosition]
                CoOccurrenceFrequencyMatrix[CenterWordPosition, ContextWordPosition] += 1.0

    # Ensure symmetry and remove diagonal self-associations
    CoOccurrenceFrequencyMatrix = (CoOccurrenceFrequencyMatrix + CoOccurrenceFrequencyMatrix.T) * np.float32(0.5)
    np.fill_diagonal(CoOccurrenceFrequencyMatrix, 0.0)

    # Compute Positive Pointwise Mutual Information (PPMI)
    AggregateCoOccurrenceMassValue = float(CoOccurrenceFrequencyMatrix.sum())
    if AggregateCoOccurrenceMassValue <= 0:
        AggregateCoOccurrenceMassValue = 1.0

    FrequencyOfRowsAccumulator = CoOccurrenceFrequencyMatrix.sum(axis=1)
    FrequencyOfColsAccumulator = CoOccurrenceFrequencyMatrix.sum(axis=0)
    MinimalNonZeroEpsilonLimit = np.float32(1e-12)
    DenominativeProbabilityGridAcrossVocabulary = np.maximum(
        np.outer(FrequencyOfRowsAccumulator, FrequencyOfColsAccumulator), 
        MinimalNonZeroEpsilonLimit
    )

    # Perform memory-efficient in-place PPMI calculations
    np.multiply(CoOccurrenceFrequencyMatrix, np.float32(AggregateCoOccurrenceMassValue), out=CoOccurrenceFrequencyMatrix)
    np.divide(CoOccurrenceFrequencyMatrix, DenominativeProbabilityGridAcrossVocabulary, out=CoOccurrenceFrequencyMatrix)
    del DenominativeProbabilityGridAcrossVocabulary
    
    np.maximum(CoOccurrenceFrequencyMatrix, MinimalNonZeroEpsilonLimit, out=CoOccurrenceFrequencyMatrix)
    np.log2(CoOccurrenceFrequencyMatrix, out=CoOccurrenceFrequencyMatrix)
    np.maximum(CoOccurrenceFrequencyMatrix, np.float32(0.0), out=CoOccurrenceFrequencyMatrix)
    
    PositivePointwiseMutualInformationMatrix = CoOccurrenceFrequencyMatrix
    np.save("embeddings/ppmi_matrix.npy", PositivePointwiseMutualInformationMatrix)

    # Phase 3: Dimensionality Reduction for Visualization
    FrequencyOfEveryLexicalType = Counter()
    for DocumentIDKey in SortedListOfDocumentKeys:
        FrequencyOfEveryLexicalType.update(
            ConvertSurfaceTokensToIntegerIndices(FullCollectionOfDocuments[DocumentIDKey], MappingFromWordsToIDs, SetOfRecognizedKeywords)
        )
    
    IndicesOfTopTwoHundredCommonTypes = [TypeID for TypeID, _freq in FrequencyOfEveryLexicalType.most_common(200)]
    SubsetOfMatrixForVisualEmbedding = np.ascontiguousarray(
        PositivePointwiseMutualInformationMatrix[IndicesOfTopTwoHundredCommonTypes], 
        dtype=np.float32
    )

    try:
        from sklearn.manifold import TSNE
        OptimizationEngineForTSNE = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
        TwoDimensionalRepresentationOfEmbeddings = OptimizationEngineForTSNE.fit_transform(SubsetOfMatrixForVisualEmbedding)
    except ImportError:
        # SVD fallback if sklearn is absent
        NormalizedCenteringMatrix = SubsetOfMatrixForVisualEmbedding - SubsetOfMatrixForVisualEmbedding.mean(axis=0)
        LeftSingularVectors, SingularValuesTable, TransposedRightSingularVectors = np.linalg.svd(NormalizedCenteringMatrix, full_matrices=False)
        TwoDimensionalRepresentationOfEmbeddings = LeftSingularVectors[:, :2] * SingularValuesTable[:2]

    # Assign labels based on majority category for each type
    TaggingTypeToCategoryDistribution = {}
    for DocumentIDKey in SortedListOfDocumentKeys:
        MetadataKeyString = str(DocumentIDKey)
        DominantCategoryOfDoc = DictionaryOfArticleMetadata.get(MetadataKeyString, {}).get("category", "general")
        IntegerSequenceInDoc = ConvertSurfaceTokensToIntegerIndices(
            FullCollectionOfDocuments[DocumentIDKey], MappingFromWordsToIDs, SetOfRecognizedKeywords
        )
        for WordTypeID in IntegerSequenceInDoc:
            TaggingTypeToCategoryDistribution.setdefault(WordTypeID, Counter())[DominantCategoryOfDoc] += 1

    OrderedLabelsForPlottedPoints = []
    for WordTypeID in IndicesOfTopTwoHundredCommonTypes:
        CountsOfCategoriesForThisType = TaggingTypeToCategoryDistribution.get(WordTypeID, Counter())
        if CountsOfCategoriesForThisType:
            MajorityCategoryWinner = CountsOfCategoriesForThisType.most_common(1)[0][0]
        else:
            MajorityCategoryWinner = "general"
        OrderedLabelsForPlottedPoints.append(MajorityCategoryWinner)

    # Render t-SNE plot
    SetOfUniqueCategoriesInPlot = sorted(set(OrderedLabelsForPlottedPoints))
    PrimaryFigureObject, AxisLayoutObject = plt.subplots(figsize=(10, 8))
    
    for CurrentCategoryTitle in SetOfUniqueCategoriesInPlot:
        FilterMaskForCurrentCategory = np.array([OrderedLabelsForPlottedPoints[i] == CurrentCategoryTitle for i in range(len(IndicesOfTopTwoHundredCommonTypes))])
        AxisLayoutObject.scatter(
            TwoDimensionalRepresentationOfEmbeddings[FilterMaskForCurrentCategory, 0], 
            TwoDimensionalRepresentationOfEmbeddings[FilterMaskForCurrentCategory, 1], 
            s=14, alpha=0.75, label=CurrentCategoryTitle
        )
        
    AxisLayoutObject.legend(title="Article Category")
    AxisLayoutObject.set_xlabel("t-SNE Coordinate 1")
    AxisLayoutObject.set_ylabel("t-SNE Coordinate 2")
    AxisLayoutObject.set_title("t-SNE Visualization of Top 200 Tokens (PPMI Based)")
    plt.tight_layout()
    plt.savefig("embeddings/tsne_ppmi.png", dpi=150)
    plt.close(PrimaryFigureObject)

    # Phase 4: Nearest Neighbors Logic
    UrduQueryLexemes = ["پاکستان", "کرکٹ", "فلم", "دنیا", "میچ", "کھلاڑی", "وزیر", "سائنس", "اداکار", "حکومت"]
    print("\nTOP 5 COSINE NEIGHBORS (Based on PPMI Row Vectors):")
    L2NormOfEachMatrixRow = np.linalg.norm(PositivePointwiseMutualInformationMatrix, axis=1)
    MappingIndicesToSurfaceForms = {IndexPos: Lexeme for Lexeme, IndexPos in MappingFromWordsToIDs.items()}

    for SingleQueryLexeme in UrduQueryLexemes:
        if SingleQueryLexeme not in MappingFromWordsToIDs:
            print(f"{SingleQueryLexeme}: [NOT PRESENT IN VOCABULARY]")
            continue
            
        IndexOfQueryLexeme = MappingFromWordsToIDs[SingleQueryLexeme]
        MatrixDotProductResults = PositivePointwiseMutualInformationMatrix @ PositivePointwiseMutualInformationMatrix[IndexOfQueryLexeme]
        CosineSimilarityScores = MatrixDotProductResults / (L2NormOfEachMatrixRow * L2NormOfEachMatrixRow[IndexOfQueryLexeme] + 1e-12)
        
        # Exclude the word itself from the neighbor list
        CosineSimilarityScores[IndexOfQueryLexeme] = -1.0
        
        IndicesOfTopNeighbors = np.argsort(-CosineSimilarityScores)[:5]
        ListOfNeighborStrings = [MappingIndicesToSurfaceForms[Idx] for Idx in IndicesOfTopNeighbors]
        print(f"{SingleQueryLexeme}: {' '.join(ListOfNeighborStrings)}")


if __name__ == "__main__":
    ExecuteMatrixGenerationPipeline()
