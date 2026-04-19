"""
WORD2VEC EVALUATION SUITE - Benchmarking and Analysis of Learned Word Embeddings.
This script assesses the quality of word embeddings through neighbor searches, 
analogy tasks, and a formalized comparison of four experimental conditions (C1 to C4).
It leverages the training logic from 'w2v_training_logic.py'.
"""

import os
import sys
import json
import numpy as np

# Updated import to reflect the new filename
import w2v_training_logic as WordVectorEngine


def InitializeUtf8StandardOutput():
    """Forces standard output to UTF-8 to ensure Urdu strings print properly."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def RetrieveWordToIndexMapping(PathToMappingJson):
    """Loads the lexical-to-integer mapping from the specified JSON file."""
    with open(PathToMappingJson, mode="r", encoding="utf-8") as JsonFileHandle:
        return json.load(JsonFileHandle)


def ConstructReverseIndexToWordMap(VocabularyToIndexDictionary):
    """Creates an inverse mapping from integer IDs back to surface word strings."""
    return {IntegerIdx: SurfaceWord for SurfaceWord, IntegerIdx in VocabularyToIndexDictionary.items()}


def ApplyRowLevelL2Normalization(InputEmbeddingMatrix):
    """
    Normalizes each row vector to unit length (L2 norm).
    Critical for accurate cosine similarity calculations.
    """
    VectorMagnitudes = np.linalg.norm(InputEmbeddingMatrix, axis=1, keepdims=True) + 1e-12
    return InputEmbeddingMatrix / VectorMagnitudes


def IdentifyTopKSemanticNeighbors(NormalizedMatrix, WordToIndex, IndexToWord, TargetQueryWord, KValue):
    """
    Finds the K most similar words to a target query word using cosine similarity.
    Calculates the dot product between the query vector and the rest of the matrix.
    """
    if TargetQueryWord not in WordToIndex:
        return []
        
    QueryWordIndex = WordToIndex[TargetQueryWord]
    SimilarityScoresVector = NormalizedMatrix @ NormalizedMatrix[QueryWordIndex]
    
    # Effectively exclude the query word itself from results
    SimilarityScoresVector[QueryWordIndex] = -1e9
    
    SortingRankOrder = np.argsort(-SimilarityScoresVector)[:KValue]
    return [(IndexToWord[Idx], float(SimilarityScoresVector[Idx])) for Idx in SortingRankOrder]


def PerformAnalogyVectortest(NormalizedMatrix, WordToIndex, WordA, WordB, WordC, CandidateQuantity, ExcludeInputWords=True):
    """
    Evaluates the vector offset analogy: WordB - WordA + WordC ≈ ?
    Example: King - Man + Woman ≈ Queen
    """
    for SpecificWord in (WordA, WordB, WordC):
        if SpecificWord not in WordToIndex:
            return []
            
    IdxA, IdxB, IdxC = WordToIndex[WordA], WordToIndex[WordB], WordToIndex[WordC]
    
    # Vector arithmetic in embedding space
    ResultantAnalogyVector = NormalizedMatrix[IdxB] - NormalizedMatrix[IdxA] + NormalizedMatrix[IdxC]
    NormalizedAnalogyVector = ResultantAnalogyVector / (np.linalg.norm(ResultantAnalogyVector) + 1e-12)
    
    SearchSimilarityScores = NormalizedMatrix @ NormalizedAnalogyVector
    
    if ExcludeInputWords:
        SearchSimilarityScores[IdxA] = -1e9
        SearchSimilarityScores[IdxB] = -1e9
        SearchSimilarityScores[IdxC] = -1e9
        
    IndicesOfBestMatches = np.argsort(-SearchSimilarityScores)[:CandidateQuantity]
    MappingFromIdxToWord = ConstructReverseIndexToWordMap(WordToIndex)
    
    return [MappingFromIdxToWord[MatchedIdx] for MatchedIdx in IndicesOfBestMatches]


def CalculateMeanReciprocalRank(NormalizedMappingMatrix, WordToIndexMap, SetOfLabeledEvaluationPairs):
    """
    Computes the MRR score across a set of (query, target) gold pairs.
    Measures the ranking quality of the target word relative to the query word.
    """
    ReciprocalRankCalculations = []
    
    for QueryLexeme, TargetGoldLexeme in SetOfLabeledEvaluationPairs:
        if QueryLexeme not in WordToIndexMap or TargetGoldLexeme not in WordToIndexMap:
            ReciprocalRankCalculations.append(0.0)
            continue
            
        QueryID = WordToIndexMap[QueryLexeme]
        GoldTargetID = WordToIndexMap[TargetGoldLexeme]
        
        ScoreVectorForQuery = NormalizedMappingMatrix @ NormalizedMappingMatrix[QueryID]
        ScoreVectorForQuery[QueryID] = -1e9
        
        RankedListIndices = np.argsort(-ScoreVectorForQuery)
        NumericalRankPosition = int(np.where(RankedListIndices == GoldTargetID)[0][0]) + 1
        
        ReciprocalRankCalculations.append(1.0 / float(NumericalRankPosition))
        
    return float(np.mean(ReciprocalRankCalculations)) if ReciprocalRankCalculations else 0.0


def FormattedDisplayOfNeighbors(SectionTitleHeader, NormalizedEmbs, WordToIdxDict, ListOfQueries, KVal):
    """Prints a block of semantic neighbors in a readable format for evaluation."""
    print(f"\n--- {SectionTitleHeader} ---")
    ReverseIndexMapper = ConstructReverseIndexToWordMap(WordToIdxDict)
    
    for CurrentQuery in ListOfQueries:
        NeighborListResults = IdentifyTopKSemanticNeighbors(NormalizedEmbs, WordToIdxDict, ReverseIndexMapper, CurrentQuery, KVal)
        
        if not NeighborListResults:
            print(f"  {CurrentQuery} -> [NOT FOUND IN VOCABULARY]")
            continue
            
        SurfaceStringsOutput = "  ".join([Wrd for Wrd, _Score in NeighborListResults])
        print(f"  {CurrentQuery}: {SurfaceStringsOutput}")


def ExecuteComprehensiveFourConditionSuite(MasterWordToIndex):
    """
    Benchmarks four different versions of the embedding space:
    - C1: PPMI Baseline
    - C2: Word2Vec on Raw Text
    - C3: Word2Vec on Cleaned Text (Default d=100)
    - C4: Word2Vec on Cleaned Text (High-dim d=200)
    """
    TotalVocabularyCount = len(MasterWordToIndex)
    RepresentativeQueryList = ["پاکستان", "کرکٹ", "حکومت", "فلم", "انڈیا"]

    ValidationPairsForMrr = [
        ("کرکٹ", "میچ"), ("میچ", "کرکٹ"), ("فلم", "اداکار"), ("اداکار", "فلم"),
        ("ٹیم", "کھلاڑی"), ("کھلاڑی", "ٹیم"), ("پاکستان", "انڈیا"), ("انڈیا", "پاکستان"),
        ("حکومت", "وزیر"), ("وزیر", "حکومت"), ("فوج", "جنرل"), ("جنرل", "فوج"),
        ("صحت", "بیماری"), ("تعلیم", "سکول"), ("رنز", "وکٹ"), ("بیٹسمین", "بولر"),
        ("سعودی", "عرب"), ("ٹرمپ", "امریکی"), ("کپتان", "ٹیم"), ("سیریز", "میچ"),
    ]

    PerformanceMetricsStorage = []

    # CONDITION 1: PPMI BASELINE
    PpmiRelativePath = "embeddings/ppmi_matrix.npy"
    if not os.path.isfile(PpmiRelativePath):
        print(f"ERROR: {PpmiRelativePath} missing. Run 'matrix_embeddings.py' first.")
        sys.exit(1)
        
    PpmiRawMatrixData = np.load(PpmiRelativePath).astype(np.float64)
    if PpmiRawMatrixData.shape[0] != TotalVocabularyCount:
        print("ERROR: Vocabulary mismatch in PPMI matrix data.")
        sys.exit(1)
        
    C1_NormalizedEmbeddings = ApplyRowLevelL2Normalization(PpmiRawMatrixData.astype(np.float32))
    FormattedDisplayOfNeighbors("C1: PPMI Baseline Row Vectors", C1_NormalizedEmbeddings, MasterWordToIndex, RepresentativeQueryList, 5)
    
    C1_MrrScoreValue = CalculateMeanReciprocalRank(C1_NormalizedEmbeddings, MasterWordToIndex, ValidationPairsForMrr)
    print(f"C1 - Mean Reciprocal Rank: {round(C1_MrrScoreValue, 5)}")
    PerformanceMetricsStorage.append(("C1", "PPMI Baseline (Frequency Based)", C1_MrrScoreValue))

    # CONDITION 2: Word2Vec trained on RAW TEXT
    print("\nINITIATING TRAINING FOR C2 (raw.txt, dim=100) - This may take a few minutes...")
    WordVectorEngine.ExecuteSkipGramNeuralTraining(
        PathToCorpusFile="raw.txt",
        WordToIndexDictionary=MasterWordToIndex,
        EmbeddingDimensionality=100,
        WindowInteractionRadius=5,
        CountOfNegativeSamples=10,
        SizeOfMiniBatch=512,
        InitialLearningRateValue=0.001,
        TotalEpochCountToTrain=5,
        OutputFilePathForEmbeddings="embeddings/embeddings_w2v_raw.npy",
        OutputFilePathForLossPlot="embeddings/loss_w2v_raw.png",
        FrequencyOfLoggingInBatches=800,
    )
    
    C2_EmbeddingsRawLoad = np.load("embeddings/embeddings_w2v_raw.npy").astype(np.float32)
    C2_NormalizedEmbeddings = ApplyRowLevelL2Normalization(C2_EmbeddingsRawLoad)
    FormattedDisplayOfNeighbors("C2: Word2Vec (Raw Dataset, d=100)", C2_NormalizedEmbeddings, MasterWordToIndex, RepresentativeQueryList, 5)
    
    C2_MrrScoreValue = CalculateMeanReciprocalRank(C2_NormalizedEmbeddings, MasterWordToIndex, ValidationPairsForMrr)
    print(f"C2 - Mean Reciprocal Rank: {round(C2_MrrScoreValue, 5)}")
    PerformanceMetricsStorage.append(("C2", "Word2Vec on Noisy 'raw.txt' (d=100)", C2_MrrScoreValue))

    # CONDITION 3: Word2Vec trained on CLEANED TEXT (Default)
    MainEmbeddingsPath = "embeddings/embeddings_w2v.npy"
    if not os.path.isfile(MainEmbeddingsPath):
        print("ERROR: C3 embeddings missing. Ensure 'w2v_training_logic.py' was successful.")
        sys.exit(1)
        
    C3_EmbeddingsCleanLoad = np.load(MainEmbeddingsPath).astype(np.float32)
    C3_NormalizedEmbeddings = ApplyRowLevelL2Normalization(C3_EmbeddingsCleanLoad)
    FormattedDisplayOfNeighbors("C3: Word2Vec (Cleaned Dataset, d=100)", C3_NormalizedEmbeddings, MasterWordToIndex, RepresentativeQueryList, 5)
    
    C3_MrrScoreValue = CalculateMeanReciprocalRank(C3_NormalizedEmbeddings, MasterWordToIndex, ValidationPairsForMrr)
    print(f"C3 - Mean Reciprocal Rank: {round(C3_MrrScoreValue, 5)}")
    PerformanceMetricsStorage.append(("C3", "Word2Vec on Purified 'cleaned.txt' (d=100)", C3_MrrScoreValue))

    # CONDITION 4: Word2Vec trained on CLEANED TEXT with HIGH DIMENSIONALITY (d=200)
    print("\nINITIATING TRAINING FOR C4 (cleaned.txt, dim=200) - Increased parameter count.")
    WordVectorEngine.ExecuteSkipGramNeuralTraining(
        PathToCorpusFile="cleaned.txt",
        WordToIndexDictionary=MasterWordToIndex,
        EmbeddingDimensionality=200,
        WindowInteractionRadius=5,
        CountOfNegativeSamples=10,
        SizeOfMiniBatch=512,
        InitialLearningRateValue=0.001,
        TotalEpochCountToTrain=5,
        OutputFilePathForEmbeddings="embeddings/embeddings_w2v_d200.npy",
        OutputFilePathForLossPlot="embeddings/loss_w2v_d200.png",
        FrequencyOfLoggingInBatches=800,
    )
    
    C4_EmbeddingsD200Load = np.load("embeddings/embeddings_w2v_d200.npy").astype(np.float32)
    C4_NormalizedEmbeddings = ApplyRowLevelL2Normalization(C4_EmbeddingsD200Load)
    FormattedDisplayOfNeighbors("C4: Word2Vec (Cleaned Dataset, d=200)", C4_NormalizedEmbeddings, MasterWordToIndex, RepresentativeQueryList, 5)
    
    C4_MrrScoreValue = CalculateMeanReciprocalRank(C4_NormalizedEmbeddings, MasterWordToIndex, ValidationPairsForMrr)
    print(f"C4 - Mean Reciprocal Rank: {round(C4_MrrScoreValue, 5)}")
    PerformanceMetricsStorage.append(("C4", "Word2Vec on Purified 'cleaned.txt' (d=200)", C4_MrrScoreValue))

    # Summary Output
    print("\n" + "="*70)
    print("{:<5} | {:<40} | {:<10}".format("ID", "CONDITION DESCRIPTION", "MRR@20"))
    print("-" * 70)
    for Ident, Desc, Val in PerformanceMetricsStorage:
        print("{:<5} | {:<40} | {:.5f}".format(Ident, Desc, Val))
    print("="*70)


def RunFullEvaluationSequence():
    """Main driver for the evaluation script."""
    InitializeUtf8StandardOutput()
    
    # Load required data
    VocabIDMapping = RetrieveWordToIndexMapping("embeddings/word2idx.json")
    NumericIndexToWordMap = ConstructReverseIndexToWordMap(VocabIDMapping)

    # 1. Evaluate Nearest Neighbors for specific query types
    QuerySpecificationsTable = [
        ("Pakistan", "پاکستان"), ("Government", "حکومت"), ("Court", "عدالت"),
        ("Economy", "معیشت"), ("Army", "فوج"), ("Health", "صحت"),
        ("Education", "تعلیم"), ("Population", "آبادی")
    ]

    PrimaryModelPath = "embeddings/embeddings_w2v.npy"
    if not os.path.isfile(PrimaryModelPath):
        print(f"FAILED: Main embeddings at {PrimaryModelPath} not found.")
        return

    NormalizedMainEmbeddings = ApplyRowLevelL2Normalization(np.load(PrimaryModelPath).astype(np.float32))

    print("\nTOP 10 SEMANTIC NEIGHBORS (Primary C3 Model - Word2Vec Cleaned d=100):")
    for Label, UrduScript in QuerySpecificationsTable:
        TopNeighborsFound = IdentifyTopKSemanticNeighbors(NormalizedMainEmbeddings, VocabIDMapping, NumericIndexToWordMap, UrduScript, 10)
        if not TopNeighborsFound:
            print(f"{Label} ({UrduScript}) -> [NOT IN VOCABULARY]")
            continue
        print(f"{Label} ({UrduScript}): {'  '.join([WordCandidate for WordCandidate, _ in TopNeighborsFound])}")

    # 2. Evaluate Analogy Tasks
    AnalogyTestData = [
        ("کرکٹ", "میچ", "فلم", ["ریلیز", "فلموں", "اداکار"]),
        ("حکومت", "وزیر", "فوج", ["جنرل", "افواج", "سربراہ"]),
        ("ٹیم", "کھلاڑی", "فلم", ["اداکار", "کردار", "ریلیز"]),
        ("پاکستان", "انڈیا", "میچ", ["کرکٹ", "سیریز", "ٹی"]),
        ("بیٹسمین", "رنز", "بولر", ["وکٹ", "گیند", "شکار"]),
        ("سعودی", "عرب", "امریکی", ["ٹرمپ", "صدر", "ریاست"]),
        ("کپتان", "ٹیم", "وزیر", ["حکومت", "کابینہ", "ملک"]),
        ("صحت", "بیماری", "تعلیم", ["طلبہ", "سکول", "یونیورسٹی"]),
        ("فوج", "جنرل", "عدالت", ["جج", "فیصلہ", "سزا"]),
        ("کرکٹر", "کھلاڑی", "اداکار", ["فلم", "کردار", "ریلیز"]),
    ]

    print("\nANALOGY COMPLETION TESTS (Top 3 Candidates per Task):")
    SuccessCountInTop3 = 0
    for A, B, C, GoldAcceptedResults in AnalogyTestData:
        TopCandidatesGenerated = PerformAnalogyVectortest(NormalizedMainEmbeddings, VocabIDMapping, A, B, C, 3)
        HitDetected = any(Candidate in TopCandidatesGenerated for Candidate in GoldAcceptedResults)
        if HitDetected:
            SuccessCountInTop3 += 1
        print(f"  {A} : {B} :: {C} : ?  => Top-3: {' '.join(TopCandidatesGenerated)} | SUCCESS: {HitDetected}")
        
    print(f"\nANALOGY ACCURACY (Top-3 Lenient): {SuccessCountInTop3} / 10")

    # 3. Run the overall 4-condition benchmarking suite
    print("\n" + "#"*40)
    print(" STARTING FOUR-CONDITION BENCHMARK BLOCK ")
    print("#"*40)
    ExecuteComprehensiveFourConditionSuite(VocabIDMapping)


if __name__ == "__main__":
    RunFullEvaluationSequence()
