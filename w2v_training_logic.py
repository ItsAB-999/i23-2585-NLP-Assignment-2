"""
WORD2VEC TRAINING LOGIC - Skip-Gram with Negative Sampling Implementation.
This component handles the neural architecture for word embeddings, including pair 
generation, negative sampling, and the logic for the Binary Cross-Entropy (BCE) loss.
"""

import os
import re
import sys
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


def EnableStandardUtf8ConsolePrinting():
    """Ensures logs remain readable when displaying Urdu text in the console."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def ExtractArticleTokensFromPurifiedCorpus(FilePathToRead):
    """
    Identifies logical article boundaries within the corpus file.
    Returns a dictionary of token lists, keyed by numeric article identifiers.
    """
    try:
        with open(FilePathToRead, mode="r", encoding="utf-8") as InputFilePointer:
            RawLinesFromDataset = InputFilePointer.read().splitlines()
    except Exception as IOErrorWhileLoading:
        print(f"ERROR: Could not access {FilePathToRead}. Logic: {IOErrorWhileLoading}")
        return {}

    AccumulatedDocTokenLinesByArticleID = {}
    CurrentDocumentKeyInLoop = None
    
    for SingleRawLineOfText in RawLinesFromDataset:
        CleanedLineText = SingleRawLineOfText.strip()
        DelimiterMarkerRegexMatch = re.match(r"^\[?(\d+)\]?\s*$", CleanedLineText)
        
        if DelimiterMarkerRegexMatch:
            CurrentDocumentKeyInLoop = int(DelimiterMarkerRegexMatch.group(1))
            AccumulatedDocTokenLinesByArticleID[CurrentDocumentKeyInLoop] = []
        elif CurrentDocumentKeyInLoop is not None:
            AccumulatedDocTokenLinesByArticleID[CurrentDocumentKeyInLoop].append(SingleRawLineOfText)

    FinalParsedArticlesAsTokens = {}
    for ArticleIDKey, LineContentList in AccumulatedDocTokenLinesByArticleID.items():
        FlattenedArticleBodyString = " ".join(LineContentList)
        FinalParsedArticlesAsTokens[ArticleIDKey] = FlattenedArticleBodyString.split()
        
    return FinalParsedArticlesAsTokens


def MapLexicalTokensToNumericIndices(ListOfSurfaceTokens, DictionaryOfWordToIDMappings):
    """
    Converts a sequence of string tokens into a list of integers.
    Fallback to the index of '<UNK>' is applied for any out-of-vocabulary terms.
    """
    IndexOfUnknownWordSentinel = DictionaryOfWordToIDMappings["<UNK>"]
    return [DictionaryOfWordToIDMappings[T] if T in DictionaryOfWordToIDMappings else IndexOfUnknownWordSentinel for T in ListOfSurfaceTokens]


def ConstructSkipGramTrainingPairArrays(ArticleDictionaryMapping, WordToIDMap, RadiusOfWindowContext):
    """
    Generates center-context word pairs required for Skip-Gram training.
    Uses vectorized operations on NumPy arrays for performance.
    """
    ListToStoreCenterWordChunks = []
    ListToStoreContextWordChunks = []
    
    for TokenSequenceInArticle in ArticleDictionaryMapping.values():
        NumericIDSequence = np.asarray(
            MapLexicalTokensToNumericIndices(TokenSequenceInArticle, WordToIDMap), 
            dtype=np.int64
        )
        SequenceLengthMetric = int(NumericIDSequence.shape[0])
        
        if SequenceLengthMetric < 2:
            continue
            
        OffsetRangeForContextSelection = list(range(-RadiusOfWindowContext, 0)) + list(range(1, RadiusOfWindowContext + 1))
        
        for RelativePositionOffset in OffsetRangeForContextSelection:
            AbsoluteStepDistance = -RelativePositionOffset if RelativePositionOffset < 0 else RelativePositionOffset
            
            if RelativePositionOffset > 0:
                CenterSlicePart = NumericIDSequence[:-AbsoluteStepDistance]
                ContextSlicePart = NumericIDSequence[AbsoluteStepDistance:]
            else:
                CenterSlicePart = NumericIDSequence[AbsoluteStepDistance:]
                ContextSlicePart = NumericIDSequence[:-AbsoluteStepDistance]
                
            ListToStoreCenterWordChunks.append(CenterSlicePart)
            ListToStoreContextWordChunks.append(ContextSlicePart)
            
    if not ListToStoreCenterWordChunks:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        
    FinalMasterCenterArray = np.concatenate(ListToStoreCenterWordChunks)
    FinalMasterContextArray = np.concatenate(ListToStoreContextWordChunks)
    
    return FinalMasterCenterArray, FinalMasterContextArray


class NeuralSkipGramBceArchitecture(nn.Module):
    """
    Implementation of the Skip-Gram objective using Binary Cross Entropy.
    Features separate weight matrices for center words (V) and context words (U).
    """

    def __init__(self, VocabularyDimensionSize, LatticeEmbeddingWidth):
        super().__init__()
        # V represents input embeddings for center words
        self.CenterWordEmbeddings = nn.Embedding(VocabularyDimensionSize, LatticeEmbeddingWidth)
        # U represents output embeddings for context words (positive and negative)
        self.ContextWordEmbeddings = nn.Embedding(VocabularyDimensionSize, LatticeEmbeddingWidth)
        
        # Initialize weights with a small uniform distribution
        SmallWeightInitializerAmplitude = 0.5 / LatticeEmbeddingWidth
        nn.init.uniform_(self.CenterWordEmbeddings.weight, -SmallWeightInitializerAmplitude, SmallWeightInitializerAmplitude)
        nn.init.uniform_(self.ContextWordEmbeddings.weight, -SmallWeightInitializerAmplitude, SmallWeightInitializerAmplitude)

    def forward(self, BatchOfCenterIDs, BatchOfPositiveContextIDs, BatchOfNegativeContextIDs):
        """
        Calculates the training loss for a mini-batch.
        Formula: -log(sigmoid(v_c · u_pos)) - sum(log(sigmoid(-v_c · u_neg)))
        """
        VectorOfCenters = self.CenterWordEmbeddings(BatchOfCenterIDs)
        VectorOfPositiveContexts = self.ContextWordEmbeddings(BatchOfPositiveContextIDs)
        
        # Calculate dot products for positive pairs
        PositivePairScores = (VectorOfCenters * VectorOfPositiveContexts).sum(dim=1)
        
        # Calculate dot products for negative samples
        VectorOfNegativeContexts = self.ContextWordEmbeddings(BatchOfNegativeContextIDs)
        NegativePairScores = (VectorOfCenters.unsqueeze(1) * VectorOfNegativeContexts).sum(dim=2)
        
        # BCE components
        PositiveLossComponent = -F.logsigmoid(PositivePairScores)
        NegativeLossComponent = -F.logsigmoid(-NegativePairScores).sum(dim=1)
        
        return (PositiveLossComponent + NegativeLossComponent).mean()


def ExecuteSkipGramNeuralTraining(
    PathToCorpusFile,
    WordToIndexDictionary,
    EmbeddingDimensionality,
    WindowInteractionRadius,
    CountOfNegativeSamples,
    SizeOfMiniBatch,
    InitialLearningRateValue,
    TotalEpochCountToTrain,
    OutputFilePathForEmbeddings,
    OutputFilePathForLossPlot,
    FrequencyOfLoggingInBatches,
):
    """
    Conducts the full training procedure for neural word embeddings.
    1. Pre-computes context pairs.
    2. Runs the Adam optimizer over specified epochs.
    3. Logs loss and generates visualization plots.
    4. Merges C and U matrices and saves the result.
    """
    ComputationDeviceTarget = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure reproducible results
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    ParsedDictionaryOfArticles = ExtractArticleTokensFromPurifiedCorpus(PathToCorpusFile)
    FullVocabSizeValue = len(WordToIndexDictionary)
    
    # Calculate unigram counts for noise distribution sampling
    ArrayOfUnigramFrequencyCounts = np.zeros(FullVocabSizeValue, dtype=np.float64)
    for TokenListPerArticle in ParsedDictionaryOfArticles.values():
        SequenceOfIDValues = np.asarray(MapLexicalTokensToNumericIndices(TokenListPerArticle, WordToIndexDictionary), dtype=np.int64)
        if SequenceOfIDValues.size:
            np.add.at(ArrayOfUnigramFrequencyCounts, SequenceOfIDValues, 1.0)

    # Pre-generate the training pairs in memory
    TrainingCenterArray, TrainingContextArray = ConstructSkipGramTrainingPairArrays(
        ParsedDictionaryOfArticles, WordToIndexDictionary, WindowInteractionRadius
    )
    TotalPairCountAvailable = TrainingCenterArray.shape[0]
    
    # Noise distribution for negative sampling according to Mikolov's paper (power of 0.75)
    PoweredFrequencyDistribution = np.power(ArrayOfUnigramFrequencyCounts + 1e-12, 0.75)
    NoiseProbabilityDistributionTensor = torch.from_numpy((PoweredFrequencyDistribution / PoweredFrequencyDistribution.sum()).astype(np.float32))

    NeuralNetworkModel = NeuralSkipGramBceArchitecture(FullVocabSizeValue, EmbeddingDimensionality).to(ComputationDeviceTarget)
    GradientOptimizer = torch.optim.Adam(NeuralNetworkModel.parameters(), lr=InitialLearningRateValue)

    HistoryOfMeanLossByEpoch = []
    CompleteGranularTraceOfBatchLoss = []

    for CurrentEpochIndex in range(TotalEpochCountToTrain):
        RandomShuffleOfDataIndices = np.random.permutation(TotalPairCountAvailable)
        AccumulatedLossForThisEpoch = 0.0
        ProcessedBatchesCounter = 0
        EpochStartTimeMoment = time.time()

        for BatchStartIndex in range(0, TotalPairCountAvailable, SizeOfMiniBatch):
            BatchSelectionIndices = RandomShuffleOfDataIndices[BatchStartIndex : BatchStartIndex + SizeOfMiniBatch]
            if BatchSelectionIndices.size == 0:
                continue

            CenterIDBatchTensor = torch.from_numpy(TrainingCenterArray[BatchSelectionIndices]).long().to(ComputationDeviceTarget)
            PositiveContextIDBatchTensor = torch.from_numpy(TrainingContextArray[BatchSelectionIndices]).long().to(ComputationDeviceTarget)
            CurrentBatchActualSize = CenterIDBatchTensor.shape[0]

            # Sample negative context words based on the unigram probability distribution
            NegativeContextSamplesBatch = torch.multinomial(
                NoiseProbabilityDistributionTensor,
                num_samples=CurrentBatchActualSize * CountOfNegativeSamples,
                replacement=True,
            ).view(CurrentBatchActualSize, CountOfNegativeSamples).to(ComputationDeviceTarget)

            GradientOptimizer.zero_grad()
            ComputedScalarLossValue = NeuralNetworkModel(CenterIDBatchTensor, PositiveContextIDBatchTensor, NegativeContextSamplesBatch)
            ComputedScalarLossValue.backward()
            GradientOptimizer.step()

            DetachedLossFloatValue = float(ComputedScalarLossValue.detach().cpu().item())
            AccumulatedLossForThisEpoch += DetachedLossFloatValue
            ProcessedBatchesCounter += 1
            CompleteGranularTraceOfBatchLoss.append(DetachedLossFloatValue)

            if FrequencyOfLoggingInBatches > 0 and ProcessedBatchesCounter % FrequencyOfLoggingInBatches == 0:
                SecondsElapsedSinceStart = time.time() - EpochStartTimeMoment
                print(
                    f"EPOCH: {CurrentEpochIndex + 1} | BATCH: {ProcessedBatchesCounter} | LOSS: {round(DetachedLossFloatValue, 5)} | ELAPSED: {round(SecondsElapsedSinceStart, 2)}s",
                    flush=True,
                )

        MeanLossResultForEpoch = AccumulatedLossForThisEpoch / max(1, ProcessedBatchesCounter)
        HistoryOfMeanLossByEpoch.append(MeanLossResultForEpoch)
        print(f"COMPLETED EPOCH {CurrentEpochIndex + 1} | MEAN LOSS: {round(MeanLossResultForEpoch, 5)}", flush=True)

    # Export learned parameters and visualization
    os.makedirs(os.path.dirname(OutputFilePathForEmbeddings), exist_ok=True)
    LearnedCenterWeights = NeuralNetworkModel.CenterWordEmbeddings.weight.detach().cpu().numpy().astype(np.float32)
    LearnedContextWeights = NeuralNetworkModel.ContextWordEmbeddings.weight.detach().cpu().numpy().astype(np.float32)
    
    # Assignment spec recommends averaging both matrices to get final embeddings
    SymmetrizedMergedEmbeddingsMatrix = (LearnedCenterWeights + LearnedContextWeights) / 2.0
    np.save(OutputFilePathForEmbeddings, SymmetrizedMergedEmbeddingsMatrix)

    # Plot the loss curve
    PlotFigureObject, PlotAxesLayout = plt.subplots(figsize=(8, 5))
    PlotAxesLayout.plot(range(1, len(CompleteGranularTraceOfBatchLoss) + 1), CompleteGranularTraceOfBatchLoss, linewidth=0.8, alpha=0.85)
    PlotAxesLayout.set_xlabel("Global Minibatch Index")
    PlotAxesLayout.set_ylabel("BCE Skip-Gram Training Loss")
    PlotAxesLayout.set_title("Evolution of Neural Word2Vec Loss")
    plt.tight_layout()
    plt.savefig(OutputFilePathForLossPlot, dpi=150)
    plt.close(PlotFigureObject)

    return HistoryOfMeanLossByEpoch, CompleteGranularTraceOfBatchLoss


def RunDefaultNeuralTrainingConfiguration():
    """
    Standard driver function for training the primary Skip-Gram model.
    Uses the hyper-parameters specified in the assignment documentation.
    """
    EnableStandardUtf8ConsolePrinting()
    
    # Load vocabulary mapping generated by the previous stage (matrix_embeddings.py)
    try:
        with open("embeddings/word2idx.json", mode="r", encoding="utf-8") as MetadataInputPointer:
            DictionaryMappingWordsToIndex = json.load(MetadataInputPointer)
    except FileNotFoundError:
        print("ERROR: 'embeddings/word2idx.json' not found. Ensure 'matrix_embeddings.py' has been executed.")
        sys.exit(1)

    # Configure training hyper-parameters
    WidthOfEmbeddings = 100
    RadiusOfContextWindow = 5
    RatioOfNegativeSamples = 10
    MiniBatchQuantity = 512
    StepLearningRateCoefficient = 0.001
    EpochIterationsCount = 5
    LoggingIntervalStepCount = 400

    ExecuteSkipGramNeuralTraining(
        PathToCorpusFile="cleaned.txt",
        WordToIndexDictionary=DictionaryMappingWordsToIndex,
        EmbeddingDimensionality=WidthOfEmbeddings,
        WindowInteractionRadius=RadiusOfContextWindow,
        CountOfNegativeSamples=RatioOfNegativeSamples,
        SizeOfMiniBatch=MiniBatchQuantity,
        InitialLearningRateValue=StepLearningRateCoefficient,
        TotalEpochCountToTrain=EpochIterationsCount,
        OutputFilePathForEmbeddings="embeddings/embeddings_w2v.npy",
        OutputFilePathForLossPlot="embeddings/loss_w2v.png",
        FrequencyOfLoggingInBatches=LoggingIntervalStepCount,
    )
    
    print("\nSUCCESS: Saved trained embeddings to 'embeddings/embeddings_w2v.npy'.")
    print("SUCCESS: Logged training loss curve to 'embeddings/loss_w2v.png'.")


if __name__ == "__main__":
    RunDefaultNeuralTrainingConfiguration()
