"""
TOPIC DATA PROCESSOR - Preparation for Article Classification Tasks.
This script cleans and segments the full corpus into document-level records, 
annotates them with categorical labels from metadata, and performs 
vocabulary indexing to produce finalized training/testing JSON datasets for the Transformer.
"""

import os
import re
import sys
import json
import random
import collections
import numpy as np


def ConfigureTerminalForUrduOutput():
    """Adjusts standard output encoding to UTF-8 for console stability."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def ExtractTokenizedArticles(PathToPurifiedCorpus):
    """
    Identifies and extracts articles from the cleaned corpus file.
    Segments articles into lists of tokens, stripping document markers.
    """
    try:
        with open(PathToPurifiedCorpus, mode="r", encoding="utf-8") as InFilePtr:
            BodyTextLines = InFilePtr.read().splitlines()
    except Exception as IOErrorVal:
        print(f"FAILED TO LOAD CORPUS: {IOErrorVal}")
        return {}

    AccumulatorBufferByArticleID = {}
    TrackedIDKey = None
    
    for CurrentLineContent in BodyTextLines:
        CleanedLine = CurrentLineContent.strip()
        DelimiterMatch = re.match(r"^\[?(\d+)\]?\s*$", CleanedLine)
        
        if DelimiterMatch:
            TrackedIDKey = int(DelimiterMatch.group(1))
            AccumulatorBufferByArticleID[TrackedIDKey] = []
        elif TrackedIDKey is not None:
            if CleanedLine and not re.match(r"^=+$", CleanedLine):
                AccumulatorBufferByArticleID[TrackedIDKey].append(CurrentLineContent)

    FinalArticleStorage = {}
    for IDKey, FragmentsList in AccumulatorBufferByArticleID.items():
        UnifiedArticleBody = " ".join(FragmentsList)
        FinalArticleStorage[IDKey] = UnifiedArticleBody.split()
        
    return FinalArticleStorage


def SampleAndBalanceCategorizedArticles(ArticleMap, MasterMetadata, TargetVolumePerCategory, TotalOutputCap):
    """
    Produces a balanced dataset for classification by sampling from each primary category.
    Targets categories like 'world', 'sport', and 'general'.
    """
    CategoryLabelPools = collections.defaultdict(list)
    for DocID, TokenList in ArticleMap.items():
        AssignedCategory = MasterMetadata.get(str(DocID), {}).get("category", "general")
        CategoryLabelPools[AssignedCategory].append((DocID, TokenList))
        
    FinalSamplingBuffer = []
    PrimaryCategoriesList = ["world", "sport", "general"]
    
    # 1. Selection from primary target categories
    for TargetLabel in PrimaryCategoriesList:
        AvailableSubset = CategoryLabelPools[TargetLabel]
        random.shuffle(AvailableSubset)
        QuantityToTake = min(len(AvailableSubset), TargetVolumePerCategory)
        FinalSamplingBuffer.extend(AvailableSubset[:QuantityToTake])
        
    # 2. Backfill if necessary from other sources
    CurrentBufferCount = len(FinalSamplingBuffer)
    if CurrentBufferCount < TotalOutputCap:
        BackfillPoolSet = []
        for Lbl, Items in CategoryLabelPools.items():
            # Gather leftover items
            if Lbl in PrimaryCategoriesList:
                BackfillPoolSet.extend(Items[TargetVolumePerCategory:])
            else:
                BackfillPoolSet.extend(Items)
                
        random.shuffle(BackfillPoolSet)
        ExtraSpace = TotalOutputCap - CurrentBufferCount
        FinalSamplingBuffer.extend(BackfillPoolSet[:ExtraSpace])
        
    random.shuffle(FinalSamplingBuffer)
    return FinalSamplingBuffer


def ConstructLexicalIndexMap(RecordList, MinimumFrequencyFloor):
    """
    Builds a word-to-index dictionary specific to the classification dataset.
    Tokens appearing fewer than N times are mapped to <UNK>.
    Index 0 is reserved for <PAD>.
    """
    TokenFrequencyAudit = collections.Counter()
    for _, TokensSequence in RecordList:
        TokenFrequencyAudit.update(TokensSequence)
        
    # Retain only significant tokens
    ValidatedVocabList = [Word for Word, Count in TokenFrequencyAudit.items() if Count >= MinimumFrequencyFloor]
    ValidatedVocabList.sort()
    
    WordToIndexDictionary = {"<PAD>": 0, "<UNK>": 1}
    for PosIdx, SurfaceWord in enumerate(ValidatedVocabList):
        WordToIndexDictionary[SurfaceWord] = PosIdx + 2
        
    return WordToIndexDictionary


def MapArticleRecordsToIntegerMatrix(RecordList, WordToIDMap, LabelToIDMap, FixedMaxSequenceLen):
    """
    Pre-processes articles into integer sequences of a fixed length.
    Truncates or pads as necessary to fit the matrix dimensions.
    """
    EncodedDatasetRecords = []
    
    for DocID, TokenSequence in RecordList:
        IntegerContentVector = []
        for SingleToken in TokenSequence[:FixedMaxSequenceLen]:
            IntegerContentVector.append(WordToIDMap.get(SingleToken, WordToIDMap["<UNK>"]))
            
        # Pad with zeros if shorter than max length
        while len(IntegerContentVector) < FixedMaxSequenceLen:
            IntegerContentVector.append(WordToIDMap["<PAD>"])
            
        CategoricalMetadata = json.load(open("Metadata.json", "r", encoding="utf-8")).get(str(DocID), {})
        CategoryStringId = CategoricalMetadata.get("category", "general")
        NumericLabelID = LabelToIDMap.get(CategoryStringId, LabelToIDMap["general"])
        
        EncodedDatasetRecords.append({
            "id": int(DocID),
            "input": IntegerContentVector,
            "label": int(NumericLabelID)
        })
        
    return EncodedDatasetRecords


def RunTopicDataPreparationPipeline():
    """Main workflow management for preparing the topic classification data files."""
    ConfigureTerminalForUrduOutput()
    
    # Ensure consistent results for dataset partitioning
    random.seed(1337)
    np.random.seed(1337)

    # 1. Load basic assets
    try:
        MetadataSource = json.load(open("Metadata.json", mode="r", encoding="utf-8"))
    except FileNotFoundError:
        print("ERROR: Metadata.json file not found.")
        return

    AllTokenizedArticles = ExtractTokenizedArticles("cleaned.txt")
    
    # 2. Balanced Sampling for Classification
    BalancedArticleCollection = SampleAndBalanceCategorizedArticles(
        AllTokenizedArticles, MetadataSource, 
        TargetVolumePerCategory=250, TotalOutputCap=800
    )

    # 3. Categorical Label Encoding
    DistinctCategoriesFound = sorted(list({MetadataSource.get(str(D), {}).get("category", "general") for D, _ in BalancedArticleCollection}))
    LabelNameToIndexMap = {Name: Idx for Idx, Name in enumerate(DistinctCategoriesFound)}
    
    # 4. Vocabulary Generation
    ClassificationVocabMap = ConstructLexicalIndexMap(BalancedArticleCollection, MinimumFrequencyFloor=3)
    
    # 5. Data Partitioning (75% Train / 25% Test)
    PartitionCutoffIndex = int(len(BalancedArticleCollection) * 0.75)
    ArticleSubsetTrain = BalancedArticleCollection[:PartitionCutoffIndex]
    ArticleSubsetTest = BalancedArticleCollection[PartitionCutoffIndex:]

    # 6. Serialization to Disk
    MaximumContextLength = 256
    
    TrainingPayload = MapArticleRecordsToIntegerMatrix(ArticleSubsetTrain, ClassificationVocabMap, LabelNameToIndexMap, MaximumContextLength)
    TestingPayload = MapArticleRecordsToIntegerMatrix(ArticleSubsetTest, ClassificationVocabMap, LabelNameToIndexMap, MaximumContextLength)

    os.makedirs("data", exist_ok=True)
    
    with open("data/topic_train.json", mode="w", encoding="utf-8") as TrF:
        json.dump(TrainingPayload, TrF, ensure_ascii=False)
        
    with open("data/topic_test.json", mode="w", encoding="utf-8") as TeF:
        json.dump(TestingPayload, TeF, ensure_ascii=False)
        
    # Persist the configuration metadata for use by the training script
    ConfigurationArtifact = {
        "word2idx": ClassificationVocabMap,
        "label2idx": LabelNameToIndexMap,
        "idx2label": DistinctCategoriesFound,
        "vocab_size": len(ClassificationVocabMap),
        "class_count": len(DistinctCategoriesFound),
        "max_len": MaximumContextLength
    }
    
    with open("data/topic_config.json", mode="w", encoding="utf-8") as CfF:
        json.dump(ConfigurationArtifact, CfF, ensure_ascii=False)

    print("\nTRANSFORMER DATA PREPARATION COMPLETE.")
    print(f"Total Samples Generated: {len(BalancedArticleCollection)}")
    print(f"Train / Test Split: {len(TrainingPayload)} / {len(TestingPayload)}")
    print(f"Vocabulary Size: {len(ClassificationVocabMap)}")
    print(f"Categories Mapped: {', '.join(DistinctCategoriesFound)}")
    print("Files created: data/topic_train.json, data/topic_test.json, data/topic_config.json")


if __name__ == "__main__":
    RunTopicDataPreparationPipeline()
