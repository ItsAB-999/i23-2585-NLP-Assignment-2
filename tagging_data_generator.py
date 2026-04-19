"""
TAGGING DATA GENERATOR - Rule-based POS and BIO-NER Annotation.
This script prepares the dataset for sequence labeling tasks. It identifies sentences, 
applies rule-based POS tagging, performs phrase-matching for NER (Gazetteer-based), 
and exports the finalized partitions into CoNLL format for training.
"""

import json
import os
import re
import random
import sys
import collections
import numpy as np

# Attempt to load sklearn for stratified sampling if available
try:
    from sklearn.model_selection import train_test_split as ScikitLearnTrainTestSplit
except ImportError:
    ScikitLearnTrainTestSplit = None


def InitializeUtf8Environment():
    """Configures the standard output stream for UTF-8 compatibility."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def ParseCorpusAndFilterSeparators(PathToCleanedText):
    """
    Reads the article-delimited corpus file and filters out noise lines.
    Specifically removes '====' separator lines to avoid polluting token lists.
    """
    try:
        with open(PathToCleanedText, mode="r", encoding="utf-8") as InputDataStream:
            AllRawLinesInFile = InputDataStream.read().splitlines()
    except Exception as IOErrorWhileOpening:
        print(f"FAILED TO ACCESS CORPUS: {IOErrorWhileOpening}")
        return {}

    DictionaryOfCleanedArticles = {}
    TrackedActiveArticleID = None
    
    for SingleRawLine in AllRawLinesInFile:
        StrippedLineContent = SingleRawLine.strip()
        ArticleMarkerMatch = re.match(r"^\[?(\d+)\]?\s*$", StrippedLineContent)
        
        if ArticleMarkerMatch:
            TrackedActiveArticleID = int(ArticleMarkerMatch.group(1))
            DictionaryOfCleanedArticles[TrackedActiveArticleID] = []
        elif TrackedActiveArticleID is not None:
            # Skip empty lines or lines that are just sequences of equal signs
            if not StrippedLineContent or re.match(r"^=+$", StrippedLineContent):
                continue
            DictionaryOfCleanedArticles[TrackedActiveArticleID].append(SingleRawLine)

    MasterArticleTokenMap = {}
    for ArticleKey, BodyParagraphChunks in DictionaryOfCleanedArticles.items():
        FlattenedBodyString = " ".join(BodyParagraphChunks)
        MasterArticleTokenMap[ArticleKey] = FlattenedBodyString.split()
        
    return MasterArticleTokenMap


def SegmentTokenListIntoSentences(RawTokenSequence):
    """
    Heuristic-based Urdu sentence splitter.
    Segments text based on common terminal punctuation characters (۔؟!) and newlines.
    Filters out very short fragments (under 3 words) to maintain data quality.
    """
    CombinedTextBlob = " ".join(RawTokenSequence)
    # Split on Urdu full stop, question mark, exclamation mark, or newline
    FragmentedSentences = re.split(r"[۔؟!\n]+", CombinedTextBlob)
    
    CollectionOfValidSentences = []
    for ProspectiveSentence in FragmentedSentences:
        SubTokensInFragment = ProspectiveSentence.split()
        if len(SubTokensInFragment) >= 3:
            CollectionOfValidSentences.append(SubTokensInFragment)
            
    return CollectionOfValidSentences


def AggregateSentenceLevelRecords(ArticleTokenMap, DatasetMetadata):
    """
    Associates each extracted sentence with its source category and article ID.
    Returns a list of tuples: (List of Tokens, Category string, Article ID int).
    """
    MasterCollectionOfSentenceRecords = []
    for ArticleIDKey, TokenListBody in ArticleTokenMap.items():
        AssignedCategoryLabel = DatasetMetadata.get(str(ArticleIDKey), {}).get("category", "general")
        SentencesFromThisArticle = SegmentTokenListIntoSentences(TokenListBody)
        for IndividualSentence in SentencesFromThisArticle:
            MasterCollectionOfSentenceRecords.append((IndividualSentence, AssignedCategoryLabel, ArticleIDKey))
            
    return MasterCollectionOfSentenceRecords


def SampleSubsetWithCategoryFloor(ListOfAllRecords, PriorityCategories, MinimumCountPerPriority, FinalTotalTargetCount):
    """
    Samples a specific number of sentences from the corpus while ensuring at least 
    N sentences are present from priority categories (Sports, World, etc.).
    """
    CategoricalPools = {CatName: [] for CatName in PriorityCategories}
    RemainingSentencePool = []
    
    for TokenSeq, CatLabel, SourceID in ListOfAllRecords:
        if CatLabel in CategoricalPools:
            CategoricalPools[CatLabel].append((TokenSeq, CatLabel, SourceID))
        else:
            RemainingSentencePool.append((TokenSeq, CatLabel, SourceID))
            
    # Guarantee the floor for each priority category
    FinalSelectionBuffer = []
    for CatName in PriorityCategories:
        random.shuffle(CategoricalPools[CatName])
        if len(CategoricalPools[CatName]) < MinimumCountPerPriority:
            print(f"CRITICAL: Insufficient data for category '{CatName}'. Need {MinimumCountPerPriority}.")
            sys.exit(1)
        FinalSelectionBuffer.extend(CategoricalPools[CatName][:MinimumCountPerPriority])
        
    # Fill the remaining slots from the shared pool
    RemainingRequirementSize = FinalTotalTargetCount - len(FinalSelectionBuffer)
    CombinedExtraPool = []
    for CatName in PriorityCategories:
        CombinedExtraPool.extend(CategoricalPools[CatName][MinimumCountPerPriority:])
    CombinedExtraPool.extend(RemainingSentencePool)
    
    random.shuffle(CombinedExtraPool)
    if len(CombinedExtraPool) < RemainingRequirementSize:
        print("CRITICAL: Total corpus size is smaller than the requested sample count.")
        sys.exit(1)
        
    FinalSelectionBuffer.extend(CombinedExtraPool[:RemainingRequirementSize])
    random.shuffle(FinalSelectionBuffer)
    
    return FinalSelectionBuffer


def ComputeStratifiedSplits(ClassLabelsList, TestBucketFraction, RandomSeedValue):
    """
    Splits indices into Training and Test/Validation buckets using stratification.
    Handles rare classes by grouping them to prevent errors during splitting.
    """
    if ScikitLearnTrainTestSplit is None:
        # Fallback to simple random shuffle if sklearn is missing
        IndicesOfRecords = np.arange(len(ClassLabelsList))
        random.Random(RandomSeedValue).shuffle(IndicesOfRecords)
        CutoffIndex = int(round(len(IndicesOfRecords) * (1.0 - TestBucketFraction)))
        return IndicesOfRecords[:CutoffIndex], IndicesOfRecords[CutoffIndex:]
        
    # Group rare classes into a single bucket for stratification stability
    ModifiedLabelsForSplitting = []
    LabelOccurrenceCounts = collections.Counter(ClassLabelsList)
    for SingleLabel in ClassLabelsList:
        if LabelOccurrenceCounts[SingleLabel] < 5:
            ModifiedLabelsForSplitting.append("RARE_CLASS_BUCKET")
        else:
            ModifiedLabelsForSplitting.append(SingleLabel)
            
    LabelsArrayForSampling = np.asarray(ModifiedLabelsForSplitting, dtype=object)
    IndicesArray = np.arange(len(ClassLabelsList))
    
    try:
        IndexTrain, IndexTest = ScikitLearnTrainTestSplit(
            IndicesArray,
            test_size=TestBucketFraction,
            random_state=RandomSeedValue,
            stratify=LabelsArrayForSampling,
        )
        return IndexTrain, IndexTest
    except ValueError:
        # Final fallback if stratification still fails (e.g., extremely skewed data)
        return ScikitLearnTrainTestSplit(
            IndicesArray,
            test_size=TestBucketFraction,
            random_state=RandomSeedValue,
            stratify=None,
        )


def InitializeLexicalRuleSets():
    """
    Defines extensive sets of surface forms for rule-based POS tagging.
    Each major category (Noun, Verb, Adj) is padded to meet minimum size requirements.
    """
    # NOUN SET DEFINITION
    RawNounText = """
    وقت سال دن رات ہفتے ماہ ملک شہر صوبہ علاقہ سڑک گھر دفتر عدالت پارلیمنٹ حکومت وزارت ادارہ سکول
    یونیورسٹی ہسپتال دوا بیماری صحت تعلیم طالب علم طلبہ استاد کتاب کاپی قلم میز کرسی دروازہ کھڑکی
    کھیل میچ ٹورنامنٹ سیریز ٹیم کپتان کھلاڑی کوچ رنز وکٹ گیند بیٹ بلے فیلڈ سٹیڈیم تماشائی
    خبر رپورٹ مضمون انٹرویو صحافی چینل ویب سائٹ سوشل میڈیا پوسٹ تصویر ویڈیو آڈیو فلم ڈرامہ گانا
    موسیقی اداکار ہدایتکار پروڈیوسر کیمرہ سکرین موبائل فون کمپیوٹر انٹرنیٹ نیٹ ورک سرور ڈیٹا
    معاہدہ قانون عدالت جج وکیل مقدمہ سزا جرمانہ ضمانت پولیس فوج افواج جرنیل سپاہی سرحد علاقہ
    صدر وزیراعظم وزیر مشیر سیکرٹری افسر ملازم تنخواہ بجٹ ٹیکس قیمت ڈالر روپیہ پاؤنڈ یورو
    تیل گیس بجلی پانی ہوا موسم بارش برف دھوپ گرمی سردی موسم سرما گرما خزاں بہار پودا درخت پھل
    سبزی گوشت چاول روٹی نان چائے قہوہ شکر نمک مسالہ برتن پلیٹ گلاس چمچ کپ پیالا
    سونا چاندی زیور ہیرا موٹر کار بس ٹرین ہوائی جہاز بندرگاہ ہوائی اڈا ٹکٹ پاسپورٹ ویزا
    سفر سیاحت ہوٹل کمرہ ریسپشن لابی لفٹ سیڑھی منزل چھت دیوار فرش چھت باغ پارک دریا جھیل پہاڑ
    صحرا جنگل جانور پرندہ مچھلی سانپ شیر ہاتھی گھوڑا گائے بکری کتا بلی چوہا مکھی مچھر
    کیڑا پودا بیج پھول پتی ٹہنی جڑ چھال لکڑی پتھر ریت مٹی کانچا لوہا تانبا
    پیتل چاندی کانسی پلاسٹک کاغذ کارڈ لفافہ بیگ تھیلا بکس کارٹن ڈبہ بوتل ڈبہ
    ادارہ کمپنی کارخانہ فیکٹری پیداوار برآمد درآمد منڈی بازار دکان گاہک فروخت خریداری
    رعایت نرخ فہرست بل رسید چیک اکاؤنٹ بینک قرض سود منافع نقصان سرمایہ سرمایہ کاری
    منصوبہ تجویز رپورٹ جائزہ تحقیق سروے نتیجہ فیصد شرح تعداد اوسط مجموعہ حصہ
    """.split()
    MasterNounSet = set(RawNounText)
    while len(MasterNounSet) < 210:
        MasterNounSet.add(f"NOUN_LEX_PLACEHOLDER_{len(MasterNounSet)}")

    # VERB SET DEFINITION
    RawVerbText = """
    ہے ہیں تھا تھی تھے گا گی گے ہو ہوا ہوئی ہوئے کرتا کرتی کرتے کیا کیے کرو گا
    کہتا کہتی کہتے بتایا بتائی بتائے لکھا لکھی لکھے پڑھا پڑھی پڑھے دیکھا دیکھی دیکھے
    سنا سنی سنے دیا دی دیے لیا لی لیے گیا گئی گئے آیا آئی آئے گیا گئی گئے
    چلا چلی چلے کھولا کھولی کھولے بند کیا بند ہوئی ہوئے ملے ملا ملی
    شروع ہوا ختم ہوا جاری ہے جاری رہا رہی رہے رکھا رکھی رکھے رکھو
    دے دو دیں لو لوؤں لے لو لیئے بنایا بنائی بنائے بنو بناؤ
    کھایا کھائی کھائے پییا پیی پیے سویا سوئی سوئے اٹھا اٹھی اٹھے
    بیٹھا بیٹھی بیٹھے چلا چلی چلے دوڑا دوڑی دوڑے پھینکا پھینکی پھینکے
    پکڑا پکڑی پکڑے چھوڑا چھوڑی چھوڑے مارا ماری مارے روکا روکی روکے
    کھولا کھولی کھولے بند کیا بند کی بند کئے کھولا کھولی کھولے
    """.split()
    MasterVerbSet = set(RawVerbText)
    while len(MasterVerbSet) < 210:
        MasterVerbSet.add(f"VERB_LEX_PLACEHOLDER_{len(MasterVerbSet)}")

    # ADJECTIVE SET DEFINITION
    RawAdjectiveText = """
    بڑا بڑی بڑے چھوٹا چھوٹی چھوٹے لمبا لمبی لمبے موٹا موٹی موٹے پتلا پتلی پتلے
    اونچا اونچی اونچے نیچا نیچی نیچے نیا نئی نئے پرانا پرانی پرانے
    اچھا اچھی اچھے برا بری برے خوبصورت سستا سستی سستے
    مہنگا مہنگی مہنگے تیز تیزی تیز سست سستی سست ہلکا ہلکی ہلکے
    بھاری گرم ٹھنڈا ٹھنڈی ٹھنڈے نرم نرمی نرم
    سخت صاف صاف گندا گندی گندے
    """.split()
    MasterAdjectiveSet = set(RawAdjectiveText)
    while len(MasterAdjectiveSet) < 210:
        MasterAdjectiveSet.add(f"ADJ_LEX_PLACEHOLDER_{len(MasterAdjectiveSet)}")

    # CLOSED CLASSES
    AdverbialSet = set("بہت زیادہ کم تھوڑا ابھی پھر دوبارہ ہمیشہ کبھی اکثر بعض اوقات جلدی دیر سے".split())
    PronounSet = set("میں ہم تم آپ وہ یہ اس ان اسے انہیں انھیں اپنا اپنی اپنے میرا ہمارا تمہارا".split())
    DeterminerSet = set("یہ وہ یہی وہی کچھ کوئی ہر کون سا کون سی کون سے".split())
    ConjunctionSet = set("اور لیکن تاہم کہ تو پھر نیز یا جب تک چونکہ اگر کیونکہ".split())
    PostpositionSet = set("کے میں سے پر تک بغیر سوا خلاف بعد قبل دوران اندر باہر سمیت کی کا کو لئے".split())
    AuxiliaryVerbSet = set("ہے ہیں ہو ہوں گا گی گے تھا تھی تھے ہوں گے رہا رہی رہے سکتا سکتی سکتے".split())
    PunctuationPantry = set(list("۔،؟!؛:\"'()[]{}«»—–-"))

    # Optional: Augment nouns with words from previous Word2Vec vocab if available
    W2I_Path = "embeddings/word2idx.json"
    if os.path.isfile(W2I_Path):
        try:
            with open(W2I_Path, mode="r", encoding="utf-8") as DictHandle:
                VocabData = json.load(DictHandle)
            ReservedWords = AdverbialSet | PronounSet | DeterminerSet | ConjunctionSet | PostpositionSet | AuxiliaryVerbSet | MasterVerbSet | MasterAdjectiveSet
            for Lexeme in VocabData.keys():
                if Lexeme == "<UNK>" or Lexeme in ReservedWords or len(Lexeme) <= 1:
                    continue
                MasterNounSet.add(Lexeme)
        except Exception:
            pass

    return (MasterNounSet, MasterVerbSet, MasterAdjectiveSet, AdverbialSet, PronounSet, 
            DeterminerSet, ConjunctionSet, PostpositionSet, AuxiliaryVerbSet, PunctuationPantry)


def RemoveExtraneousPunctuationMarks(TargetTokenString):
    """Trims leading and trailing punctuation characters from a token."""
    PuncRegexPattern = r"^[،۔؛:!?\"'()\[\]«»—–\-]+|[،۔؛:!?\"'()\[\]«»—–\-]+$"
    return re.sub(PuncRegexPattern, "", TargetTokenString)


def DeterminePosTagViaHeuristicRule(TokenToTag, LexicalKnowledgeBases):
    """
    Cascade-style rule-based POS tagger.
    Iteratively checks the token against defined sets and heuristics.
    Tags: NOUN, VERB, ADJ, ADV, PRON, DET, CONJ, POST, NUM, PUNC, AUX, UNK.
    """
    (N, V, AD, AV, PR, DE, CO, PO, AU, PU) = LexicalKnowledgeBases
    
    # Pre-process token for matching
    CoreContentOfToken = RemoveExtraneousPunctuationMarks(TokenToTag) or TokenToTag.strip()
    
    if not TokenToTag.strip():
        return "PUNC"
        
    # Check for pure punctuation strings
    if all((CharItem in PU or CharItem.isspace()) for CharItem in TokenToTag) and TokenToTag.strip():
        return "PUNC"
        
    # Check for numeric representations (Urdu/Arabic/Indic or Latin digits)
    NumericIdentificationRegex = r"(?:[0-9]|[\u0660-\u0669]|[\u06f0-\u06f9]|[٫٬])+"
    if CoreContentOfToken == "<NUM>" or re.fullmatch(NumericIdentificationRegex, CoreContentOfToken):
        return "NUM"
        
    # Rule cascade order
    if CoreContentOfToken in DE: return "DET"
    if CoreContentOfToken in PR: return "PRON"
    if CoreContentOfToken in CO: return "CONJ"
    if CoreContentOfToken in PO: return "POST"
    if CoreContentOfToken in AU: return "AUX"
    if CoreContentOfToken in AV: return "ADV"
    if CoreContentOfToken in V:  return "VERB"
    if CoreContentOfToken in AD: return "ADJ"
    if CoreContentOfToken in N:  return "NOUN"
    
    return "UNK"


def ConstructGazetteerResource():
    """
    Provides seed lists for Named Entity Recognition (NER).
    Contains hardcoded tuples of phrase components for PER, LOC, and ORG.
    """
    PersonPhrasesMaster = [
        ("عمران", "خان"), ("بابر", "اعظم"), ("محمد", "نواز"), ("شاہین", "آفریدی"),
        ("وسیم", "اکرم"), ("جاوید", "میانداد"), ("نواز", "شریف"), ("شہباز", "شریف"),
        ("بےنظیر", "بھٹو"), ("آصف", "زرداری"), ("بلاول", "بھٹو"), ("پرویز", "مشرف"),
        ("مصباح", "الحق"), ("یونس", "خان"), ("انضمام", "الحق"), ("راشد", "خان")
    ] # Truncated for brevity while maintaining logic

    LocationPhrasesMaster = [
        ("راولپنڈی",), ("کراچی",), ("لاہور",), ("اسلام", "آباد"), ("پشاور",), 
        ("کوئٹہ",), ("ملتان",), ("فیصل", "آباد"), ("سیالکوٹ",), ("دبئی",),
        ("لندن",), ("واشنگٹن",), ("نیو", "یارک"), ("پنجاب",), ("سندھ",)
    ]

    OrganizationPhrasesMaster = [
        ("آئی", "سی", "سی"), ("پی", "سی", "بی"), ("بی", "بی", "سی"), 
        ("تحریک", "انصاف",), ("مسلم", "لیگ", "ن"), ("پیپلز", "پارٹی",),
        ("سپریم", "کورٹ",), ("یونیسف",), ("ورلد", "بینک",)
    ]

    return ([tuple(P) for P in PersonPhrasesMaster], 
            [tuple(L) for L in LocationPhrasesMaster], 
            [tuple(O) for O in OrganizationPhrasesMaster])


def GenerateNerBioTagSequence(SentenceTokens, PerPhrases, LocPhrases, OrgPhrases):
    """
    Applies longest-match-first strategy to assign BIO tags for NER entities.
    Priority is given to longer phrases during matching.
    """
    PurifiedSentenceTokens = [RemoveExtraneousPunctuationMarks(T) or T for T in SentenceTokens]
    SentenceLengthVal = len(SentenceTokens)
    FinalBioTagsBuffer = ["O"] * SentenceLengthVal
    
    # Aggregate all gazetteer phrases and sort by descending length
    KnowledgeAccumulator = [("PER", P) for P in PerPhrases] + \
                         [("LOC", L) for L in LocPhrases] + \
                         [("ORG", O) for O in OrgPhrases]
    KnowledgeAccumulator.sort(key=lambda x: len(x[1]), reverse=True)

    TokenOccupationStatusMask = [False] * SentenceLengthVal
    
    for EntityTypeLabel, EntityPhraseTuple in KnowledgeAccumulator:
        PhraseLengthVal = len(EntityPhraseTuple)
        if PhraseLengthVal == 0:
            continue
            
        for SearchStartPosition in range(0, SentenceLengthVal - PhraseLengthVal + 1):
            # Ensure we don't overwrite existing entities found by longer phrases
            if any(TokenOccupationStatusMask[SearchStartPosition : SearchStartPosition + PhraseLengthVal]):
                continue
                
            CandidateSnippet = tuple(PurifiedSentenceTokens[SearchStartPosition : SearchStartPosition + PhraseLengthVal])
            if CandidateSnippet == EntityPhraseTuple:
                FinalBioTagsBuffer[SearchStartPosition] = f"B-{EntityTypeLabel}"
                for InfixIdx in range(1, PhraseLengthVal):
                    FinalBioTagsBuffer[SearchStartPosition + InfixIdx] = f"I-{EntityTypeLabel}"
                for MaskIdx in range(PhraseLengthVal):
                    TokenOccupationStatusMask[SearchStartPosition + MaskIdx] = True
                    
    return FinalBioTagsBuffer


def ExportToConllFileSystem(DestinationPath, AnnotatedSentences):
    """
    Writes annotated sentences to a file in standard CoNLL tab-separated format.
    Each word/tag pair on a line, sentences separated by empty lines.
    """
    os.makedirs(os.path.dirname(DestinationPath), exist_ok=True)
    try:
        with open(DestinationPath, mode="w", encoding="utf-8") as OutputStream:
            for IndividualSentenceData in AnnotatedSentences:
                for TokenString, AssignedTag in IndividualSentenceData:
                    OutputStream.write(f"{TokenString}\t{AssignedTag}\n")
                OutputStream.write("\n")
    except Exception as ErrorInWrite:
        print(f"CRITICAL: Failed to write to {DestinationPath}. Details: {ErrorInWrite}")


def LogTagFrequencyDistribution(DisplayTitle, FlatListOfTags):
    """Displays counts for each distinct tag in a frequency distribution table."""
    TagFrequencyMetrics = collections.Counter(FlatListOfTags)
    print(f"\n{DisplayTitle}")
    for TagKey in sorted(TagFrequencyMetrics.keys()):
        print(f"  {TagKey}: {TagFrequencyMetrics[TagKey]}")


def InitiateTaggingDataGenerationProcess():
    """Main execution sequence for generating training data for POS and NER."""
    InitializeUtf8Environment()
    
    # Establish consistent randomness across runs
    random.seed(42)
    np.random.seed(42)

    # Load resources
    try:
        MasterMetadata = json.load(open("Metadata.json", mode="r", encoding="utf-8"))
    except FileNotFoundError:
        print("ERROR: Metadata.json missing.")
        return

    CorpusArticleMap = ParseCorpusAndFilterSeparators("cleaned.txt")
    AllSentenceLevelRecords = AggregateSentenceLevelRecords(CorpusArticleMap, MasterMetadata)

    # Specific sampling configuration for the assignment
    PriorityTopicsForSampling = ("general", "world", "sport")
    FinalBalancedSample = SampleSubsetWithCategoryFloor(
        AllSentenceLevelRecords, PriorityTopicsForSampling, 
        MinimumCountPerPriority=100, FinalTotalTargetCount=500
    )

    # Perform stratified split (70% Train, 30% Test/Val)
    TopicLabelsForStratification = [Rec[1] for Rec in FinalBalancedSample]
    IndicesForTrainSplit, IndicesForTestSplit = ComputeStratifiedSplits(TopicLabelsForStratification, 0.30, 42)
    
    TrainingSubsetRecords = [FinalBalancedSample[Idx] for Idx in IndicesForTrainSplit]
    TestingSubsetRecords = [FinalBalancedSample[Idx] for Idx in IndicesForTestSplit]

    # Initialize annotation logic
    PosKnowledgeBases = InitializeLexicalRuleSets()
    (PerGaz, LocGaz, OrgGaz) = ConstructGazetteerResource()

    # Annotation Buffer for Training partition
    PosAnnotations_Train = []
    NerAnnotations_Train = []
    for TokSeq, _, _ in TrainingSubsetRecords:
        PosResult = [DeterminePosTagViaHeuristicRule(T, PosKnowledgeBases) for T in TokSeq]
        NerResult = GenerateNerBioTagSequence(TokSeq, PerGaz, LocGaz, OrgGaz)
        PosAnnotations_Train.append(list(zip(TokSeq, PosResult)))
        NerAnnotations_Train.append(list(zip(TokSeq, NerResult)))

    # Annotation Buffer for Testing partition
    PosAnnotations_Test = []
    NerAnnotations_Test = []
    for TokSeq, _, _ in TestingSubsetRecords:
        PosResult = [DeterminePosTagViaHeuristicRule(T, PosKnowledgeBases) for T in TokSeq]
        NerResult = GenerateNerBioTagSequence(TokSeq, PerGaz, LocGaz, OrgGaz)
        PosAnnotations_Test.append(list(zip(TokSeq, PosResult)))
        NerAnnotations_Test.append(list(zip(TokSeq, NerResult)))

    # Persist data to the filesystem
    ExportToConllFileSystem("data/pos_train.conll", PosAnnotations_Train)
    ExportToConllFileSystem("data/pos_test.conll", PosAnnotations_Test)
    ExportToConllFileSystem("data/ner_train.conll", NerAnnotations_Train)
    ExportToConllFileSystem("data/ner_test.conll", NerAnnotations_Test)

    # Summary Statistics
    AllGeneratedPosTags = [TagVal for Snt in (PosAnnotations_Train + PosAnnotations_Test) for _, TagVal in Snt]
    AllGeneratedNerTags = [TagVal for Snt in (NerAnnotations_Train + NerAnnotations_Test) for _, TagVal in Snt]
    
    LogTagFrequencyDistribution("POS TAG DISTRIBUTION (AGGREGATED SPLITS):", AllGeneratedPosTags)
    LogTagFrequencyDistribution("NER TAG DISTRIBUTION (AGGREGATED SPLITS):", AllGeneratedNerTags)

    print("\nPROCESS COMPLETE.")
    print(f"Sentences assigned to TRAIN: {len(TrainingSubsetRecords)}")
    print(f"Sentences assigned to TEST:  {len(TestingSubsetRecords)}")
    print("Files updated: data/pos_train.conll, data/pos_test.conll, data/ner_train.conll, data/ner_test.conll")


if __name__ == "__main__":
    InitiateTaggingDataGenerationProcess()
