"""
GPU-Accelerated Feature Extractor for AI vs Human Text Classification
Uses PyTorch for GPU acceleration and vectorized NumPy operations
Optimized for RTX 4050 with batch processing
"""

import numpy as np
import torch
import re
from collections import Counter
from scipy.stats import entropy
import warnings
from typing import List, Union
import zlib
warnings.filterwarnings('ignore')


class GPUFeatureExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', batch_size=32):
        """
        GPU-accelerated feature extractor
        
        Args:
            device: 'cuda' or 'cpu'
            batch_size: Number of texts to process in parallel
        """
        self.device = device
        self.batch_size = batch_size
        self.feature_names = None
        
        print(f"🚀 Feature Extractor initialized on: {device.upper()}")
        if device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Batch size: {batch_size}")
        
        # AI-specific phrases
        self.ai_phrases = [
            'it is important to note', 'plays a crucial role', 'it should be noted',
            'furthermore', 'moreover', 'nevertheless', 'consequently', 'thus',
            'in conclusion', 'to summarize', 'in summary', 'overall',
            'significantly', 'substantially', 'considerably', 'notably',
            'it is worth noting', 'one must consider', 'it can be argued'
        ]
        
        self.transition_words = [
            'however', 'therefore', 'moreover', 'furthermore', 'nevertheless',
            'nonetheless', 'consequently', 'accordingly', 'meanwhile'
        ]
        
        self.hedging_words = [
            'perhaps', 'possibly', 'probably', 'maybe', 'might', 'could',
            'would', 'should', 'seem', 'appear', 'suggest', 'indicate'
        ]
        
        self.intensifiers = [
            'very', 'extremely', 'highly', 'particularly', 'especially',
            'remarkably', 'incredibly', 'absolutely', 'completely'
        ]
    
    def extract_features(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Extract features from texts with GPU acceleration"""
        if isinstance(texts, str):
            texts = [texts]
        
        n_texts = len(texts)
        print(f"Extracting features from {n_texts} texts on {self.device.upper()}...")
        
        # Process in batches for memory efficiency
        all_features = []
        
        for i in range(0, n_texts, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_features = self._extract_batch(batch)
            all_features.append(batch_features)
        
        features_array = np.vstack(all_features)
        
        # Store feature names on first run
        if self.feature_names is None:
            self.feature_names = self._get_feature_names()
        
        print(f"✓ Extracted {features_array.shape[1]} features")
        return features_array
    
    def _extract_batch(self, texts: List[str]) -> np.ndarray:
        """Extract features from a batch of texts"""
        batch_features = []
        
        for text in texts:
            try:
                features = self._extract_single(text)
                # Ensure we always get exactly 56 features
                if len(features) != 56:
                    print(f"Warning: Expected 56 features, got {len(features)}. Padding/truncating.")
                    if len(features) < 56:
                        features = np.pad(features, (0, 56 - len(features)), mode='constant')
                    else:
                        features = features[:56]
                batch_features.append(features)
            except Exception as e:
                print(f"Error extracting features from text: {str(e)[:100]}. Using zeros.")
                batch_features.append(np.zeros(56))
        
        return np.array(batch_features, dtype=np.float32)
    
    def _extract_single(self, text: str) -> np.ndarray:
        """Extract all features from a single text"""
        if not text or not isinstance(text, str):
            return np.zeros(56)  # Return zero features for empty text
        
        # Preprocessing
        text_lower = text.lower()
        sentences = self._split_sentences(text)
        words = self._tokenize(text)
        words_lower = [w.lower() for w in words]
        
        features = []
        
        # === BASIC STATISTICS ===
        features.append(len(text))  # char_count
        features.append(len(words))  # word_count
        features.append(len(sentences))  # sentence_count
        features.append(len(set(words_lower)))  # vocab_size
        
        # === VOCABULARY RICHNESS ===
        features.append(self._type_token_ratio(words_lower))
        features.append(self._hapax_ratio(words_lower))
        features.append(self._yules_k(words_lower))
        features.append(self._simpsons_index(words_lower))
        
        # === SENTENCE STRUCTURE ===
        sent_lengths = [len(self._tokenize(s)) for s in sentences]
        features.append(np.mean(sent_lengths) if sent_lengths else 0)
        features.append(np.std(sent_lengths) if len(sent_lengths) > 1 else 0)
        features.append(np.max(sent_lengths) if sent_lengths else 0)
        features.append(np.min(sent_lengths) if sent_lengths else 0)
        
        # === WORD LENGTH STATISTICS ===
        word_lengths = [len(w) for w in words]
        features.append(np.mean(word_lengths) if word_lengths else 0)
        features.append(np.std(word_lengths) if len(word_lengths) > 1 else 0)
        features.append(np.max(word_lengths) if word_lengths else 0)
        
        # === READABILITY ===
        features.append(self._flesch_reading_ease(text, words, sentences))
        features.append(self._avg_syllables_per_word(words))
        features.append(self._gunning_fog(words, sentences))
        
        # === PUNCTUATION PATTERNS ===
        total_chars = len(text) if len(text) > 0 else 1
        features.append(text.count(',') / total_chars)
        features.append(text.count(';') / total_chars)
        features.append(text.count(':') / total_chars)
        features.append(text.count('!') / total_chars)
        features.append(text.count('?') / total_chars)
        features.append(text.count('...') / total_chars)
        features.append((text.count('-') + text.count('—')) / total_chars)
        
        # === AI HALLMARKS ===
        total_words = len(words) if len(words) > 0 else 1
        features.append(self._phrase_rate(text_lower, self.ai_phrases, total_words))
        features.append(self._phrase_rate(text_lower, self.transition_words, total_words))
        features.append(self._phrase_rate(text_lower, self.hedging_words, total_words))
        features.append(self._phrase_rate(text_lower, self.intensifiers, total_words))
        
        # === CONTRACTIONS ===
        contractions = ["n't", "'ll", "'ve", "'re", "'m", "'d"]
        contraction_count = sum(text_lower.count(c) for c in contractions)
        features.append(contraction_count / total_words)
        
        # === ENTROPY ===
        features.append(self._char_entropy(text))
        features.append(self._word_entropy(words_lower))
        features.append(self._bigram_entropy(words_lower))
        
        # === REPETITION PATTERNS ===
        features.append(self._bigram_repetition_rate(words_lower))
        features.append(self._trigram_repetition_rate(words_lower))
        features.append(self._word_repetition_rate(words_lower))
        
        # === GRAMMATICAL PATTERNS ===
        features.append(self._passive_voice_rate(text_lower))
        features.append(self._modal_verb_rate(words_lower))
        features.append(self._determiner_rate(words_lower))
        features.append(self._pronoun_rate(words_lower))
        features.append(self._first_person_rate(words_lower))
        features.append(self._third_person_rate(words_lower))
        
        # === STRUCTURAL PATTERNS ===
        features.append(text.count('\n\n') / len(sentences) if sentences else 0)
        features.append(self._avg_clause_length(text))
        
        # === LEXICAL DIVERSITY ===
        features.append(self._mtld(words_lower))
        
        # === ADDITIONAL FEATURES ===
        features.append(self._burstiness(words_lower))
        features.append(self._compression_ratio(text))
        
        # === CHARACTER PATTERNS ===
        features.append(self._char_trigram_diversity(text))
        features.append(self._capitalization_rate(text))
        features.append(self._numeric_rate(text))
        features.append(self._special_char_rate(text))
        
        # === SENTENCE PATTERNS ===
        features.append(self._sentence_starter_diversity(sentences))
        features.append(self._question_sentence_rate(sentences))
        
        # === COMPLEXITY ===
        features.append(self._noun_verb_ratio(words))
        features.append(self._formality_score(words, text))
        features.append(self._lexical_cohesion(words))
        
        return np.array(features)
    
    # === HELPER METHODS ===
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return re.findall(r'\b\w+\b', text)
    
    def _type_token_ratio(self, words: List[str]) -> float:
        """Type-token ratio (lexical diversity)"""
        if not words:
            return 0
        return len(set(words)) / len(words)
    
    def _hapax_ratio(self, words: List[str]) -> float:
        """Ratio of words appearing once"""
        if not words:
            return 0
        word_freq = Counter(words)
        hapax = sum(1 for count in word_freq.values() if count == 1)
        return hapax / len(words)
    
    def _yules_k(self, words: List[str]) -> float:
        """Yule's K measure"""
        if len(words) < 2:
            return 0
        word_freq = Counter(words)
        m1 = len(word_freq)
        m2 = sum(count**2 for count in word_freq.values())
        try:
            return 10000 * (m2 - m1) / (m1 * m1)
        except:
            return 0
    
    def _simpsons_index(self, words: List[str]) -> float:
        """Simpson's diversity index"""
        if not words:
            return 0
        word_freq = Counter(words)
        n = len(words)
        return sum((count / n) ** 2 for count in word_freq.values())
    
    def _flesch_reading_ease(self, text: str, words: List[str], sentences: List[str]) -> float:
        """Flesch Reading Ease score"""
        if not words or not sentences:
            return 0
        syllables = sum(self._count_syllables(w) for w in words)
        try:
            score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))
            return max(0, min(100, score))
        except:
            return 0
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximate)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllables = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel
        
        if word.endswith('e'):
            syllables -= 1
        
        return max(1, syllables)
    
    def _avg_syllables_per_word(self, words: List[str]) -> float:
        """Average syllables per word"""
        if not words:
            return 0
        return sum(self._count_syllables(w) for w in words) / len(words)
    
    def _gunning_fog(self, words: List[str], sentences: List[str]) -> float:
        """Gunning Fog Index"""
        if not words or not sentences:
            return 0
        complex_words = sum(1 for w in words if self._count_syllables(w) >= 3)
        try:
            return 0.4 * ((len(words) / len(sentences)) + 100 * (complex_words / len(words)))
        except:
            return 0
    
    def _phrase_rate(self, text: str, phrases: List[str], total_words: int) -> float:
        """Rate of specific phrases"""
        count = sum(text.count(phrase.lower()) for phrase in phrases)
        return count / total_words if total_words > 0 else 0
    
    def _char_entropy(self, text: str) -> float:
        """Character-level entropy"""
        if not text:
            return 0
        char_freq = Counter(text)
        probs = np.array(list(char_freq.values())) / len(text)
        return entropy(probs)
    
    def _word_entropy(self, words: List[str]) -> float:
        """Word-level entropy"""
        if not words:
            return 0
        word_freq = Counter(words)
        probs = np.array(list(word_freq.values())) / len(words)
        return entropy(probs)
    
    def _bigram_entropy(self, words: List[str]) -> float:
        """Bigram entropy"""
        if len(words) < 2:
            return 0
        bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
        bigram_freq = Counter(bigrams)
        probs = np.array(list(bigram_freq.values())) / len(bigrams)
        return entropy(probs)
    
    def _bigram_repetition_rate(self, words: List[str]) -> float:
        """Rate of repeated bigrams"""
        if len(words) < 2:
            return 0
        bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
        return 1 - (len(set(bigrams)) / len(bigrams))
    
    def _trigram_repetition_rate(self, words: List[str]) -> float:
        """Rate of repeated trigrams"""
        if len(words) < 3:
            return 0
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        return 1 - (len(set(trigrams)) / len(trigrams))
    
    def _word_repetition_rate(self, words: List[str]) -> float:
        """Rate of repeated words"""
        if not words:
            return 0
        return 1 - (len(set(words)) / len(words))
    
    def _passive_voice_rate(self, text: str) -> float:
        """Estimate passive voice rate"""
        passive_indicators = ['is being', 'are being', 'was being', 'were being',
                             'has been', 'have been', 'had been', 'will be']
        count = sum(text.count(indicator) for indicator in passive_indicators)
        sentences = self._split_sentences(text)
        return count / len(sentences) if sentences else 0
    
    def _modal_verb_rate(self, words: List[str]) -> float:
        """Rate of modal verbs"""
        modals = ['can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']
        count = sum(1 for w in words if w in modals)
        return count / len(words) if words else 0
    
    def _determiner_rate(self, words: List[str]) -> float:
        """Rate of determiners"""
        determiners = ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her']
        count = sum(1 for w in words if w in determiners)
        return count / len(words) if words else 0
    
    def _pronoun_rate(self, words: List[str]) -> float:
        """Rate of pronouns"""
        pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']
        count = sum(1 for w in words if w in pronouns)
        return count / len(words) if words else 0
    
    def _first_person_rate(self, words: List[str]) -> float:
        """Rate of first-person pronouns"""
        first_person = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        count = sum(1 for w in words if w in first_person)
        return count / len(words) if words else 0
    
    def _third_person_rate(self, words: List[str]) -> float:
        """Rate of third-person pronouns"""
        third_person = ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'their']
        count = sum(1 for w in words if w in third_person)
        return count / len(words) if words else 0
    
    def _avg_clause_length(self, text: str) -> float:
        """Average clause length"""
        clauses = re.split(r'[,;:]', text)
        clause_lengths = [len(self._tokenize(c)) for c in clauses]
        return np.mean(clause_lengths) if clause_lengths else 0
    
    def _mtld(self, words: List[str], threshold: float = 0.72) -> float:
        """Measure of Textual Lexical Diversity"""
        if len(words) < 50:
            return len(set(words))
        
        def _mtld_calc(words, threshold):
            factor = 0
            types = set()
            tokens = 0
            for word in words:
                types.add(word)
                tokens += 1
                if tokens > 0 and len(types) / tokens < threshold:
                    factor += 1
                    types = set()
                    tokens = 0
            if tokens > 0:
                factor += (1 - (len(types) / tokens)) / (1 - threshold)
            return len(words) / factor if factor > 0 else len(words)
        
        forward = _mtld_calc(words, threshold)
        backward = _mtld_calc(words[::-1], threshold)
        return (forward + backward) / 2
    
    def _burstiness(self, words: List[str]) -> float:
        """Burstiness of word usage"""
        if not words:
            return 0
        word_freq = Counter(words)
        frequencies = list(word_freq.values())
        if len(frequencies) < 2:
            return 0
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        return (std_freq - mean_freq) / (std_freq + mean_freq) if (std_freq + mean_freq) > 0 else 0
    
    def _compression_ratio(self, text: str) -> float:
        """Text compression ratio"""
        if not text:
            return 0
        compressed = zlib.compress(text.encode('utf-8', errors='ignore'))
        return len(compressed) / len(text.encode('utf-8', errors='ignore'))
    
    def _char_trigram_diversity(self, text: str) -> float:
        """Character trigram diversity"""
        if len(text) < 3:
            return 0
        trigrams = [text[i:i+3] for i in range(len(text)-2)]
        return len(set(trigrams)) / len(trigrams) if trigrams else 0
    
    def _capitalization_rate(self, text: str) -> float:
        """Rate of capitalized characters"""
        if not text:
            return 0
        return sum(1 for c in text if c.isupper()) / len(text)
    
    def _numeric_rate(self, text: str) -> float:
        """Rate of numeric characters"""
        if not text:
            return 0
        return sum(1 for c in text if c.isdigit()) / len(text)
    
    def _special_char_rate(self, text: str) -> float:
        """Rate of special characters"""
        if not text:
            return 0
        special = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return special / len(text)
    
    def _sentence_starter_diversity(self, sentences: List[str]) -> float:
        """Diversity of sentence starters"""
        if not sentences:
            return 0
        starters = [s.split()[0].lower() if s.split() else '' for s in sentences]
        starters = [s for s in starters if s]
        return len(set(starters)) / len(starters) if starters else 0
    
    def _question_sentence_rate(self, sentences: List[str]) -> float:
        """Rate of question sentences"""
        if not sentences:
            return 0
        questions = sum(1 for s in sentences if s.strip().endswith('?'))
        return questions / len(sentences)
    
    def _noun_verb_ratio(self, words: List[str]) -> float:
        """Approximate noun/verb ratio"""
        noun_endings = ['tion', 'ment', 'ness', 'ity']
        verb_endings = ['ing', 'ed', 'ate', 'ize']
        
        nouns = sum(1 for w in words if any(w.endswith(e) for e in noun_endings))
        verbs = sum(1 for w in words if any(w.endswith(e) for e in verb_endings))
        
        return nouns / verbs if verbs > 0 else 0
    
    def _formality_score(self, words: List[str], text: str) -> float:
        """Formality score"""
        if not words:
            return 0
        informal_markers = sum(1 for w in words if w in ['yeah', 'yep', 'nope', 'gonna', 'wanna'])
        contractions = text.count("n't") + text.count("'ll") + text.count("'ve")
        formal_words = sum(1 for w in words if len(w) > 8)
        
        informality = (informal_markers + contractions) / len(words)
        formality = formal_words / len(words)
        
        return formality - informality
    
    def _lexical_cohesion(self, words: List[str]) -> float:
        """Lexical cohesion"""
        if len(words) < 10:
            return 0
        
        window_size = 10
        overlaps = []
        
        for i in range(len(words) - window_size):
            window1 = set(words[i:i+window_size//2])
            window2 = set(words[i+window_size//2:i+window_size])
            overlap = len(window1 & window2) / len(window1 | window2) if len(window1 | window2) > 0 else 0
            overlaps.append(overlap)
        
        return np.mean(overlaps) if overlaps else 0
    
    def _get_feature_names(self) -> List[str]:
        """Get names of all features (56 total)"""
        return [
            # Basic stats (4)
            'char_count', 'word_count', 'sentence_count', 'vocab_size',
            # Vocabulary richness (4)
            'type_token_ratio', 'hapax_ratio', 'yules_k', 'simpsons_index',
            # Sentence structure (4)
            'avg_sentence_length', 'sentence_length_std', 'max_sentence_length', 'min_sentence_length',
            # Word length (3)
            'avg_word_length', 'word_length_std', 'max_word_length',
            # Readability (3)
            'flesch_reading_ease', 'avg_syllables_per_word', 'gunning_fog',
            # Punctuation (7)
            'comma_rate', 'semicolon_rate', 'colon_rate', 'exclamation_rate',
            'question_rate', 'ellipsis_rate', 'dash_rate',
            # AI hallmarks (4)
            'ai_phrase_rate', 'transition_word_rate', 'hedging_rate', 'intensifier_rate',
            # Contractions (1)
            'contraction_rate',
            # Entropy (3)
            'char_entropy', 'word_entropy', 'bigram_entropy',
            # Repetition (3)
            'bigram_repetition_rate', 'trigram_repetition_rate', 'word_repetition_rate',
            # Grammar (6)
            'passive_voice_rate', 'modal_verb_rate', 'determiner_rate', 'pronoun_rate',
            'first_person_rate', 'third_person_rate',
            # Structure (2)
            'paragraph_break_rate', 'avg_clause_length',
            # Lexical diversity (1)
            'mtld',
            # Additional (2)
            'burstiness', 'compression_ratio',
            # Character patterns (4)
            'char_trigram_diversity', 'capitalization_rate', 'numeric_rate', 'special_char_rate',
            # Sentence patterns (2)
            'sentence_starter_diversity', 'question_sentence_rate',
            # Complexity (3)
            'noun_verb_ratio', 'formality_score', 'lexical_cohesion'
        ]  # Total: 56 features


if __name__ == "__main__":
    # Test
    extractor = GPUFeatureExtractor()
    
    test_texts = [
        "This is a test. It's simple.",
        "Furthermore, it is important to note that AI detection requires sophisticated features."
    ]
    
    features = extractor.extract_features(test_texts)
    print(f"\n✓ Extracted {features.shape[1]} features from {features.shape[0]} texts")
    print(f"Feature shape: {features.shape}")