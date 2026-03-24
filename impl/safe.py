import numpy as np
import string
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from collections import Counter
from saxpy.znorm import znorm
from saxpy.alphabet import cuts_for_asize
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD


# ============== Embedding Strategy Interface ==============

class EmbeddingStrategy(ABC):
    """Abstract base class for embedding strategies."""
    
    @abstractmethod
    def fit(self, all_word_sequences: List[List[str]], config: Dict[str, Any]) -> None:
        """Train the embedding model."""
        pass
    
    @abstractmethod
    def embed(self, all_word_sequences: List[List[str]], embedding_dim: int) -> np.ndarray:
        """Embed word sequences using trained model."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this strategy."""
        pass
    
    @abstractmethod
    def uses_epochs(self) -> bool:
        """Return whether this strategy uses the epochs parameter."""
        pass


# ============== Word2Vec Strategy ==============

class Word2VecStrategy(EmbeddingStrategy):
    """
    Word2Vec embedding strategy (original SAFE approach).
    
    Key parameters:
    - embedding_epochs: Number of training epochs (default: 100)
    - window_size: Context window size (default: 5)
    """
    
    def __init__(self):
        self.model = None
    
    def fit(self, all_word_sequences: List[List[str]], config: Dict[str, Any]) -> None:
        if not config.get('verbose', True):
            logging.getLogger('gensim').setLevel(logging.WARNING)
        
        self.model = Word2Vec(
            sentences=all_word_sequences,
            vector_size=config['embedding_dim'],
            window=config.get('window_size', 5),
            min_count=1,
            epochs=config['embedding_epochs'],
            sg=1,  # Skip-gram
            seed=config.get('seed', 42),
        )
    
    def embed(self, all_word_sequences: List[List[str]], embedding_dim: int) -> np.ndarray:
        embedded_sequences = []
        for words in all_word_sequences:
            seq_embeddings = []
            for w in words:
                if w in self.model.wv:
                    seq_embeddings.append(self.model.wv[w])
                else:
                    seq_embeddings.append(np.zeros(embedding_dim))
            embedded_sequences.append(seq_embeddings)
        return np.array(embedded_sequences)
    
    def get_name(self) -> str:
        return "word2vec"
    
    def uses_epochs(self) -> bool:
        return True


# ============== FastText Strategy ==============

class FastTextStrategy(EmbeddingStrategy):
    """
    FastText embedding strategy - handles OOV words via character n-grams.
    
    Key parameters:
    - embedding_epochs: Number of training epochs (default: 100)
    - window_size: Context window size (default: 5)
    - fasttext_min_ngram: Min character n-gram length (default: auto based on word_length)
    - fasttext_max_ngram: Max character n-gram length (default: auto based on word_length)
    
    Auto n-gram logic: For SAX words of length L, max_ngram = min(L, 4)
    """
    
    def __init__(self):
        self.model = None
    
    def fit(self, all_word_sequences: List[List[str]], config: Dict[str, Any]) -> None:
        if not config.get('verbose', True):
            logging.getLogger('gensim').setLevel(logging.WARNING)
        
        word_length = config.get('word_length', 6)
        
        # Auto-adjust n-gram range based on word length
        min_n = config.get('fasttext_min_ngram', 2)
        max_n = config.get('fasttext_max_ngram', min(word_length, 4))
        
        # Ensure valid range
        min_n = max(1, min(min_n, word_length))
        max_n = max(min_n, min(max_n, word_length))
        
        self.model = FastText(
            sentences=all_word_sequences,
            vector_size=config['embedding_dim'],
            window=config.get('window_size', 5),
            min_count=1,
            epochs=config['embedding_epochs'],
            sg=1,
            seed=config.get('seed', 42),
            min_n=min_n,
            max_n=max_n,
        )
    
    def embed(self, all_word_sequences: List[List[str]], embedding_dim: int) -> np.ndarray:
        embedded_sequences = []
        for words in all_word_sequences:
            seq_embeddings = []
            for w in words:
                # FastText can produce embeddings for OOV words via subword n-grams
                seq_embeddings.append(self.model.wv[w])
            embedded_sequences.append(seq_embeddings)
        return np.array(embedded_sequences)
    
    def get_name(self) -> str:
        return "fasttext"
    
    def uses_epochs(self) -> bool:
        return True


# ============== Doc2Vec Strategy ==============

class Doc2VecStrategy(EmbeddingStrategy):
    """
    Doc2Vec embedding strategy - one vector per sequence (time series).
    
    Key parameters:
    - embedding_epochs: Number of training epochs (RECOMMENDED: 200+ for good inference)
    - window_size: Context window size (default: 5)
    - doc2vec_dm: Architecture type (1=PV-DM, 0=PV-DBOW, default: 1)
    
    Note: Doc2Vec typically needs MORE epochs than Word2Vec for stable inference.
    """
    
    def __init__(self):
        self.model = None
    
    def fit(self, all_word_sequences: List[List[str]], config: Dict[str, Any]) -> None:
        if not config.get('verbose', True):
            logging.getLogger('gensim').setLevel(logging.WARNING)
        
        documents = [TaggedDocument(words=seq, tags=[str(i)]) 
                     for i, seq in enumerate(all_word_sequences)]
        
        dm = config.get('doc2vec_dm', 1)  # 1=PV-DM, 0=PV-DBOW
        
        self.model = Doc2Vec(
            documents=documents,
            vector_size=config['embedding_dim'],
            window=config.get('window_size', 5),
            min_count=1,
            epochs=config['embedding_epochs'],
            dm=dm,
            seed=config.get('seed', 42),
        )
    
    def embed(self, all_word_sequences: List[List[str]], embedding_dim: int) -> np.ndarray:
        # Doc2Vec produces one vector per document/sequence
        # We expand it to 3D by repeating across a "word" dimension
        embeddings = []
        for seq in all_word_sequences:
            vec = self.model.infer_vector(seq)
            # Expand to (n_words, embedding_dim) by repeating
            seq_embedding = np.tile(vec, (len(seq), 1))
            embeddings.append(seq_embedding)
        return np.array(embeddings)
    
    def get_name(self) -> str:
        return "doc2vec"
    
    def uses_epochs(self) -> bool:
        return True


# ============== PPMI + SVD Strategy ==============

class PPMISVDStrategy(EmbeddingStrategy):
    """
    PPMI + TruncatedSVD embedding strategy - count-based, deterministic.
    
    Key parameters:
    - window_size: Co-occurrence window size (default: 5)
    - embedding_dim: Output dimension (via SVD)
    
    Note: This method is DETERMINISTIC - embedding_epochs is IGNORED.
    Advantage: Stable, reproducible results regardless of random seed.
    """
    
    def __init__(self):
        self.svd = None
        self.word_vectors = None
        self.w2i = None
        self.vocab = None
    
    def _build_cooccurrence(self, seqs: List[List[str]], window: int = 5):
        """Build word-context co-occurrence matrix."""
        self.vocab = sorted({w for s in seqs for w in s})
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        counts = Counter()
        
        for s in seqs:
            for i, w in enumerate(s):
                wi = self.w2i[w]
                left = max(0, i - window)
                right = min(len(s), i + window + 1)
                for j in range(left, right):
                    if j != i:
                        cj = self.w2i[s[j]]
                        counts[(wi, cj)] += 1
        
        row, col, data = [], [], []
        for (wi, cj), v in counts.items():
            row.append(wi)
            col.append(cj)
            data.append(v)
        
        X = coo_matrix((data, (row, col)), shape=(len(self.vocab), len(self.vocab)))
        return X
    
    def _compute_ppmi(self, X):
        """Compute Positive Pointwise Mutual Information matrix."""
        X = X.tocsr()
        total = X.sum()
        if total == 0:
            return X
        
        sum_w = np.array(X.sum(axis=1)).ravel()
        sum_c = np.array(X.sum(axis=0)).ravel()
        
        rows, cols = X.nonzero()
        ppmi_vals = []
        
        for r, c in zip(rows, cols):
            v = X[r, c]
            denom = sum_w[r] * sum_c[c]
            if denom > 0:
                pmi = np.log((v * total) / denom)
                ppmi_vals.append(max(pmi, 0.0))
            else:
                ppmi_vals.append(0.0)
        
        return coo_matrix((ppmi_vals, (rows, cols)), shape=X.shape)
    
    def fit(self, all_word_sequences: List[List[str]], config: Dict[str, Any]) -> None:
        window = config.get('window_size', 5)
        embedding_dim = config['embedding_dim']
        
        # Build co-occurrence matrix
        X = self._build_cooccurrence(all_word_sequences, window=window)
        
        # Compute PPMI
        M = self._compute_ppmi(X)
        
        # Apply SVD
        n_components = min(embedding_dim, len(self.vocab) - 1, M.shape[0] - 1)
        if n_components < 1:
            n_components = 1
        
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.word_vectors = self.svd.fit_transform(M)
        
        # Pad if needed
        if self.word_vectors.shape[1] < embedding_dim:
            padding = np.zeros((self.word_vectors.shape[0], 
                               embedding_dim - self.word_vectors.shape[1]))
            self.word_vectors = np.hstack([self.word_vectors, padding])
    
    def embed(self, all_word_sequences: List[List[str]], embedding_dim: int) -> np.ndarray:
        embedded_sequences = []
        for words in all_word_sequences:
            seq_embeddings = []
            for w in words:
                if w in self.w2i:
                    seq_embeddings.append(self.word_vectors[self.w2i[w]])
                else:
                    seq_embeddings.append(np.zeros(embedding_dim))
            embedded_sequences.append(seq_embeddings)
        return np.array(embedded_sequences)
    
    def get_name(self) -> str:
        return "ppmi_svd"
    
    def uses_epochs(self) -> bool:
        return False  # Deterministic method - epochs ignored


# ============== Strategy Factory ==============

def get_embedding_strategy(method: str) -> EmbeddingStrategy:
    """Factory function to get embedding strategy by name."""
    strategies = {
        'word2vec': Word2VecStrategy,
        'fasttext': FastTextStrategy,
        'doc2vec': Doc2VecStrategy,
        'ppmi_svd': PPMISVDStrategy,
    }
    
    method_lower = method.lower()
    if method_lower not in strategies:
        available = ', '.join(strategies.keys())
        raise ValueError(f"Unknown embedding method '{method}'. Available: {available}")
    
    return strategies[method_lower]()


# ============== Main SafeTransformer Class ==============

class SafeTransformer:
    """
    SAFE-style time series transformer with pluggable embedding methods.
    
    Supported embedding methods:
    - 'word2vec': Original SAFE approach (Skip-gram)
    - 'fasttext': Subword-aware, handles OOV words
    - 'doc2vec': Document-level embeddings (needs more epochs)
    - 'ppmi_svd': Deterministic count-based method (ignores epochs)
    
    Parameters:
    -----------
    word_length : int
        Length of SAX words (default: 6)
    alphabet_size : int
        Number of SAX symbols (default: 4)
    embedding_dim : int
        Output embedding dimension (default: 50)
    embedding_epochs : int
        Training epochs for neural methods (default: 100)
        NOTE: Ignored by 'ppmi_svd' method
    embedding_method : str
        Embedding strategy to use (default: 'word2vec')
    window_size : int
        Context window size for all methods (default: 5)
    stride : int, optional
        Step when extracting words from symbol string. If None, stride=word_length (non-overlapping).
        E.g. word_length=4, stride=2 gives 50% overlapping windows.
    seed : int, optional
        Random seed for embedding training. If None, uses 42.
    fasttext_min_ngram : int
        Min n-gram length for FastText (default: 2)
    fasttext_max_ngram : int or None
        Max n-gram length for FastText (default: auto = min(word_length, 4))
    doc2vec_dm : int
        Doc2Vec architecture: 1=PV-DM, 0=PV-DBOW (default: 1)
    force_square_dimension : bool
        Force embedding_dim = n_words (default: False)
    verbose : bool
        Print progress information (default: True)
    """

    def __init__(self, 
                 word_length: int = 6, 
                 alphabet_size: int = 4, 
                 embedding_dim: int = 50, 
                 embedding_epochs: int = 100, 
                 embedding_method: str = 'word2vec',
                 window_size: int = 5,
                 stride: Optional[int] = None,
                 seed: Optional[int] = None,
                 fasttext_min_ngram: int = 2,
                 fasttext_max_ngram: Optional[int] = None,
                 doc2vec_dm: int = 1,
                 force_square_dimension: bool = False,
                 verbose: bool = True):
        
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.embedding_dim = embedding_dim
        self.embedding_epochs = embedding_epochs
        self.embedding_method = embedding_method
        self.window_size = window_size
        self.stride = stride if stride is not None else word_length
        self.seed = seed if seed is not None else 42
        self.fasttext_min_ngram = fasttext_min_ngram
        self.fasttext_max_ngram = fasttext_max_ngram if fasttext_max_ngram else min(word_length, 4)
        self.doc2vec_dm = doc2vec_dm
        self.force_square_dimension = force_square_dimension
        self.verbose = verbose
        
        # Initialize embedding strategy
        self.embedding_strategy = get_embedding_strategy(embedding_method)
        
        # Normalization parameters
        self.normalize_min_val = 0
        self.normalize_max_val = 1

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _znormalize(self, data):
        return znorm(data)

    def _fit_normalize(self, data):
        """Fit normalization parameters to 0-1 range."""
        self.normalize_min_val = data.min()
        self.normalize_max_val = data.max()

    def _normalize(self, data):
        """Normalize data to 0-1 range."""
        if self.normalize_max_val - self.normalize_min_val > 0:
            return (data - self.normalize_min_val) / (self.normalize_max_val - self.normalize_min_val)
        return data

    def _get_config(self) -> Dict[str, Any]:
        """Build configuration dict for embedding strategy."""
        return {
            'embedding_dim': self.embedding_dim,
            'embedding_epochs': self.embedding_epochs,
            'window_size': self.window_size,
            'word_length': self.word_length,
            'seed': self.seed,
            'fasttext_min_ngram': self.fasttext_min_ngram,
            'fasttext_max_ngram': self.fasttext_max_ngram,
            'doc2vec_dm': self.doc2vec_dm,
            'verbose': self.verbose,
        }

    def fit(self, X: np.ndarray, retrain_embedding: bool = True):
        """Fit the transformer and return embeddings."""
        X = np.array(X)
        
        self._log("\n" + "=" * 60)
        self._log("SAFE TRANSFORMER")
        self._log("=" * 60)
        self._log(f"  Input shape: {X.shape}")
        self._log(f"  Embedding method: {self.embedding_method}")
        self._log(f"  Embedding dim: {self.embedding_dim}")
        self._log(f"  Word length: {self.word_length}")
        self._log(f"  Alphabet size: {self.alphabet_size}")
        self._log(f"  Window size: {self.window_size}")
        if self.embedding_strategy.uses_epochs():
            self._log(f"  Epochs: {self.embedding_epochs}")
        else:
            self._log(f"  Epochs: N/A (deterministic method)")
        self._log("=" * 60)

        # Convert all series to word sequences
        all_word_sequences = []
        for series in X:
            normalized = self._znormalize(series)
            symbol_string = self._series_to_symbols(normalized)
            words = self._extract_words(symbol_string)
            all_word_sequences.append(words)
        
        self._log(f"Word sequences shape: {np.array(all_word_sequences).shape}")
        
        if self.force_square_dimension:
            self.embedding_dim = len(all_word_sequences[0])
        
        # Train embedding model
        if retrain_embedding:
            vocab_size = len(set(w for seq in all_word_sequences for w in seq))
            self._log(f"Training {self.embedding_method}: {vocab_size} unique words...")
            
            config = self._get_config()
            self.embedding_strategy.fit(all_word_sequences, config)
        
        # Get embeddings
        embeddings = self.embedding_strategy.embed(all_word_sequences, self.embedding_dim)
        
        # Normalize
        if retrain_embedding:
            self._fit_normalize(embeddings)
        
        return self._normalize(embeddings)

    def _series_to_symbols(self, series):
        """Convert series to symbol string."""
        alphabet = string.ascii_lowercase[:self.alphabet_size]
        breakpoints = cuts_for_asize(self.alphabet_size)

        symbols = []
        for v in series:
            for i, bp in enumerate(breakpoints):
                if v < bp:
                    symbols.append(alphabet[i-1])
                    break
            else:
                symbols.append(alphabet[-1])
        return ''.join(symbols)

    def _extract_words(self, symbol_string):
        """Extract words from symbol string using stride (stride=word_length -> non-overlapping)."""
        words = []
        for i in range(0, len(symbol_string) - self.word_length + 1, self.stride):
            word = symbol_string[i:i + self.word_length]
            words.append(word)
        return words

    def fit_on_train_and_test(self, X_train: np.ndarray, X_test: np.ndarray):
        """Fit on combined train and test data."""
        train_test_merged = np.concatenate([X_train, X_test], axis=0)
        self.fit(train_test_merged)

    def transform(self, X: np.ndarray):
        """Transform data using fitted model."""
        return self.fit(X, retrain_embedding=False)

    def transform_with_word_augmentation(self, X: np.ndarray, Y: np.ndarray):
        """Augment by shifting sequences to balance classes."""
        X, Y = np.array(X), np.array(Y)
        unique, counts = np.unique(Y, return_counts=True)
        max_count = counts.max()
        
        X_aug, Y_aug = list(X), list(Y)
        for cls, count in zip(unique, counts):
            for _ in range(max_count - count):
                seq = X[np.random.choice(np.where(Y == cls)[0])].copy()
                C = np.clip(int(np.random.normal(self.word_length / 2, self.word_length / 4)), 
                           1, self.word_length - 1)
                X_aug.append(np.concatenate([seq[C:], np.full(C, seq[-1])]))
                Y_aug.append(cls)
        
        self._log(f"Augmented: {len(X)} → {len(X_aug)} samples")
        return self.transform(np.array(X_aug)), np.array(Y_aug)


# ============== Available Methods and Parameters ==============

AVAILABLE_EMBEDDING_METHODS = ['word2vec', 'fasttext', 'doc2vec', 'ppmi_svd']

# Parameter recommendations per method
PARAMETER_RECOMMENDATIONS = {
    'word2vec': {
        'embedding_epochs': 100,
        'window_size': 5,
        'notes': 'Standard baseline. Good default choice.'
    },
    'fasttext': {
        'embedding_epochs': 100,
        'window_size': 5,
        'fasttext_max_ngram': 'auto (min of word_length and 4)',
        'notes': 'Better OOV handling. Recommended when vocab fragmentation is expected.'
    },
    'doc2vec': {
        'embedding_epochs': 200,  # Needs more epochs!
        'window_size': 5,
        'doc2vec_dm': 1,
        'notes': 'Needs 2x epochs vs Word2Vec. Try dm=0 (DBOW) for faster training.'
    },
    'ppmi_svd': {
        'embedding_epochs': 'IGNORED',
        'window_size': 5,
        'notes': 'Deterministic. Use when reproducibility is critical.'
    },
}
