import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import re
from collections import Counter
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade
import warnings

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

def load_sample_data(dataset_name: str, sample_size: int = 1000) -> pd.DataFrame:
    if dataset_name.lower() == 'liar':
        from data.sample_datasets import generate_liar_sample
        return generate_liar_sample(sample_size)
    elif dataset_name.lower() == 'truthfulqa':
        from data.sample_datasets import generate_truthfulqa_sample
        return generate_truthfulqa_sample(sample_size)
    elif dataset_name.lower() == 'combined':
        from data.sample_datasets import generate_liar_sample, generate_truthfulqa_sample
        liar_df = generate_liar_sample(sample_size // 2)
        truthful_df = generate_truthfulqa_sample(sample_size // 2)
        combined_df = pd.concat([liar_df, truthful_df], ignore_index=True)
        return combined_df
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def analyze_text_patterns(texts: pd.Series) -> Dict[str, Any]:
    texts = texts.fillna('')

    analysis = {}

    analysis['total_texts'] = len(texts)
    analysis['non_empty_texts'] = len([t for t in texts if str(t).strip()])

    char_lengths = texts.str.len()
    word_lengths = texts.str.split().str.len()

    analysis['char_length_stats'] = {
        'mean': char_lengths.mean(),
        'median': char_lengths.median(),
        'std': char_lengths.std(),
        'min': char_lengths.min(),
        'max': char_lengths.max(),
        'q25': char_lengths.quantile(0.25),
        'q75': char_lengths.quantile(0.75)
    }

    analysis['word_length_stats'] = {
        'mean': word_lengths.mean(),
        'median': word_lengths.median(),
        'std': word_lengths.std(),
        'min': word_lengths.min(),
        'max': word_lengths.max(),
        'q25': word_lengths.quantile(0.25),
        'q75': word_lengths.quantile(0.75)
    }

    analysis['linguistic_patterns'] = analyze_linguistic_patterns(texts)
    analysis['readability'] = calculate_readability_metrics(texts)
    analysis['common_patterns'] = find_common_patterns(texts)

    return analysis

def analyze_linguistic_patterns(texts: pd.Series) -> Dict[str, Any]:
    patterns = {}

    corpus = ' '.join(texts.fillna('').astype(str))

    punctuation_counts = {
        'exclamation_marks': corpus.count('!'),
        'question_marks': corpus.count('?'),
        'periods': corpus.count('.'),
        'commas': corpus.count(','),
        'semicolons': corpus.count(';'),
        'colons': corpus.count(':'),
        'quotation_marks': corpus.count('"') + corpus.count("'")
    }
    patterns['punctuation'] = punctuation_counts

    patterns['capitalization'] = {
        'all_caps_words': len(re.findall(r'\b[A-Z]{2,}\b', corpus)),
        'title_case_ratio': len(re.findall(r'\b[A-Z][a-z]+\b', corpus)) / max(len(corpus.split()), 1),
        'sentence_start_caps': len(re.findall(r'(?:^|[.!?]\s+)([A-Z])', corpus))
    }

    patterns['numbers'] = {
        'digit_count': len(re.findall(r'\d', corpus)),
        'number_sequences': len(re.findall(r'\d+', corpus)),
        'percentage_mentions': len(re.findall(r'\d+%', corpus)),
        'year_mentions': len(re.findall(r'\b(19|20)\d{2}\b', corpus))
    }

    patterns['special_chars'] = {
        'urls': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', corpus)),
        'mentions': len(re.findall(r'@\w+', corpus)),
        'hashtags': len(re.findall(r'#\w+', corpus)),
        'parentheses': corpus.count('(') + corpus.count(')')
    }

    return patterns

def calculate_readability_metrics(texts: pd.Series) -> Dict[str, float]:
    readability = {}

    sample_texts = texts.dropna().sample(min(100, len(texts))).tolist()

    flesch_scores = []
    fk_grades = []

    for text in sample_texts:
        text_str = str(text)
        if len(text_str.strip()) > 0:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    flesch_score = flesch_reading_ease(text_str)
                    fk_grade = flesch_kincaid_grade(text_str)

                    if not np.isnan(flesch_score):
                        flesch_scores.append(flesch_score)
                    if not np.isnan(fk_grade):
                        fk_grades.append(fk_grade)
            except:
                continue

    readability['flesch_reading_ease'] = {
        'mean': np.mean(flesch_scores) if flesch_scores else 0,
        'std': np.std(flesch_scores) if flesch_scores else 0
    }

    readability['flesch_kincaid_grade'] = {
        'mean': np.mean(fk_grades) if fk_grades else 0,
        'std': np.std(fk_grades) if fk_grades else 0
    }

    avg_words_per_sentence = []
    avg_chars_per_word = []

    for text in sample_texts:
        text_str = str(text).strip()
        if text_str:
            sentences = sent_tokenize(text_str)
            words = word_tokenize(text_str)

            if sentences and words:
                avg_words_per_sentence.append(len(words) / len(sentences))
                avg_chars_per_word.append(np.mean([len(word) for word in words if word.isalpha()]))

    readability['avg_words_per_sentence'] = np.mean(avg_words_per_sentence) if avg_words_per_sentence else 0
    readability['avg_chars_per_word'] = np.mean(avg_chars_per_word) if avg_chars_per_word else 0

    return readability

def find_common_patterns(texts: pd.Series, top_k: int = 20) -> Dict[str, List[Tuple[str, int]]]:
    patterns = {}

    all_texts = texts.fillna('').astype(str)
    corpus = ' '.join(all_texts)

    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()

    words = word_tokenize(corpus.lower())
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]

    word_freq = Counter(filtered_words)
    patterns['common_words'] = word_freq.most_common(top_k)

    bigrams = [(filtered_words[i], filtered_words[i+1]) for i in range(len(filtered_words)-1)]
    bigram_freq = Counter([' '.join(bigram) for bigram in bigrams])
    patterns['common_bigrams'] = bigram_freq.most_common(top_k)

    phrase_patterns = [
        r'\b(?:according to|based on|reports show|studies indicate)\b',
        r'\b(?:however|therefore|furthermore|moreover|nevertheless)\b',
        r'\b(?:believe|think|claim|argue|suggest)\b',
        r'\b(?:fact|truth|evidence|proof|data)\b',
        r'\b(?:false|fake|misleading|incorrect|wrong)\b'
    ]

    phrase_counts = {}
    for pattern in phrase_patterns:
        matches = re.findall(pattern, corpus, re.IGNORECASE)
        if matches:
            phrase_counts[pattern] = len(matches)

    patterns['common_phrases'] = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)

    return patterns

def analyze_label_distribution(df: pd.DataFrame, label_column: str = 'label') -> Dict[str, Any]:
    if label_column not in df.columns:
        return {'error': f'Column {label_column} not found in dataset'}

    analysis = {}

    label_counts = df[label_column].value_counts()
    analysis['label_counts'] = label_counts.to_dict()
    analysis['label_proportions'] = (label_counts / len(df)).to_dict()

    analysis['num_unique_labels'] = len(label_counts)
    analysis['most_common_label'] = label_counts.index[0]
    analysis['least_common_label'] = label_counts.index[-1]
    analysis['balance_ratio'] = label_counts.min() / label_counts.max()

    expected_freq = len(df) / len(label_counts)
    chi_square = sum((count - expected_freq) ** 2 / expected_freq for count in label_counts)
    analysis['imbalance_severity'] = chi_square

    if analysis['balance_ratio'] > 0.8:
        analysis['balance_class'] = 'Well-balanced'
    elif analysis['balance_ratio'] > 0.5:
        analysis['balance_class'] = 'Moderately imbalanced'
    else:
        analysis['balance_class'] = 'Highly imbalanced'

    return analysis

def detect_data_quality_issues(df: pd.DataFrame, text_columns: List[str]) -> Dict[str, Any]:
    issues = {}

    missing_data = df.isnull().sum()
    issues['missing_data'] = {
        'columns_with_missing': missing_data[missing_data > 0].to_dict(),
        'total_missing_values': missing_data.sum(),
        'rows_with_missing': df.isnull().any(axis=1).sum(),
        'percentage_complete': (1 - missing_data.sum() / (len(df) * len(df.columns))) * 100
    }

    duplicate_rows = df.duplicated().sum()
    issues['duplicates'] = {
        'duplicate_rows': duplicate_rows,
        'duplicate_percentage': (duplicate_rows / len(df)) * 100
    }

    text_issues = {}
    for col in text_columns:
        if col in df.columns:
            col_issues = {}
            texts = df[col].fillna('')

            empty_texts = (texts.str.len() <= 2).sum()
            col_issues['empty_or_very_short'] = empty_texts

            very_long_texts = (texts.str.len() > 1000).sum()
            col_issues['extremely_long'] = very_long_texts

            special_char_ratio = texts.str.count(r'[^a-zA-Z0-9\s\.,!?;:\'"()]').sum() / texts.str.len().sum()
            col_issues['high_special_char_ratio'] = special_char_ratio > 0.1

            repeated_patterns = 0
            for text in texts.sample(min(100, len(texts))):
                if isinstance(text, str) and len(text) > 10:
                    words = text.split()
                    if len(words) > 4:
                        word_counts = Counter(words)
                        max_repetition = max(word_counts.values()) if word_counts else 0
                        if max_repetition > len(words) * 0.3:
                            repeated_patterns += 1

            col_issues['potential_spam'] = repeated_patterns
            text_issues[col] = col_issues

    issues['text_quality'] = text_issues

    inconsistencies = {}

    if 'statement' in df.columns and 'label' in df.columns:
        text_label_groups = df.groupby('statement')['label'].nunique()
        inconsistent_texts = (text_label_groups > 1).sum()
        inconsistencies['label_inconsistencies'] = inconsistent_texts

    issues['inconsistencies'] = inconsistencies

    quality_score = 100

    quality_score -= min(20, issues['missing_data']['percentage_complete'] * 0.2)
    quality_score -= min(10, issues['duplicates']['duplicate_percentage'] * 0.5)

    for col_issues in text_issues.values():
        if col_issues.get('empty_or_very_short', 0) > len(df) * 0.05:
            quality_score -= 5
        if col_issues.get('potential_spam', 0) > 0:
            quality_score -= 5

    issues['overall_quality_score'] = max(0, quality_score)

    return issues

def preprocess_text_data(df: pd.DataFrame, text_columns: List[str],
                        operations: List[str] = None) -> pd.DataFrame:
    if operations is None:
        operations = ['lowercase', 'remove_extra_whitespace', 'handle_special_chars']

    df_processed = df.copy()

    for col in text_columns:
        if col in df_processed.columns:
            for operation in operations:
                if operation == 'lowercase':
                    df_processed[col] = df_processed[col].str.lower()
                elif operation == 'remove_extra_whitespace':
                    df_processed[col] = df_processed[col].str.replace(r'\s+', ' ', regex=True).str.strip()
                elif operation == 'handle_special_chars':
                    df_processed[col] = df_processed[col].str.replace(r'[^\w\s\.,!?;:\'"()-]', '', regex=True)
                elif operation == 'remove_urls':
                    df_processed[col] = df_processed[col].str.replace(
                        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                        '', regex=True
                    )
                elif operation == 'remove_mentions':
                    df_processed[col] = df_processed[col].str.replace(r'@\w+', '', regex=True)
                elif operation == 'remove_hashtags':
                    df_processed[col] = df_processed[col].str.replace(r'#\w+', '', regex=True)
                elif operation == 'expand_contractions':
                    contractions = {
                        "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
                        "'d": " would", "'m": " am", "can't": "cannot", "won't": "will not"
                    }
                    for contraction, expansion in contractions.items():
                        df_processed[col] = df_processed[col].str.replace(contraction, expansion, regex=False)

    return df_processed

def create_train_test_split(df: pd.DataFrame, test_size: float = 0.2,
                          stratify_column: str = None, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    if stratify_column and stratify_column in df.columns:
        stratify = df[stratify_column]
    else:
        stratify = None

    train_indices, test_indices = train_test_split(
        range(len(df)),
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    return train_df, test_df

class DatasetAnalyzer:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.analysis_cache = {}

    def analyze(self, include_advanced: bool = False) -> Dict[str, Any]:
        analysis = {}

        analysis['basic_stats'] = self._analyze_basic_stats()

        text_columns = self._identify_text_columns()
        if text_columns:
            analysis['text_analysis'] = {}
            for col in text_columns:
                analysis['text_analysis'][col] = analyze_text_patterns(self.df[col])

        label_columns = self._identify_label_columns()
        if label_columns:
            analysis['label_analysis'] = {}
            for col in label_columns:
                analysis['label_analysis'][col] = analyze_label_distribution(self.df, col)

        analysis['data_quality'] = detect_data_quality_issues(self.df, text_columns)

        if include_advanced:
            analysis['advanced'] = self._advanced_analysis(text_columns, label_columns)

        self.analysis_cache['comprehensive'] = analysis

        return analysis

    def _analyze_basic_stats(self) -> Dict[str, Any]:
        return {
            'num_rows': len(self.df),
            'num_columns': len(self.df.columns),
            'column_names': list(self.df.columns),
            'column_types': self.df.dtypes.to_dict(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        }

    def _identify_text_columns(self) -> List[str]:
        text_columns = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                sample_values = self.df[col].dropna().head(10)
                avg_length = sample_values.astype(str).str.len().mean()
                if avg_length > 10:
                    text_columns.append(col)
        return text_columns

    def _identify_label_columns(self) -> List[str]:
        label_keywords = ['label', 'class', 'target', 'truth', 'verdict']
        label_columns = []

        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in label_keywords):
                label_columns.append(col)
            elif self.df[col].dtype in ['object', 'category'] and self.df[col].nunique() < 20:
                label_columns.append(col)

        return label_columns

    def _advanced_analysis(self, text_columns: List[str], label_columns: List[str]) -> Dict[str, Any]:
        advanced = {}

        if len(text_columns) > 0 and len(label_columns) > 0:
            text_col = text_columns[0]
            label_col = label_columns[0]

            text_by_label = self.df.groupby(label_col)[text_col].agg([
                'count', lambda x: x.str.len().mean(), lambda x: x.str.len().std()
            ]).round(2)
            advanced['text_length_by_label'] = text_by_label.to_dict()

        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            correlation_matrix = self.df[numeric_columns].corr()
            advanced['numeric_correlations'] = correlation_matrix.to_dict()

        return advanced

    def generate_report(self, output_format: str = 'dict') -> Union[Dict[str, Any], str]:
        if 'comprehensive' not in self.analysis_cache:
            self.analyze(include_advanced=True)

        analysis = self.analysis_cache['comprehensive']

        if output_format == 'text':
            return self._format_text_report(analysis)
        else:
            return analysis

    def _format_text_report(self, analysis: Dict[str, Any]) -> str:
        report = []
        report.append("Dataset Analysis Report")
        report.append("=" * 50)

        basic = analysis['basic_stats']
        report.append(f"\nBasic Statistics:")
        report.append(f"  Rows: {basic['num_rows']:,}")
        report.append(f"  Columns: {basic['num_columns']}")
        report.append(f"  Memory Usage: {basic['memory_usage_mb']:.2f} MB")

        quality = analysis['data_quality']
        report.append(f"\nData Quality:")
        report.append(f"  Overall Score: {quality['overall_quality_score']:.1f}/100")
        report.append(f"  Missing Values: {quality['missing_data']['total_missing_values']}")
        report.append(f"  Duplicate Rows: {quality['duplicates']['duplicate_rows']}")

        if 'label_analysis' in analysis:
            report.append(f"\nLabel Analysis:")
            for col, label_info in analysis['label_analysis'].items():
                report.append(f"  {col}:")
                report.append(f"    Unique Labels: {label_info['num_unique_labels']}")
                report.append(f"    Balance: {label_info['balance_class']}")
                report.append(f"    Balance Ratio: {label_info['balance_ratio']:.3f}")

        return "\n".join(report)
